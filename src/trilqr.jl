# An implementation of TRILQR for the solution of square or
# rectangular consistent linear adjoint systems Ax = b and Aᵀy = c.
#
# This method is described in
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, July 2019.

export trilqr, trilqr!

"""
    (x, y, stats) = trilqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                           atol::T=√eps(T), rtol::T=√eps(T), transfer_to_usymcg::Bool=true,
                           itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Combine USYMLQ and USYMQR to solve adjoint systems.

    [0  A] [y] = [b]
    [Aᵀ 0] [x]   [c]

USYMLQ is used for solving primal system `Ax = b`.
USYMQR is used for solving dual system `Aᵀy = c`.

An option gives the possibility of transferring from the USYMLQ point to the
USYMCG point, when it exists. The transfer is based on the residual norm.

TriLQR can be warm-started from initial guesses `x0` and `y0` with the method

    (x, y, stats) = trilqr(A, b, c, x0, y0; kwargs...)

where `kwargs` are the same keyword arguments as above.

#### Reference

* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function trilqr end

function trilqr(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}, x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = TrilqrSolver(A, b)
  trilqr!(solver, A, b, c, x0, y0; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

function trilqr(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = TrilqrSolver(A, b)
  trilqr!(solver, A, b, c; kwargs...)
  return (solver.x, solver.y, solver.stats)
end

"""
    solver = trilqr!(solver::TrilqrSolver, A, b, c; kwargs...)
    solver = trilqr!(solver::TrilqrSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`trilqr`](@ref).

See [`TrilqrSolver`](@ref) for more details about the `solver`.
"""
function trilqr! end

function trilqr!(solver :: TrilqrSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC},
                x0 :: AbstractVector, y0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0, y0)
  trilqr!(solver, A, b, c; kwargs...)
  return solver
end

function trilqr!(solver :: TrilqrSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC};
                 atol :: T=√eps(T), rtol :: T=√eps(T), transfer_to_usymcg :: Bool=true,
                 itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("TRILQR: primal system of %d equations in %d variables\n", m, n)
  (verbose > 0) && @printf("TRILQR: dual system of %d equations in %d variables\n", n, m)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  uₖ₋₁, uₖ, p, d̅, x, stats = solver.uₖ₋₁, solver.uₖ, solver.p, solver.d̅, solver.x, solver.stats
  vₖ₋₁, vₖ, q, t, wₖ₋₃, wₖ₋₂ = solver.vₖ₋₁, solver.vₖ, solver.q, solver.y, solver.wₖ₋₃, solver.wₖ₋₂
  Δx, Δy, warm_start = solver.Δx, solver.Δy, solver.warm_start
  rNorms, sNorms = stats.residuals_primal, stats.residuals_dual
  reset!(stats)
  r₀ = warm_start ? q : b
  s₀ = warm_start ? p : c

  if warm_start
    mul!(r₀, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), r₀)
    mul!(s₀, Aᵀ, Δy)
    @kaxpby!(n, one(FC), c, -one(FC), s₀)
  end

  # Initial solution x₀ and residual r₀ = b - Ax₀.
  x .= zero(FC)          # x₀
  bNorm = @knrm2(m, r₀)  # rNorm = ‖r₀‖

  # Initial solution y₀ and residual s₀ = c - Aᵀy₀.
  t .= zero(FC)          # t₀
  cNorm = @knrm2(n, s₀)  # sNorm = ‖s₀‖

  iter = 0
  itmax == 0 && (itmax = m+n)

  history && push!(rNorms, bNorm)
  history && push!(sNorms, cNorm)
  εL = atol + rtol * bNorm
  εQ = atol + rtol * cNorm
  ξ = zero(T)
  (verbose > 0) && @printf("%5s  %7s  %7s\n", "k", "‖rₖ‖", "‖sₖ‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e\n", iter, bNorm, cNorm)

  # Set up workspace.
  βₖ = @knrm2(m, r₀)          # β₁ = ‖r₀‖ = ‖v₁‖
  γₖ = @knrm2(n, s₀)          # γ₁ = ‖s₀‖ = ‖u₁‖
  vₖ₋₁ .= zero(FC)            # v₀ = 0
  uₖ₋₁ .= zero(FC)            # u₀ = 0
  vₖ .= r₀ ./ βₖ              # v₁ = (b - Ax₀) / β₁
  uₖ .= s₀ ./ γₖ              # u₁ = (c - Aᵀy₀) / γ₁
  cₖ₋₁ = cₖ = -one(T)         # Givens cosines used for the LQ factorization of Tₖ
  sₖ₋₁ = sₖ = zero(FC)        # Givens sines used for the LQ factorization of Tₖ
  d̅ .= zero(FC)               # Last column of D̅ₖ = Uₖ(Qₖ)ᵀ
  ζₖ₋₁ = ζbarₖ = zero(FC)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
  ζₖ₋₂ = ηₖ = zero(FC)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
  δbarₖ₋₁ = δbarₖ = zero(FC)  # Coefficients of Lₖ₋₁ and L̅ₖ modified over the course of two iterations
  ψbarₖ₋₁ = ψₖ₋₁ = zero(FC)   # ψₖ₋₁ and ψbarₖ are the last components of h̅ₖ = Qₖγ₁e₁
  ϵₖ₋₃ = λₖ₋₂ = zero(FC)      # Components of Lₖ₋₁
  wₖ₋₃ .= zero(FC)            # Column k-3 of Wₖ = Vₖ(Lₖ)⁻ᵀ
  wₖ₋₂ .= zero(FC)            # Column k-2 of Wₖ = Vₖ(Lₖ)⁻ᵀ

  # Stopping criterion.
  inconsistent = false
  solved_lq = bNorm == 0
  solved_lq_tol = solved_lq_mach = false
  solved_cg = solved_cg_tol = solved_cg_mach = false
  solved_primal = solved_lq || solved_cg
  solved_qr_tol = solved_qr_mach = false
  solved_dual = cNorm == 0
  tired = iter ≥ itmax
  status = "unknown"

  while !((solved_primal && solved_dual) || tired)
    # Update iteration index.
    iter = iter + 1

    # Continue the SSY tridiagonalization process.
    # AUₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
    # AᵀVₖ = Uₖ(Tₖ)ᵀ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    mul!(q, A , uₖ)  # Forms vₖ₊₁ : q ← Auₖ
    mul!(p, Aᵀ, vₖ)  # Forms uₖ₊₁ : p ← Aᵀvₖ

    @kaxpy!(m, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
    @kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - βₖ * uₖ₋₁

    αₖ = @kdot(m, vₖ, q)      # αₖ = ⟨vₖ,q⟩

    @kaxpy!(m, -     αₖ , vₖ, q)    # q ← q - αₖ * vₖ
    @kaxpy!(n, -conj(αₖ), uₖ, p)    # p ← p - ᾱₖ * uₖ

    βₖ₊₁ = @knrm2(m, q)       # βₖ₊₁ = ‖q‖
    γₖ₊₁ = @knrm2(n, p)       # γₖ₊₁ = ‖p‖

    # Update the LQ factorization of Tₖ = L̅ₖQₖ.
    # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
    # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
    # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
    # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
    # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
    # [ •        •  •  •  γₖ]   [ •         •   •  λₖ₋₂ δₖ₋₁  0   ]
    # [ 0  •  •  •  0  βₖ αₖ]   [ •    •    •   0  ϵₖ₋₂ λₖ₋₁ δbarₖ]

    if iter == 1
      δbarₖ = αₖ
    elseif iter == 2
      # [δbar₁ γ₂] [c₂  s̄₂] = [δ₁   0  ]
      # [ β₂   α₂] [s₂ -c₂]   [λ₁ δbar₂]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
      λₖ₋₁  =      cₖ  * βₖ + sₖ * αₖ
      δbarₖ = conj(sₖ) * βₖ - cₖ * αₖ
    else
      # [0  βₖ  αₖ] [cₖ₋₁   s̄ₖ₋₁   0] = [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ]
      #             [sₖ₋₁  -cₖ₋₁   0]
      #             [ 0      0     1]
      #
      # [ λₖ₋₂   δbarₖ₋₁  γₖ] [1   0   0 ] = [λₖ₋₂  δₖ₋₁    0  ]
      # [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ] [0   cₖ  s̄ₖ]   [ϵₖ₋₂  λₖ₋₁  δbarₖ]
      #                       [0   sₖ -cₖ]
      (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
      ϵₖ₋₂  =  sₖ₋₁ * βₖ
      λₖ₋₁  = -cₖ₋₁ *      cₖ  * βₖ + sₖ * αₖ
      δbarₖ = -cₖ₋₁ * conj(sₖ) * βₖ - cₖ * αₖ
    end

    if !solved_primal
      # Compute ζₖ₋₁ and ζbarₖ, last components of the solution of L̅ₖz̅ₖ = β₁e₁
      # [δbar₁] [ζbar₁] = [β₁]
      if iter == 1
        ηₖ = βₖ
      end
      # [δ₁    0  ] [  ζ₁ ] = [β₁]
      # [λ₁  δbar₂] [ζbar₂]   [0 ]
      if iter == 2
        ηₖ₋₁ = ηₖ
        ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ηₖ   = -λₖ₋₁ * ζₖ₋₁
      end
      # [λₖ₋₂  δₖ₋₁    0  ] [ζₖ₋₂ ] = [0]
      # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ζₖ₋₁ ]   [0]
      #                     [ζbarₖ]
      if iter ≥ 3
        ζₖ₋₂ = ζₖ₋₁
        ηₖ₋₁ = ηₖ
        ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ηₖ   = -ϵₖ₋₂ * ζₖ₋₂ - λₖ₋₁ * ζₖ₋₁
      end

      # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Uₖ(Qₖ)ᵀ.
      # [d̅ₖ₋₁ uₖ] [cₖ  s̄ₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * uₖ
      #           [sₖ -cₖ]             ⟷ d̅ₖ   = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
      if iter ≥ 2
        # Compute solution xₖ.
        # (xᴸ)ₖ ← (xᴸ)ₖ₋₁ + ζₖ₋₁ * dₖ₋₁
        @kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
        @kaxpy!(n, ζₖ₋₁ * sₖ, uₖ, x)
      end

      # Compute d̅ₖ.
      if iter == 1
        # d̅₁ = u₁
        @. d̅ = uₖ
      else
        # d̅ₖ = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
        @kaxpby!(n, -cₖ, uₖ, conj(sₖ), d̅)
      end

      # Compute USYMLQ residual norm
      # ‖rₖ‖ = √(|μₖ|² + |ωₖ|²)
      if iter == 1
        rNorm_lq = bNorm
      else
        μₖ = βₖ * (sₖ₋₁ * ζₖ₋₂ - cₖ₋₁ * cₖ * ζₖ₋₁) + αₖ * sₖ * ζₖ₋₁
        ωₖ = βₖ₊₁ * sₖ * ζₖ₋₁
        rNorm_lq = sqrt(abs2(μₖ) + abs2(ωₖ))
      end
      history && push!(rNorms, rNorm_lq)

      # Compute USYMCG residual norm
      # ‖rₖ‖ = |ρₖ|
      if transfer_to_usymcg && (abs(δbarₖ) > eps(T))
        ζbarₖ = ηₖ / δbarₖ
        ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
        rNorm_cg = abs(ρₖ)
      end

      # Update primal stopping criterion
      solved_lq_tol = rNorm_lq ≤ εL
      solved_lq_mach = rNorm_lq + 1 ≤ 1
      solved_lq = solved_lq_tol || solved_lq_mach
      solved_cg_tol = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ εL)
      solved_cg_mach = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg + 1 ≤ 1)
      solved_cg = solved_cg_tol || solved_cg_mach
      solved_primal = solved_lq || solved_cg
    end

    if !solved_dual
      # Compute ψₖ₋₁ and ψbarₖ the last coefficients of h̅ₖ = Qₖγ₁e₁.
      if iter == 1
        ψbarₖ = γₖ
      else
        # [cₖ  s̄ₖ] [ψbarₖ₋₁] = [ ψₖ₋₁ ]
        # [sₖ -cₖ] [   0   ]   [ ψbarₖ]
        ψₖ₋₁  = cₖ * ψbarₖ₋₁
        ψbarₖ = sₖ * ψbarₖ₋₁
      end

      # Compute the direction wₖ₋₁, the last column of Wₖ₋₁ = (Vₖ₋₁)(Lₖ₋₁)⁻ᵀ ⟷ (L̄ₖ₋₁)(Wₖ₋₁)ᵀ = (Vₖ₋₁)ᵀ.
      # w₁ = v₁ / δ̄₁
      if iter == 2
        wₖ₋₁ = wₖ₋₂
        @kaxpy!(m, one(FC), vₖ₋₁, wₖ₋₁)
        @. wₖ₋₁ = vₖ₋₁ / conj(δₖ₋₁)
      end
      # w₂ = (v₂ - λ̄₁w₁) / δ̄₂
      if iter == 3
        wₖ₋₁ = wₖ₋₃
        @kaxpy!(m, one(FC), vₖ₋₁, wₖ₋₁)
        @kaxpy!(m, -conj(λₖ₋₂), wₖ₋₂, wₖ₋₁)
        @. wₖ₋₁ = wₖ₋₁ / conj(δₖ₋₁)
      end
      # wₖ₋₁ = (vₖ₋₁ - λ̄ₖ₋₂wₖ₋₂ - ϵ̄ₖ₋₃wₖ₋₃) / δ̄ₖ₋₁
      if iter ≥ 4
        @kscal!(m, -conj(ϵₖ₋₃), wₖ₋₃)
        wₖ₋₁ = wₖ₋₃
        @kaxpy!(m, one(FC), vₖ₋₁, wₖ₋₁)
        @kaxpy!(m, -conj(λₖ₋₂), wₖ₋₂, wₖ₋₁)
        @. wₖ₋₁ = wₖ₋₁ / conj(δₖ₋₁)
      end

      if iter ≥ 3
        # Swap pointers.
        @kswap(wₖ₋₃, wₖ₋₂)
      end

      if iter ≥ 2
        # Compute solution tₖ₋₁.
        # tₖ₋₁ ← tₖ₋₂ + ψₖ₋₁ * wₖ₋₁
        @kaxpy!(m, ψₖ₋₁, wₖ₋₁, t)
      end

      # Update ψbarₖ₋₁
      ψbarₖ₋₁ = ψbarₖ

      # Compute USYMQR residual norm ‖sₖ₋₁‖ = |ψbarₖ|.
      sNorm = abs(ψbarₖ)
      history && push!(sNorms, sNorm)

      # Compute ‖Asₖ₋₁‖ = |ψbarₖ| * √(|δbarₖ|² + |λbarₖ|²).
      AsNorm = abs(ψbarₖ) * √(abs2(δbarₖ) + abs2(cₖ * βₖ₊₁))

      # Update dual stopping criterion
      iter == 1 && (ξ = atol + rtol * AsNorm)
      solved_qr_tol = sNorm ≤ εQ
      solved_qr_mach = sNorm + 1 ≤ 1
      inconsistent = AsNorm ≤ ξ
      solved_dual = solved_qr_tol || solved_qr_mach || inconsistent
    end

    # Compute uₖ₊₁ and uₖ₊₁.
    @. vₖ₋₁ = vₖ  # vₖ₋₁ ← vₖ
    @. uₖ₋₁ = uₖ  # uₖ₋₁ ← uₖ

    if βₖ₊₁ ≠ zero(T)
      @. vₖ = q / βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
    end
    if γₖ₊₁ ≠ zero(T)
      @. uₖ = p / γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
    end

    # Update ϵₖ₋₃, λₖ₋₂, δbarₖ₋₁, cₖ₋₁, sₖ₋₁, γₖ and βₖ.
    if iter ≥ 3
      ϵₖ₋₃ = ϵₖ₋₂
    end
    if iter ≥ 2
      λₖ₋₂ = λₖ₋₁
    end
    δbarₖ₋₁ = δbarₖ
    cₖ₋₁    = cₖ
    sₖ₋₁    = sₖ
    γₖ      = γₖ₊₁
    βₖ      = βₖ₊₁

    tired = iter ≥ itmax

    kdisplay(iter, verbose) &&  solved_primal && !solved_dual && @printf("%5d  %7s  %7.1e\n", iter, "", sNorm)
    kdisplay(iter, verbose) && !solved_primal &&  solved_dual && @printf("%5d  %7.1e  %7s\n", iter, rNorm_lq, "")
    kdisplay(iter, verbose) && !solved_primal && !solved_dual && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm_lq, sNorm)
  end
  (verbose > 0) && @printf("\n")

  # Compute USYMCG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
  if solved_cg
    @kaxpy!(n, ζbarₖ, d̅, x)
  end

  tired                            && (status = "maximum number of iterations exceeded")
  solved_lq_tol  && !solved_dual   && (status = "Only the primal solution xᴸ is good enough given atol and rtol")
  solved_cg_tol  && !solved_dual   && (status = "Only the primal solution xᶜ is good enough given atol and rtol")
  !solved_primal && solved_qr_tol  && (status = "Only the dual solution t is good enough given atol and rtol")
  solved_lq_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᴸ, t) are good enough given atol and rtol")
  solved_cg_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᶜ, t) are good enough given atol and rtol")
  solved_lq_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᴸ")
  solved_cg_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᶜ")
  !solved_primal && solved_qr_mach && (status = "Only found approximate zero-residual dual solution t")
  solved_lq_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᴸ, t)")
  solved_cg_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᶜ, t)")
  solved_lq_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᴸ and a dual solution t good enough given atol and rtol")
  solved_cg_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᶜ and a dual solution t good enough given atol and rtol")
  solved_lq_tol  && solved_qr_mach && (status = "Found a primal solution xᴸ good enough given atol and rtol and an approximate zero-residual dual solutions t")
  solved_cg_tol  && solved_qr_mach && (status = "Found a primal solution xᶜ good enough given atol and rtol and an approximate zero-residual dual solutions t")

  # Update x and y
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  warm_start && @kaxpy!(m, one(FC), Δy, t)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.status = status
  stats.solved_primal = solved_primal
  stats.solved_dual = solved_dual
  return solver
end
