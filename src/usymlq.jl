# An implementation of USYMLQ for the solution of linear system Ax = b.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point and quasi-definite systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, November 2018.

export usymlq, usymlq!

"""
    (x, stats) = usymlq(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                        atol::T=√eps(T), rtol::T=√eps(T), transfer_to_usymcg::Bool=true,
                        itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the linear system Ax = b using the USYMLQ method.

USYMLQ is based on the orthogonal tridiagonalization process and requires two initial nonzero vectors `b` and `c`.
The vector `c` is only used to initialize the process and a default value can be `b` or `Aᵀb` depending on the shape of `A`.
The error norm ‖x - x*‖ monotonously decreases in USYMLQ.
It's considered as a generalization of SYMMLQ.

It can also be applied to under-determined and over-determined problems.
In all cases, problems must be consistent.

An option gives the possibility of transferring to the USYMCG point,
when it exists. The transfer is based on the residual norm.

USYMLQ can be warm-started from an initial guess `x0` with the method

    (x, stats) = usymlq(A, b, c, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function usymlq end

function usymlq(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = UsymlqSolver(A, b)
  usymlq!(solver, A, b, c, x0; kwargs...)
  return (solver.x, solver.stats)
end

function usymlq(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = UsymlqSolver(A, b)
  usymlq!(solver, A, b, c; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = usymlq!(solver::UsymlqSolver, A, b, c; kwargs...)
    solver = usymlq!(solver::UsymlqSolver, A, b, c, x0; kwargs...)

where `kwargs` are keyword arguments of [`usymlq`](@ref).

See [`UsymlqSolver`](@ref) for more details about the `solver`.
"""
function usymlq! end

function usymlq!(solver :: UsymlqSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC},
                 x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  usymlq!(solver, A, b, c; kwargs...)
  return solver
end

function usymlq!(solver :: UsymlqSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC};
                 atol :: T=√eps(T), rtol :: T=√eps(T), transfer_to_usymcg :: Bool=true,
                 itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("USYMLQ: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  uₖ₋₁, uₖ, p, Δx, x = solver.uₖ₋₁, solver.uₖ, solver.p, solver.Δx, solver.x
  vₖ₋₁, vₖ, q, d̅, stats = solver.vₖ₋₁, solver.vₖ, solver.q, solver.d̅, solver.stats
  warm_start = solver.warm_start
  rNorms = stats.residuals
  reset!(stats)
  r₀ = warm_start ? q : b

  if warm_start
    mul!(r₀, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), r₀)
  end

  # Initial solution x₀ and residual norm ‖r₀‖.
  x .= zero(FC)
  bNorm = @knrm2(m, r₀)
  history && push!(rNorms, bNorm)
  if bNorm == 0
    stats.niter = 0
    stats.solved = true
    stats.inconsistent = false
    stats.status = "x = 0 is a zero-residual solution"
    solver.warm_start = false
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = m+n)

  ε = atol + rtol * bNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, bNorm)

  βₖ = @knrm2(m, r₀)          # β₁ = ‖v₁‖ = ‖r₀‖
  γₖ = @knrm2(n, c)           # γ₁ = ‖u₁‖ = ‖c‖
  vₖ₋₁ .= zero(FC)            # v₀ = 0
  uₖ₋₁ .= zero(FC)            # u₀ = 0
  vₖ .= r₀ ./ βₖ              # v₁ = (b - Ax₀) / β₁
  uₖ .= c ./ γₖ               # u₁ = c / γ₁
  cₖ₋₁ = cₖ = -one(T)         # Givens cosines used for the LQ factorization of Tₖ
  sₖ₋₁ = sₖ = zero(FC)        # Givens sines used for the LQ factorization of Tₖ
  d̅ .= zero(FC)               # Last column of D̅ₖ = Uₖ(Qₖ)ᵀ
  ζₖ₋₁ = ζbarₖ = zero(FC)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
  ζₖ₋₂ = ηₖ = zero(FC)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
  δbarₖ₋₁ = δbarₖ = zero(FC)  # Coefficients of Lₖ₋₁ and Lₖ modified over the course of two iterations

  # Stopping criterion.
  solved_lq = bNorm ≤ ε
  solved_cg = false
  tired     = iter ≥ itmax
  status    = "unknown"

  while !(solved_lq || solved_cg || tired)
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
    # [ •        •  •  •  γₖ]   [ •         •   •   •    •    0   ]
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
      # (xᴸ)ₖ₋₁ ← (xᴸ)ₖ₋₂ + ζₖ₋₁ * dₖ₋₁
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

    # Compute uₖ₊₁ and uₖ₊₁.
    @. vₖ₋₁ = vₖ  # vₖ₋₁ ← vₖ
    @. uₖ₋₁ = uₖ  # uₖ₋₁ ← uₖ

    if βₖ₊₁ ≠ zero(T)
      @. vₖ = q / βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
    end
    if γₖ₊₁ ≠ zero(T)
      @. uₖ = p / γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
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

    # Update sₖ₋₁, cₖ₋₁, γₖ, βₖ and δbarₖ₋₁.
    sₖ₋₁    = sₖ
    cₖ₋₁    = cₖ
    γₖ      = γₖ₊₁
    βₖ      = βₖ₊₁
    δbarₖ₋₁ = δbarₖ

    # Update stopping criterion.
    solved_lq = rNorm_lq ≤ ε
    solved_cg = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ ε)
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm_lq)
  end
  (verbose > 0) && @printf("\n")

  # Compute USYMCG point
  # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
  if solved_cg
    @kaxpy!(n, ζbarₖ, d̅, x)
  end

  tired     && (status = "maximum number of iterations exceeded")
  solved_lq && (status = "solution xᴸ good enough given atol and rtol")
  solved_cg && (status = "solution xᶜ good enough given atol and rtol")

  # Update x
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved_lq || solved_cg
  stats.inconsistent = false
  stats.status = status
  return solver
end
