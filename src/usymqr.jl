# An implementation of USYMQR for the solution of linear system Ax = b.
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

export usymqr, usymqr!

"""
    (x, stats) = usymqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                        atol::T=√eps(T), rtol::T=√eps(T),
                        itmax::Int=0, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

Solve the linear system Ax = b using the USYMQR method.

USYMQR is based on the orthogonal tridiagonalization process and requires two initial nonzero vectors `b` and `c`.
The vector `c` is only used to initialize the process and a default value can be `b` or `Aᵀb` depending on the shape of `A`.
The residual norm ‖b - Ax‖ monotonously decreases in USYMQR.
It's considered as a generalization of MINRES.

It can also be applied to under-determined and over-determined problems.
USYMQR finds the minimum-norm solution if problems are inconsistent.

USYMQR can be warm-started from an initial guess `x0` with the method

    (x, stats) = usymqr(A, b, c, x0; kwargs...)

where `kwargs` are the same keyword arguments as above.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function usymqr end

function usymqr(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}, x0 :: AbstractVector; kwargs...) where FC <: FloatOrComplex
  solver = UsymqrSolver(A, b)
  usymqr!(solver, A, b, c, x0; kwargs...)
  return (solver.x, solver.stats)
end

function usymqr(A, b :: AbstractVector{FC}, c :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = UsymqrSolver(A, b)
  usymqr!(solver, A, b, c; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = usymqr!(solver::UsymqrSolver, A, b, c; kwargs...)
    solver = usymqr!(solver::UsymqrSolver, A, b, c, x0; kwargs...)

where `kwargs` are keyword arguments of [`usymqr`](@ref).

See [`UsymqrSolver`](@ref) for more details about the `solver`.
"""
function usymqr! end

function usymqr!(solver :: UsymqrSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC},
                 x0 :: AbstractVector; kwargs...) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}
  warm_start!(solver, x0)
  usymqr!(solver, A, b, c; kwargs...)
  return solver
end

function usymqr!(solver :: UsymqrSolver{T,FC,S}, A, b :: AbstractVector{FC}, c :: AbstractVector{FC};
                 atol :: T=√eps(T), rtol :: T=√eps(T),
                 itmax :: Int=0, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  m, n = size(A)
  length(b) == m || error("Inconsistent problem size")
  length(c) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("USYMQR: system of %d equations in %d variables\n", m, n)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $FC")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")
  ktypeof(c) == S || error("ktypeof(c) ≠ $S")

  # Compute the adjoint of A
  Aᵀ = A'

  # Set up workspace.
  vₖ₋₁, vₖ, q, Δx, x, p = solver.vₖ₋₁, solver.vₖ, solver.q, solver.Δx, solver.x, solver.p
  wₖ₋₂, wₖ₋₁, uₖ₋₁, uₖ, stats = solver.wₖ₋₂, solver.wₖ₋₁, solver.uₖ₋₁, solver.uₖ, solver.stats
  warm_start = solver.warm_start
  rNorms, AᵀrNorms = stats.residuals, stats.Aresiduals
  reset!(stats)
  r₀ = warm_start ? q : b

  if warm_start
    mul!(r₀, A, Δx)
    @kaxpby!(n, one(FC), b, -one(FC), r₀)
  end

  # Initial solution x₀ and residual norm ‖r₀‖.
  x .= zero(FC)
  rNorm = @knrm2(m, r₀)
  history && push!(rNorms, rNorm)
  if rNorm == 0
    stats.niter = 0
    stats.solved = true
    stats.inconsistent = false
    stats.status = "x = 0 is a zero-residual solution"
    solver.warm_start = false
    return solver
  end

  iter = 0
  itmax == 0 && (itmax = m+n)

  ε = atol + rtol * rNorm
  κ = zero(T)
  (verbose > 0) && @printf("%5s  %7s  %7s\n", "k", "‖rₖ‖", "‖Aᵀrₖ₋₁‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7s\n", iter, rNorm, "✗ ✗ ✗ ✗")

  βₖ = @knrm2(m, r₀)           # β₁ = ‖v₁‖ = ‖r₀‖
  γₖ = @knrm2(n, c)            # γ₁ = ‖u₁‖ = ‖c‖
  vₖ₋₁ .= zero(FC)             # v₀ = 0
  uₖ₋₁ .= zero(FC)             # u₀ = 0
  vₖ .= r₀ ./ βₖ               # v₁ = (b - Ax₀) / β₁
  uₖ .= c ./ γₖ                # u₁ = c / γ₁
  cₖ₋₂ = cₖ₋₁ = cₖ = one(T)    # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
  sₖ₋₂ = sₖ₋₁ = sₖ = zero(FC)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
  wₖ₋₂ .= zero(FC)             # Column k-2 of Wₖ = Uₖ(Rₖ)⁻¹
  wₖ₋₁ .= zero(FC)             # Column k-1 of Wₖ = Uₖ(Rₖ)⁻¹
  ζbarₖ = βₖ                   # ζbarₖ is the last component of z̅ₖ = (Qₖ)ᵀβ₁e₁

  # Stopping criterion.
  solved = rNorm ≤ ε
  inconsistent = false
  tired = iter ≥ itmax
  status = "unknown"

  while !(solved || tired || inconsistent)
    # Update iteration index.
    iter = iter + 1

    # Continue the SSY tridiagonalization process.
    # AUₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
    # AᵀVₖ = Uₖ(Tₖ)ᵀ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᵀ

    mul!(q, A , uₖ)  # Forms vₖ₊₁ : q ← Auₖ
    mul!(p, Aᵀ, vₖ)  # Forms uₖ₊₁ : p ← Aᵀvₖ

    @kaxpy!(m, -γₖ, vₖ₋₁, q) # q ← q - γₖ * vₖ₋₁
    @kaxpy!(n, -βₖ, uₖ₋₁, p) # p ← p - βₖ * uₖ₋₁

    αₖ = @kdot(m, vₖ, q)     # αₖ = ⟨vₖ,q⟩

    @kaxpy!(m, -     αₖ , vₖ, q)   # q ← q - αₖ * vₖ
    @kaxpy!(n, -conj(αₖ), uₖ, p)   # p ← p - ᾱₖ * uₖ

    βₖ₊₁ = @knrm2(m, q)      # βₖ₊₁ = ‖q‖
    γₖ₊₁ = @knrm2(n, p)      # γₖ₊₁ = ‖p‖

    # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
    #                                            [ Oᵀ ]
    # [ α₁ γ₂ 0  •  •  •   0  ]      [ δ₁ λ₁ ϵ₁ 0  •  •  0  ]
    # [ β₂ α₂ γ₃ •         •  ]      [ 0  δ₂ λ₂ •  •     •  ]
    # [ 0  •  •  •  •      •  ]      [ •  •  δ₃ •  •  •  •  ]
    # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
    # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
    # [ •        •  •  •   γₖ ]      [ •           •  • λₖ₋₁]
    # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  δₖ ]
    # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
    #
    # If k = 1, we don't have any previous reflexion.
    # If k = 2, we apply the last reflexion.
    # If k ≥ 3, we only apply the two previous reflexions.

    # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
    if iter ≥ 3
      # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
      # [s̄ₖ₋₂ -cₖ₋₂] [γₖ]   [λbarₖ₋₁]
      ϵₖ₋₂    =  sₖ₋₂ * γₖ
      λbarₖ₋₁ = -cₖ₋₂ * γₖ
    end

    # Apply previous Givens reflections Qₖ₋₁.ₖ
    if iter ≥ 2
      iter == 2 && (λbarₖ₋₁ = γₖ)
      # [cₖ₋₁  sₖ₋₁] [λbarₖ₋₁] = [λₖ₋₁ ]
      # [s̄ₖ₋₁ -cₖ₋₁] [   αₖ  ]   [δbarₖ]
      λₖ₋₁  =      cₖ₋₁  * λbarₖ₋₁ + sₖ₋₁ * αₖ
      δbarₖ = conj(sₖ₋₁) * λbarₖ₋₁ - cₖ₋₁ * αₖ
    end

    # Compute and apply current Givens reflection Qₖ.ₖ₊₁
    iter == 1 && (δbarₖ = αₖ)
    # [cₖ  sₖ] [δbarₖ] = [δₖ]
    # [s̄ₖ -cₖ] [βₖ₊₁ ]   [0 ]
    (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

    # Update z̅ₖ₊₁ = Qₖ.ₖ₊₁ [ z̄ₖ ]
    #                      [ 0  ]
    #
    # [cₖ  sₖ] [ζbarₖ] = [   ζₖ  ]
    # [s̄ₖ -cₖ] [  0  ]   [ζbarₖ₊₁]
    ζₖ      =      cₖ  * ζbarₖ
    ζbarₖ₊₁ = conj(sₖ) * ζbarₖ

    # Compute the direction wₖ, the last column of Wₖ = Uₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Uₖ)ᵀ.
    # w₁ = u₁ / δ₁
    if iter == 1
      wₖ = wₖ₋₁
      @kaxpy!(n, one(FC), uₖ, wₖ)
      @. wₖ = wₖ / δₖ
    end
    # w₂ = (u₂ - λ₁w₁) / δ₂
    if iter == 2
      wₖ = wₖ₋₂
      @kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
      @kaxpy!(n, one(FC), uₖ, wₖ)
      @. wₖ = wₖ / δₖ
    end
    # wₖ = (uₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
    if iter ≥ 3
      @kscal!(n, -ϵₖ₋₂, wₖ₋₂)
      wₖ = wₖ₋₂
      @kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
      @kaxpy!(n, one(FC), uₖ, wₖ)
      @. wₖ = wₖ / δₖ
    end

    # Compute solution xₖ.
    # xₖ ← xₖ₋₁ + ζₖ * wₖ
    @kaxpy!(n, ζₖ, wₖ, x)

    # Compute ‖rₖ‖ = |ζbarₖ₊₁|.
    rNorm = abs(ζbarₖ₊₁)
    history && push!(rNorms, rNorm)

    # Compute ‖Aᵀrₖ₋₁‖ = |ζbarₖ| * √(|δbarₖ|² + |λbarₖ|²).
    AᵀrNorm = abs(ζbarₖ) * √(abs2(δbarₖ) + abs2(cₖ₋₁ * γₖ₊₁))
    history && push!(AᵀrNorms, AᵀrNorm)

    # Compute uₖ₊₁ and uₖ₊₁.
    @. vₖ₋₁ = vₖ # vₖ₋₁ ← vₖ
    @. uₖ₋₁ = uₖ # uₖ₋₁ ← uₖ

    if βₖ₊₁ ≠ zero(T)
      @. vₖ = q / βₖ₊₁ # βₖ₊₁vₖ₊₁ = q
    end
    if γₖ₊₁ ≠ zero(T)
      @. uₖ = p / γₖ₊₁ # γₖ₊₁uₖ₊₁ = p
    end

    # Update directions for x.
    if iter ≥ 2
      @kswap(wₖ₋₂, wₖ₋₁)
    end

    # Update sₖ₋₂, cₖ₋₂, sₖ₋₁, cₖ₋₁, ζbarₖ, γₖ, βₖ.
    if iter ≥ 2
      sₖ₋₂ = sₖ₋₁
      cₖ₋₂ = cₖ₋₁
    end
    sₖ₋₁  = sₖ
    cₖ₋₁  = cₖ
    ζbarₖ = ζbarₖ₊₁
    γₖ    = γₖ₊₁
    βₖ    = βₖ₊₁

    # Update stopping criterion.
    iter == 1 && (κ = atol + rtol * AᵀrNorm)
    solved = rNorm ≤ ε
    inconsistent = !solved && AᵀrNorm ≤ κ
    tired = iter ≥ itmax
    kdisplay(iter, verbose) && @printf("%5d  %7.1e  %7.1e\n", iter, rNorm, AᵀrNorm)
  end
  (verbose > 0) && @printf("\n")
  status = tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol"

  # Update x
  warm_start && @kaxpy!(n, one(FC), Δx, x)
  solver.warm_start = false

  # Update stats
  stats.niter = iter
  stats.solved = solved
  stats.inconsistent = inconsistent
  stats.status = status
  return solver
end
