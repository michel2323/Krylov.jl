# An implementation of the Lanczos version of the conjugate gradient method.
#
# The implementation follows
# A. Frommer and P. Maass, Fast CG-Based Methods for Tikhonov-Phillips Regularization,
# SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
#
# C. C. Paige and M. A. Saunders, Solution of Sparse Indefinite Systems of Linear Equations,
# SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Princeton, NJ, March 2015.

export cg_lanczos, cg_lanczos!


"""
    (x, stats) = cg_lanczos(A, b::AbstractVector{FC};
                            M=I, atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                            check_curvature::Bool=false, verbose::Int=0, history::Bool=false)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

The Lanczos version of the conjugate gradient method to solve the
symmetric linear system

    Ax = b

The method does _not_ abort if A is not definite.

A preconditioner M may be provided in the form of a linear operator and is
assumed to be hermitian and positive definite.

#### References

* A. Frommer and P. Maass, [*Fast CG-Based Methods for Tikhonov-Phillips Regularization*](https://doi.org/10.1137/S1064827596313310), SIAM Journal on Scientific Computing, 20(5), pp. 1831--1850, 1999.
* C. C. Paige and M. A. Saunders, [*Solution of Sparse Indefinite Systems of Linear Equations*](https://doi.org/10.1137/0712047), SIAM Journal on Numerical Analysis, 12(4), pp. 617--629, 1975.
"""
function cg_lanczos end

function cg_lanczos(A, b :: AbstractVector{FC}; kwargs...) where FC <: FloatOrComplex
  solver = CgLanczosSolver(A, b)
  cg_lanczos!(solver, A, b; kwargs...)
  return (solver.x, solver.stats)
end

"""
    solver = cg_lanczos!(solver::CgLanczosSolver, A, b; kwargs...)

where `kwargs` are keyword arguments of [`cg_lanczos`](@ref).

See [`CgLanczosSolver`](@ref) for more details about the `solver`.
"""
function cg_lanczos! end

function cg_lanczos!(solver :: CgLanczosSolver{T,FC,S}, A, b :: AbstractVector{FC};
                     M=I, atol :: T=√eps(T), rtol :: T=√eps(T), itmax :: Int=0,
                     check_curvature :: Bool=false, verbose :: Int=0, history :: Bool=false) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: DenseVector{FC}}

  n, m = size(A)
  m == n || error("System must be square")
  length(b) == n || error("Inconsistent problem size")
  (verbose > 0) && @printf("CG Lanczos: system of %d equations in %d variables\n", n, n)

  # Tests M = Iₙ
  MisI = (M === I)

  # Check type consistency
  eltype(A) == FC || error("eltype(A) ≠ $T")
  ktypeof(b) == S || error("ktypeof(b) ≠ $S")

  # Set up workspace.
  allocate_if(!MisI, solver, :v, S, n)
  x, Mv, Mv_prev = solver.x, solver.Mv, solver.Mv_prev
  p, Mv_next, stats = solver.p, solver.Mv_next, solver.stats
  rNorms = stats.residuals
  reset!(stats)
  v = MisI ? Mv : solver.v

  # Initial state.
  x .= zero(FC)               # x₀
  Mv .= b                     # Mv₁ ← b
  MisI || mul!(v, M, Mv)      # v₁ = M⁻¹ * Mv₁
  β = sqrt(@kdotr(n, v, Mv))  # β₁ = v₁ᵀ M v₁
  σ = β
  rNorm = σ
  history && push!(rNorms, rNorm)
  if β == 0
    stats.niter = 0
    stats.solved = true
    stats.Anorm = zero(T)
    stats.indefinite = false
    stats.status = "x = 0 is a zero-residual solution"
    return solver
  end
  p .= v

  # Initialize Lanczos process.
  # β₁Mv₁ = b
  @kscal!(n, one(FC) / β, v)           # v₁  ←  v₁ / β₁
  MisI || @kscal!(n, one(FC) / β, Mv)  # Mv₁ ← Mv₁ / β₁
  Mv_prev .= Mv

  iter = 0
  itmax == 0 && (itmax = 2 * n)

  # Initialize some constants used in recursions below.
  ω = zero(T)
  γ = one(T)
  Anorm2 = zero(T)
  β_prev = zero(T)

  # Define stopping tolerance.
  ε = atol + rtol * rNorm
  (verbose > 0) && @printf("%5s  %7s\n", "k", "‖rₖ‖")
  kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)

  indefinite = false
  solved = rNorm ≤ ε
  tired = iter ≥ itmax
  status = "unknown"

  # Main loop.
  while ! (solved || tired || (check_curvature & indefinite))
    # Form next Lanczos vector.
    # βₖ₊₁Mvₖ₊₁ = Avₖ - δₖMvₖ - βₖMvₖ₋₁
    mul!(Mv_next, A, v)        # Mvₖ₊₁ ← Avₖ
    δ = @kdotr(n, v, Mv_next)  # δₖ = vₖᵀ A vₖ

    # Check curvature. Exit fast if requested.
    # It is possible to show that σₖ² (δₖ - ωₖ₋₁ / γₖ₋₁) = pₖᵀ A pₖ.
    γ = one(T) / (δ - ω / γ)  # γₖ = 1 / (δₖ - ωₖ₋₁ / γₖ₋₁)
    indefinite |= (γ ≤ 0)
    (check_curvature & indefinite) && continue

    @kaxpy!(n, -δ, Mv, Mv_next)        # Mvₖ₊₁ ← Mvₖ₊₁ - δₖMvₖ
    if iter > 0
      @kaxpy!(n, -β, Mv_prev, Mv_next) # Mvₖ₊₁ ← Mvₖ₊₁ - βₖMvₖ₋₁
      @. Mv_prev = Mv                  # Mvₖ₋₁ ← Mvₖ
    end
    @. Mv = Mv_next                      # Mvₖ ← Mvₖ₊₁
    MisI || mul!(v, M, Mv)               # vₖ₊₁ = M⁻¹ * Mvₖ₊₁
    β = sqrt(@kdotr(n, v, Mv))           # βₖ₊₁ = vₖ₊₁ᵀ M vₖ₊₁
    @kscal!(n, one(FC) / β, v)           # vₖ₊₁  ←  vₖ₊₁ / βₖ₊₁
    MisI || @kscal!(n, one(FC) / β, Mv)  # Mvₖ₊₁ ← Mvₖ₊₁ / βₖ₊₁
    Anorm2 += β_prev^2 + β^2 + δ^2       # Use ‖Tₖ₊₁‖₂ as increasing approximation of ‖A‖₂.
    β_prev = β

    # Compute next CG iterate.
    @kaxpy!(n, γ, p, x)     # xₖ₊₁ = xₖ + γₖ * pₖ
    ω = β * γ
    σ = -ω * σ              # σₖ₊₁ = - βₖ₊₁ * γₖ * σₖ
    ω = ω * ω               # ωₖ = (βₖ₊₁ * γₖ)²
    @kaxpby!(n, σ, v, ω, p) # pₖ₊₁ = σₖ₊₁ * vₖ₊₁ + ωₖ * pₖ
    rNorm = abs(σ)          # ‖rₖ₊₁‖_M = |σₖ₊₁| because rₖ₊₁ = σₖ₊₁ * vₖ₊₁ and ‖vₖ₊₁‖_M = 1
    history && push!(rNorms, rNorm)
    iter = iter + 1
    kdisplay(iter, verbose) && @printf("%5d  %7.1e\n", iter, rNorm)
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
  end
  (verbose > 0) && @printf("\n")

  status = tired ? "maximum number of iterations exceeded" : (check_curvature & indefinite) ? "negative curvature" : "solution good enough given atol and rtol"

  # Update stats. TODO: Estimate Acond.
  stats.niter = iter
  stats.solved = solved
  stats.Anorm = sqrt(Anorm2)
  stats.indefinite = indefinite
  stats.status = status
  return solver
end
