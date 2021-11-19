"""    (x, flags, stats) = usymlqr(A, b, c)

Solve the symmetric saddle-point system

    [ I  A ] [ s ] = [ b ]
    [ A'   ] [ t ]   [ c ]

by way of the Saunders-Simon-Yip tridiagonalization using the USYMQR and USYMLQ methods.
The method solves the least-squares problem

    [ I  A ] [ r ] = [ b ]
    [ A'   ] [ x ]   [ 0 ]

and the least-norm problem

    [ I  A ] [ y ] = [ 0 ]
    [ A'   ] [ z ]   [ c ]

and simply adds the solutions.

M. A. Saunders, H. D. Simon and E. L. Yip
Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations
SIAM Journal on Numerical Analysis, 25(4), 927-940, 1988.
"""
function usymlqr(A, b::Vector{Float64}, c::Vector{Float64};
                 M :: AbstractLinearOperator=opEye(),
                 N :: AbstractLinearOperator=opEye(),
                 itnlim::Int=maximum(size(A)),
                 atol_ls::Float64=1.0e-6, rtol_ls::Float64=1.0e-6,
                 atol_ln::Float64=1.0e-6, rtol_ln::Float64=1.0e-6,
                 sigma::Float64=0.0, conlim::Float64=1.0e+8)

  m, n = size(A)
  if length(b) ≠ m || length(c) ≠ n
    error("USYMLQR: Dimensions mismatch")
  end

  # Tests M == Iₙ and N == Iₘ
  MisI = isa(M, opEye)
  NisI = isa(N, opEye)

  # compute initial vectors
  Mu = copy(b)
  u = M * Mu
  Nv = copy(c)
  v = N * Nv

  # Exit fast if b or c is zero.
  beta1 = sqrt(@kdot(m, u, Mu)) # β₁ = ‖u₁‖_M
  beta1 > 0.0 || error("USYMLQR: b must be nonzero.")
  gamma1 = sqrt(@kdot(n, v, Nv)) # γ₁ = ‖v₁‖_N
  gamma1 > 0.0 || error("USYMLQR: c must be nonzero.")
  iter = 0
  ctol = conlim > 0.0 ? 1/conlim : 0.0

  ls_zero_resid_tol = atol_ls + rtol_ls * beta1
  ls_optimality_tol = atol_ls + rtol_ls * norm(A' * b)  # FIXME
  ln_tol = atol_ln + rtol_ln * gamma1

  @info "" ls_optimality_tol ln_tol

  @info @sprintf("USYMLQR with %d rows and %d columns", m, n)

  # Initial SSY vectors.
  u_prev = fill!(similar(b), 0)   #  u₀ = 0
  Mu_prev = fill!(similar(b), 0)  # Mu₀ = 0
  v_prev = fill!(similar(c), 0)   #  v₀ = 0
  Nv_prev = fill!(similar(c), 0)  # Nv₀ = 0

  # normalize initial vectors
  @. u /= beta1                   # β₁Mu₁ = b
  MisI || (@. Mu /= beta1)
  @. v /= gamma1                  # γ₁Nv₁ = c
  NisI || (@. Nv /= gamma1)

  q = A * v
  alpha = dot(u, q)  # alpha₁

  Nvv = copy(Nv)
  vv = copy(v)
  beta = beta1
  gamma = gamma1

  @debug "" u v

  # initial norm estimates
  Anorm2 = alpha * alpha + sigma * sigma
  Anorm = sqrt(Anorm2)
  sigma_min = sigma_max = abs(alpha)  # extreme singular values estimates.
  Acond = 1.0

  # initial residual of least-squares problem
  phibar = beta1
  rNorm_qr = phibar
  rNorms_qr = [rNorm_qr]
  ArNorm_qr = 0.0  # just so it exists at the end of the loop!
  ArNorms_qr = Float64[]

  # initialization for QR factorization of T{k+1,k}
  cs = -1.0
  sn = 0.0
  deltabar = alpha
  lambda = 0.0
  epsilon = 0.0
  eta = 0.0
  @info @sprintf("%4s %8s %7s %7s %7s %7s %7s %7s %7s\n",
                 "iter", "alpha", "beta", "gamma", "‖A‖", "κ(A)", "‖Ax-b‖", "‖A'r‖", "‖A'y-c‖")
  infoline = @sprintf("%4d %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e ", iter, alpha, beta, gamma, Anorm, Acond, rNorm_qr)

  # initialize x and z update directions
  x = fill!(similar(c), 0)
  xNorm = 0.0
  z = fill!(similar(c), 0)
  wbar = v / deltabar
  w = fill!(similar(v), 0)
  wold = fill!(similar(v), 0)
  Wnorm2 = 0.0

  # quantities related to the update of y
  etabar = gamma / deltabar
  p = fill!(similar(u), 0)
  pbar = copy(u)
  y = fill!(similar(b), 0)
  yC = etabar * pbar
  zC = -etabar * wbar
  @debug "‖A * zC + yC‖" norm(A * zC + yC)
  yNorm2 = 0.0
  yNorm = 0.0

  @debug "" y

  # quantities related to the computation of ‖x‖
  # TODO

  # quantities related to regularization
  if sigma ≠ 0
    sigmak = sigma
    psibar = 0.0
    gNorm2 = 0.0
  end

  # residual of the least-norm problem
  rNorm_lq = 2 * ln_tol  # just so it exists at the end of the loop!
  rNorms_lq = Float64[]

  status = "unknown"
  transition_to_cg = false

  # stopping conditions that apply to both problems
  tired  = iter ≥ itnlim
  ill_cond_lim = 1/Acond ≤ ctol
  ill_cond_mach = 1.0 + 1/Acond ≤ 1.0
  ill_cond = ill_cond_mach | ill_cond_lim

  # stopping conditions related to the least-squares problem
  test_LS = rNorm_qr / (1.0 + Anorm * xNorm)
  zero_resid_lim_LS = test_LS ≤ ls_zero_resid_tol
  zero_resid_mach_LS = 1.0 + test_LS ≤ 1.0
  zero_resid_LS = zero_resid_mach_LS | zero_resid_lim_LS
  test_LS = ArNorm_qr / (Anorm * max(1.0, rNorm_qr))
  solved_lim_LS = test_LS ≤ ls_optimality_tol
  solved_mach_LS = 1.0 + test_LS ≤ 1.0
  # TODO: check this
  solved_LS = false  # solved_mach_LS | solved_lim_LS | zero_resid_LS

  # stopping conditions related to the least-norm problem
  test_LN = rNorm_lq / sqrt(gamma1^2 + Anorm2 * yNorm2)
  solved_lim_LN = test_LN ≤ ln_tol
  solved_mach_LN = 1.0 + test_LN ≤ 1.0
  # TODO: check this
  solved_LN = false # solved_lim_LN | solved_mach_LN

  solved = solved_LS & solved_LN

  # TODO: remove this when finished
  tests_LS = Float64[]
  tests_LN = Float64[]

  while ! (solved | tired | ill_cond)

    iter = iter + 1

    # continue tridiagonalization
    @. Mu_prev = Mu
    @. u_prev = u

    @. Mu = q - alpha * Mu
    Atuprev = A' * u_prev
    @. Nv = Atuprev - alpha * Nv - beta * Nv_prev  # this is the previous beta

    # compute preconditioned basis vectors
    u = M * Mu
    v = N * Nv

    # normalize basis vectors
    beta = sqrt(@kdot(m, u, Mu))
    if beta > 0
      @. u /= beta
      MisI || (@. Mu /= beta)
    end
    gamma = sqrt(@kdot(n, v, Nv))
    if gamma > 0
      @. v /= gamma
      NisI || (@. Nv /= gamma)
    end

    @debug "" alpha gamma beta
    @debug "" u v

    # save vectors for next iteration (TODO: can we get rid of one of them?)
    @. Nv_prev = Nvv
    @. v_prev = vv
    @. Nvv = Nv
    @. vv = v

    # update norm estimates
    Anorm2 += beta * beta + gamma * gamma
    Anorm = sqrt(Anorm2)

    # continue QR factorization of T{k+1,k}
    lambdabar = -cs * gamma  # = gamma2 = lambdabar1 at the first pass
    epsilon = sn * gamma     # = 0 = epsilon0 at the first pass

    @debug "" lambdabar epsilon

    # compute optimality residual of least-squares problem at x{k-1}
    # TODO: use recurrence formula for QR residual
    if !solved_LS
      if sigma ≠ 0
        ArNorm_qr_computed = sqrt(phibar^2 * (deltabar^2 + lambdabar^2) + sigma^4 * xNorm^2)
      else
        ArNorm_qr_computed = abs(phibar) * sqrt(deltabar^2 + lambdabar^2)
      end
      ArNorm_qr = norm(A' * (b - A * x) + sigma^2 * x)  # FIXME
      @debug "" ArNorm_qr_computed ArNorm_qr abs(ArNorm_qr_computed - ArNorm_qr) / ArNorm_qr
      ArNorm_qr = ArNorm_qr_computed
      push!(ArNorms_qr, ArNorm_qr)

      test_LS = ArNorm_qr / (Anorm * max(1.0, rNorm_qr))
      solved_lim_LS = test_LS ≤ ls_optimality_tol
      solved_mach_LS = 1.0 + test_LS ≤ 1.0
      solved_LS = solved_mach_LS | solved_lim_LS

      # TODO: remove this when finished
      push!(tests_LS, test_LS)

      if solved_LS
        @info "solved LS problem with" x
      end
    end
    infoline *= @sprintf("%7.1e ", ArNorm_qr)

    # perform rotations related to regularization if applicable
    if sigma ≠ 0.0
      # first rotation
      olddeltabar = deltabar
      deltabar = sqrt(olddeltabar^2 + sigmak^2)
      cbar = olddeltabar / deltabar
      sbar = sigmak / deltabar
      sigmahat = sbar * lambdabar
      lambdabar = cbar * lambdabar
      psitilde = sbar * phibar - cbar * psibar
      phibar = cbar * phibar + sbar * psibar  # overwrite phibar instead of using a separate variable

      # second rotation
      sigmak = sqrt(sigmahat^2 + sigma^2)
      ctilde = -sigma / sigmak
      stilde = sigmahat / sigmak
      psi = ctilde * psitilde
      psibar = stilde * psitilde
      gNorm2 += psi^2
    end

    # continue QR factorization
    delta = sqrt(deltabar^2 + beta^2)
    csold = cs  # used later to compute the residual at yCG
    snold = sn
    cs = deltabar/ delta
    sn = beta / delta

    # update w (used to update x and z)
    @. wold = w
    @. w = cs * wbar

    if !solved_LS
      # the optimality conditions of the LS problem were not triggerred
      # update x and see if we have a zero residual

      phi = cs * phibar
      phibar = sn * phibar
      @. x += phi * w
      xNorm = norm(x)  # FIXME

      @debug "" x xNorm w / norm(w)

      # update least-squares residual
      if sigma ≠ 0
        rNorm_qr_computed = sqrt(phibar^2 + gNorm2 + psibar^2)
        # TODO: this should = sqrt(phibar^2 + sigma^2 xNorm^2). Is this a quick way to compute xNorm?
      else
        rNorm_qr_computed = abs(phibar)
      end
      rNorm_qr = norm([b - A * x ; sigma * x])  # FIXME
      @debug "" rNorm_qr_computed rNorm_qr abs(rNorm_qr_computed - rNorm_qr) / rNorm_qr
      rNorm_qr = rNorm_qr_computed
      push!(rNorms_qr, rNorm_qr)

      # stopping conditions related to the least-squares problem
      test_LS = rNorm_qr / (1.0 + Anorm * xNorm)
      zero_resid_lim_LS = test_LS ≤ ls_zero_resid_tol
      zero_resid_mach_LS = 1.0 + test_LS ≤ 1.0
      zero_resid_LS = zero_resid_mach_LS | zero_resid_lim_LS
      solved_LS |= zero_resid_LS

      if zero_resid_LS
        @info "solved LS problem to zero residual with" x
      end
    end

    # continue tridiagonalization
    q = A * v
    @. q -= gamma * Mu_prev
    alpha = dot(u, q)

    # update norm estimates
    Anorm2 += alpha * alpha + sigma * sigma
    Anorm = sqrt(Anorm2)
    # Wnorm2 += dot(w, w)
    # Acond = Anorm * sqrt(Wnorm2)
    # Estimate κ₂(A) based on the diagonal of L.
    sigma_min = min(delta, sigma_min)
    sigma_max = max(delta, sigma_max)
    # @info "" sigma_min sigma_max
    Acond = sigma_max / sigma_min

    # continue QR factorization of T{k+1,k}
    lambda = cs * lambdabar + sn * alpha
    deltabar = sn * lambdabar - cs * alpha

    @debug "" lambda deltabar

    if !solved_LN

      etaold = eta
      eta = cs * etabar # = etak

      # compute residual of least-norm problem at y{k-1}
      # TODO: use recurrence formula for LQ residual
      rNorm_lq_computed = sqrt((delta * eta)^2 + (epsilon * etaold)^2)
      rNorm_lq = norm(A' * y - c)  # FIXME
      @debug "" rNorm_lq_computed rNorm_lq abs(rNorm_lq_computed - rNorm_lq) / rNorm_lq
      rNorm_lq = rNorm_lq_computed
      push!(rNorms_lq, rNorm_lq)

      # stopping conditions related to the least-norm problem
      test_LN = rNorm_lq / sqrt(gamma1^2 + Anorm2 * yNorm2)
      solved_lim_LN = test_LN ≤ ln_tol
      solved_mach_LN = 1.0 + test_LN ≤ 1.0
      solved_LN = solved_lim_LN | solved_mach_LN

      # TODO: remove this when finished
      push!(tests_LN, test_LN)

      if solved_LN
        @info "solved LN problem with" y z
      end

      @. wbar = (v - lambda * w - epsilon * wold) / deltabar
      # @debug "" wbar

      if !solved_LN

          # prepare to update y and z
          @. p = cs * pbar + sn * u

          # update y and z
          @. y += eta * p
          @. z -= eta * w
          yNorm2 += eta * eta
          yNorm = sqrt(yNorm2)

          @debug "" y z

          @. pbar = sn * pbar - cs * u
          etabarold = etabar
          etabar = -(lambda * eta + epsilon * etaold) / deltabar # = etabar{k+1}

          # see if CG iterate has smaller residual
          # TODO: use recurrence formula for CG residual
          @. yC = y + etabar * pbar
          @. zC = z - etabar * wbar
          yCNorm2 = yNorm2 + etabar* etabar
          rNorm_cg_computed = gamma * abs(snold * etaold - csold * etabarold)
          rNorm_cg = norm(A' * yC - c)
          @debug "" rNorm_cg_computed rNorm_cg
          # if rNorm_cg < rNorm_lq
          #   # stopping conditions related to the least-norm problem
          # test_cg = rNorm_cg / sqrt(gamma1^2 + Anorm2 * yCNorm2)
          #   solved_lim_LN = test_cg ≤ ln_tol
          #   solved_mach_LN = 1.0 + test_cg ≤ 1.0
          #   solved_LN = solved_lim_LN | solved_mach_LN
          #   # transition_to_cg = solved_LN
          #   transition_to_cg = false
          # end

          if transition_to_cg
            # @. yC = y + etabar* pbar
            # @. zC = z - etabar* wbar
            @info "solved LN problem with CG point" yC zC
          end
      end
    end

    infoline *= @sprintf("%7.1e\n", rNorm_lq)
    @info infoline

    infoline = @sprintf("%4d %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e ", iter, alpha, beta, gamma, Anorm, Acond, rNorm_qr)

    # stopping conditions that apply to both problems
    tired  = iter ≥ itnlim
    ill_cond_lim = 1/Acond ≤ ctol
    ill_cond_mach = 1.0 + 1/Acond ≤ 1.0
    ill_cond = ill_cond_mach | ill_cond_lim

    solved = solved_LS & solved_LN
  end

  @info infoline

  @info "final LS status" zero_resid_lim_LS zero_resid_mach_LS solved_lim_LS solved_mach_LS
  @info "final LN status" solved_lim_LN solved_mach_LN
  @info "" solved  tired ill_cond

  # at the very end, recover r, yC and zC
  r = b - A * x
  # yC = y + etabar* pbar  # these might suffer from cancellation
  # zC = z - etabar* wbar  # if the last step is small

  return (x, r, y, z, yC, zC, rNorms_qr, ArNorms_qr, rNorms_lq, tests_LS, tests_LN)
end
