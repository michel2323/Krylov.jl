@testset "gmres" begin
  gmres_tol = 1.0e-6

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      # Symmetric and positive definite system.
      A, b = symmetric_definite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant.
      A, b = symmetric_indefinite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Nonsymmetric and positive definite systems.
      A, b = nonsymmetric_definite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Nonsymmetric indefinite variant.
      A, b = nonsymmetric_indefinite(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Sparse Laplacian.
      A, b = sparse_laplacian(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Symmetric indefinite variant, almost singular.
      A, b = almost_singular(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ 100 * gmres_tol)
      @test(stats.solved)

      # Singular system.
      A, b = square_inconsistent(FC=FC)
      (x, stats) = gmres(A, b)
      r = b - A * x
      Aresid = norm(A' * r) / norm(A' * b)
      @test(Aresid ≤ gmres_tol)
      @test(stats.inconsistent)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = gmres(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Poisson equation in polar coordinates.
      A, b = polar_poisson(FC=FC)
      (x, stats) = gmres(A, b, reorthogonalization=true)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Test with Jacobi (or diagonal) preconditioner
      A, b, M = square_preconditioned(FC=FC)
      (x, stats) = gmres(A, b, M=M)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Right preconditioning
      A, b, N = square_preconditioned(FC=FC)
      (x, stats) = gmres(A, b, N=N)
      r = b - A * x
      resid = norm(r) / norm(b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)

      # Split preconditioning
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = gmres(A, b, M=M, N=N)
      r = b - A * x
      resid = norm(M * r) / norm(M * b)
      @test(resid ≤ gmres_tol)
      @test(stats.solved)
    end
  end
end
