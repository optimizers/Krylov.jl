@testset "lsqr" begin
  lsqr_tol = 1.0e-4

  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin

      for npower = 1 : 4
        (b, A, D, HY, HZ, Acond, rnorm) = test(40, 40, 4, npower, 0)  # No regularization.

        (x, stats) = lsqr(A, b)
        r = b - A * x
        resid = norm(A' * r) / norm(b)
        @test(resid ≤ lsqr_tol)
        @test(stats.solved)

        λ = 1.0e-3
        (x, stats) = lsqr(A, b, λ=λ)
        r = b - A * x
        resid = norm(A' * r - λ * λ * x) / norm(b)
        @test(resid ≤ lsqr_tol)
        @test(stats.solved)
      end

      A = [i/j - j/i for i=1:10, j=1:6]
      b = A * ones(6)

      # test trust-region constraint
      (x, stats) = lsqr(A, b)

      radius = 0.75 * norm(x)
      (x, stats) = lsqr(A, b, radius=radius)
      @test(stats.solved)
      @test(abs(radius - norm(x)) ≤ lsqr_tol * radius)

      # Test b == 0
      A, b = zero_rhs(FC=FC)
      (x, stats) = lsqr(A, b)
      @test norm(x) == 0
      @test stats.status == "x = 0 is a zero-residual solution"

      # Test with preconditioners
      A, b, M, N = two_preconditioners(FC=FC)
      (x, stats) = lsqr(A, b, M=M, N=N)
      r = b - A * x
      resid = sqrt(dot(r, M * r)) / norm(b)
      @test(resid ≤ lsqr_tol)
      @test(stats.solved)

      # Test regularization
      A, b, λ = regularization(FC=FC)
      (x, stats) = lsqr(A, b, λ=λ)
      r = b - A * x
      resid = norm(A' * r - λ^2 * x) / norm(b)
      @test(resid ≤ lsqr_tol)

      # Test saddle-point systems
      A, b, D = saddle_point(FC=FC)
      D⁻¹ = inv(D)
      (x, stats) = lsqr(A, b, M=D⁻¹)
      r = D⁻¹ * (b - A * x)
      resid = norm(A' * r) / norm(b)
      @test(resid ≤ lsqr_tol)

      # Test symmetric and quasi-definite systems
      A, b, M, N = sqd(FC=FC)
      M⁻¹ = inv(M)
      N⁻¹ = inv(N)
      (x, stats) = lsqr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
      r = M⁻¹ * (b - A * x)
      resid = norm(A' * r - N * x) / norm(b)
      @test(resid ≤ lsqr_tol)

      λ = 4.0
      (x, stats) = lsqr(A, b, M=M⁻¹, N=N⁻¹, λ=λ)
      r = M⁻¹ * (b - A * x)
      resid = norm(A' * r - λ^2 * N * x) / norm(b)
      @test(resid ≤ lsqr_tol)

      # Test dimension of additional vectors
      for transpose ∈ (false, true)
        A, b, c, D = small_sp(transpose, FC=FC)
        D⁻¹ = inv(D)
        (x, stats) = lsqr(A, b, M=D⁻¹)

        A, b, c, M, N = small_sqd(transpose, FC=FC)
        M⁻¹ = inv(M)
        N⁻¹ = inv(N)
        (x, stats) = lsqr(A, b, M=M⁻¹, N=N⁻¹, sqd=true)
      end
    end
  end
end
