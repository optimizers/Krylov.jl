function test_trilqr()
  trilqr_tol = 1.0e-6

  # Test underdetermined adjoint systems.
  A, b, c = underdetermined_adjoint()
  (x, t, stats) = trilqr(A, b, c)
  show(stats)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @printf("TriLQR: Primal relative residual: %8.1e\n", resid_primal)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @printf("TriLQR: Dual relative residual: %8.1e\n", resid_dual)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test square adjoint systems.
  A, b, c = square_adjoint()
  (x, t, stats) = trilqr(A, b, c)
  show(stats)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @printf("TriLQR: Primal relative residual: %8.1e\n", resid_primal)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @printf("TriLQR: Dual relative residual: %8.1e\n", resid_dual)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test overdetermined adjoint systems
  A, b, c = overdetermined_adjoint()
  (x, t, stats) = trilqr(A, b, c)
  show(stats)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @printf("TriLQR: Primal relative residual: %8.1e\n", resid_primal)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @printf("TriLQR: Dual relative residual: %8.1e\n", resid_dual)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)
end

test_trilqr()
