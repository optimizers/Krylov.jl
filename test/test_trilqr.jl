function test_trilqr()
  trilqr_tol = 1.0e-6

  # Test underdetermined adjoint systems.
  A, b, c = underdetermined_adjoint()
  (x, t, stats) = trilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test square adjoint systems.
  A, b, c = square_adjoint()
  (x, t, stats) = trilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test overdetermined adjoint systems
  A, b, c = overdetermined_adjoint()
  (x, t, stats) = trilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test adjoint ODEs.
  A, b, c = adjoint_ode()
  (x, t, stats) = trilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test adjoint PDEs.
  A, b, c = adjoint_pde()
  (x, t, stats) = trilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  @test(resid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)

  # Test consistent Ax = b and inconsistent Aᵀt = c.
  A, b, c = rectangular_adjoint()
  (x, t, stats) = trilqr(A, b, c)

  r = b - A * x
  resid_primal = norm(r) / norm(b)
  @test(resid_primal ≤ trilqr_tol)
  @test(stats.solved_primal)

  s = c - A' * t
  resid_dual = norm(s) / norm(c)
  Aresid_dual = norm(A * s) / norm(A * c)
  @test(Aresid_dual ≤ trilqr_tol)
  @test(stats.solved_dual)
end

@testset "trilqr" begin
  test_trilqr()
end
