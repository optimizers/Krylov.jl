function test_warm_start(FC)
  A, b = warm_start(FC=FC)
  c = copy(b)
  n, m = size(A)
  x0 = 1.2 * ones(FC, n)
  y0 = 0.8 * ones(FC, n)
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5
  tol = 1.0e-6

  # BILQR
  x, y, stats = bilqr(A, b, c, x0, y0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
  s = c - A' * y
  resid = norm(s) / norm(c)
  @test(resid ≤ tol)

  solver = BilqrSolver(A, b)
  solve!(solver, A, b, c, x0, y0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
  s = c - A' * solver.y
  resid = norm(s) / norm(c)
  @test(resid ≤ tol)

  # TRILQR
  x, y, stats = trilqr(A, b, c, x0, y0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
  s = c - A' * y
  resid = norm(s) / norm(c)
  @test(resid ≤ tol)

  solver = TrilqrSolver(A, b)
  solve!(solver, A, b, c, x0, y0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
  s = c - A' * solver.y
  resid = norm(s) / norm(c)
  @test(resid ≤ tol)

  # TRICG
  x, y, stats = tricg(A, b, b, x0, y0)
  r = [b - x - A * y; b - A' * x + y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  solver = TricgSolver(A, b)
  solve!(solver, A, b, b, x0, y0)
  r = [b - solver.x - A * solver.y; b - A' * solver.x + solver.y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  # TRIMR
  x, y, stats = trimr(A, b, b, x0, y0)
  r = [b - x - A * y; b - A' * x + y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  solver = TrimrSolver(A, b)
  solve!(solver, A, b, b, x0, y0)
  r = [b - solver.x - A * solver.y; b - A' * solver.x + solver.y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  # USYMLQR
  x, y, stats = usymlqr(A, b, b, x0, y0)
  r = [b - x - A * y; b - A' * x]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  solver = UsymlqrSolver(A, b)
  solve!(solver, A, b, b, x0, y0)
  r = [b - solver.x - A * solver.y; b - A' * solver.x]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  # GPMR
  x, y, stats = gpmr(A, A', b, b, x0, y0)
  r = [b - x - A * y; b - A' * x - y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  solver = GpmrSolver(A, b)
  solve!(solver, A, A', b, b, x0, y0)
  r = [b - solver.x - A * solver.y; b - A' * solver.x - solver.y]
  resid = norm(r) / norm([b; b])
  @test(resid ≤ tol)

  # MINRES-QLP
  x, stats = minres_qlp(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = MinresQlpSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # SYMMLQ
  x, stats = symmlq(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = SymmlqSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # CG
  x, stats = cg(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = CgSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # CR
  x, stats = cr(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = CrSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # CAR
  x, stats = car(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = CarSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # CG-LANCZOS
  x, stats = cg_lanczos(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = CgLanczosSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # MINRES
  x, stats = minres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = MinresSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # MINARES
  x, stats = minares(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = MinaresSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # DIOM
  x, stats = diom(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = DiomSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # DQGMRES
  x, stats = dqgmres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = DqgmresSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # FOM
  x, stats = fom(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = FomSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # GMRES
  x, stats = gmres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = GmresSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # FGMRES
  x, stats = fgmres(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = FgmresSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # BICGSTAB
  x, stats = bicgstab(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = BicgstabSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # CGS
  x, stats = cgs(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = CgsSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # BILQ
  x, stats = bilq(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = BilqSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # QMR
  x, stats = qmr(A, b, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = QmrSolver(A, b)
  solve!(solver, A, b, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # USYMLQ
  x, stats = usymlq(A, b, c, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = UsymlqSolver(A, b)
  solve!(solver, A, b, c, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  # USYMQR
  x, stats = usymqr(A, b, c, x0)
  r = b - A * x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)

  solver = UsymqrSolver(A, b)
  solve!(solver, A, b, c, x0)
  r = b - A * solver.x
  resid = norm(r) / norm(b)
  @test(resid ≤ tol)
end

@testset "warm-start" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_warm_start(FC)
    end
  end
end
