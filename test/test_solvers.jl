function test_solvers(FC)
  A   = FC.(get_div_grad(4, 4, 4))  # Dimension n x n
  n   = size(A, 1)
  m   = div(n, 2)
  Au  = A[1:m,:]  # Dimension m x n
  Ao  = A[:,1:m]  # Dimension n x m
  b   = Ao * ones(FC, m) # Dimension n
  c   = Au * ones(FC, n) # Dimension m
  mem = 10
  shifts = [1.0; 2.0; 3.0; 4.0; 5.0]
  nshifts = 5
  T = real(FC)
  S = Vector{FC}
  solvers = Dict{Symbol, KrylovSolver}()

  @eval begin
    $solvers[:cg] = $(KRYLOV_SOLVERS[:cg])($n, $n, $S)
    $solvers[:car] = $(KRYLOV_SOLVERS[:car])($n, $n, $S)
    $solvers[:symmlq] = $(KRYLOV_SOLVERS[:symmlq])($n, $n, $S)
    $solvers[:minres] = $(KRYLOV_SOLVERS[:minres])($n, $n, $S)
    $solvers[:minares] = $(KRYLOV_SOLVERS[:minares])($n, $n, $S)
    $solvers[:cg_lanczos] = $(KRYLOV_SOLVERS[:cg_lanczos])($n, $n, $S)
    $solvers[:cg_lanczos_shift] = $(KRYLOV_SOLVERS[:cg_lanczos_shift])($n, $n, $nshifts, $S)
    $solvers[:diom] = $(KRYLOV_SOLVERS[:diom])($n, $n, $mem, $S)
    $solvers[:fom] = $(KRYLOV_SOLVERS[:fom])($n, $n, $mem, $S)
    $solvers[:dqgmres] = $(KRYLOV_SOLVERS[:dqgmres])($n, $n, $mem, $S)
    $solvers[:gmres] = $(KRYLOV_SOLVERS[:gmres])($n, $n, $mem, $S)
    $solvers[:fgmres] = $(KRYLOV_SOLVERS[:fgmres])($n, $n, $mem, $S)
    $solvers[:cr] = $(KRYLOV_SOLVERS[:cr])($n, $n, $S)
    $solvers[:crmr] = $(KRYLOV_SOLVERS[:crmr])($m, $n, $S)
    $solvers[:cgs] = $(KRYLOV_SOLVERS[:cgs])($n, $n, $S)
    $solvers[:bicgstab] = $(KRYLOV_SOLVERS[:bicgstab])($n, $n, $S)
    $solvers[:craigmr] = $(KRYLOV_SOLVERS[:craigmr])($m, $n, $S)
    $solvers[:cgne] = $(KRYLOV_SOLVERS[:cgne])($m, $n, $S)
    $solvers[:lnlq] = $(KRYLOV_SOLVERS[:lnlq])($m, $n, $S)
    $solvers[:craig] = $(KRYLOV_SOLVERS[:craig])($m, $n, $S)
    $solvers[:lslq] = $(KRYLOV_SOLVERS[:lslq])($n, $m, $S)
    $solvers[:cgls] = $(KRYLOV_SOLVERS[:cgls])($n, $m, $S)
    $solvers[:cgls_lanczos_shift] = $(KRYLOV_SOLVERS[:cgls_lanczos_shift])($n, $m, $nshifts, $S)
    $solvers[:lsqr] = $(KRYLOV_SOLVERS[:lsqr])($n, $m, $S)
    $solvers[:crls] = $(KRYLOV_SOLVERS[:crls])($n, $m, $S)
    $solvers[:lsmr] = $(KRYLOV_SOLVERS[:lsmr])($n, $m, $S)
    $solvers[:usymqr] = $(KRYLOV_SOLVERS[:usymqr])($n, $m, $S)
    $solvers[:trilqr] = $(KRYLOV_SOLVERS[:trilqr])($n, $n, $S)
    $solvers[:bilq] = $(KRYLOV_SOLVERS[:bilq])($n, $n, $S)
    $solvers[:bilqr] = $(KRYLOV_SOLVERS[:bilqr])($n, $n, $S)
    $solvers[:minres_qlp] = $(KRYLOV_SOLVERS[:minres_qlp])($n, $n, $S)
    $solvers[:qmr] = $(KRYLOV_SOLVERS[:qmr])($n, $n, $S)
    $solvers[:usymlq] = $(KRYLOV_SOLVERS[:usymlq])($m, $n, $S)
    $solvers[:tricg] = $(KRYLOV_SOLVERS[:tricg])($m, $n, $S)
    $solvers[:trimr] = $(KRYLOV_SOLVERS[:trimr])($m, $n, $S)
    $solvers[:usymlqr] = $(KRYLOV_SOLVERS[:usymlqr])($m, $n, $S)
    $solvers[:gpmr] = $(KRYLOV_SOLVERS[:gpmr])($n, $m, $mem, $S)
    $solvers[:cg_lanczos_shift] = $(KRYLOV_SOLVERS[:cg_lanczos_shift])($n, $n, $nshifts, $S)
  end

  @testset "Check compatibility between KrylovSolvers and the dimension of the linear problems" begin
    A2  = FC.(get_div_grad(2, 2, 2))
    n2  = size(A2, 1)
    m2  = div(n2, 2)
    Au2 = A2[1:m2,:]
    Ao2 = A2[:,1:m2]
    b2  = Ao2 * ones(FC, m2)
    c2  = Au2 * ones(FC, n2)
    shifts2 = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0]
    for (method, solver) in solvers
      if method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom, :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr)
        @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $n2)") solve!(solver, A2, b2)
      end
      method == :cg_lanczos_shift && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $n2)") solve!(solver, A2, b2, shifts2)
      method == :cg_lanczos_shift && @test_throws ErrorException("solver.nshifts = $(solver.nshifts) is inconsistent with length(shifts) = $(length(shifts2))") solve!(solver, A, b, shifts2)
      method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m2, $n2)") solve!(solver, Au2, c2)
      method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") solve!(solver, Ao2, b2)
      method == :cgls_lanczos_shift && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") solve!(solver, Ao2, b2, shifts2)
      method == :cgls_lanczos_shift && @test_throws ErrorException("solver.nshifts = $(solver.nshifts) is inconsistent with length(shifts) = $(length(shifts2))") solve!(solver, Ao, b, shifts2)
      method ∈ (:bilqr, :trilqr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $n2)") solve!(solver, A2, b2, b2)
      method == :gpmr && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") solve!(solver, Ao2, Au2, b2, c2)
      method ∈ (:tricg, :trimr, :usymlqr) && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") solve!(solver, Ao2, b2, c2)
      method == :usymlq && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m2, $n2)") solve!(solver, Au2, c2, b2)
      method == :usymqr && @test_throws ErrorException("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($n2, $m2)") solve!(solver, Ao2, b2, c2)
    end
  end

  @testset "Test the keyword argument timemax" begin
    timemax = 0.0
    for (method, solver) in solvers
      method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom, :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr) && solve!(solver, A, b, timemax=timemax)
      method == :cg_lanczos_shift && solve!(solver, A, b, shifts, timemax=timemax)
      method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr) && solve!(solver, Au, c, timemax=timemax)
      method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr) && solve!(solver, Ao, b, timemax=timemax)
      method == :cgls_lanczos_shift && solve!(solver, Ao, b, shifts, timemax=timemax)
      method ∈ (:bilqr, :trilqr) && solve!(solver, A, b, b, timemax=timemax)
      method == :gpmr && solve!(solver, Ao, Au, b, c, timemax=timemax)
      method ∈ (:tricg, :trimr, :usymlqr) && solve!(solver, Au, c, b, timemax=timemax)
      method == :usymlq && solve!(solver, Au, c, b, timemax=timemax)
      method == :usymqr && solve!(solver, Ao, b, c, timemax=timemax)
      @test solver.stats.status == "time limit exceeded"
    end
  end

  for (method, solver) in solvers
    @testset "$(method)" begin
      for i = 1 : 3
        A  = i * A
        Au = i * Au
        Ao = i * Ao
        b  = 5 * b
        c  = 3 * c

        if method ∈ (:cg, :cr, :car, :symmlq, :minares, :minres, :minres_qlp, :cg_lanczos, :diom, :fom,
                     :dqgmres, :gmres, :fgmres, :cgs, :bicgstab, :bilq, :qmr, :cg_lanczos_shift)
          method == :cg_lanczos_shift ? solve!(solver, A, b, shifts) : solve!(solver, A, b)
          niter = niterations(solver)
          @test Aprod(solver) == (method ∈ (:cgs, :bicgstab) ? 2 * niter : niter)
          @test Atprod(solver) == (method ∈ (:bilq, :qmr) ? niter : 0)
          @test solution(solver) === solver.x
          @test nsolution(solver) == 1
        end

        if method ∈ (:cgne, :crmr, :lnlq, :craig, :craigmr)
          solve!(solver, Au, c)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver, 1) === solver.x
          @test nsolution(solver) == (method ∈ (:cgne, :crmr) ? 1 : 2)
          (nsolution == 2) && (@test solution(solver, 2) == solver.y)
        end

        if method ∈ (:cgls, :crls, :lslq, :lsqr, :lsmr, :cgls_lanczos_shift)
          method == :cgls_lanczos_shift ? solve!(solver, Ao, b, shifts) : solve!(solver, Ao, b)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver) === solver.x
          @test nsolution(solver) == 1
        end

        if method ∈ (:bilqr, :trilqr)
          solve!(solver, A, b, b)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver, 1) === solver.x
          @test solution(solver, 2) === solver.y
          @test nsolution(solver) == 2
          @test issolved_primal(solver)
          @test issolved_dual(solver)
        end

        if method ∈ (:tricg, :trimr, :usymlqr, :gpmr)
          method == :gpmr ? solve!(solver, Ao, Au, b, c) : solve!(solver, Au, c, b)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          method != :gpmr && (@test Atprod(solver) == niter)
          method == :gpmr && (@test Bprod(solver) == niter)
          @test solution(solver, 1) === solver.x
          @test solution(solver, 2) === solver.y
          @test nsolution(solver) == 2
        end

        if method ∈ (:usymlq, :usymqr)
          method == :usymlq ? solve!(solver, Au, c, b) : solve!(solver, Ao, b, c)
          niter = niterations(solver)
          @test Aprod(solver) == niter
          @test Atprod(solver) == niter
          @test solution(solver) === solver.x
          @test nsolution(solver) == 1
        end

        @test niter > 0
        @test statistics(solver) === solver.stats
        @test issolved(solver)
      end

      io = IOBuffer()
      show(io, solver, show_stats=false)
      showed = String(take!(io))

      # Test that the lines have the same length
      str = split(showed, '\n', keepempty=false)
      len_row = length(str[1])
      @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_row, &, str)

      # Test that the columns have the same length
      str2 = split(showed, ['│','┌','┬','┐','├','┼','┤','└','┴','┴','┘','\n'], keepempty=false)
      len_col1 = length(str2[1])
      len_col2 = length(str2[2])
      len_col3 = length(str2[3])
      @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col1, &, str2[1:3:end-2])
      @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col2, &, str2[2:3:end-1])
      @test mapreduce(x -> length(x) - mapreduce(y -> occursin(y, x), |, ["w̅","w̄","d̅"]) == len_col3, &, str2[3:3:end])

      # Code coverage
      show(io, solver, show_stats=true)
    end
  end
end

@testset "solvers" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_solvers(FC)
    end
  end
end
