using AMDGPU, AMDGPU.rocSPARSE

include("gpu.jl")

@testset "AMD -- AMDGPU.jl" begin

  @test AMDGPU.functional()
  AMDGPU.allowscalar(false)

  @testset "documentation" begin
    A_cpu = rand(ComplexF64, 20, 20)
    A_cpu = A_cpu + A_cpu'
    b_cpu = rand(ComplexF64, 20)
    A_gpu = ROCMatrix(A_cpu)
    b_gpu = ROCVector(b_cpu)
    x, stats = minres(A_gpu, b_gpu)
    r_gpu = b_gpu - A_gpu * x
    @test norm(r_gpu) ≤ 1e-4

    A_cpu = sprand(100, 200, 0.3)
    b_cpu = rand(100)
    A_csc_gpu = ROCSparseMatrixCSC(A_cpu)
    A_csr_gpu = ROCSparseMatrixCSR(A_cpu)
    A_coo_gpu = ROCSparseMatrixCOO(A_cpu)
    b_gpu = ROXVector(b_cpu)
    b_gpu = ROCVector(b_cpu)
    x_csc, y_csc, stats_csc = lnlq(A_csc_gpu, b_gpu)
    x_csr, y_csr, stats_csr = craig(A_csr_gpu, b_gpu)
    x_coo, y_coo, stats_coo = craigmr(A_coo_gpu, b_gpu)
    r_csc = b_gpu - A_csc_gpu * x_csc
    r_csr = b_gpu - A_csr_gpu * x_csr
    r_coo = b_gpu - A_coo_gpu * x_coo
    @test norm(r_csc) ≤ 1e-4
    @test norm(r_csr) ≤ 1e-4
    @test norm(r_coo) ≤ 1e-4
  end

  for FC in (Float32, Float64, ComplexF32, ComplexF64)
    S = ROCVector{FC}
    M = ROCMatrix{FC}
    T = real(FC)
    n = 10
    x = rand(FC, n)
    x = S(x)
    y = rand(FC, n)
    y = S(y)
    a = rand(FC)
    b = rand(FC)
    s = rand(FC)
    a2 = rand(T)
    b2 = rand(T)
    c = rand(T)

    @testset "kdot -- $FC" begin
      Krylov.@kdot(n, x, y)
    end

    @testset "kdotr -- $FC" begin
      Krylov.@kdotr(n, x, y)
    end

    @testset "knrm2 -- $FC" begin
      Krylov.@knrm2(n, x)
    end

    @testset "kaxpy! -- $FC" begin
      Krylov.@kaxpy!(n, a, x, y)
      Krylov.@kaxpy!(n, a2, x, y)
    end

    @testset "kaxpby! -- $FC" begin
      Krylov.@kaxpby!(n, a, x, b, y)
      Krylov.@kaxpby!(n, a2, x, b, y)
      Krylov.@kaxpby!(n, a, x, b2, y)
      Krylov.@kaxpby!(n, a2, x, b2, y)
    end

    @testset "kcopy! -- $FC" begin
      Krylov.@kcopy!(n, x, y)
    end

    @testset "kswap -- $FC" begin
      Krylov.@kswap(x, y)
    end

    @testset "kref! -- $FC" begin
      Krylov.@kref!(n, x, y, c, s)
    end

    @testset "conversion -- $FC" begin
      test_conversion(S, M)
    end

    ε = eps(T)
    atol = √ε
    rtol = √ε

    @testset "GMRES -- $FC" begin
      A, b = nonsymmetric_indefinite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = gmres(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "CG -- $FC" begin
      A, b = symmetric_definite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = cg(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    @testset "MINRES-QLP -- $FC" begin
      A, b = symmetric_indefinite(FC=FC)
      A = M(A)
      b = S(b)
      x, stats = minres_qlp(A, b)
      @test norm(b - A * x) ≤ atol + rtol * norm(b)
    end

    # @testset "processes -- $FC" begin
    #   test_processes(S, M)
    # end

    @testset "solver -- $FC" begin
      test_solver(S, M)
    end
  end
end
