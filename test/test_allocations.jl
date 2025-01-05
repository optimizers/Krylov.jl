@testset "allocations" begin

  for FC in (Float32, Float64, ComplexF32, ComplexF64)
    @testset "Data Type: $FC" begin

      A   = FC.(get_div_grad(18, 18, 18))  # Dimension m x n
      m,n = size(A)
      p   = 5
      k   = div(n, 2)
      Au  = A[1:k,:]          # Dimension k x n
      Ao  = A[:,1:k]          # Dimension m x k
      b   = Ao * ones(FC, k)  # Dimension m
      c   = Au * ones(FC, n)  # Dimension k
      B   = A * Matrix{FC}(I, m, p)  # Dimension m × p
      mem = 200

      T = real(FC)
      shifts = T[1; 2; 3; 4; 5]
      nshifts = 5
      nbits_FC = sizeof(FC)  # 8 bits for ComplexF32 and 16 bits for ComplexF64
      nbits_T = sizeof(T)    # 4 bits for Float32 and 8 bits for Float64

      @testset "SYMMLQ" begin
        # SYMMLQ needs:
        # 5 n-vectors: x, Mvold, Mv, Mv_next, w̅
        storage_symmlq_bytes(n) = nbits_FC * 5 * n

        expected_symmlq_bytes = storage_symmlq_bytes(n)
        symmlq(A, b)  # warmup
        actual_symmlq_bytes = @allocated symmlq(A, b)
        @test expected_symmlq_bytes ≤ actual_symmlq_bytes ≤ 1.02 * expected_symmlq_bytes

        solver = SymmlqSolver(A, b)
        symmlq!(solver, A, b)  # warmup
        inplace_symmlq_bytes = @allocated symmlq!(solver, A, b)
        @test inplace_symmlq_bytes == 0
      end

      @testset "CG" begin
        # CG needs:
        # 4 n-vectors: x, r, p, Ap
        storage_cg_bytes(n) = nbits_FC * 4 * n

        expected_cg_bytes = storage_cg_bytes(n)
        cg(A, b)  # warmup
        actual_cg_bytes = @allocated cg(A, b)
        @test expected_cg_bytes ≤ actual_cg_bytes ≤ 1.02 * expected_cg_bytes

        solver = CgSolver(A, b)
        cg!(solver, A, b)  # warmup
        inplace_cg_bytes = @allocated cg!(solver, A, b)
        @test inplace_cg_bytes == 0
      end

      @testset "CG-LANCZOS" begin
        # CG-LANCZOS needs:
        # 5 n-vectors: x, Mv, Mv_prev, p, Mv_next
        storage_cg_lanczos_bytes(n) = nbits_FC * 5 * n

        expected_cg_lanczos_bytes = storage_cg_lanczos_bytes(n)
        cg_lanczos(A, b)  # warmup
        actual_cg_lanczos_bytes = @allocated cg_lanczos(A, b)
        @test expected_cg_lanczos_bytes ≤ actual_cg_lanczos_bytes ≤ 1.02 * expected_cg_lanczos_bytes

        solver = CgLanczosSolver(A, b)
        cg_lanczos!(solver, A, b)  # warmup
        inplace_cg_lanczos_bytes = @allocated cg_lanczos!(solver, A, b)
        @test inplace_cg_lanczos_bytes == 0
      end

      @testset "CG-LANCZOS-SHIFT" begin
        # CG-LANCZOS-SHIFT needs:
        # - 3 n-vectors: Mv, Mv_prev, Mv_next
        # - 2 (n*nshifts)-matrices: x, p
        # - 5 nshifts-vectors: σ, δhat, ω, γ, rNorms
        # - 3 nshifts-bitVector: indefinite, converged, not_cv
        storage_cg_lanczos_shift_bytes(n, nshifts) = nbits_FC * ((3 * n) + (2 * n * nshifts)) + nbits_T * (5 * nshifts) + (3 * nshifts)
        expected_cg_lanczos_shift_bytes = storage_cg_lanczos_shift_bytes(n, nshifts)
        cg_lanczos_shift(A, b, shifts)  # warmup
        actual_cg_lanczos_shift_bytes = @allocated cg_lanczos_shift(A, b, shifts)
        @test expected_cg_lanczos_shift_bytes ≤ actual_cg_lanczos_shift_bytes ≤ 1.02 * expected_cg_lanczos_shift_bytes

        solver = CgLanczosShiftSolver(A, b, nshifts)
        cg_lanczos_shift!(solver, A, b, shifts)  # warmup
        inplace_cg_lanczos_shift_bytes = @allocated cg_lanczos_shift!(solver, A, b, shifts)
        @test inplace_cg_lanczos_shift_bytes == 0
      end

      @testset "CR" begin
        # CR needs:
        # 5 n-vectors: x, r, p, q, Ar
        storage_cr_bytes(n) = nbits_FC * 5 * n

        expected_cr_bytes = storage_cr_bytes(n)
        cr(A, b)  # warmup
        actual_cr_bytes = @allocated cr(A, b)
        @test expected_cr_bytes ≤ actual_cr_bytes ≤ 1.02 * expected_cr_bytes

        solver = CrSolver(A, b)
        cr!(solver, A, b)  # warmup
        inplace_cr_bytes = @allocated cr!(solver, A, b)
        @test inplace_cr_bytes == 0
      end

      @testset "CAR" begin
        # CAR needs:
        # 7 n-vectors: x, r, p, s, q, t, u
        storage_car_bytes(n) = nbits_FC * 7 * n

        expected_car_bytes = storage_car_bytes(n)
        car(A, b)  # warmup
        actual_car_bytes = @allocated car(A, b)
        @test expected_car_bytes ≤ actual_car_bytes ≤ 1.02 * expected_car_bytes

        solver = CarSolver(A, b)
        car!(solver, A, b)  # warmup
        inplace_car_bytes = @allocated car!(solver, A, b)
        @test inplace_car_bytes == 0
      end

      @testset "MINRES" begin
        # MINRES needs:
        # 6 n-vectors: x, r1, r2, w1, w2, y
        storage_minres_bytes(n) = nbits_FC * 6 * n

        expected_minres_bytes = storage_minres_bytes(n)
        minres(A, b)  # warmup
        actual_minres_bytes = @allocated minres(A, b)
        @test expected_minres_bytes ≤ actual_minres_bytes ≤ 1.02 * expected_minres_bytes

        solver = MinresSolver(A, b)
        minres!(solver, A, b)  # warmup
        inplace_minres_bytes = @allocated minres!(solver, A, b)
        @test inplace_minres_bytes == 0
      end

      @testset "MINRES-QLP" begin
        # MINRES-QLP needs:
        # - 6 n-vectors: wₖ₋₁, wₖ, vₖ₋₁, vₖ, x, p
        storage_minres_qlp_bytes(n) = nbits_FC * 6 * n

        expected_minres_qlp_bytes = storage_minres_qlp_bytes(n)
        minres_qlp(A, b)  # warmup
        actual_minres_qlp_bytes = @allocated minres_qlp(A, b)
        @test expected_minres_qlp_bytes ≤ actual_minres_qlp_bytes ≤ 1.02 * expected_minres_qlp_bytes

        solver = MinresQlpSolver(A, b)
        minres_qlp!(solver, A, b)  # warmup
        inplace_minres_qlp_bytes = @allocated minres_qlp!(solver, A, b)
        @test inplace_minres_qlp_bytes == 0
      end

      @testset "MINARES" begin
        # MINARES needs:
        # 8 n-vectors: vₖ, vₖ₊₁, x, wₖ₋₂, wₖ₋₁, dₖ₋₂, dₖ₋₁, q
        storage_minares_bytes(n) = nbits_FC * 8 * n

        expected_minares_bytes = storage_minares_bytes(n)
        minares(A, b)  # warmup
        actual_minares_bytes = @allocated minares(A, b)
        @test expected_minares_bytes ≤ actual_minares_bytes ≤ 1.02 * expected_minares_bytes

        solver = MinaresSolver(A, b)
        minares!(solver, A, b)  # warmup
        inplace_minares_bytes = @allocated minares!(solver, A, b)
        @test inplace_minares_bytes == 0
      end

      @testset "DIOM" begin
        # DIOM needs:
        # - 2 n-vectors: x, t
        # - 1 (n*mem)-matrix: V
        # - 1 n*(mem-1)-matrix: P
        # - 1 (mem-1)-vector: L
        # - 1 mem-vector: H
        storage_diom_bytes(mem, n) = nbits_FC * ((2 * n) + (n * mem) + (n * (mem-1)) + (mem-1) + (mem))

        expected_diom_bytes = storage_diom_bytes(mem, n)
        diom(A, b, memory=mem)  # warmup
        actual_diom_bytes = @allocated diom(A, b, memory=mem)
        @test expected_diom_bytes ≤ actual_diom_bytes ≤ 1.02 * expected_diom_bytes

        solver = DiomSolver(A, b, mem)
        diom!(solver, A, b)  # warmup
        inplace_diom_bytes = @allocated diom!(solver, A, b)
        @test inplace_diom_bytes == 0
      end

      @testset "FOM" begin
        # FOM needs:
        # - 2 n-vectors: x, w
        # - 1 (n*mem)-matrix: V
        # - 2 mem-vectors: l, z
        # - 1 (mem*(mem+1)/2)-vector: U
        storage_fom_bytes(mem, n) = nbits_FC * ((2 * n) + (n * mem) + (2 * mem) + (mem * (mem+1) / 2))

        expected_fom_bytes = storage_fom_bytes(mem, n)
        fom(A, b, memory=mem)  # warmup
        actual_fom_bytes = @allocated fom(A, b, memory=mem)
        @test expected_fom_bytes ≤ actual_fom_bytes ≤ 1.02 * expected_fom_bytes

        solver = FomSolver(A, b, mem)
        fom!(solver, A, b)  # warmup
        inplace_fom_bytes = @allocated fom!(solver, A, b)
        @test inplace_fom_bytes == 0
      end

      @testset "DQGMRES" begin
        # DQGMRES needs:
        # - 2 n-vectors: x, t
        # - 2 (n*mem)-matrices: P, V
        # - 2 mem-vectors: c, s
        # - 1 (mem+1)-vector: H
        storage_dqgmres_bytes(mem, n) = nbits_FC * ((2 * n) + (2 * n * mem) + mem + (mem + 1)) + nbits_T * mem

        expected_dqgmres_bytes = storage_dqgmres_bytes(mem, n)
        dqgmres(A, b, memory=mem)  # warmup
        actual_dqgmres_bytes = @allocated dqgmres(A, b, memory=mem)
        @test expected_dqgmres_bytes ≤ actual_dqgmres_bytes ≤ 1.02 * expected_dqgmres_bytes

        solver = DqgmresSolver(A, b, mem)
        dqgmres!(solver, A, b)  # warmup
        inplace_dqgmres_bytes = @allocated dqgmres!(solver, A, b)
        @test inplace_dqgmres_bytes == 0
      end

      @testset "GMRES" begin
        # GMRES needs:
        # - 2 n-vectors: x, w
        # - 1 n*(mem)-matrix: V
        # - 3 mem-vectors: c, s, z
        # - 1 (mem*(mem+1)/2)-vector: R
        storage_gmres_bytes(mem, n) = nbits_FC * ((2 * n) + (n * mem) + (2 * mem) + (mem * (mem+1) / 2)) + nbits_T * mem

        expected_gmres_bytes = storage_gmres_bytes(mem, n)
        gmres(A, b, memory=mem)  # warmup
        actual_gmres_bytes = @allocated gmres(A, b, memory=mem)
        @test expected_gmres_bytes ≤ actual_gmres_bytes ≤ 1.02 * expected_gmres_bytes

        solver = GmresSolver(A, b, mem)
        gmres!(solver, A, b)  # warmup
        inplace_gmres_bytes = @allocated gmres!(solver, A, b)
        @test inplace_gmres_bytes == 0
      end

      @testset "FGMRES" begin
        # FGMRES needs:
        # - 2 n-vectors: x, w
        # - 2 n*(mem)-matrix: V, Z
        # - 3 mem-vectors: c, s, z
        # - 1 (mem*(mem+1)/2)-vector: R
        storage_fgmres_bytes(mem, n) = nbits_FC * ((2 * n) + (2 * n * mem) + (2 * mem) + (mem * (mem+1) / 2)) + nbits_T * mem

        expected_fgmres_bytes = storage_fgmres_bytes(mem, n)
        fgmres(A, b, memory=mem)  # warmup
        actual_fgmres_bytes = @allocated fgmres(A, b, memory=mem)
        @test expected_fgmres_bytes ≤ actual_fgmres_bytes ≤ 1.02 * expected_fgmres_bytes

        solver = FgmresSolver(A, b, mem)
        fgmres!(solver, A, b)  # warmup
        inplace_fgmres_bytes = @allocated fgmres!(solver, A, b)
        @test inplace_fgmres_bytes == 0
      end

      @testset "CGS" begin
        # CGS needs:
        # 6 n-vectors: x, r, u, p, q, ts
        storage_cgs_bytes(n) = nbits_FC * 6 * n

        expected_cgs_bytes = storage_cgs_bytes(n)
        cgs(A, b)  # warmup
        actual_cgs_bytes = @allocated cgs(A, b)
        @test expected_cgs_bytes ≤ actual_cgs_bytes ≤ 1.02 * expected_cgs_bytes

        solver = CgsSolver(A, b)
        cgs!(solver, A, b)  # warmup
        inplace_cgs_bytes = @allocated cgs!(solver, A, b)
        @test inplace_cgs_bytes == 0
      end

      @testset "BICGSTAB" begin
        # BICGSTAB needs:
        # 6 n-vectors: x, r, p, v, s, qd
        storage_bicgstab_bytes(n) = nbits_FC * 6 * n

        expected_bicgstab_bytes = storage_bicgstab_bytes(n)
        bicgstab(A, b)  # warmup
        actual_bicgstab_bytes = @allocated bicgstab(A, b)
        @test expected_bicgstab_bytes ≤ actual_bicgstab_bytes ≤ 1.02 * expected_bicgstab_bytes

        solver = BicgstabSolver(A, b)
        bicgstab!(solver, A, b)  # warmup
        inplace_bicgstab_bytes = @allocated bicgstab!(solver, A, b)
        @test inplace_bicgstab_bytes == 0
      end

      @testset "CGNE" begin
        # CGNE needs:
        # - 3 n-vectors: x, p, Aᴴz
        # - 2 m-vectors: r, q
        storage_cgne_bytes(m, n) = nbits_FC * (3 * n + 2 * m)

        expected_cgne_bytes = storage_cgne_bytes(k, n)
        (x, stats) = cgne(Au, c)  # warmup
        actual_cgne_bytes = @allocated cgne(Au, c)
        @test expected_cgne_bytes ≤ actual_cgne_bytes ≤ 1.02 * expected_cgne_bytes

        solver = CgneSolver(Au, c)
        cgne!(solver, Au, c)  # warmup
        inplace_cgne_bytes = @allocated cgne!(solver, Au, c)
        @test inplace_cgne_bytes == 0
      end

      @testset "CRMR" begin
        # CRMR needs:
        # - 3 n-vectors: x, p, Aᴴr
        # - 2 m-vectors: r, q
        storage_crmr_bytes(m, n) = nbits_FC * (3 * n + 2 * m)

        expected_crmr_bytes = storage_crmr_bytes(k, n)
        (x, stats) = crmr(Au, c)  # warmup
        actual_crmr_bytes = @allocated crmr(Au, c)
        @test expected_crmr_bytes ≤ actual_crmr_bytes ≤ 1.02 * expected_crmr_bytes

        solver = CrmrSolver(Au, c)
        crmr!(solver, Au, c)  # warmup
        inplace_crmr_bytes = @allocated crmr!(solver, Au, c)
        @test inplace_crmr_bytes == 0
      end

      @testset "LNLQ" begin
        # LNLQ needs:
        # - 3 n-vectors: x, v, Aᴴu
        # - 4 m-vectors: y, w̄, u, Av
        storage_lnlq_bytes(m, n) = nbits_FC * (3 * n + 4 * m)

        expected_lnlq_bytes = storage_lnlq_bytes(k, n)
        lnlq(Au, c)  # warmup
        actual_lnlq_bytes = @allocated lnlq(Au, c)
        @test expected_lnlq_bytes ≤ actual_lnlq_bytes ≤ 1.02 * expected_lnlq_bytes

        solver = LnlqSolver(Au, c)
        lnlq!(solver, Au, c)  # warmup
        inplace_lnlq_bytes = @allocated lnlq!(solver, Au, c)
        @test inplace_lnlq_bytes == 0
      end

      @testset "CRAIG" begin
        # CRAIG needs:
        # - 3 n-vectors: x, v, Aᴴu
        # - 4 m-vectors: y, w, u, Av
        storage_craig_bytes(m, n) = nbits_FC * (3 * n + 4 * m)

        expected_craig_bytes = storage_craig_bytes(k, n)
        craig(Au, c)  # warmup
        actual_craig_bytes = @allocated craig(Au, c)
        @test expected_craig_bytes ≤ actual_craig_bytes ≤ 1.02 * expected_craig_bytes

        solver = CraigSolver(Au, c)
        craig!(solver, Au, c)  # warmup
        inplace_craig_bytes = @allocated craig!(solver, Au, c)
        @test inplace_craig_bytes == 0
      end

      @testset "CRAIGMR" begin
        # CRAIGMR needs:
        # - 4 n-vectors: x, v, Aᴴu, d
        # - 5 m-vectors: y, u, w, wbar, Av
        storage_craigmr_bytes(m, n) = nbits_FC * (4 * n + 5 * m)

        expected_craigmr_bytes = storage_craigmr_bytes(k, n)
        craigmr(Au, c)  # warmup
        actual_craigmr_bytes = @allocated craigmr(Au, c)
        @test expected_craigmr_bytes ≤ actual_craigmr_bytes ≤ 1.02 * expected_craigmr_bytes

        solver = CraigmrSolver(Au, c)
        craigmr!(solver, Au, c)  # warmup
        inplace_craigmr_bytes = @allocated craigmr!(solver, Au, c)
        @test inplace_craigmr_bytes == 0
      end

      @testset "CGLS" begin
        # CGLS needs:
        # - 3 n-vectors: x, p, s
        # - 2 m-vectors: r, q
        storage_cgls_bytes(m, n) = nbits_FC * (3 * n + 2 * m)

        expected_cgls_bytes = storage_cgls_bytes(m, k)
        (x, stats) = cgls(Ao, b)  # warmup
        actual_cgls_bytes = @allocated cgls(Ao, b)
        @test expected_cgls_bytes ≤ actual_cgls_bytes ≤ 1.02 * expected_cgls_bytes

        solver = CglsSolver(Ao, b)
        cgls!(solver, Ao, b)  # warmup
        inplace_cgls_bytes = @allocated cgls!(solver, Ao, b)
        @test inplace_cgls_bytes == 0
      end

      @testset "CGLS-LANCZOS-SHIFT" begin
        # CGLS-LANCZOS-SHIFT needs:
        # - 2 n-vectors: Mv, v
        # - 3 m-vectors: Mv_prev, Mv_next, u
        # - 2 (n*nshifts)-matrices: x, p
        # - 5 nshifts-vectors: σ, δhat, ω, γ, rNorms
        # - 3 nshifts-bitVector: converged, indefinite, not_cv
        storage_cgls_lanczos_shift_bytes(m, n, nshifts) = nbits_FC * (2 * n + 3 * m + 2 * n * nshifts) + nbits_T * (5 * nshifts) + (3 * nshifts)

        expected_cgls_lanczos_shift_bytes = storage_cgls_lanczos_shift_bytes(m, k, nshifts)
        (x, stats) = cgls_lanczos_shift(Ao, b, shifts)  # warmup
        actual_cgls_lanczos_shift_bytes = @allocated cgls_lanczos_shift(Ao, b, shifts)
        @test expected_cgls_lanczos_shift_bytes ≤ actual_cgls_lanczos_shift_bytes ≤ 1.03 * expected_cgls_lanczos_shift_bytes

        solver = CglsLanczosShiftSolver(Ao, b, nshifts)
        cgls_lanczos_shift!(solver, Ao, b, shifts)  # warmup
        inplace_cgls_lanczos_shift_bytes = @allocated cgls_lanczos_shift!(solver, Ao, b, shifts)
        @test inplace_cgls_lanczos_shift_bytes == 0
      end

      @testset "LSLQ" begin
        # LSLQ needs:
        # - 4 n-vectors: x_lq, v, Aᴴu, w̄ (= x_cg)
        # - 2 m-vectors: u, Av
        storage_lslq_bytes(m, n) = nbits_FC * (4 * n + 2 * m)

        expected_lslq_bytes = storage_lslq_bytes(m, k)
        (x, stats) = lslq(Ao, b)  # warmup
        actual_lslq_bytes = @allocated lslq(Ao, b)
        @test expected_lslq_bytes ≤ actual_lslq_bytes ≤ 1.025 * expected_lslq_bytes

        solver = LslqSolver(Ao, b)
        lslq!(solver, Ao, b)  # warmup
        inplace_lslq_bytes = @allocated lslq!(solver, Ao, b)
        @test inplace_lslq_bytes == 0
      end

      @testset "CRLS" begin
        # CRLS needs:
        # - 4 n-vectors: x, p, Ar, q
        # - 3 m-vectors: r, Ap, s
        storage_crls_bytes(m, n) = nbits_FC * (4 * n + 3 * m)

        expected_crls_bytes = storage_crls_bytes(m, k)
        (x, stats) = crls(Ao, b)  # warmup
        actual_crls_bytes = @allocated crls(Ao, b)
        @test expected_crls_bytes ≤ actual_crls_bytes ≤ 1.02 * expected_crls_bytes

        solver = CrlsSolver(Ao, b)
        crls!(solver, Ao, b)  # warmup
        inplace_crls_bytes = @allocated crls!(solver, Ao, b)
        @test inplace_crls_bytes == 0
      end

      @testset "LSQR" begin
        # LSQR needs:
        # - 4 n-vectors: x, v, w, Aᴴu
        # - 2 m-vectors: u, Av
        storage_lsqr_bytes(m, n) = nbits_FC * (4 * n + 2 * m)

        expected_lsqr_bytes = storage_lsqr_bytes(m, k)
        (x, stats) = lsqr(Ao, b)  # warmup
        actual_lsqr_bytes = @allocated lsqr(Ao, b)
        @test expected_lsqr_bytes ≤ actual_lsqr_bytes ≤ 1.02 * expected_lsqr_bytes

        solver = LsqrSolver(Ao, b)
        lsqr!(solver, Ao, b)  # warmup
        inplace_lsqr_bytes = @allocated lsqr!(solver, Ao, b)
        @test inplace_lsqr_bytes == 0
      end

      @testset "LSMR" begin
        # LSMR needs:
        # - 5 n-vectors: x, v, h, hbar, Aᴴu
        # - 2 m-vectors: u, Av
        storage_lsmr_bytes(m, n) = nbits_FC * (5 * n + 2 * m)

        expected_lsmr_bytes = storage_lsmr_bytes(m, k)
        (x, stats) = lsmr(Ao, b)  # warmup
        actual_lsmr_bytes = @allocated lsmr(Ao, b)
        @test expected_lsmr_bytes ≤ actual_lsmr_bytes ≤ 1.02 * expected_lsmr_bytes

        solver = LsmrSolver(Ao, b)
        lsmr!(solver, Ao, b)  # warmup
        inplace_lsmr_bytes = @allocated lsmr!(solver, Ao, b)
        @test inplace_lsmr_bytes == 0
      end

      @testset "BiLQ" begin
        # BILQ needs:
        # - 8 n-vectors: uₖ₋₁, uₖ, vₖ₋₁, vₖ, x, d̅, p, q
        storage_bilq_bytes(n) = nbits_FC * 8 * n

        expected_bilq_bytes = storage_bilq_bytes(n)
        bilq(A, b)  # warmup
        actual_bilq_bytes = @allocated bilq(A, b)
        @test expected_bilq_bytes ≤ actual_bilq_bytes ≤ 1.02 * expected_bilq_bytes

        solver = BilqSolver(A, b)
        bilq!(solver, A, b)  # warmup
        inplace_bilq_bytes = @allocated bilq!(solver, A, b)
        @test inplace_bilq_bytes == 0
      end

      @testset "QMR" begin
        # QMR needs:
        # - 9 n-vectors: uₖ₋₁, uₖ, vₖ₋₁, vₖ, x, wₖ₋₁, wₖ, p, q
        storage_qmr_bytes(n) = nbits_FC * 9 * n

        expected_qmr_bytes = storage_qmr_bytes(n)
        qmr(A, b)  # warmup
        actual_qmr_bytes = @allocated qmr(A, b)
        @test expected_qmr_bytes ≤ actual_qmr_bytes ≤ 1.02 * expected_qmr_bytes

        solver = QmrSolver(A, b)
        qmr!(solver, A, b)  # warmup
        inplace_qmr_bytes = @allocated qmr!(solver, A, b)
        @test inplace_qmr_bytes == 0
      end

      @testset "BiLQR" begin
        # BILQR needs:
        # - 11 n-vectors: uₖ₋₁, uₖ, vₖ₋₁, vₖ, x, t, d̅, wₖ₋₁, wₖ, p, q
        storage_bilqr_bytes(n) = nbits_FC * 11 * n

        expected_bilqr_bytes = storage_bilqr_bytes(n)
        bilqr(A, b, b)  # warmup
        actual_bilqr_bytes = @allocated bilqr(A, b, b)
        @test expected_bilqr_bytes ≤ actual_bilqr_bytes ≤ 1.02 * expected_bilqr_bytes

        solver = BilqrSolver(A, b)
        bilqr!(solver, A, b, b)  # warmup
        inplace_bilqr_bytes = @allocated bilqr!(solver, A, b, b)
        @test inplace_bilqr_bytes == 0
      end

      @testset "USYMLQ" begin
        # USYMLQ needs:
        # - 5 n-vectors: uₖ₋₁, uₖ, x, d̅, p
        # - 3 m-vectors: vₖ₋₁, vₖ, q
        storage_usymlq_bytes(m, n) = nbits_FC * (5 * n + 3 * m)

        expected_usymlq_bytes = storage_usymlq_bytes(k, n)
        usymlq(Au, c, b)  # warmup
        actual_usymlq_bytes = @allocated usymlq(Au, c, b)
        @test expected_usymlq_bytes ≤ actual_usymlq_bytes ≤ 1.02 * expected_usymlq_bytes

        solver = UsymlqSolver(Au, c)
        usymlq!(solver, Au, c, b)  # warmup
        inplace_usymlq_bytes = @allocated usymlq!(solver, Au, c, b)
        @test inplace_usymlq_bytes == 0
      end

      @testset "USYMQR" begin
        # USYMQR needs:
        # - 6 n-vectors: vₖ₋₁, vₖ, x, wₖ₋₁, wₖ, p
        # - 3 m-vectors: uₖ₋₁, uₖ, q
        storage_usymqr_bytes(m, n) = nbits_FC * (6 * n + 3 * m)

        expected_usymqr_bytes = storage_usymqr_bytes(m, k)
        (x, stats) = usymqr(Ao, b, c) # warmup
        actual_usymqr_bytes = @allocated usymqr(Ao, b, c)
        @test expected_usymqr_bytes ≤ actual_usymqr_bytes ≤ 1.02 * expected_usymqr_bytes

        solver = UsymqrSolver(Ao, b)
        usymqr!(solver, Ao, b, c)  # warmup
        inplace_usymqr_bytes = @allocated usymqr!(solver, Ao, b, c)
        @test inplace_usymqr_bytes == 0
      end

      @testset "TriLQR" begin
        # TRILQR needs:
        # - 6 m-vectors: vₖ₋₁, vₖ, t, wₖ₋₁, wₖ, q
        # - 5 n-vectors: uₖ₋₁, uₖ, x, d̅, p
        storage_trilqr_bytes(m, n) = nbits_FC * (6 * m + 5 * n)

        expected_trilqr_bytes = storage_trilqr_bytes(n, n)
        trilqr(A, b, b)  # warmup
        actual_trilqr_bytes = @allocated trilqr(A, b, b)
        @test expected_trilqr_bytes ≤ actual_trilqr_bytes ≤ 1.02 * expected_trilqr_bytes

        solver = TrilqrSolver(A, b)
        trilqr!(solver, A, b, b)  # warmup
        inplace_trilqr_bytes = @allocated trilqr!(solver, A, b, b)
        @test inplace_trilqr_bytes == 0
      end

      @testset "TriCG" begin
        # TriCG needs:
        # - 6 n-vectors: yₖ, uₖ₋₁, uₖ, gy₂ₖ₋₁, gy₂ₖ, p
        # - 6 m-vectors: xₖ, vₖ₋₁, vₖ, gx₂ₖ₋₁, gx₂ₖ, q
        storage_tricg_bytes(m, n) = nbits_FC * (6 * n + 6 * m)

        expected_tricg_bytes = storage_tricg_bytes(k, n)
        tricg(Au, c, b)  # warmup
        actual_tricg_bytes = @allocated tricg(Au, c, b)
        @test expected_tricg_bytes ≤ actual_tricg_bytes ≤ 1.02 * expected_tricg_bytes

        solver = TricgSolver(Au, c)
        tricg!(solver, Au, c, b)  # warmup
        inplace_tricg_bytes = @allocated tricg!(solver, Au, c, b)
        @test inplace_tricg_bytes == 0
      end

      @testset "TriMR" begin
        # TriMR needs:
        # - 8 n-vectors: yₖ, uₖ₋₁, uₖ, gy₂ₖ₋₃, gy₂ₖ₋₂, gy₂ₖ₋₁, gy₂ₖ, p
        # - 8 m-vectors: xₖ, vₖ₋₁, vₖ, gx₂ₖ₋₃, gx₂ₖ₋₂, gx₂ₖ₋₁, gx₂ₖ, q
        storage_trimr_bytes(m, n) = nbits_FC * (8 * n + 8 * m)

        expected_trimr_bytes = storage_trimr_bytes(k, n)
        trimr(Au, c, b)  # warmup
        actual_trimr_bytes = @allocated trimr(Au, c, b)
        @test expected_trimr_bytes ≤ actual_trimr_bytes ≤ 1.02 * expected_trimr_bytes

        solver = TrimrSolver(Au, c)
        trimr!(solver, Au, c, b)  # warmup
        inplace_trimr_bytes = @allocated trimr!(solver, Au, c, b)
        @test inplace_trimr_bytes == 0
      end

      @testset "GPMR" begin
        # GPMR needs:
        # - 2 m-vectors: x, q
        # - 2 n-vectors: y, p
        # - 1 (m*mem)-matrix: V
        # - 1 (n*mem)-matrix: U
        # - 1 (2*mem)-vector: zt
        # - 2 (4*mem)-vectors: gc, gs
        # - 1 (mem*(2mem+1))-vector: R
        storage_gpmr_bytes(mem, m, n) = nbits_FC * ((mem + 2) * (n + m) + mem * (2 * mem + 7)) + nbits_T * 4 * mem

        expected_gpmr_bytes = storage_gpmr_bytes(mem, m, k)
        gpmr(Ao, Au, b, c, memory=mem, itmax=mem)  # warmup
        actual_gpmr_bytes = @allocated gpmr(Ao, Au, b, c, memory=mem, itmax=mem)
        @test expected_gpmr_bytes ≤ actual_gpmr_bytes ≤ 1.02 * expected_gpmr_bytes

        solver = GpmrSolver(Ao, b, mem)
        gpmr!(solver, Ao, Au, b, c)  # warmup
        inplace_gpmr_bytes = @allocated gpmr!(solver, Ao, Au, b, c)
        @test inplace_gpmr_bytes == 0
      end

      @testset "BLOCK-GMRES" begin
        # BLOCK-GMRES needs:
        # - 2 (n*p)-matrices: X, W
        # - 1 (p*p)-matrix: C
        # - 1 (2p*p)-matrix: D
        # - mem p-vectors: τ
        # - mem (n*p)-matrices: V
        # - mem (p*p)-matrices: Z
        # - mem*(mem+1)/2 (p*p)-matrices: R
        # - mem (2p*p)-matrices: H
        function storage_block_gmres_bytes(mem, n, p)
          res = (2*n*p + p*p + 2p*p + mem*p + mem*n*p + mem*p*p + mem*(mem+1)*p*p/2 + mem*2p*p)
          return nbits_FC * res
        end

        expected_block_gmres_bytes = storage_block_gmres_bytes(mem, n, p)
        block_gmres(A, B, memory=mem, itmax=mem)  # warmup
        actual_block_gmres_bytes = @allocated block_gmres(A, B, memory=mem, itmax=mem)
        @test expected_block_gmres_bytes ≤ actual_block_gmres_bytes ≤ 1.02 * expected_block_gmres_bytes

        solver = BlockGmresSolver(A, B, mem)
        block_gmres!(solver, A, B)  # warmup
        inplace_block_gmres_bytes = @allocated block_gmres!(solver, A, B)
        @test inplace_block_gmres_bytes == 0
      end

      # @testset "BLOCK-MINRES" begin
      #   # BLOCK-MINRES needs:
      #   # - 2 (n*p)-matrices: X, W
      #   # - 1 (p*p)-matrix: C
      #   # - 1 (2p*p)-matrix: D
      #   # - mem p-vectors: τ
      #   # - mem (n*p)-matrices: V
      #   # - mem (p*p)-matrices: Z
      #   # - mem*(mem+1)/2 (p*p)-matrices: R
      #   # - mem (2p*p)-matrices: H
      #   function storage_block_minres_bytes(mem, n, p)
      #     res = (2*n*p + p*p + 2p*p + mem*p + mem*n*p + mem*p*p + mem*(mem+1)*p*p/2 + mem*2p*p)
      #     return nbits_FC * res
      #   end

      #   expected_block_minres_bytes = storage_block_minres_bytes(mem, n, p)
      #   block_minres(A, B)  # warmup
      #   actual_block_minres_bytes = @allocated block_minres(A, B)
      #   @test expected_block_minres_bytes ≤ actual_block_minres_bytes ≤ 1.02 * expected_block_minres_bytes

      #   solver = BlockGmresSolver(A, B, mem)
      #   block_minres!(solver, A, B)  # warmup
      #   inplace_block_minres_bytes = @allocated block_minres!(solver, A, B)
      #   @test inplace_block_minres_bytes == 0
      # end
    end
  end
end
