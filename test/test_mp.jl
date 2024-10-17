@testset "mp" begin
  n = 5
  for fn in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos, :dqgmres, :diom, :cr, :gpmr,
             :lslq, :lsqr, :lsmr, :lnlq, :craig, :bicgstab, :craigmr, :crls, :symmlq, :minres,
             :bilq, :minres_qlp, :qmr, :usymlq, :tricg, :trimr, :trilqr, :bilqr, :gmres, :fom,
             :car, :minares, :fgmres, :cg_lanczos_shift, :cgls_lanczos_shift)
    @testset "$fn" begin
      for T in (Float16, Float32, Float64, BigFloat)
        for FC in (T, Complex{T})
          A = spdiagm(-1 => -ones(FC,n-1), 0 => 3*ones(FC,n), 1 => -ones(FC,n-1))
          B = spdiagm(-1 => -ones(FC,n-1), 0 => 5*ones(FC,n), 1 => -ones(FC,n-1))
          b = ones(FC, n)
          c = -ones(FC, n)
          shifts = [-one(T), one(T)]
          if fn in (:usymlq, :usymqr)
            x, _ = @eval $fn($A, $b, $c)
          elseif fn in (:trilqr, :bilqr)
            x, t, _ = @eval $fn($A, $b, $c)
          elseif fn in (:tricg, :trimr, :usymlqr)
            x, y, _ = @eval $fn($A, $b, $c)
          elseif fn == :gpmr
            x, y, _ = @eval $fn($A, $B, $b, $c)
          elseif fn in (:lnlq, :craig, :craigmr)
            x, y, _ = @eval $fn($A, $b)
          elseif fn in (:cg_lanczos_shift, :cgls_lanczos_shift)
            x, _ = @eval $fn($A, $b, $shifts)
          else
            x, _ = @eval $fn($A, $b)
          end
          atol = √eps(T)
          rtol = √eps(T)
          Κ = (T == Float16 ? 10 : 1)
          if fn in (:tricg, :trimr)
            @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
            @test norm(A' * x - y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
            @test eltype(y) == FC
          if fn == :usymlqr
            @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
            @test norm(A' * x - c) ≤ Κ * (atol + norm([b; c]) * rtol)
            @test eltype(y) == FC
          elseif fn == :gpmr
            @test norm(x + A * y - b) ≤ Κ * (atol + norm([b; c]) * rtol)
            @test norm(B * x + y - c) ≤ Κ * (atol + norm([b; c]) * rtol)
            @test eltype(y) == FC
          elseif fn == :cg_lanczos_shift
            @test norm((A - I) * x[1] - b) ≤ Κ * (atol + norm(b) * rtol)
            @test norm((A + I) * x[2] - b) ≤ Κ * (atol + norm(b) * rtol)
            @test eltype(x) == Vector{FC}
          elseif fn == :cgls_lanczos_shift
            @test norm(A' * (b - A * x[1]) + x[1]) ≤ Κ * (atol + norm(A' * b) * rtol)
            @test norm(A' * (b - A * x[2]) - x[2]) ≤ Κ * (atol + norm(A' * b) * rtol)
            @test eltype(x) == Vector{FC}
          else
            @test norm(A * x - b) ≤ Κ * (atol + norm(b) * rtol)
            @test eltype(x) == FC
          end
          if fn in (:trilqr, :bilqr)
            @test norm(A' * t - c) ≤ Κ * (atol + norm(c) * rtol)
            @test eltype(t) == FC
          end
          if fn in (:lnlq, :craig, :craigmr)
            @test norm(A * A' * y - b) ≤ Κ * (atol + norm(b) * rtol)
            @test eltype(y) == FC
          end
        end
      end
    end
  end
end
