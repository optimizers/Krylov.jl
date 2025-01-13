function test_verbose(FC)
  A   = FC.(get_div_grad(4, 4, 4))  # Dimension m x n
  m,n = size(A)
  k   = div(n, 2)
  Au  = A[1:k,:]          # Dimension k x n
  Ao  = A[:,1:k]          # Dimension m x k
  b   = Ao * ones(FC, k)  # Dimension m
  c   = Au * ones(FC, n)  # Dimension k
  mem = 10

  T = real(FC)
  shifts  = T[1; 2; 3; 4; 5]
  nshifts = 5

  for fn in (:cg, :cgls, :usymqr, :cgne, :cgs, :crmr, :cg_lanczos, :dqgmres, :diom, :cr, :gpmr,
             :lslq, :lsqr, :lsmr, :lnlq, :craig, :bicgstab, :craigmr, :crls, :symmlq, :minres,
             :bilq, :minres_qlp, :qmr, :usymlq, :tricg, :trimr, :trilqr, :bilqr, :gmres, :fom,
             :car, :minares, :fgmres, :usymlqr :cg_lanczos_shift, :cgls_lanczos_shift)

    @testset "$fn" begin
      io = IOBuffer()
      if fn in (:trilqr, :bilqr)
        @eval $fn($A, $b, $b, verbose=1, iostream=$io)
      elseif fn in (:tricg, :trimr, :usymlqr)
        @eval $fn($Au, $c, $b, verbose=1, iostream=$io)
      elseif fn in (:lnlq, :craig, :craigmr, :cgne, :crmr)
        @eval $fn($Au, $c, verbose=1, iostream=$io)
      elseif fn in (:lslq, :lsqr, :lsmr, :cgls, :crls)
        @eval $fn($Ao, $b, verbose=1, iostream=$io)
      elseif fn == :usymlq
        @eval $fn($Au, $c, $b, verbose=1, iostream=$io)
      elseif fn == :usymqr
        @eval $fn($Ao, $b, $c, verbose=1, iostream=$io)
      elseif fn == :gpmr
        @eval $fn($Ao, $Au, $b, $c, verbose=1, iostream=$io)
      elseif fn in (:cg_lanczos_shift, :cgls_lanczos_shift)
        @eval $fn($A, $b, $shifts, verbose=1, iostream=$io)
      else
        @eval $fn($A, $b, verbose=1, iostream=$io)
      end

      showed = String(take!(io))
      str = split(showed, '\n', keepempty=false)
      nrows = length(str)
      first_row = fn in (:bilqr, :trilqr) ? 3 : 2
      last_row = fn == :cg ? nrows-1 : nrows
      str = str[first_row:last_row]
      len_header = length(str[1])
      @test mapreduce(x -> length(x) == len_header, &, str)
    end
  end
end

@testset "verbose" begin
  for FC in (Float64, ComplexF64)
    @testset "Data Type: $FC" begin
      test_verbose(FC)
    end
  end
end
