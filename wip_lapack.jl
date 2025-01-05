import LinearAlgebra.BLAS.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK.liblapack

for (Xgeqrf, Xorgqr, Xormqr, T) in ((:sgeqrf, :sorgqr, :sormqr, :Float32),
                                    (:dgeqrf, :dorgqr, :dormqr, :Float64),
                                    (:cgeqrf, :cungqr, :cunmqr, :ComplexF32),
                                    (:zgeqrf, :zungqr, :zunmqr, :ComplexF64))
    @eval begin
        function kgeqrf_buffer!(A :: Matrix{$T}, tau :: Vector{$T})
            symb = @blasfunc($Xgeqrf)
            m, n = size(A)
            work  = Ref{$T}(0)
            lwork = -1
            info  = Ref{BlasInt}()
            lda = max(1, stride(A,2))
            @ccall liblapack.dgeqrf_64_(m::Ref{BlasInt}, n::Ref{BlasInt}, A::Ptr{$T},
                                        lda::Ref{BlasInt}, tau::Ptr{$T}, work::Ptr{$T},
                                        lwork::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
            return work[] |> Int
        end

        function kgeqrf!(A :: Matrix{$T}, tau :: Vector{$T}, work :: Vector{$T})
            symb = @blasfunc($Xgeqrf)
            m, n = size(A)
            # lwork = length(work)
            lwork = kgeqrf_buffer!(A, tau)
            @assert lwork ≤ length(work)
            info  = Ref{BlasInt}()
            lda = max(1,stride(A,2))
            @ccall liblapack.dgeqrf_64_(m::Ref{BlasInt}, n::Ref{BlasInt}, A::Ptr{$T},
                                        lda::Ref{BlasInt}, tau::Ptr{$T}, work::Ptr{$T},
                                        lwork::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
            return nothing
        end

        function korgqr_buffer!(A :: Matrix{$T}, tau::Vector{$T})
            symb = @blasfunc($Xorgqr)
            m, n = size(A)
            k = length(tau)
            work  = Ref{$T}(0)
            lwork = -1
            info  = Ref{BlasInt}()
            lda = max(1, stride(A,2))
            info  = Ref{BlasInt}()
            @ccall liblapack.dorgqr_64_(m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                        A::Ptr{$T}, lda::Ref{BlasInt}, tau::Ptr{$T}, work::Ptr{$T},
                                        lwork::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
            return work[] |> Int
        end

        function korgqr!(A :: Matrix{$T}, tau::Vector{$T}, work::Vector{$T})
            symb = @blasfunc($Xorgqr)
            m, n = size(A)
            k = length(tau)
            # lwork = length(work)
            lwork = korgqr_buffer!(A, tau)
            @assert lwork ≤ length(work)
            info  = Ref{BlasInt}()
            lda = max(1,stride(A,2))
            info  = Ref{BlasInt}()
            @ccall liblapack.dorgqr_64_(m::Ref{BlasInt}, n::Ref{BlasInt}, k::Ref{BlasInt},
                                        A::Ptr{$T}, lda::Ref{BlasInt}, tau::Ptr{$T}, work::Ptr{$T},
                                        lwork::Ref{BlasInt}, info::Ptr{BlasInt})::Cvoid
            return nothing
        end

        function kormqr_buffer!(side::Char, trans::Char, A::Matrix{$T}, tau::Vector{$T}, C::Matrix{$T})
            symb = @blasfunc($Xormqr)
            m, n = size(A)
            k = length(tau)
            work = Ref{$T}(0)
            lwork = -1
            info = Ref{BlasInt}()
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            @ccall liblapack.dormqr_64_(side::Ref{UInt8}, trans::Ref{UInt8}, m::Ref{BlasInt},
                                        n::Ref{BlasInt}, k::Ref{BlasInt}, A::Ptr{$T},
                                        lda::Ref{BlasInt}, tau::Ptr{$T}, C::Ptr{$T},
                                        ldc::Ref{BlasInt}, work::Ptr{$T}, lwork::Ref{BlasInt},
                                        info::Ref{BlasInt}, 1::Clong, 1::Clong)::Cvoid
            return work[] |> BlasInt
        end

        function kormqr!(side::Char, trans::Char, A::Matrix{$T}, tau::Vector{$T}, C::Matrix{$T}, work::Vector{$T})
            symb = @blasfunc($Xormqr)
            m, n = size(A)
            k = length(tau)
            # lwork = length(work)
            lwork = kormqr_buffer!(side, trans, A, tau, C)
            @assert lwork ≤ length(work)
            info = Ref{BlasInt}()
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            @ccall liblapack.dormqr_64_(side::Ref{UInt8}, trans::Ref{UInt8}, m::Ref{BlasInt},
                                        n::Ref{BlasInt}, k::Ref{BlasInt}, A::Ptr{$T},
                                        lda::Ref{BlasInt}, tau::Ptr{$T}, C::Ptr{$T},
                                        ldc::Ref{BlasInt}, work::Ptr{$T}, lwork::Ref{BlasInt},
                                        info::Ref{BlasInt}, 1::Clong, 1::Clong)::Cvoid
            return nothing
        end
    end
end
