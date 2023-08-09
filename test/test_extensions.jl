using ComponentArrays
using FillArrays
using StaticArrays

@testset "extensions" begin
    @testset "ComponentArrays" begin
        n = 5
        for T in (Float32, Float64)
            A = rand(T, n, n)

            b = ComponentVector(; b1=rand(T, n - 1), b2=rand(T))
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved
        end
    end

    @testset "FillArrays" begin
        n = 5
        for T in (Float32, Float64)
            A = rand(T, n, n)

            b = Ones(T, n)
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved

            b = Zeros(T, n)
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved
        end
    end

    @testset "StaticArrays" begin
        n = 5
        for T in (Float32, Float64)
            A = rand(T, n, n)

            b = SVector{n}(rand(T, n))
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved

            b = MVector{n}(rand(T, n))
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved

            b = SizedVector{n}(rand(T, n))
            @test Krylov.ktypeof(b) == Vector{T}
            x, stats = gmres(A, b)
            @test stats.solved
        end
    end
end
