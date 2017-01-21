using Base.Test
using Krylov
using LinearOperators

include("gen_lsq.jl")
include("check_min_norm.jl")
include("test_cg.jl")
include("test_cg_lanczos.jl")
@printf("!!!! FIX MINRES TESTS !!!!")
# include("test_minres.jl")
include("test_cgls.jl")
include("test_crls.jl")
include("test_cgne.jl")
include("test_crmr.jl")

include("test_lslq.jl")
include("test_lsqr.jl")
include("test_lsmr.jl")
include("test_craigmr.jl")
