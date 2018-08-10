using Base.Test
using Krylov
using LinearOperators

include("test_dqgmres.jl")
include("gen_lsq.jl")
include("check_min_norm.jl")
include("test_aux.jl")
include("test_cg.jl")
include("test_cg_lanczos.jl")
include("test_minres.jl")
include("test_symmlq.jl")
include("test_cgls.jl")
include("test_crls.jl")
include("test_cgne.jl")
include("test_crmr.jl")
include("test_variants.jl")
include("test_lslq.jl")
include("test_lsqr.jl")
include("test_lsmr.jl")
include("test_craig.jl")
include("test_craigmr.jl")
include("test_cr.jl")
