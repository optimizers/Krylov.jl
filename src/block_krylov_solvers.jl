export BlockKrylovSolver

export BlockMinresSolver, BlockGmresSolver

const BLOCK_KRYLOV_SOLVERS = Dict(:block_minres => :BlockMinresSolver,
                                  :block_gmres  => :BlockGmresSolver )

"Abstract type for using block Krylov solvers in-place"
abstract type BlockKrylovSolver{T,FC,SV,SM} end

"""
Type for storing the vectors required by the in-place version of BLOCK-MINRES.

The outer constructors

    solver = BlockMinresSolver(m, n, p, SV, SM)
    solver = BlockMinresSolver(A, B)

may be used in order to create these vectors.
`memory` is set to `div(n,p)` if the value given is larger than `div(n,p)`.
"""
mutable struct BlockMinresSolver{T,FC,SV,SM} <: BlockKrylovSolver{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  P          :: SM
  Q          :: SM
  C          :: SM
  D          :: SM
  Φ          :: SM
  tmp        :: SM
  Vₖ₋₁       :: SM
  Vₖ         :: SM
  wₖ₋₂       :: SM
  wₖ₋₁       :: SM
  Hₖ₋₂       :: SM
  Hₖ₋₁       :: SM
  τₖ₋₂       :: SV
  τₖ₋₁       :: SV
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockMinresSolver(m, n, p, SV, SM)
  FC = eltype(SV)
  T  = real(FC)
  ΔX = SM(undef, 0, 0)
  X  = SM(undef, n, p)
  P  = SM(undef, 0, 0)
  Q  = SM(undef, n, p)
  C  = SM(undef, p, p)
  D  = SM(undef, 2p, p)
  Φ  = SM(undef, p, p)
  tmp = C isa Matrix ? SM(undef, 0, 0) : SM(undef, p, p)
  Vₖ₋₁ = SM(undef, n, p)
  Vₖ   = SM(undef, n, p)
  wₖ₋₂ = SM(undef, n, p)
  wₖ₋₁ = SM(undef, n, p)
  Hₖ₋₂ = SM(undef, 2p, p)
  Hₖ₋₁ = SM(undef, 2p, p)
  τₖ₋₂ = SV(undef, p)
  τₖ₋₁ = SV(undef, p)
  stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
  solver = BlockMinresSolver{T,FC,SV,SM}(m, n, p, ΔX, X, P, Q, C, D, Φ, tmp, Vₖ₋₁, Vₖ, wₖ₋₂, wₖ₋₁, Hₖ₋₂, Hₖ₋₁, τₖ₋₂, τₖ₋₁, false, stats)
  return solver
end

function BlockMinresSolver(A, B)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockMinresSolver(m, n, p, SV, SM)
end

"""
Type for storing the vectors required by the in-place version of BLOCK-GMRES.

The outer constructors

    solver = BlockGmresSolver(m, n, p, memory, SV, SM)
    solver = BlockGmresSolver(A, B, memory = 5)

may be used in order to create these vectors.
`memory` is set to `div(n,p)` if the value given is larger than `div(n,p)`.
`memory` is an optional argument in the second constructor.
"""
mutable struct BlockGmresSolver{T,FC,SV,SM} <: BlockKrylovSolver{T,FC,SV,SM}
  m          :: Int
  n          :: Int
  p          :: Int
  ΔX         :: SM
  X          :: SM
  W          :: SM
  P          :: SM
  Q          :: SM
  C          :: SM
  D          :: SM
  V          :: Vector{SM}
  Z          :: Vector{SM}
  R          :: Vector{SM}
  H          :: Vector{SM}
  τ          :: Vector{SV}
  tmp        :: SM
  size_buffer:: Vector{Int}
  buffer     :: SV
  warm_start :: Bool
  stats      :: SimpleStats{T}
end

function BlockGmresSolver(m, n, p, memory, SV, SM)
  memory = min(div(n,p), memory)
  FC = eltype(SV)
  T  = real(FC)
  ΔX = SM(undef, 0, 0)
  X  = SM(undef, n, p)
  W  = SM(undef, n, p)
  P  = SM(undef, 0, 0)
  Q  = SM(undef, 0, 0)
  C  = SM(undef, p, p)
  D  = SM(undef, 2p, p)
  V  = SM[SM(undef, n, p) for i = 1 : memory]
  Z  = SM[SM(undef, p, p) for i = 1 : memory]
  R  = SM[SM(undef, p, p) for i = 1 : div(memory * (memory+1), 2)]
  H  = SM[SM(undef, 2p, p) for i = 1 : memory]
  τ  = SV[SV(undef, p) for i = 1 : memory]
  tmp = C isa Matrix ? SM(undef, 0, 0) : SM(undef, p, p)
  trans = FC <: AbstractFloat ? 'T' : 'C'
  size_buffer = Int[kgeqrf_buffer!(V[1], τ[1]), korgqr_buffer!(V[1], τ[1]), kgeqrf_buffer!(H[1], τ[1]), korgqr_buffer!(H[1], τ[1]), kormqr_buffer!('L', trans, H[1], τ[1], D)]
  buffer = C isa Matrix ? SV(undef, 0) : SV(undef, size_buffer |> maximum)
  stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
  solver = BlockGmresSolver{T,FC,SV,SM}(m, n, p, ΔX, X, W, P, Q, C, D, V, Z, R, H, τ, tmp, size_buffer, buffer, false, stats)
  return solver
end

function BlockGmresSolver(A, B, memory = 5)
  m, n = size(A)
  s, p = size(B)
  SM = typeof(B)
  SV = matrix_to_vector(SM)
  BlockGmresSolver(m, n, p, memory, SV, SM)
end

for (KS, fun, nsol, nA, nAt, warm_start) in [
  (:BlockMinresSolver, :block_minres!, 1, 1, 0, true)
  (:BlockGmresSolver , :block_gmres! , 1, 1, 0, true)
]
  @eval begin
    size(solver :: $KS) = solver.m, solver.n
    nrhs(solver :: $KS) = solver.p
    statistics(solver :: $KS) = solver.stats
    niterations(solver :: $KS) = solver.stats.niter
    Aprod(solver :: $KS) = $nA * solver.stats.niter
    Atprod(solver :: $KS) = $nAt * solver.stats.niter
    nsolution(solver :: $KS) = $nsol
    if $nsol == 1
      solution(solver :: $KS) = solver.X
      solution(solver :: $KS, p :: Integer) = (p == 1) ? solution(solver) : error("solution(solver) has only one output.")
      results(solver :: $KS) = (solver.X, solver.stats)
    end
    issolved(solver :: $KS) = solver.stats.solved
    if $warm_start
      function warm_start!(solver :: $KS, X0)
        n, p = size(solver.X)
        n2, p2 = size(X0)
        SM = typeof(solver.X)
        (n == n2 && p == p2) || error("X0 should have size ($n, $p)")
        allocate_if(true, solver, :ΔX, SM, n, p)
        copyto!(solver.ΔX, X0)
        solver.warm_start = true
        return solver
      end
    end
  end
end

function ksizeof(attribute)
  if isa(attribute, Vector{<:AbstractVector}) && !isempty(attribute)
    # A vector of vectors is a vector of pointers in Julia.
    # All vectors inside a vector have the same size in Krylov.jl
    size_attribute = sizeof(attribute) + length(attribute) * ksizeof(attribute[1])
  else
    size_attribute = sizeof(attribute)
  end
  return size_attribute
end

function sizeof(stats_solver :: Union{KrylovStats, KrylovSolver, BlockKrylovSolver})
  type = typeof(stats_solver)
  nfields = fieldcount(type)
  storage = 0
  for i = 1:nfields
    field_i = getfield(stats_solver, i)
    size_i = ksizeof(field_i)
    storage += size_i
  end
  return storage
end

"""
    show(io, solver; show_stats=true)

Statistics of `solver` are displayed if `show_stats` is set to true.
"""
function show(io :: IO, solver :: Union{KrylovSolver{T,FC,S}, BlockKrylovSolver{T,FC,S}}; show_stats :: Bool=true) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
  workspace = typeof(solver)
  name_solver = string(workspace.name.name)
  name_stats = string(typeof(solver.stats).name.name)
  nbytes = sizeof(solver)
  storage = format_bytes(nbytes)
  architecture = S <: Vector ? "CPU" : "GPU"
  l1 = max(length(name_solver), length(string(FC)) + 11)  # length("Precision: ") = 11
  nchar = workspace <: Union{CgLanczosShiftSolver, FomSolver, DiomSolver, DqgmresSolver, GmresSolver, FgmresSolver, GpmrSolver, BlockGmresSolver} ? 8 : 0  # length("Vector{}") = 8
  l2 = max(ndigits(solver.m) + 7, length(architecture) + 14, length(string(S)) + nchar)  # length("nrows: ") = 7 and length("Architecture: ") = 14
  l2 = max(l2, length(name_stats) + 2 + length(string(T)))  # length("{}") = 2
  l3 = max(ndigits(solver.n) + 7, length(storage) + 9)  # length("Storage: ") = 9 and length("cols: ") = 7
  format = Printf.Format("│%$(l1)s│%$(l2)s│%$(l3)s│\n")
  format2 = Printf.Format("│%$(l1+1)s│%$(l2)s│%$(l3)s│\n")
  @printf(io, "┌%s┬%s┬%s┐\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "$(name_solver)", "nrows: $(solver.m)", "ncols: $(solver.n)")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Precision: $FC", "Architecture: $architecture","Storage: $storage")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  Printf.format(io, format, "Attribute", "Type", "Size")
  @printf(io, "├%s┼%s┼%s┤\n", "─"^l1, "─"^l2, "─"^l3)
  for i=1:fieldcount(workspace)
    name_i = fieldname(workspace, i)
    type_i = fieldtype(workspace, i)
    field_i = getfield(solver, name_i)
    size_i = ksizeof(field_i)
    if (name_i::Symbol in [:w̅, :w̄, :d̅]) && (VERSION < v"1.8.0-DEV")
      (size_i ≠ 0) && Printf.format(io, format2, string(name_i), type_i, format_bytes(size_i))
    else
      (size_i ≠ 0) && Printf.format(io, format, string(name_i), type_i, format_bytes(size_i))
    end
  end
  @printf(io, "└%s┴%s┴%s┘\n","─"^l1,"─"^l2,"─"^l3)
  if show_stats
    @printf(io, "\n")
    show(io, solver.stats)
  end
  return nothing
end
