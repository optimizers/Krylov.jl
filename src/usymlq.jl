# An implementation of USYMLQ for the solution of linear system Ax = b.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point and quasi-definite systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, November 2018.

export usymlq, usymlq!

"""
    (x, stats) = usymlq(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                        transfer_to_usymcg::Bool=true, atol::T=√eps(T),
                        rtol::T=√eps(T), itmax::Int=0,
                        timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                        callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, stats) = usymlq(A, b, c, x0::AbstractVector; kwargs...)

USYMLQ can be warm-started from an initial guess `x0` where `kwargs` are the same keyword arguments as above.

USYMLQ determines the least-norm solution of the consistent linear system Ax = b of size m × n.

USYMLQ is based on the orthogonal tridiagonalization process and requires two initial nonzero vectors `b` and `c`.
The vector `c` is only used to initialize the process and a default value can be `b` or `Aᴴb` depending on the shape of `A`.
The error norm ‖x - x*‖ monotonously decreases in USYMLQ.
When `A` is Hermitian and `b = c`, USYMLQ is equivalent to SYMMLQ.
USYMLQ is considered as a generalization of SYMMLQ.

It can also be applied to under-determined and over-determined problems.
In all cases, problems must be consistent.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`.

#### Optional argument

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`.

#### Keyword arguments

* `transfer_to_usymcg`: transfer from the USYMLQ point to the USYMCG point, when it exists. The transfer is based on the residual norm;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length `n`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function usymlq end

"""
    solver = usymlq!(solver::UsymlqSolver, A, b, c; kwargs...)
    solver = usymlq!(solver::UsymlqSolver, A, b, c, x0; kwargs...)

where `kwargs` are keyword arguments of [`usymlq`](@ref).

See [`UsymlqSolver`](@ref) for more details about the `solver`.
"""
function usymlq! end

def_args_usymlq = (:(A                    ),
                   :(b::AbstractVector{FC}),
                   :(c::AbstractVector{FC}))

def_optargs_usymlq = (:(x0::AbstractVector),)

def_kwargs_usymlq = (:(; transfer_to_usymcg::Bool = true),
                     :(; atol::T = √eps(T)              ),
                     :(; rtol::T = √eps(T)              ),
                     :(; itmax::Int = 0                 ),
                     :(; timemax::Float64 = Inf         ),
                     :(; verbose::Int = 0               ),
                     :(; history::Bool = false          ),
                     :(; callback = solver -> false     ),
                     :(; iostream::IO = kstdout         ))

def_kwargs_usymlq = extract_parameters.(def_kwargs_usymlq)

args_usymlq = (:A, :b, :c)
optargs_usymlq = (:x0,)
kwargs_usymlq = (:transfer_to_usymcg, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function usymlq!(solver :: UsymlqSolver{T,FC,S}, $(def_args_usymlq...); $(def_kwargs_usymlq...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "USYMLQ: system of %d equations in %d variables\n", m, n)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    uₖ₋₁, uₖ, p, Δx, x = solver.uₖ₋₁, solver.uₖ, solver.p, solver.Δx, solver.x
    vₖ₋₁, vₖ, q, d̅, stats = solver.vₖ₋₁, solver.vₖ, solver.q, solver.d̅, solver.stats
    warm_start = solver.warm_start
    rNorms = stats.residuals
    reset!(stats)
    r₀ = warm_start ? q : b

    if warm_start
      mul!(r₀, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r₀)
    end

    # Initial solution x₀ and residual norm ‖r₀‖.
    kfill!(x, zero(FC))
    bNorm = knorm(m, r₀)
    history && push!(rNorms, bNorm)
    if bNorm == 0
      stats.niter = 0
      stats.solved = true
      stats.inconsistent = false
      stats.timer = start_time |> ktimer
      stats.status = "x = 0 is a zero-residual solution"
      solver.warm_start = false
      return solver
    end

    iter = 0
    itmax == 0 && (itmax = m+n)

    ε = atol + rtol * bNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s\n", "k", "‖rₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, bNorm, start_time |> ktimer)

    βₖ = knorm(m, r₀)           # β₁ = ‖v₁‖ = ‖r₀‖
    γₖ = knorm(n, c)            # γ₁ = ‖u₁‖ = ‖c‖
    kfill!(vₖ₋₁, zero(FC))      # v₀ = 0
    kfill!(uₖ₋₁, zero(FC))      # u₀ = 0
    vₖ .= r₀ ./ βₖ              # v₁ = (b - Ax₀) / β₁
    uₖ .= c ./ γₖ               # u₁ = c / γ₁
    cₖ₋₁ = cₖ = -one(T)         # Givens cosines used for the LQ factorization of Tₖ
    sₖ₋₁ = sₖ = zero(FC)        # Givens sines used for the LQ factorization of Tₖ
    kfill!(d̅, zero(FC))         # Last column of D̅ₖ = Uₖ(Qₖ)ᴴ
    ζₖ₋₁ = ζbarₖ = zero(FC)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
    ζₖ₋₂ = ηₖ = zero(FC)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
    δbarₖ₋₁ = δbarₖ = zero(FC)  # Coefficients of Lₖ₋₁ and Lₖ modified over the course of two iterations

    # Stopping criterion.
    solved_lq = bNorm ≤ ε
    solved_cg = false
    tired     = iter ≥ itmax
    status    = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved_lq || solved_cg || tired || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the SSY tridiagonalization process.
      # AUₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = Uₖ(Tₖ)ᴴ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms vₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms uₖ₊₁ : p ← Aᴴvₖ

      kaxpy!(m, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
      kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - βₖ * uₖ₋₁

      αₖ = kdot(m, vₖ, q)      # αₖ = ⟨vₖ,q⟩

      if iter ≥ 2
        kaxpy!(m, -     αₖ , vₖ, q)    # q ← q - αₖ * vₖ
        kaxpy!(n, -conj(αₖ), uₖ, p)    # p ← p - ᾱₖ * uₖ
      end

      βₖ₊₁ = knorm(m, q)       # βₖ₊₁ = ‖q‖
      γₖ₊₁ = knorm(n, p)       # γₖ₊₁ = ‖p‖

      # Update the LQ factorization of Tₖ = L̅ₖQₖ.
      # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
      # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
      # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
      # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
      # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
      # [ •        •  •  •  γₖ]   [ •         •   •   •    •    0   ]
      # [ 0  •  •  •  0  βₖ αₖ]   [ •    •    •   0  ϵₖ₋₂ λₖ₋₁ δbarₖ]

      if iter == 1
        δbarₖ = αₖ
      elseif iter == 2
        # [δbar₁ γ₂] [c₂  s̄₂] = [δ₁   0  ]
        # [ β₂   α₂] [s₂ -c₂]   [λ₁ δbar₂]
        (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
        λₖ₋₁  =      cₖ  * βₖ + sₖ * αₖ
        δbarₖ = conj(sₖ) * βₖ - cₖ * αₖ
      else
        # [0  βₖ  αₖ] [cₖ₋₁   s̄ₖ₋₁   0] = [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ]
        #             [sₖ₋₁  -cₖ₋₁   0]
        #             [ 0      0     1]
        #
        # [ λₖ₋₂   δbarₖ₋₁  γₖ] [1   0   0 ] = [λₖ₋₂  δₖ₋₁    0  ]
        # [sₖ₋₁βₖ  -cₖ₋₁βₖ  αₖ] [0   cₖ  s̄ₖ]   [ϵₖ₋₂  λₖ₋₁  δbarₖ]
        #                       [0   sₖ -cₖ]
        (cₖ, sₖ, δₖ₋₁) = sym_givens(δbarₖ₋₁, γₖ)
        ϵₖ₋₂  =  sₖ₋₁ * βₖ
        λₖ₋₁  = -cₖ₋₁ *      cₖ  * βₖ + sₖ * αₖ
        δbarₖ = -cₖ₋₁ * conj(sₖ) * βₖ - cₖ * αₖ
      end

      # Compute ζₖ₋₁ and ζbarₖ, last components of the solution of L̅ₖz̅ₖ = β₁e₁
      # [δbar₁] [ζbar₁] = [β₁]
      if iter == 1
        ηₖ = βₖ
      end
      # [δ₁    0  ] [  ζ₁ ] = [β₁]
      # [λ₁  δbar₂] [ζbar₂]   [0 ]
      if iter == 2
        ηₖ₋₁ = ηₖ
        ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ηₖ   = -λₖ₋₁ * ζₖ₋₁
      end
      # [λₖ₋₂  δₖ₋₁    0  ] [ζₖ₋₂ ] = [0]
      # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ζₖ₋₁ ]   [0]
      #                     [ζbarₖ]
      if iter ≥ 3
        ζₖ₋₂ = ζₖ₋₁
        ηₖ₋₁ = ηₖ
        ζₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ηₖ   = -ϵₖ₋₂ * ζₖ₋₂ - λₖ₋₁ * ζₖ₋₁
      end

      # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = Uₖ(Qₖ)ᴴ.
      # [d̅ₖ₋₁ uₖ] [cₖ  s̄ₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * uₖ
      #           [sₖ -cₖ]             ⟷ d̅ₖ   = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
      if iter ≥ 2
        # Compute solution xₖ.
        # (xᴸ)ₖ₋₁ ← (xᴸ)ₖ₋₂ + ζₖ₋₁ * dₖ₋₁
        kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
        kaxpy!(n, ζₖ₋₁ * sₖ, uₖ, x)
      end

      # Compute d̅ₖ.
      if iter == 1
        # d̅₁ = u₁
        kcopy!(n, d̅, uₖ)  # d̅ ← vₖ
      else
        # d̅ₖ = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
        kaxpby!(n, -cₖ, uₖ, conj(sₖ), d̅)
      end

      # Compute vₖ₊₁ and uₖ₊₁.
      kcopy!(m, vₖ₋₁, vₖ)  # vₖ₋₁ ← vₖ
      kcopy!(n, uₖ₋₁, uₖ)  # uₖ₋₁ ← uₖ

      if βₖ₊₁ ≠ zero(T)
        vₖ .= q ./ βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
      end
      if γₖ₊₁ ≠ zero(T)
        uₖ .= p ./ γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
      end

      # Compute USYMLQ residual norm
      # ‖rₖ‖ = √(|μₖ|² + |ωₖ|²)
      if iter == 1
        rNorm_lq = bNorm
      else
        μₖ = βₖ * (sₖ₋₁ * ζₖ₋₂ - cₖ₋₁ * cₖ * ζₖ₋₁) + αₖ * sₖ * ζₖ₋₁
        ωₖ = βₖ₊₁ * sₖ * ζₖ₋₁
        rNorm_lq = sqrt(abs2(μₖ) + abs2(ωₖ))
      end
      history && push!(rNorms, rNorm_lq)

      # Compute USYMCG residual norm
      # ‖rₖ‖ = |ρₖ|
      if transfer_to_usymcg && (abs(δbarₖ) > eps(T))
        ζbarₖ = ηₖ / δbarₖ
        ρₖ = βₖ₊₁ * (sₖ * ζₖ₋₁ - cₖ * ζbarₖ)
        rNorm_cg = abs(ρₖ)
      end

      # Update sₖ₋₁, cₖ₋₁, γₖ, βₖ and δbarₖ₋₁.
      sₖ₋₁    = sₖ
      cₖ₋₁    = cₖ
      γₖ      = γₖ₊₁
      βₖ      = βₖ₊₁
      δbarₖ₋₁ = δbarₖ

      # Update stopping criterion.
      user_requested_exit = callback(solver) :: Bool
      solved_lq = rNorm_lq ≤ ε
      solved_cg = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ ε)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %.2fs\n", iter, rNorm_lq, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Compute USYMCG point
    # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
    if solved_cg
      kaxpy!(n, ζbarₖ, d̅, x)
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    solved_lq           && (status = "solution xᴸ good enough given atol and rtol")
    solved_cg           && (status = "solution xᶜ good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Update x
    warm_start && kaxpy!(n, one(FC), Δx, x)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved_lq || solved_cg
    stats.inconsistent = false
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
