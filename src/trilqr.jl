# An implementation of TRILQR for the solution of square or
# rectangular consistent linear adjoint systems Ax = b and Aᴴy = c.
#
# This method is described in
#
# A. Montoison and D. Orban
# BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property.
# SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
#
# Alexis Montoison, <alexis.montoison@polymtl.ca>
# Montreal, July 2019.

export trilqr, trilqr!

"""
    (x, y, stats) = trilqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                           transfer_to_usymcg::Bool=true, atol::T=√eps(T),
                           rtol::T=√eps(T), itmax::Int=0,
                           timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                           callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = trilqr(A, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

TriLQR can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Combine USYMLQ and USYMQR to solve adjoint systems.

    [0  A] [y] = [b]
    [Aᴴ 0] [x]   [c]

USYMLQ is used for solving primal system `Ax = b` of size m × n.
USYMQR is used for solving dual system `Aᴴy = c` of size n × m.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension `m × n`;
* `b`: a vector of length `m`;
* `c`: a vector of length `n`.

#### Optional arguments

* `x0`: a vector of length `n` that represents an initial guess of the solution `x`;
* `y0`: a vector of length `m` that represents an initial guess of the solution `y`.

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
* `y`: a dense vector of length `m`;
* `stats`: statistics collected on the run in an [`AdjointStats`](@ref) structure.

#### Reference

* A. Montoison and D. Orban, [*BiLQ: An Iterative Method for Nonsymmetric Linear Systems with a Quasi-Minimum Error Property*](https://doi.org/10.1137/19M1290991), SIAM Journal on Matrix Analysis and Applications, 41(3), pp. 1145--1166, 2020.
"""
function trilqr end

"""
    solver = trilqr!(solver::TrilqrSolver, A, b, c; kwargs...)
    solver = trilqr!(solver::TrilqrSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`trilqr`](@ref).

See [`TrilqrSolver`](@ref) for more details about the `solver`.
"""
function trilqr! end

def_args_trilqr = (:(A                    ),
                   :(b::AbstractVector{FC}),
                   :(c::AbstractVector{FC}))

def_optargs_trilqr = (:(x0::AbstractVector),
                      :(y0::AbstractVector))

def_kwargs_trilqr = (:(; transfer_to_usymcg::Bool = true),
                     :(; atol::T = √eps(T)              ),
                     :(; rtol::T = √eps(T)              ),
                     :(; itmax::Int = 0                 ),
                     :(; timemax::Float64 = Inf         ),
                     :(; verbose::Int = 0               ),
                     :(; history::Bool = false          ),
                     :(; callback = solver -> false     ),
                     :(; iostream::IO = kstdout         ))

def_kwargs_trilqr = extract_parameters.(def_kwargs_trilqr)

args_trilqr = (:A, :b, :c)
optargs_trilqr = (:x0, :y0)
kwargs_trilqr = (:transfer_to_usymcg, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function trilqr!(solver :: TrilqrSolver{T,FC,S}, $(def_args_trilqr...); $(def_kwargs_trilqr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "TRILQR: primal system of %d equations in %d variables\n", m, n)
    (verbose > 0) && @printf(iostream, "TRILQR: dual system of %d equations in %d variables\n", n, m)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    uₖ₋₁, uₖ, p, d̅, x, stats = solver.uₖ₋₁, solver.uₖ, solver.p, solver.d̅, solver.x, solver.stats
    vₖ₋₁, vₖ, q, t, wₖ₋₃, wₖ₋₂ = solver.vₖ₋₁, solver.vₖ, solver.q, solver.y, solver.wₖ₋₃, solver.wₖ₋₂
    Δx, Δy, warm_start = solver.Δx, solver.Δy, solver.warm_start
    rNorms, sNorms = stats.residuals_primal, stats.residuals_dual
    reset!(stats)
    r₀ = warm_start ? q : b
    s₀ = warm_start ? p : c

    if warm_start
      mul!(r₀, A, Δx)
      kaxpby!(n, one(FC), b, -one(FC), r₀)
      mul!(s₀, Aᴴ, Δy)
      kaxpby!(n, one(FC), c, -one(FC), s₀)
    end

    # Initial solution x₀ and residual r₀ = b - Ax₀.
    kfill!(x, zero(FC))   # x₀
    bNorm = knorm(m, r₀)  # rNorm = ‖r₀‖

    # Initial solution y₀ and residual s₀ = c - Aᴴy₀.
    kfill!(t, zero(FC))   # t₀
    cNorm = knorm(n, s₀)  # sNorm = ‖s₀‖

    iter = 0
    itmax == 0 && (itmax = m+n)

    history && push!(rNorms, bNorm)
    history && push!(sNorms, cNorm)
    εL = atol + rtol * bNorm
    εQ = atol + rtol * cNorm
    ξ = zero(T)
    (verbose > 0) && @printf(iostream, "%5s  %7s  %7s  %5s\n", "k", "‖rₖ‖", "‖sₖ‖", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %7.1e  %.2fs\n", iter, bNorm, cNorm, start_time |> ktimer)

    # Set up workspace.
    βₖ = knorm(m, r₀)           # β₁ = ‖r₀‖ = ‖v₁‖
    γₖ = knorm(n, s₀)           # γ₁ = ‖s₀‖ = ‖u₁‖
    kfill!(vₖ₋₁, zero(FC))      # v₀ = 0
    kfill!(uₖ₋₁, zero(FC))      # u₀ = 0
    vₖ .= r₀ ./ βₖ              # v₁ = (b - Ax₀) / β₁
    uₖ .= s₀ ./ γₖ              # u₁ = (c - Aᴴy₀) / γ₁
    cₖ₋₁ = cₖ = -one(T)         # Givens cosines used for the LQ factorization of Tₖ
    sₖ₋₁ = sₖ = zero(FC)        # Givens sines used for the LQ factorization of Tₖ
    kfill!(d̅, zero(FC))         # Last column of D̅ₖ = Uₖ(Qₖ)ᴴ
    ζₖ₋₁ = ζbarₖ = zero(FC)     # ζₖ₋₁ and ζbarₖ are the last components of z̅ₖ = (L̅ₖ)⁻¹β₁e₁
    ζₖ₋₂ = ηₖ = zero(FC)        # ζₖ₋₂ and ηₖ are used to update ζₖ₋₁ and ζbarₖ
    δbarₖ₋₁ = δbarₖ = zero(FC)  # Coefficients of Lₖ₋₁ and L̅ₖ modified over the course of two iterations
    ψbarₖ₋₁ = ψₖ₋₁ = zero(FC)   # ψₖ₋₁ and ψbarₖ are the last components of h̅ₖ = Qₖγ₁e₁
    ϵₖ₋₃ = λₖ₋₂ = zero(FC)      # Components of Lₖ₋₁
    kfill!(wₖ₋₃, zero(FC))      # Column k-3 of Wₖ = Vₖ(Lₖ)⁻ᴴ
    kfill!(wₖ₋₂, zero(FC))      # Column k-2 of Wₖ = Vₖ(Lₖ)⁻ᴴ

    # Stopping criterion.
    inconsistent = false
    solved_lq = bNorm == 0
    solved_lq_tol = solved_lq_mach = false
    solved_cg = solved_cg_tol = solved_cg_mach = false
    solved_primal = solved_lq || solved_cg
    solved_qr_tol = solved_qr_mach = false
    solved_dual = cNorm == 0
    tired = iter ≥ itmax
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !((solved_primal && solved_dual) || tired || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the SSY tridiagonalization process.
      # AUₖ  = VₖTₖ    + βₖ₊₁vₖ₊₁(eₖ)ᵀ = Vₖ₊₁Tₖ₊₁.ₖ
      # AᴴVₖ = Uₖ(Tₖ)ᴴ + γₖ₊₁uₖ₊₁(eₖ)ᵀ = Uₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , uₖ)  # Forms vₖ₊₁ : q ← Auₖ
      mul!(p, Aᴴ, vₖ)  # Forms uₖ₊₁ : p ← Aᴴvₖ

      if iter ≥ 2
        kaxpy!(m, -γₖ, vₖ₋₁, q)  # q ← q - γₖ * vₖ₋₁
        kaxpy!(n, -βₖ, uₖ₋₁, p)  # p ← p - βₖ * uₖ₋₁
      end

      αₖ = kdot(m, vₖ, q)  # αₖ = ⟨vₖ,q⟩

      kaxpy!(m, -     αₖ , vₖ, q)  # q ← q - αₖ * vₖ
      kaxpy!(n, -conj(αₖ), uₖ, p)  # p ← p - ᾱₖ * uₖ

      βₖ₊₁ = knorm(m, q)  # βₖ₊₁ = ‖q‖
      γₖ₊₁ = knorm(n, p)  # γₖ₊₁ = ‖p‖

      # Update the LQ factorization of Tₖ = L̅ₖQₖ.
      # [ α₁ γ₂ 0  •  •  •  0 ]   [ δ₁   0    •   •   •    •    0   ]
      # [ β₂ α₂ γ₃ •        • ]   [ λ₁   δ₂   •                 •   ]
      # [ 0  •  •  •  •     • ]   [ ϵ₁   λ₂   δ₃  •             •   ]
      # [ •  •  •  •  •  •  • ] = [ 0    •    •   •   •         •   ] Qₖ
      # [ •     •  •  •  •  0 ]   [ •    •    •   •   •    •    •   ]
      # [ •        •  •  •  γₖ]   [ •         •   •  λₖ₋₂ δₖ₋₁  0   ]
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

      if !solved_primal
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
          # (xᴸ)ₖ ← (xᴸ)ₖ₋₁ + ζₖ₋₁ * dₖ₋₁
          kaxpy!(n, ζₖ₋₁ * cₖ,  d̅, x)
          kaxpy!(n, ζₖ₋₁ * sₖ, uₖ, x)
        end

        # Compute d̅ₖ.
        if iter == 1
          # d̅₁ = u₁
          kcopy!(n, d̅, uₖ)  # d̅ ← uₖ
        else
          # d̅ₖ = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
          kaxpby!(n, -cₖ, uₖ, conj(sₖ), d̅)
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

        # Update primal stopping criterion
        solved_lq_tol = rNorm_lq ≤ εL
        solved_lq_mach = rNorm_lq + 1 ≤ 1
        solved_lq = solved_lq_tol || solved_lq_mach
        solved_cg_tol = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ εL)
        solved_cg_mach = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg + 1 ≤ 1)
        solved_cg = solved_cg_tol || solved_cg_mach
        solved_primal = solved_lq || solved_cg
      end

      if !solved_dual
        # Compute ψₖ₋₁ and ψbarₖ the last coefficients of h̅ₖ = Qₖγ₁e₁.
        if iter == 1
          ψbarₖ = γₖ
        else
          # [cₖ  s̄ₖ] [ψbarₖ₋₁] = [ ψₖ₋₁ ]
          # [sₖ -cₖ] [   0   ]   [ ψbarₖ]
          ψₖ₋₁  = cₖ * ψbarₖ₋₁
          ψbarₖ = sₖ * ψbarₖ₋₁
        end

        # Compute the direction wₖ₋₁, the last column of Wₖ₋₁ = (Vₖ₋₁)(Lₖ₋₁)⁻ᴴ ⟷ (L̄ₖ₋₁)(Wₖ₋₁)ᵀ = (Vₖ₋₁)ᵀ.
        # w₁ = v₁ / δ̄₁
        if iter == 2
          wₖ₋₁ = wₖ₋₂
          kaxpy!(m, one(FC), vₖ₋₁, wₖ₋₁)
          wₖ₋₁ .= vₖ₋₁ ./ conj(δₖ₋₁)
        end
        # w₂ = (v₂ - λ̄₁w₁) / δ̄₂
        if iter == 3
          wₖ₋₁ = wₖ₋₃
          kaxpy!(m, one(FC), vₖ₋₁, wₖ₋₁)
          kaxpy!(m, -conj(λₖ₋₂), wₖ₋₂, wₖ₋₁)
          wₖ₋₁ .= wₖ₋₁ ./ conj(δₖ₋₁)
        end
        # wₖ₋₁ = (vₖ₋₁ - λ̄ₖ₋₂wₖ₋₂ - ϵ̄ₖ₋₃wₖ₋₃) / δ̄ₖ₋₁
        if iter ≥ 4
          kscal!(m, -conj(ϵₖ₋₃), wₖ₋₃)
          wₖ₋₁ = wₖ₋₃
          kaxpy!(m, one(FC), vₖ₋₁, wₖ₋₁)
          kaxpy!(m, -conj(λₖ₋₂), wₖ₋₂, wₖ₋₁)
          wₖ₋₁ .= wₖ₋₁ ./ conj(δₖ₋₁)
        end

        if iter ≥ 3
          # Swap pointers.
          @kswap!(wₖ₋₃, wₖ₋₂)
        end

        if iter ≥ 2
          # Compute solution tₖ₋₁.
          # tₖ₋₁ ← tₖ₋₂ + ψₖ₋₁ * wₖ₋₁
          kaxpy!(m, ψₖ₋₁, wₖ₋₁, t)
        end

        # Update ψbarₖ₋₁
        ψbarₖ₋₁ = ψbarₖ

        # Compute USYMQR residual norm ‖sₖ₋₁‖ = |ψbarₖ|.
        sNorm = abs(ψbarₖ)
        history && push!(sNorms, sNorm)

        # Compute ‖Asₖ₋₁‖ = |ψbarₖ| * √(|δbarₖ|² + |λbarₖ|²).
        AsNorm = abs(ψbarₖ) * √(abs2(δbarₖ) + abs2(cₖ * βₖ₊₁))

        # Update dual stopping criterion
        iter == 1 && (ξ = atol + rtol * AsNorm)
        solved_qr_tol = sNorm ≤ εQ
        solved_qr_mach = sNorm + 1 ≤ 1
        inconsistent = AsNorm ≤ ξ
        solved_dual = solved_qr_tol || solved_qr_mach || inconsistent
      end

      # Compute uₖ₊₁ and uₖ₊₁.
      kcopy!(m, vₖ₋₁, vₖ)  # vₖ₋₁ ← vₖ
      kcopy!(n, uₖ₋₁, uₖ)  # uₖ₋₁ ← uₖ

      if βₖ₊₁ ≠ zero(T)
        vₖ .= q ./ βₖ₊₁  # βₖ₊₁vₖ₊₁ = q
      end
      if γₖ₊₁ ≠ zero(T)
        uₖ .= p ./ γₖ₊₁  # γₖ₊₁uₖ₊₁ = p
      end

      # Update ϵₖ₋₃, λₖ₋₂, δbarₖ₋₁, cₖ₋₁, sₖ₋₁, γₖ and βₖ.
      if iter ≥ 3
        ϵₖ₋₃ = ϵₖ₋₂
      end
      if iter ≥ 2
        λₖ₋₂ = λₖ₋₁
      end
      δbarₖ₋₁ = δbarₖ
      cₖ₋₁    = cₖ
      sₖ₋₁    = sₖ
      γₖ      = γₖ₊₁
      βₖ      = βₖ₊₁

      user_requested_exit = callback(solver) :: Bool
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns

      kdisplay(iter, verbose) &&  solved_primal && !solved_dual && @printf(iostream, "%5d  %7s  %7.1e  %.2fs\n", iter, "✗ ✗ ✗ ✗", sNorm, start_time |> ktimer)
      kdisplay(iter, verbose) && !solved_primal &&  solved_dual && @printf(iostream, "%5d  %7.1e  %7s  %.2fs\n", iter, rNorm_lq, "✗ ✗ ✗ ✗", start_time |> ktimer)
      kdisplay(iter, verbose) && !solved_primal && !solved_dual && @printf(iostream, "%5d  %7.1e  %7.1e  %.2fs\n", iter, rNorm_lq, sNorm, start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Compute USYMCG point
    # (xᶜ)ₖ ← (xᴸ)ₖ₋₁ + ζbarₖ * d̅ₖ
    if solved_cg
      kaxpy!(n, ζbarₖ, d̅, x)
    end

    # Termination status
    tired                            && (status = "maximum number of iterations exceeded")
    solved_lq_tol  && !solved_dual   && (status = "Only the primal solution xᴸ is good enough given atol and rtol")
    solved_cg_tol  && !solved_dual   && (status = "Only the primal solution xᶜ is good enough given atol and rtol")
    !solved_primal && solved_qr_tol  && (status = "Only the dual solution t is good enough given atol and rtol")
    solved_lq_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᴸ, t) are good enough given atol and rtol")
    solved_cg_tol  && solved_qr_tol  && (status = "Both primal and dual solutions (xᶜ, t) are good enough given atol and rtol")
    solved_lq_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᴸ")
    solved_cg_mach && !solved_dual   && (status = "Only found approximate zero-residual primal solution xᶜ")
    !solved_primal && solved_qr_mach && (status = "Only found approximate zero-residual dual solution t")
    solved_lq_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᴸ, t)")
    solved_cg_mach && solved_qr_mach && (status = "Found approximate zero-residual primal and dual solutions (xᶜ, t)")
    solved_lq_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᴸ and a dual solution t good enough given atol and rtol")
    solved_cg_mach && solved_qr_tol  && (status = "Found approximate zero-residual primal solutions xᶜ and a dual solution t good enough given atol and rtol")
    solved_lq_tol  && solved_qr_mach && (status = "Found a primal solution xᴸ good enough given atol and rtol and an approximate zero-residual dual solutions t")
    solved_cg_tol  && solved_qr_mach && (status = "Found a primal solution xᶜ good enough given atol and rtol and an approximate zero-residual dual solutions t")
    user_requested_exit              && (status = "user-requested exit")
    overtimed                        && (status = "time limit exceeded")

    # Update x and y
    warm_start && kaxpy!(n, one(FC), Δx, x)
    warm_start && kaxpy!(m, one(FC), Δy, t)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved_primal = solved_primal
    stats.solved_dual = solved_dual
    stats.timer = start_time |> ktimer
    stats.status = status
    return solver
  end
end
