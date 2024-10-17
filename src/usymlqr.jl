# An implementation of USYMLQR for the solution of symmetric saddle-point systems.
#
# This method is described in
#
# M. A. Saunders, H. D. Simon, and E. L. Yip
# Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations.
# SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
#
# A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin
# A tridiagonalization method for symmetric saddle-point systems.
# SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Alexis Montoison, <alexis.montoison@polymtl.ca> -- <amontoison@anl.gov>
# Montréal, November 2021 -- Chicago, October 2024.

export usymlqr, usymlqr!

"""
   (x, y, stats) = usymlqr(A, b::AbstractVector{FC}, c::AbstractVector{FC};
                           M=I, N=I, ldiv::Bool=false, atol::T=√eps(T),
                           rtol::T=√eps(T), itmax::Int=0, timemax::Float64=Inf,
                           verbose::Int=0, history::Bool=false,
                           callback=solver->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (x, y, stats) = usymlqr(A, b, c, x0::AbstractVector, y0::AbstractVector; kwargs...)

USYMLQR can be warm-started from initial guesses `x0` and `y0` where `kwargs` are the same keyword arguments as above.

Solve the symmetric saddle-point system

    [ E   A ] [ x ] = [ b ]
    [ Aᴴ    ] [ y ]   [ c ]

where E = M⁻¹ ≻ 0 by way of the Saunders-Simon-Yip tridiagonalization using USYMLQ and USYMQR methods.
The method solves the least-squares problem

    [ E   A ] [ s ] = [ b ]
    [ Aᴴ    ] [ t ]   [ 0 ]

and the least-norm problem

    [ E   A ] [ w ] = [ 0 ]
    [ Aᴴ    ] [ z ]   [ c ]

and simply adds the solutions.

    [ M   O ]
    [ 0   N ]

indicates the weighted norm in which residuals are measured.
It's the Euclidean norm when `M` and `N` are identity operators.

#### Input arguments

* `A`: a linear operator that models a matrix of dimension m × n;
* `b`: a vector of length m;
* `c`: a vector of length n.

#### Optional arguments

* `x0`: a vector of length m that represents an initial guess of the solution x;
* `y0`: a vector of length n that represents an initial guess of the solution y.

#### Keyword arguments

* `M`: linear operator that models a Hermitian positive-definite matrix of size `m` used for centered preconditioning of the partitioned system;
* `N`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning of the partitioned system;
* `ldiv`: define whether the preconditioners use `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm;
* `rtol`: relative stopping tolerance based on the residual norm;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `m+n`;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be kdisplayed if verbose mode is enabled (verbose > 0). Information will be kdisplayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms, or Aᴴ-residual norms;
* `callback`: function or functor called as `callback(solver)` that returns `true` if the Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `x`: a dense vector of length m;
* `y`: a dense vector of length n;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### References

* M. A. Saunders, H. D. Simon, and E. L. Yip, [*Two Conjugate-Gradient-Type Methods for Unsymmetric Linear Equations*](https://doi.org/10.1137/0725052), SIAM Journal on Numerical Analysis, 25(4), pp. 927--940, 1988.
* A. Buttari, D. Orban, D. Ruiz and D. Titley-Peloquin, [*A tridiagonalization method for symmetric saddle-point and quasi-definite systems*](https://doi.org/10.1137/18M1194900), SIAM Journal on Scientific Computing, 41(5), pp. 409--432, 2019.
"""
function usymlqr end

"""
    solver = usymlqr!(solver::UsymlqrSolver, A, b, c; kwargs...)
    solver = usymlqr!(solver::UsymlqrSolver, A, b, c, x0, y0; kwargs...)

where `kwargs` are keyword arguments of [`usymlqr`](@ref).

See [`UsymlqrSolver`](@ref) for more details about the `solver`.
"""
function usymlqr! end

def_args_usymlqr = (:(A                    ),
                    :(b::AbstractVector{FC}),
                    :(c::AbstractVector{FC}))

def_optargs_usymlqr = (:(x0::AbstractVector),
                       :(y0::AbstractVector))

def_kwargs_usymlqr = (:(; transfer_to_usymcg::Bool = true),
                      :(; M = I                          ),
                      :(; N = I                          ),
                      :(; ldiv::Bool = false             ),
                      :(; atol::T = √eps(T)              ),
                      :(; rtol::T = √eps(T)              ),
                      :(; itmax::Int = 0                 ),
                      :(; timemax::Float64 = Inf         ),
                      :(; verbose::Int = 0               ),
                      :(; history::Bool = false          ),
                      :(; callback = solver -> false     ),
                      :(; iostream::IO = kstdout         ))

def_kwargs_usymlqr = extract_parameters.(def_kwargs_usymlqr)

args_usymlqr = (:A, :b, :c)
optargs_usymlqr = (:x0, :y0)
kwargs_usymlqr = (:transfer_to_usymcg, :M, :N, :ldiv, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)

@eval begin
  function usymlqr!(solver :: UsymlqrSolver{T,FC,S}, $(def_args_usymlqr...); $(def_kwargs_usymlqr...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    (m == solver.m && n == solver.n) || error("(solver.m, solver.n) = ($(solver.m), $(solver.n)) is inconsistent with size(A) = ($m, $n)")
    length(b) == m || error("Inconsistent problem size")
    length(c) == n || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "USYMLQR: system of %d equations in %d variables\n", m+n, m+n)

    # Check M = Iₘ and N = Iₙ
    MisI = (M === I)
    NisI = (N === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) <: S || error("ktypeof(b) is not a subtype of $S")
    ktypeof(c) <: S || error("ktypeof(c) is not a subtype of $S")

    # Compute the adjoint of A
    Aᴴ = A'

    # Set up workspace.
    allocate_if(!MisI, solver, :uₖ, S, m)
    allocate_if(!NisI, solver, :vₖ, S, n)
    Δxz, Δry = solver.Δx, solver.Δy
    M⁻¹uₖ₋₁, M⁻¹uₖ, N⁻¹vₖ₋₁, N⁻¹vₖ = solver.M⁻¹uₖ₋₁, solver.M⁻¹uₖ, solver.N⁻¹vₖ₋₁, solver.N⁻¹vₖ
    rₖ, xₖ, yₖ, zₖ, p, q = solver.r, solver.x, solver.y, solver.z, solver.p, solver.q
    d̅, wₖ₋₂, wₖ₋₁ = solver.d̅, solver.wₖ₋₂, solver.wₖ₋₁
    uₖ = MisI ? M⁻¹uₖ : solver.uₖ
    vₖ = NisI ? N⁻¹vₖ : solver.vₖ
    warm_start = solver.warm_start
    b₀ = warm_start ? q : b
    c₀ = warm_start ? p : c

    stats = solver.stats
    rNorms = stats.residuals
    reset!(stats)

    iter = 0
    itmax == 0 && (itmax = n+m)

    # Initial solutions r₀, x₀, y₀ and z₀.
    @kfill!(rₖ, zero(FC))
    @kfill!(xₖ, zero(FC))
    @kfill!(yₖ, zero(FC))
    @kfill!(zₖ, zero(FC))

    # Initialize preconditioned orthogonal tridiagonalization process.
    @kfill!(M⁻¹uₖ₋₁, zero(FC))  # u₀ = 0
    @kfill!(N⁻¹vₖ₋₁, zero(FC))  # v₀ = 0

    # [ I   A ] [ xₖ ] = [ b -   Δx - AΔy ] = [ b₀ ]
    # [ Aᴴ    ] [ yₖ ]   [ c - AᴴΔx       ]   [ c₀ ]
    if warm_start
      mul!(b₀, A, Δy)
      @kaxpy!(m, one(T), Δx, b₀)
      @kaxpby!(m, one(T), b, -one(T), b₀)
      mul!(c₀, Aᴴ, Δx)
      @kaxpby!(n, one(T), c, -one(T), c₀)
    end

    # β₁Eu₁ = b ↔ β₁u₁ = Mb
    M⁻¹uₖ .= b₀
    MisI || mul!(uₖ, M, M⁻¹uₖ)
    βₖ = sqrt(@kdot(m, uₖ, M⁻¹uₖ))  # β₁ = ‖u₁‖_E
    if βₖ ≠ 0
      @kscal!(m, 1 / βₖ, M⁻¹uₖ)
      MisI || @kscal!(m, 1 / βₖ, uₖ)
    else
      error("b must be nonzero")
    end

    # γ₁Fv₁ = c ↔ γ₁v₁ = Nc
    N⁻¹vₖ .= c₀
    NisI || mul!(vₖ, N, N⁻¹vₖ)
    γₖ = sqrt(@kdot(n, vₖ, N⁻¹vₖ))  # γ₁ = ‖v₁‖_F
    if γₖ ≠ 0
      @kscal!(n, 1 / γₖ, N⁻¹vₖ)
      NisI || @kscal!(n, 1 / γₖ, vₖ)
    else
      error("c must be nonzero")
    end

    ε = atol + rtol * rNorm
    κ = zero(T)
    (verbose > 0) && @printf(iostream, "%4s %7s %7s %7s\n", "k", "αₖ", "βₖ", "γₖ")
    kdisplay(iter, verbose) && @printf(iostream, "%4d %7.1e %7.1e %7.1e\n", iter, αₖ, βₖ, γₖ)

    cₖ₋₂ = cₖ₋₁ = cₖ = one(T)    # Givens cosines used for the QR factorization of Tₖ₊₁.ₖ
    sₖ₋₂ = sₖ₋₁ = sₖ = zero(FC)  # Givens sines used for the QR factorization of Tₖ₊₁.ₖ
    @kfill!(wₖ₋₂, zero(FC))      # Column k-2 of Wₖ = Vₖ(Rₖ)⁻¹
    @kfill!(wₖ₋₁, zero(FC))      # Column k-1 of Wₖ = Vₖ(Rₖ)⁻¹
    ϕbarₖ = βₖ                   # ϕbarₖ is the last component of f̄ₖ = (Qₖ)ᴴβ₁e₁
    @kfill!(d̅, zero(FC))         # Last column of D̅ₖ = UₖQₖ
    ηₖ₋₁ = ηbarₖ = zero(FC)      # ηₖ₋₁ and ηbarₖ are the last components of h̄ₖ = (Rₖ)⁻ᵀγ₁e₁
    ηₖ₋₂ = θₖ = zero(FC)         # ζₖ₋₂ and θₖ are used to update ηₖ₋₁ and ηbarₖ
    δbarₖ₋₁ = δbarₖ = zero(FC)   # Coefficients of Rₖ₋₁ and Rₖ modified over the course of two iterations

    # Stopping criterion.
    rNorm_LS = β₁
    rNorm_LN = γ₁
    solved_LS = rNorm_LS ≤ ε
    solved_LN = rNorm_LN ≤ ε
    solved = solved_LS && solved_LN
    inconsistent = false
    tired = iter ≥ itmax
    status = "unknown"
    ill_cond = false
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || ill_cond || inconsistent || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Continue the SSY tridiagonalization process.
      # AVₖ  = EUₖTₖ    + βₖ₊₁Euₖ₊₁(eₖ)ᵀ = EUₖ₊₁Tₖ₊₁.ₖ
      # AᴴUₖ = FVₖ(Tₖ)ᴴ + γₖ₊₁Fvₖ₊₁(eₖ)ᵀ = FVₖ₊₁(Tₖ.ₖ₊₁)ᴴ

      mul!(q, A , vₖ)  # Forms Euₖ₊₁ : q ← Avₖ
      mul!(p, Aᴴ, uₖ)  # Forms Fvₖ₊₁ : p ← Aᴴuₖ

      if iter ≥ 2
        @kaxpy!(m, -γₖ, M⁻¹uₖ₋₁, q) # q ← q - γₖ * M⁻¹uₖ₋₁
        @kaxpy!(n, -βₖ, N⁻¹vₖ₋₁, p) # p ← p - βₖ * N⁻¹vₖ₋₁
      end

      αₖ = @kdot(m, uₖ, q)  # αₖ = ⟨uₖ,q⟩

      @kaxpy!(m, -     αₖ , M⁻¹uₖ, q)   # q ← q - αₖ * M⁻¹uₖ
      @kaxpy!(n, -conj(αₖ), N⁻¹vₖ, p)   # p ← p - ᾱₖ * N⁻¹vₖ

      # Compute vₖ₊₁ and uₖ₊₁
      MisI || mulorldiv!(uₖ₊₁, M, q, ldiv)  # βₖ₊₁uₖ₊₁ = MAvₖ  - γₖuₖ₋₁ - αₖuₖ
      NisI || mulorldiv!(vₖ₊₁, N, p, ldiv)  # γₖ₊₁vₖ₊₁ = NAᴴuₖ - βₖvₖ₋₁ - ᾱₖvₖ

      βₖ₊₁ = sqrt(@kdotr(m, uₖ₊₁, q))  # βₖ₊₁ = ‖uₖ₊₁‖_E
      γₖ₊₁ = sqrt(@kdotr(n, vₖ₊₁, p))  # γₖ₊₁ = ‖vₖ₊₁‖_F

      # Update M⁻¹uₖ₋₁ and N⁻¹vₖ₋₁
      M⁻¹uₖ₋₁ .= M⁻¹uₖ
      N⁻¹vₖ₋₁ .= N⁻¹vₖ

      # Update the QR factorization of Tₖ₊₁.ₖ = Qₖ [ Rₖ ].
      #                                            [ Oᵀ ]
      # [ α₁ γ₂ 0  •  •  •   0  ]      [ δ₁ λ₁ ϵ₁ 0  •  •  0  ]
      # [ β₂ α₂ γ₃ •         •  ]      [ 0  δ₂ λ₂ •  •     •  ]
      # [ 0  •  •  •  •      •  ]      [ •  •  δ₃ •  •  •  •  ]
      # [ •  •  •  •  •  •   •  ] = Qₖ [ •     •  •  •  •  0  ]
      # [ •     •  •  •  •   0  ]      [ •        •  •  • ϵₖ₋₂]
      # [ •        •  •  •   γₖ ]      [ •           •  • λₖ₋₁]
      # [ •           •  βₖ  αₖ ]      [ 0  •  •  •  •  0  δₖ ]
      # [ 0  •  •  •  •  0  βₖ₊₁]      [ 0  •  •  •  •  •  0  ]
      #
      # If k = 1, we don't have any previous reflexion.
      # If k = 2, we apply the last reflexion.
      # If k ≥ 3, we only apply the two previous reflexions.

      # Apply previous Givens reflections Qₖ₋₂.ₖ₋₁
      if iter ≥ 3
        # [cₖ₋₂  sₖ₋₂] [0 ] = [  ϵₖ₋₂ ]
        # [s̄ₖ₋₂ -cₖ₋₂] [γₖ]   [λbarₖ₋₁]
        ϵₖ₋₂    =  sₖ₋₂ * γₖ
        λbarₖ₋₁ = -cₖ₋₂ * γₖ
      end

      # Apply previous Givens reflections Qₖ₋₁.ₖ
      if iter ≥ 2
        iter == 2 && (λbarₖ₋₁ = γₖ)
        # [cₖ₋₁  sₖ₋₁] [λbarₖ₋₁] = [λₖ₋₁ ]
        # [s̄ₖ₋₁ -cₖ₋₁] [   αₖ  ]   [δbarₖ]
        λₖ₋₁  =      cₖ₋₁  * λbarₖ₋₁ + sₖ₋₁ * αₖ
        δbarₖ = conj(sₖ₋₁) * λbarₖ₋₁ - cₖ₋₁ * αₖ
      end

      # Compute and apply current Givens reflection Qₖ.ₖ₊₁
      iter == 1 && (δbarₖ = αₖ)
      # [cₖ  sₖ] [δbarₖ] = [δₖ]
      # [s̄ₖ -cₖ] [βₖ₊₁ ]   [0 ]
      (cₖ, sₖ, δₖ) = sym_givens(δbarₖ, βₖ₊₁)

      # Compute f̄ₖ₊₁ = [   fₖ  ] = (Qₖ)ᴴβ₁e₁
      #                [ϕbarₖ₊₁]
      #
      # [cₖ  sₖ] [ϕbarₖ] = [   ϕₖ  ]
      # [s̄ₖ -cₖ] [  0  ]   [ϕbarₖ₊₁]
      ϕₖ      =      cₖ  * ϕbarₖ
      ϕbarₖ₊₁ = conj(sₖ) * ϕbarₖ

      # Compute the direction wₖ, the last column of Wₖ = Vₖ(Rₖ)⁻¹ ⟷ (Rₖ)ᵀ(Wₖ)ᵀ = (Vₖ)ᵀ.
      # w₁ = v₁ / δ₁
      if iter == 1
        wₖ = wₖ₋₁
        @kaxpy!(n, one(FC), vₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end
      # w₂ = (v₂ - λ₁w₁) / δ₂
      if iter == 2
        wₖ = wₖ₋₂
        @kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        @kaxpy!(n, one(FC), vₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end
      # wₖ = (vₖ - λₖ₋₁wₖ₋₁ - ϵₖ₋₂wₖ₋₂) / δₖ
      if iter ≥ 3
        @kscal!(n, -ϵₖ₋₂, wₖ₋₂)
        wₖ = wₖ₋₂
        @kaxpy!(n, -λₖ₋₁, wₖ₋₁, wₖ)
        @kaxpy!(n, one(FC), vₖ, wₖ)
        wₖ .= wₖ ./ δₖ
      end

      # Update the solution xₖ.
      # xₖ ← xₖ₋₁ + ϕₖ * wₖ
      @kaxpy!(n, ϕₖ, wₖ, xₖ)

      # Update the residual rₖ.
      # rₖ ← |sₖ|² * rₖ₋₁ - cₖ * ϕbarₖ₊₁ * uₖ₊₁
      @kaxpby!(n, cₖ * ϕbarₖ₊₁, q, abs2(sₖ), rₖ)

      # Compute ‖rₖ‖ = |ϕbarₖ₊₁|.
      rNorm = abs(ϕbarₖ₊₁)
      history && push!(rNorms, rNorm)

      # Compute ‖Aᴴrₖ₋₁‖ = |ϕbarₖ| * √(|δbarₖ|² + |λbarₖ|²).
      AᴴrNorm = abs(ϕbarₖ) * √(abs2(δbarₖ) + abs2(cₖ₋₁ * γₖ₊₁))
      history && push!(AᴴrNorms, AᴴrNorm)

      # Compute ηₖ₋₁ and ηbarₖ, last components of the solution of (Rₖ)ᵀh̄ₖ = γ₁e₁
      # [δbar₁] [ηbar₁] = [γ₁]
      if iter == 1
        ωₖ = γₖ
      end
      # [δ₁    0  ] [  η₁ ] = [γ₁]
      # [λ₁  δbar₂] [ηbar₂]   [0 ]
      if iter == 2
        ωₖ₋₁ = ηₖ
        ηₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ωₖ   = -λₖ₋₁ * ζₖ₋₁
      end
      # [λₖ₋₂  δₖ₋₁    0  ] [ηₖ₋₂ ] = [0]
      # [ϵₖ₋₂  λₖ₋₁  δbarₖ] [ηₖ₋₁ ]   [0]
      #                     [ηbarₖ]
      if iter ≥ 3
        ηₖ₋₂ = ζₖ₋₁
        ωₖ₋₁ = ηₖ
        ηₖ₋₁ = ηₖ₋₁ / δₖ₋₁
        ωₖ   = -ϵₖ₋₂ * ηₖ₋₂ - λₖ₋₁ * ηₖ₋₁
      end

      # Relations for the directions dₖ₋₁ and d̅ₖ, the last two columns of D̅ₖ = UₖQₖ.
      # Note: D̄ₖ represents the matrix P̄ₖ in the paper of USYMLQR.
      # [d̅ₖ₋₁ uₖ] [cₖ  s̄ₖ] = [dₖ₋₁ d̅ₖ] ⟷ dₖ₋₁ = cₖ * d̅ₖ₋₁ + sₖ * uₖ
      #           [sₖ -cₖ]             ⟷ d̅ₖ   = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
      if iter ≥ 2
        # Compute solution yₖ.
        # (yᴸ)ₖ₋₁ ← (yᴸ)ₖ₋₂ + ηₖ₋₁ * dₖ₋₁
        @kaxpy!(n, ηₖ₋₁ * cₖ,  d̅, x)
        @kaxpy!(n, ηₖ₋₁ * sₖ, uₖ, x)
      end

      # Compute d̅ₖ.
      if iter == 1
        # d̅₁ = u₁
        @kcopy!(n, uₖ, d̅)  # d̅ ← vₖ
      else
        # d̅ₖ = s̄ₖ * d̅ₖ₋₁ - cₖ * uₖ
        @kaxpby!(n, -cₖ, uₖ, conj(sₖ), d̅)
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

      # Compute zₖ.
      @kaxpy!(n, -ηₖ, wₖ, zₖ)

      # Compute uₖ₊₁ and vₖ₊₁.
      @kcopy!(m, uₖ, uₖ₋₁)  # uₖ₋₁ ← uₖ
      @kcopy!(n, vₖ, vₖ₋₁)  # vₖ₋₁ ← vₖ

      if βₖ₊₁ ≠ zero(T)
        uₖ .= q ./ βₖ₊₁ # βₖ₊₁uₖ₊₁ = q
      end
      if γₖ₊₁ ≠ zero(T)
        vₖ .= p ./ γₖ₊₁ # γₖ₊₁vₖ₊₁ = p
      end

      # Update directions for x.
      if iter ≥ 2
        @kswap(wₖ₋₂, wₖ₋₁)
      end

      # Update sₖ₋₂, cₖ₋₂, sₖ₋₁, cₖ₋₁, ϕbarₖ, γₖ, βₖ.
      if iter ≥ 2
        sₖ₋₂ = sₖ₋₁
        cₖ₋₂ = cₖ₋₁
      end
      sₖ₋₁  = sₖ
      cₖ₋₁  = cₖ
      ϕbarₖ = ϕbarₖ₊₁
      γₖ    = γₖ₊₁
      βₖ    = βₖ₊₁

      # Update δbarₖ₋₁.
      # δbarₖ₋₁ = δbarₖ

      # Update stopping criterion.
      ill_cond_lim = one(T) / Acond ≤ ctol
      ill_cond_mach = one(T) + one(T) / Acond ≤ one(T)
      ill_cond = ill_cond_mach || ill_cond_lim
      tired = iter ≥ itmax
      solved = solved_LS && solved_LN

      iter == 1 && (κ = atol + rtol * AᴴrNorm)
      user_requested_exit = callback(solver) :: Bool
      solved = rNorm ≤ ε
      solved_lq = rNorm_lq ≤ ε
      solved_cg = transfer_to_usymcg && (abs(δbarₖ) > eps(T)) && (rNorm_cg ≤ ε)
      inconsistent = !solved && AᴴrNorm ≤ κ
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%7.1e\n", rNorm_lq)
      kdisplay(iter, verbose) && @printf(iostream, "%4d %8.1e %7.1e %7.1e %7.1e %7.1e %7.1e ", iter, αₖ, βₖ, γₖ, Anorm, Acond, rNorm_qr)
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %.2fs\n", iter, rNorm, AᴴrNorm, ktimer(start_time))
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Compute USYMCG point
    # (yᶜ)ₖ ← (yᴸ)ₖ₋₁ + ηbarₖ * d̅ₖ
    # (zᶜ)ₖ ← (zᴸ)ₖ₋₁ - ηbarₖ * w̄ₖ
    if solved_cg
      @kaxpy!(n,  ηbarₖ, d̅, yₖ)
      @kaxpy!(m, -ηbarₖ, w̄, zₖ)
    end

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    # solved_lq           && (status = "solution xᴸ good enough given atol and rtol")
    # solved_cg           && (status = "solution xᶜ good enough given atol and rtol")
    solved              && (status = "solution good enough given atol and rtol")
    ill_cond_mach       && (status = "condition number seems too large for this machine")
    ill_cond_lim        && (status = "condition number exceeds tolerance")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    # Compute the solution the saddle point system
    # xₖ ← xₖ + zₖ
    # yₖ ← yₖ + rₖ
    @kaxpy!(n, one(FC), zₖ, xₖ)
    @kaxpy!(m, one(FC), rₖ, yₖ)

    # Update xₖ and yₖ
    warm_start && @kaxpy!(n, one(FC), Δxz, xₖ)
    warm_start && @kaxpy!(m, one(FC), Δyr, yₖ)
    solver.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = ktimer(start_time)
    stats.status = status
    return solver
  end
end
