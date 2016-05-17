function diff2mat(N)
  dx = 1/(N+1)
  SymTridiagonal(-2ones(N),ones(N-1))/(dx^2)
end

import Base: -
  -(T::SymTridiagonal) = SymTridiagonal(-diag(T),-diag(T,1))

Ham(N) = -diff2mat(N)/2

function initial_state{T<:Number}(H::AbstractArray,
                                  c::Vector{T})
  d,v = eigs(H; nev = length(c),
             which = :SM)

  c_norm = sqrt(sumabs2(c))
  c /= c_norm

  v0 = v*c
  E0 = dot(c.^2,d)

  N = size(H, 1)
  dx = 1/(N+1)

  # Return initial state, as well as spectrum
  E0,v0/(norm(v0)*sqrt(dx)),d,v/sqrt(dx),c
end

psi_norm(psi, dx) = norm(psi)*sqrt(dx)
tot_energy(H0, psi, dx) = real(dot(psi,H0*psi))*dx

Base.factorize{T<:Complex}(A::SymTridiagonal{T}) = factorize(Tridiagonal(diag(A,-1),diag(A), diag(A,1)))

function test_propagation(m, N, t, c = nothing;
                          verbose = false,
                          cn_dist_tol = 1e-7)
  dx = 1/(N+1)
  x = linspace(dx,1-dx, N)

  c == nothing && (c = [1])

  @time H0 = Ham(N)
  println("Finding initial state")
  @time E0,psi0,E_phi,phi,c = initial_state(H0, c)
  DE = Diagonal(E_phi)
  @printf("Initial energy: %e\n", E0)

  τ = -im*(t[2]-t[1])


  psi = begin
    psi = zeros(Complex128,N,length(t))
    psi[:,1] = psi0

    H0_op = LinearOperator(H0)
    println("Allocating work arrays")
    @time l_work = exp_lanczos_work(H0_op, psi[:,1], m, verbose)

    println("Propagating")
    @time for i = 2:length(t)
      do_print = (verbose || mod(i,div(length(t), 10)) == 0)
      do_print && println("----------------------")
      do_print && println("Step: $i")
      exp_lanczos!(H0_op, sub(psi, :, i-1), τ, m, sub(psi, :, i), l_work...;
                   rtol = 1e-5, verbose = do_print)
      do_print && println("----------------------")
    end
    
    psi
  end

  psi_cmp = begin
    # A three-point discretization of the 1D time-dependent
    # Schrödinger equation can be efficiently propagated using
    # Crank–Nicolson.
    I = Diagonal(ones(N))
    F = I + τ/2*H0
    B = factorize(I - τ/2*H0)
    psi_cmp = similar(psi)
    psi_cmp[:,1] = psi0
    println("Propagating using Crank–Nicolson")
    @time for i = 2:length(t)
      psi_cmp[:,i] = B \ (F*psi_cmp[:,i-1])
    end
    psi_cmp
  end

  @test_approx_eq_eps tot_energy(H0, psi[:,end], dx) E0 1e-8
  @test_approx_eq_eps psi_norm(psi[:,end], dx) 1 1e-8
  @test_approx_eq_eps sumabs2(psi[:,end]-psi_cmp[:,end])*dx 0 cn_dist_tol
end

m = 15
N = max(m+1,101)
nt = 2001

# For propagation of a single eigenstate, the propagator is
# essentially exact
test_propagation(m, N, linspace(0,1,nt), [1]; verbose = false)
println()
println("+++++++++++++++++++++++++++++++")
println()
test_propagation(m, N, linspace(0,1,nt), [1,1]; verbose = false)
println()
println("+++++++++++++++++++++++++++++++")
println()
test_propagation(m, N, linspace(0,1,nt), [1,1,1]; verbose = false, cn_dist_tol = 2e-6)

