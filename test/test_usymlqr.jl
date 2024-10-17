@testset "usymlqr" begin
  usymlqr_tol = 1.0e-6

  # Test saddle-point systems
  A, b, D = saddle_point()
  m, n = size(A)
  c = -b
  D⁻¹ = sparse(inv(D))
  N⁻¹ = eye(n)
  H⁻¹ = blockdiag(D⁻¹, N⁻¹)

  (x, y, stats) = usymlqr(A, b, c, M=D⁻¹)
  K = [D A; A' zeros(n, n)]
  B = [b; c]
  r =  B - K * [x; y]
  resid = sqrt(dot(r, H⁻¹ * r)) / sqrt(dot(B, H⁻¹ * B))
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlqr_tol)

  (x, y, stats) = usymlqr(A, b, c)
  K = [eye(m) A; A' zeros(n, n)]
  B = [b; c]
  r =  B - K * [x; y]
  resid = norm(r) / norm(B)
  @printf("USYMLQR: Relative residual: %8.1e\n", resid)
  @test(resid ≤ usymlqr_tol)
end

test_usymlqr()
