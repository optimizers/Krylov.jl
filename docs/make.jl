using Documenter, Krylov

makedocs(
  modules = [Krylov],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"],
                           ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "Krylov.jl",
  pages = ["Home" => "index.md",
           "API" => "api.md",
           "Krylov processes" => "processes.md",
           "Krylov methods" => ["Hermitian positive definite linear systems" => "solvers/spd.md",
                                "Hermitian indefinite linear systems" => "solvers/sid.md",
                                "Non-Hermitian square linear systems" => "solvers/unsymmetric.md",
                                "Least-norm problems" => "solvers/ln.md",
                                "Least-squares problems" => "solvers/ls.md",
                                "Adjoint systems" => "solvers/as.md",
                                "Saddle-point and Hermitian quasi-definite systems" => "solvers/sp_sqd.md",
                                "Generalized saddle-point and non-Hermitian partitioned systems" => "solvers/gsp.md"],
           "In-place methods" => "inplace.md",
           "Preconditioners" => "preconditioners.md",
           "Storage requirements" => "storage.md",
           "GPU support" => "gpu.md",
           "Warm-start" => "warm-start.md",
           "Factorization-free operators" => "factorization-free.md",
           "Callbacks" => "callbacks.md",
           "Performance tips" => "tips.md",
           "Tutorials" => ["CG" => "examples/cg.md",
                           "CAR" => "examples/car.md",
                           "CG-LANCZOS-SHIFT" => "examples/cg_lanczos_shift.md",
                           "SYMMLQ" => "examples/symmlq.md",
                           "MINRES-QLP" => "examples/minres_qlp.md",
                           "MINARES" => "examples/minares.md",
                           "TriCG" => "examples/tricg.md",
                           "TriMR" => "examples/trimr.md",
                           "BICGSTAB" => "examples/bicgstab.md",
                           "DQGMRES" => "examples/dqgmres.md",
                           "CGNE" => "examples/cgne.md",
                           "CRMR" => "examples/crmr.md",
                           "CRAIG" => "examples/craig.md",
                           "CRAIGMR" => "examples/craigmr.md",
                           "CGLS" => "examples/cgls.md",
                           "CRLS" => "examples/crls.md",
                           "LSQR" => "examples/lsqr.md",
                           "LSMR" => "examples/lsmr.md"],
           "Reference" => "reference.md"
          ]
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/Krylov.jl.git",
  push_preview = true,
  devbranch = "main",
)
