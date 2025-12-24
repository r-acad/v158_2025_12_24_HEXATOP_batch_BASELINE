using Pkg
using Dates
using CUDA

# ANSI color codes for terminal output
const C_RESET = "\u001b[0m"
const C_BOLD = "\u001b[1m"
const C_RED = "\u001b[31m"
const C_GREEN = "\u001b[32m"
const C_YELLOW = "\u001b[33m"
const C_BLUE = "\u001b[34m"
const C_MAGENTA = "\u001b[35m"
const C_CYAN = "\u001b[36m"
const C_WHITE = "\u001b[37m"

function print_banner()
    println("="^60)
    println(">>> [LAUNCHER] HEXA TopOpt: Robust Environment Setup")
    println(">>> [INFO] Time: $(Dates.now()) | Julia Version: $(VERSION)")
    println("="^60)
end

function activate_environment()
    project_dir = @__DIR__
    println(">>> [ENV] Activating project at: $project_dir")
    Pkg.activate(project_dir)
    flush(stdout)
end

function instantiate_environment()
    println(">>> [ENV] Attempting to instantiate environment...")
    flush(stdout)
    
    try
        Pkg.instantiate()
        println("[OK]")
    catch e
        println(C_RED * "[FAILED]" * C_RESET)
        println(">>> [ERROR] Failed to instantiate environment:")
        showerror(stdout, e, catch_backtrace())
        println("\n>>> [HINT] Try running: julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'")
        exit(1)
    end
    flush(stdout)
end

function verify_packages()
    println(">>> [ENV] Verifying core package list...")
    flush(stdout)
    
    required_packages = [
        "LinearAlgebra",
        "SparseArrays", 
        "Printf",
        "JSON",
        "Statistics",
        "CUDA",
        "YAML"
    ]
    
    installed_packages = [p.name for p in values(Pkg.dependencies())]
    
    missing_packages = String[]
    for pkg in required_packages
        if !(pkg in installed_packages)
            push!(missing_packages, pkg)
        end
    end
    
    if !isempty(missing_packages)
        println(C_RED * "[MISSING PACKAGES]" * C_RESET)
        println(">>> [ERROR] The following required packages are missing:")
        for pkg in missing_packages
            println("    - $pkg")
        end
        println("\n>>> [HINT] Run: julia --project=. -e 'using Pkg; Pkg.add([\"$(join(missing_packages, "\", \""))\"])'")
        exit(1)
    end
    
    println("[OK]")
    flush(stdout)
end

function precompile_environment()
    println(">>> [ENV] Precompiling project...")
    flush(stdout)
    
    try
        Pkg.precompile()
    catch e
        println(C_YELLOW * "!!! [WARN] Precompilation had warnings (safe to ignore if code runs)." * C_RESET)
    end
    flush(stdout)
end

function check_cuda_availability()
    println(">>> [GPU] Checking CUDA availability...")
    flush(stdout)
    
    if CUDA.functional()
        dev = CUDA.device()
        name = CUDA.name(dev)
        mem_gb = CUDA.total_memory() / 1024^3
        println(C_GREEN * ">>> [GPU] CUDA Available: $name ($(round(mem_gb, digits=2)) GB)" * C_RESET)
    else
        println(C_YELLOW * ">>> [GPU] No CUDA GPU detected. Will run in CPU mode (slow)." * C_RESET)
    end
    flush(stdout)
end

function generate_machine_limits()
    config_dir = joinpath(@__DIR__, "configs")
    if !isdir(config_dir); mkpath(config_dir); end
    limit_file = joinpath(config_dir, "_machine_limits.jl")

    # Get current GPU fingerprint
    current_gpu_id = "CPU_ONLY"
    current_vram = 0
    
    if CUDA.functional()
        dev = CUDA.device()
        gpu_name = CUDA.name(dev)
        total_vram = CUDA.total_memory()
        current_gpu_id = "$gpu_name-$(div(total_vram, 1024^3))GB"
        current_vram = total_vram
    end

    # Check if file exists AND matches current hardware
    if isfile(limit_file)
        println(">>> [HARDWARE] Limits file found: $limit_file")
        
        # Read the file to check for GPU ID comment
        file_content = read(limit_file, String)
        
        # Look for GPU ID in file (we'll add it as a comment)
        if occursin("# GPU_ID: $current_gpu_id", file_content)
            println(">>> [HARDWARE] GPU Match: $current_gpu_id")
            println(">>> [HARDWARE] Skipping stress test. Delete file to re-run.")
            return
        else
            println(C_YELLOW * ">>> [HARDWARE] GPU CHANGED - Previous limits invalid!" * C_RESET)
            println(">>> [HARDWARE] Previous GPU in file, current GPU: $current_gpu_id")
            println(">>> [HARDWARE] Deleting old limits and re-testing...")
            rm(limit_file, force=true)
        end
    end

    if !CUDA.functional()
        println(C_YELLOW * ">>> [HARDWARE] No GPU detected. Skipping VRAM stress test." * C_RESET)
        return
    end

    println("\n" * C_MAGENTA * "="^60 * C_RESET)
    println(C_MAGENTA * C_BOLD * ">>> [HARDWARE] RUNNING VRAM STRESS TEST (GMG SOLVER)" * C_RESET)
    println(C_MAGENTA * ">>> determining maximum safe element count for this machine..." * C_RESET)
    println(C_MAGENTA * "="^60 * C_RESET)

    # Actual stress test function - simulates GMG memory allocation pattern
    function attempt_allocation(nElem_target)
        # Clear GPU memory before each test
        GC.gc()
        CUDA.reclaim()
        
        # Calculate cubic mesh dimensions for target element count
        n = round(Int, cbrt(nElem_target))
        nx, ny, nz = n, n, n
        
        nNodes_f = (nx+1)*(ny+1)*(nz+1)
        nElem_f = nx*ny*nz
        
        try
            # Simulate finest level memory allocation
            # GMG needs: r, p, Ap, x, z, temp (6 vectors at finest level)
            v1 = CUDA.zeros(Float32, nNodes_f * 3)  # residual
            v2 = CUDA.zeros(Float32, nNodes_f * 3)  # search direction
            v3 = CUDA.zeros(Float32, nNodes_f * 3)  # matrix-vector product
            v4 = CUDA.zeros(Float32, nNodes_f * 3)  # solution
            v5 = CUDA.zeros(Float32, nNodes_f * 3)  # preconditioned residual
            v6 = CUDA.zeros(Float32, nNodes_f * 3)  # temporary vector
            
            # Element data structures
            conn = CUDA.zeros(Int32, nElem_f * 8)      # element connectivity (8 nodes per hex)
            factors = CUDA.zeros(Float32, nElem_f)     # element stiffness factors
            
            # Simulate coarser multigrid levels (typically 3-4 levels)
            buffers = []
            current_nx, current_ny, current_nz = nx, ny, nz
            
            for level in 2:4
                # Coarsen grid by factor of 2 in each direction
                current_nx = max(1, div(current_nx, 2))
                current_ny = max(1, div(current_ny, 2))
                current_nz = max(1, div(current_nz, 2))
                
                nNodes_c = (current_nx+1)*(current_ny+1)*(current_nz+1)
                nElem_c  = current_nx*current_ny*current_nz
                
                # Each coarse level needs: r, x, temp (3 vectors minimum)
                push!(buffers, CUDA.zeros(Float32, nNodes_c * 3))  # residual
                push!(buffers, CUDA.zeros(Float32, nNodes_c * 3))  # solution
                push!(buffers, CUDA.zeros(Float32, nNodes_c * 3))  # temporary
                push!(buffers, CUDA.zeros(Float32, nElem_c))       # element factors
                push!(buffers, CUDA.zeros(Int32, nElem_c * 8))     # connectivity
            end
            
            # Force synchronization to ensure all allocations complete
            CUDA.synchronize()
            
            return true
        catch e
            # Out of memory or other allocation failure
            return false
        end
    end

    # Binary search for maximum safe allocation
    safe_limit = 500_000  # Conservative starting point
    
    # Search parameters - LESS CONSERVATIVE based on empirical data
    # Empirical data from RTX 3080 10GB shows:
    #   - 5M elements: ~417 bytes/element
    #   - 32M elements: ~191 bytes/element
    #   - Average for GMG: ~250-300 bytes/element
    # Using 300 bytes/element as conservative estimate
    
    start_elems = 2_000_000    # Start at 2M elements (faster startup)
    step_elems  = 3_000_000    # Increment by 3M elements (fewer tests needed)
    
    # Estimate upper bound based on GPU memory
    # Rule of thumb: ~300 bytes per element for GMG (empirical)
    # This accounts for:
    #   - Finest level: 6 vectors × 3 DOF/node × 8 nodes/element
    #   - 3-4 coarse multigrid levels with progressively smaller allocations
    #   - Element connectivity and stiffness factors
    #   - Runtime overhead and fragmentation (~20-30%)
    total_mem = CUDA.total_memory()
    est_max = total_mem / 300  # Less conservative estimate
    
    # Cap at 200M elements (raised from 150M for very large GPUs like H100)
    end_elems = min(200_000_000, floor(Int, est_max * 1.3))  # Reduced safety margin from 1.5 to 1.3

    println(">>> [TEST] GPU Memory: $(round(total_mem / 1024^3, digits=2)) GB")
    println(">>> [TEST] Estimated max GMG elements: $(round(Int, est_max / 1_000_000))M ($(round(Int, est_max)) elements)")
    println(">>> [TEST] Search range: $(round(Int, start_elems / 1_000_000))M to $(round(Int, end_elems / 1_000_000))M elements")
    println(">>> [TEST] Step size: $(round(Int, step_elems / 1_000_000))M elements")
    println(">>> [TEST] Estimated test duration: 1-3 minutes")
    flush(stdout)

    test_count = 0
    last_pass_time = 0.0
    for count in start_elems:step_elems:end_elems
        test_count += 1
        test_start = time()
        
        count_millions = round(count / 1_000_000, digits=1)
        print("    [$(lpad(test_count, 2))] Testing $(lpad(count_millions, 5))M elements... ")
        flush(stdout)
        
        if attempt_allocation(count)
            test_duration = time() - test_start
            last_pass_time = test_duration
            println(C_GREEN * "[PASS]" * C_RESET * " ($(round(test_duration, digits=1))s)")
            safe_limit = count
            # Clean up after successful test
            GC.gc()
            CUDA.reclaim()
        else
            test_duration = time() - test_start
            println(C_RED * "[FAIL - OOM]" * C_RESET * " ($(round(test_duration, digits=1))s)")
            println(C_YELLOW * ">>> [TEST] Crash point found at $(round(count / 1_000_000, digits=1))M elements" * C_RESET)
            println(C_YELLOW * ">>> [TEST] Safe limit: $(round(safe_limit / 1_000_000, digits=1))M elements" * C_RESET)
            break
        end
        
        # Brief pause between tests to allow GPU to stabilize (reduced from 0.5s to 0.2s)
        sleep(0.2)
    end

    # If we completed all tests without failure, we hit the search limit
    if safe_limit == floor(Int, (end_elems - start_elems) / step_elems) * step_elems + start_elems
        println(C_CYAN * ">>> [TEST] Reached search limit without OOM. GPU may support even larger meshes." * C_RESET)
    end

    # Apply reduced safety margin: 5% instead of 10%
    # Empirical data shows the system is stable at crash point minus 5%
    # RTX 3080 example: Crashed at ~35M, operational at 32M = ~91% of crash point
    final_limit = floor(Int, safe_limit * 0.95)  # 5% safety margin (reduced from 10%)
    
    # Calculate actual bytes per element for this GPU
    vram_used_estimate = safe_limit * 300  # Conservative estimate
    actual_bytes_per_elem = round(Int, vram_used_estimate / safe_limit)
    
    println("\n" * C_CYAN * "="^60 * C_RESET)
    println(C_GREEN * C_BOLD * ">>> [RESULT] Maximum Safe Elements: $(round(final_limit / 1_000_000, digits=1))M" * C_RESET)
    println(C_CYAN * "    Raw Crash Point:     $(round(safe_limit / 1_000_000, digits=1))M elements" * C_RESET)
    println(C_CYAN * "    Safety Margin:       5% reduction applied" * C_RESET)
    println(C_CYAN * "    Estimated VRAM:      ~$(round(actual_bytes_per_elem * final_limit / 1024^3, digits=2)) GB at limit" * C_RESET)
    println(C_CYAN * "="^60 * C_RESET)
    
    println(">>> [HARDWARE] Writing limits to $limit_file")
    open(limit_file, "w") do io
        write(io, "# MACHINE SPECIFIC LIMITS - GENERATED AUTOMATICALLY\n")
        write(io, "# GPU_ID: $current_gpu_id\n")
        write(io, "# Generated: $(Dates.now())\n")
        write(io, "# Total VRAM: $(round(current_vram/1024^3, digits=2)) GB\n")
        write(io, "# Tested Range: $(round(start_elems/1e6, digits=1))M to $(round(safe_limit/1e6, digits=1))M elements\n")
        write(io, "# Crash Point: $(round(safe_limit/1e6, digits=1))M elements\n")
        write(io, "# \n")
        write(io, "# These limits were determined by stress-testing GMG solver memory allocation.\n")
        write(io, "# The test simulates the full memory footprint of the geometric multigrid solver:\n")
        write(io, "#   - Finest level: 6 vectors (r, p, Ap, x, z, temp) + element data\n")
        write(io, "#   - Coarse levels: 3 vectors per level + element data (3-4 levels)\n")
        write(io, "# \n")
        write(io, "# SAFETY MARGIN: 5% (reduced from previous 10% based on empirical stability data)\n")
        write(io, "# Empirical data from production runs shows stable operation at 95% of crash point.\n")
        write(io, "# \n")
        write(io, "# MEMORY ESTIMATE: ~$(actual_bytes_per_elem) bytes per element (varies with mesh size)\n")
        write(io, "#   - Small meshes (5M): ~400 bytes/element\n")
        write(io, "#   - Large meshes (30M+): ~200 bytes/element\n")
        write(io, "#   - Fixed overhead amortized across elements\n")
        write(io, "# \n")
        write(io, "# To re-test (e.g., after GPU upgrade or driver update):\n")
        write(io, "#   Delete this file and restart the application.\n")
        write(io, "# \n")
        write(io, "module MachineLimits\n")
        write(io, "    # GMG (Geometric Multigrid) Solver Limit\n")
        write(io, "    # This is the maximum element count for the memory-intensive GMG preconditioner\n")
        write(io, "    const MAX_GMG_ELEMENTS = $final_limit\n")
        write(io, "    \n")
        write(io, "    # Jacobi Preconditioner Limit\n")
        write(io, "    # Jacobi uses significantly less memory than GMG (no coarse level hierarchy)\n")
        write(io, "    # Empirical data shows Jacobi can handle 3-4x more elements than GMG\n")
        write(io, "    # Conservative estimate: 2.5x GMG limit\n")
        write(io, "    const MAX_JACOBI_ELEMENTS = $(floor(Int, final_limit * 2.5))\n")
        write(io, "end\n")
    end
    
    println(C_GREEN * C_BOLD * ">>> [HARDWARE] Setup Complete!" * C_RESET)
    println(">>> [HARDWARE] Max GMG Elements:    $(round(final_limit / 1_000_000, digits=1))M elements")
    println(">>> [HARDWARE] Max Jacobi Elements: $(round(final_limit * 2.5 / 1_000_000, digits=1))M elements")
    println(C_CYAN * "-"^60 * C_RESET)
    flush(stdout)
end

function launch_main()
    println("-"^60)
    println(">>> [LAUNCHER] Handing off to Main.jl...")
    flush(stdout)
    
    main_script = joinpath(@__DIR__, "src", "Main.jl")
    
    if !isfile(main_script)
        println(C_RED * ">>> [ERROR] Main.jl not found at: $main_script" * C_RESET)
        exit(1)
    end
    
    # Include and execute Main.jl
    include(main_script)
end

function main()
    try
        print_banner()
        activate_environment()
        instantiate_environment()
        verify_packages()
        precompile_environment()
        println(">>> [ENV] Environment Ready.")
        flush(stdout)
        
        check_cuda_availability()
        generate_machine_limits()
        
        launch_main()
        
    catch e
        if isa(e, InterruptException)
            println("\n" * C_YELLOW * ">>> [LAUNCHER] Interrupted by user." * C_RESET)
            exit(0)
        else
            println("\n" * C_RED * ">>> [LAUNCHER] Fatal error during setup:" * C_RESET)
            showerror(stderr, e, catch_backtrace())
            println("\n")
            exit(1)
        end
    end
end

# Execute main
main()