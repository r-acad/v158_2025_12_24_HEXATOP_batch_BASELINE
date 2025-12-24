# FILE: tools/test_vram_limit.jl
using CUDA
using Printf
using LinearAlgebra

# --- Mock Structures to Mimic Real Memory Usage ---
# We replicate the MGWorkspace logic to ensure the memory footprint is identical.

function estimate_memory_usage(nElem_target)
    # 1. Calculate Grid Dimensions (Cube approximation)
    n = round(Int, cbrt(nElem_target))
    nx = n; ny = n; nz = n
    
    # 2. Fine Grid Size
    nNodes_f = (nx+1)*(ny+1)*(nz+1)
    nElem_f  = nx*ny*nz
    
    # 3. Memory Accumulator (in Bytes)
    total_bytes = 0
    
    # --- Solver Vectors (Fine Grid) ---
    # r, p, z, x, Ap, diag, etc. (Float32)
    # The solver uses roughly 6 vectors of size (nNodes * 3)
    vec_size = nNodes_f * 3 * 4 # 4 bytes for Float32
    total_bytes += vec_size * 6 
    
    # --- Connectivity & Factors ---
    total_bytes += nElem_f * 8 * 4 # conn (Int32)
    total_bytes += nElem_f * 4     # factors (Float32)
    
    # --- Multigrid Hierarchy ---
    # Calculate sizes for L2, L3, L4, L5
    # Each level needs: r, x, diag (Vectors) + density (Scalar) + conn
    
    current_nx, current_ny, current_nz = nx, ny, nz
    
    for level in 2:5
        current_nx = max(1, div(current_nx, 2))
        current_ny = max(1, div(current_ny, 2))
        current_nz = max(1, div(current_nz, 2))
        
        nNodes_c = (current_nx+1)*(current_ny+1)*(current_nz+1)
        nElem_c  = current_nx*current_ny*current_nz
        
        # Vectors: r, x, diag
        total_bytes += (nNodes_c * 3 * 4) * 3
        # Density
        total_bytes += nElem_c * 4
        # Connectivity
        total_bytes += nElem_c * 8 * 4
        # BCs
        total_bytes += nNodes_c * 3 * 4
    end
    
    return total_bytes / 1024^3 # Return in GB
end

function attempt_allocation(nElem_target)
    # 1. Clean Slate
    GC.gc()
    CUDA.reclaim()
    
    # 2. Dimensions
    n = round(Int, cbrt(nElem_target))
    nx, ny, nz = n, n, n
    
    print("   > Allocating $(Base.format_bytes(nElem_target)) elements ($nx x $ny x $nz)... ")
    
    try
        # --- ALLOCATE FINE GRID ---
        nNodes_f = (nx+1)*(ny+1)*(nz+1)
        nElem_f = nx*ny*nz
        
        # Solver Vectors (The big ones)
        v1 = CUDA.zeros(Float32, nNodes_f * 3) # r
        v2 = CUDA.zeros(Float32, nNodes_f * 3) # p
        v3 = CUDA.zeros(Float32, nNodes_f * 3) # z
        v4 = CUDA.zeros(Float32, nNodes_f * 3) # x
        v5 = CUDA.zeros(Float32, nNodes_f * 3) # Ap
        v6 = CUDA.zeros(Float32, nNodes_f * 3) # diag
        
        # Connectivity
        conn = CUDA.zeros(Int32, nElem_f * 8)
        factors = CUDA.zeros(Float32, nElem_f)
        
        # --- ALLOCATE MULTIGRID LEVELS ---
        buffers = []
        current_nx, current_ny, current_nz = nx, ny, nz
        
        for level in 2:5
            current_nx = max(1, div(current_nx, 2))
            current_ny = max(1, div(current_ny, 2))
            current_nz = max(1, div(current_nz, 2))
            
            nNodes_c = (current_nx+1)*(current_ny+1)*(current_nz+1)
            nElem_c  = current_nx*current_ny*current_nz
            
            # MG Buffers
            push!(buffers, CUDA.zeros(Float32, nNodes_c * 3)) # r
            push!(buffers, CUDA.zeros(Float32, nNodes_c * 3)) # x
            push!(buffers, CUDA.zeros(Float32, nNodes_c * 3)) # diag
            push!(buffers, CUDA.zeros(Float32, nElem_c))      # density
            push!(buffers, CUDA.zeros(Int32, nElem_c * 8))    # conn
            push!(buffers, CUDA.zeros(Float32, nNodes_c * 3)) # bc
        end
        
        # If we reached here, success!
        # Force synchronization to ensure lazy allocation actually happened
        CUDA.synchronize()
        
        free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory()
        print("\u001b[32m[SUCCESS]\u001b[0m Free VRAM: $(round(free_mem/1024^3, digits=2)) GB\n")
        
        # Cleanup
        v1=nothing; v2=nothing; v3=nothing; v4=nothing; v5=nothing; v6=nothing
        conn=nothing; factors=nothing; buffers=nothing
        return true
        
    catch e
        print("\u001b[31m[FAILED]\u001b[0m OOM or Error.\n")
        return false
    end
end

function run_stress_test()
    println("==================================================================")
    println(">>> HEXA VRAM STRESS TEST (GMG 5-LEVEL PREDICTION)")
    println("==================================================================")
    
    if !CUDA.functional()
        println("[ERROR] No GPU detected.")
        return
    end
    
    free_start, total_start = CUDA.available_memory(), CUDA.total_memory()
    println("[INFO] GPU: $(CUDA.name(CUDA.device()))")
    println("[INFO] Available VRAM: $(round(free_start/1024^3, digits=2)) GB")
    
    # Range to test: From 10 Million to 100 Million
    # Adjust 'step' for finer resolution
    start_elems = 10_000_000
    end_elems   = 100_000_000
    step_elems  = 5_000_000
    
    last_success = 0
    
    for count in start_elems:step_elems:end_elems
        estimated_gb = estimate_memory_usage(count)
        
        # If estimate > Total VRAM, warn but try anyway (maybe estimate is wrong)
        if estimated_gb > (total_start/1024^3)
            println("\n[WARN] Theoretical size ($estimated_gb GB) exceeds physical VRAM.")
        end
        
        success = attempt_allocation(count)
        
        if success
            last_success = count
        else
            println("\n>>> CRASH POINT DETECTED AROUND: $(Base.format_bytes(count))")
            println(">>> SAFE LIMIT IS LIKELY: $(Base.format_bytes(last_success))")
            break
        end
    end
end

run_stress_test()