// # FILE: .\src\Utils\Helpers.jl
module Helpers 

using CUDA 
using Printf

export expand_element_indices, nodes_from_location, parse_location_component 
export calculate_element_distribution, has_enough_gpu_memory, clear_gpu_memory, get_max_feasible_elements
export enforce_gpu_memory_safety, log_gpu_state, is_gmg_feasible_on_gpu, cleanup_memory

"""
    cleanup_memory()

Aggressively reclaims GPU and System memory. 
Should be called before large allocations.
"""
function cleanup_memory()
    GC.gc()
    if CUDA.functional()
        CUDA.reclaim()
    end
end

function expand_element_indices(elem_inds, dims) 
    nElem_x = dims[1] - 1 
    nElem_y = dims[2] - 1 
    nElem_z = dims[3] - 1 
    inds = Vector{Vector{Int}}() 
    for d in 1:3 
        if (typeof(elem_inds[d]) == String && elem_inds[d] == ":") 
            if d == 1 
                push!(inds, collect(1:nElem_x)) 
            elseif d == 2 
                push!(inds, collect(1:nElem_y)) 
            elseif d == 3 
                push!(inds, collect(1:nElem_z)) 
            end 
        else 
            push!(inds, [Int(elem_inds[d])]) 
        end 
    end 
    result = Int[] 
    for i in inds[1], j in inds[2], k in inds[3] 
        eidx = i + (j-1)*nElem_x + (k-1)*nElem_x*nElem_y 
        push!(result, eidx) 
    end 
    return result 
end 

function nodes_from_location(loc::Vector, dims) 
    nNodes_x, nNodes_y, nNodes_z = dims 
    ix = parse_location_component(loc[1], nNodes_x) 
    iy = parse_location_component(loc[2], nNodes_y) 
    iz = parse_location_component(loc[3], nNodes_z) 
    nodes = Int[] 
    for k in iz, j in iy, i in ix 
        node = i + (j-1)*nNodes_x + (k-1)*nNodes_x*nNodes_y 
        push!(nodes, node) 
    end 
    return nodes 
end 

function parse_location_component(val, nNodes::Int) 
    if val == ":" 
        return collect(1:nNodes) 
    elseif isa(val, String) && endswith(val, "%") 
        perc = parse(Float64, replace(val, "%"=>"")) / 100.0 
        idx = round(Int, 1 + perc*(nNodes-1)) 
        return [idx] 
    elseif isa(val, Number) 
        if 0.0 <= val <= 1.0 
            idx = round(Int, 1 + val*(nNodes-1)) 
            return [idx] 
        else 
            idx = clamp(round(Int, val), 1, nNodes) 
            return [idx] 
        end 
    else 
        error("Invalid location component: $val") 
    end 
end 

function clear_gpu_memory() 
    if !CUDA.functional() 
        return (0, 0) 
    end 
    GC.gc() 
    CUDA.reclaim() 

    final_free, total = CUDA.available_memory(), CUDA.total_memory() 
    return (final_free, total) 
end 

function log_gpu_state(label::String)
    if CUDA.functional()
        free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory()
        used_mem = total_mem - free_mem
        @printf("   [GPU STATE] %-25s | Used: %6.2f GB | Free: %6.2f GB\n", 
                label, used_mem/1024^3, free_mem/1024^3)
        flush(stdout)
    end
end

"""
    estimate_bytes_per_element(matrix_free::Bool=true)

Revised to account for IMPLICIT connectivity in GMG mode.
"""
function estimate_bytes_per_element(matrix_free::Bool=true, use_double::Bool=false)
    prec_mult = use_double ? 2.0 : 1.0

    if matrix_free
        return 80.0 * prec_mult 
    else
        return 12000.0
    end
end

"""
    is_gmg_feasible_on_gpu(nElem::Int, use_double::Bool)

Updated to check the Machine Limits from config.
"""
function is_gmg_feasible_on_gpu(nElem::Int, use_double::Bool; config::Dict=Dict())
    if !CUDA.functional()
        return (false, 0.0, 0.0)
    end
    
    # 1. Check for Empirical Limits (Preferred)
    if haskey(config, "machine_limits")
        limits = config["machine_limits"]
        max_safe = get(limits, "MAX_GMG_ELEMENTS", 5_000_000)
        
        # If float64, reduce limit by half as a rough heuristic (since test was float32)
        if use_double
            max_safe = div(max_safe, 2)
        end
        
        if nElem <= max_safe
             return (true, 0.0, 0.0) # Assume safe
        else
             return (false, Float64(nElem), Float64(max_safe))
        end
    end

    # 2. Fallback Heuristic (If test didn't run)
    cleanup_memory()
    free_mem = Float64(CUDA.available_memory())
    
    prec_mult = use_double ? 2.0 : 1.0
    bytes_per_elem_total = 80.0 * prec_mult * 1.15
    required_mem = nElem * bytes_per_elem_total
    safety_buffer = 400 * 1024^2
    available_for_job = free_mem - safety_buffer
    
    required_gb = required_mem / 1024^3
    free_gb = free_mem / 1024^3
    
    return (required_mem < available_for_job, required_gb, free_gb)
end

function get_max_feasible_elements(matrix_free::Bool=true; safety_factor::Float64=0.95, bytes_per_elem::Int=0)
    if !CUDA.functional() 
        return 5_000_000 
    end 
      
    free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory() 
    if safety_factor == 0.95; safety_factor = 0.99; end 
    usable_mem = free_mem * safety_factor 
    bpe = (bytes_per_elem > 0) ? bytes_per_elem : estimate_bytes_per_element(matrix_free) 
    max_elems = floor(Int, usable_mem / bpe) 
    return max_elems
end
 
function estimate_gpu_memory_required(nNodes, nElem, matrix_free::Bool=true) 
    return nElem * estimate_bytes_per_element(matrix_free)
end
 
function has_enough_gpu_memory(nNodes, nElem, matrix_free::Bool=true) 
    if !CUDA.functional(); return false; end 
    try 
        free_mem, total_mem = CUDA.available_memory(), CUDA.total_memory() 
        required_mem = estimate_gpu_memory_required(nNodes, nElem, matrix_free) 
        utilization_limit = 0.99 
        usable_mem = free_mem * utilization_limit 
        req_gb = required_mem / 1024^3 
        avail_gb = usable_mem / 1024^3

        if required_mem > usable_mem 
            @warn "GPU Memory Estimate:" 
            @printf("   Required:  %.2f GB\n", req_gb) 
            @printf("   Available: %.2f GB\n", avail_gb) 
            return true 
        end 
        return true 
    catch e 
        println("Error checking GPU memory: $e") 
        return true 
    end 
end 

function calculate_element_distribution(length_x, length_y, length_z, target_elem_count) 
    total_volume = length_x * length_y * length_z 
    k = cbrt(target_elem_count / total_volume) 
    nElem_x = max(1, round(Int, k * length_x)) 
    nElem_y = max(1, round(Int, k * length_y)) 
    nElem_z = max(1, round(Int, k * length_z)) 
    dx = length_x / nElem_x 
    dy = length_y / nElem_y 
    dz = length_z / nElem_z 
    actual_elem_count = nElem_x * nElem_y * nElem_z 
    return nElem_x, nElem_y, nElem_z, Float32(dx), Float32(dy), Float32(dz), actual_elem_count
end

function enforce_gpu_memory_safety(n_active_elem::Int, n_nodes::Int, use_double_precision::Bool, use_multigrid::Bool)
    if !CUDA.functional(); return; end
    cleanup_memory()
    free_mem = CUDA.available_memory()
    
    bytes_per = estimate_bytes_per_element(true, use_double_precision)
    
    if use_multigrid
        bytes_per *= 1.2
    end

    mem_est = n_active_elem * bytes_per
    
    req_gb = mem_est / 1024^3
    avail_gb = free_mem / 1024^3
    
    if mem_est > free_mem
        println("\n\u001b[31m>>> [MEMORY GUARD] VRAM DEFICIT DETECTED (Active: $(Base.format_bytes(n_active_elem)))")
        @printf("   Req: %.2f GB | Free: %.2f GB\n", req_gb, avail_gb)
        println("   [WARNING] Expect SEVERE slowdowns (PCIe swapping) or Crash.")
        flush(stdout)
    else
        @printf("   [Memory Guard] %.2f GB est / %.2f GB free. Safe.\n", req_gb, avail_gb)
    end
end
 
end