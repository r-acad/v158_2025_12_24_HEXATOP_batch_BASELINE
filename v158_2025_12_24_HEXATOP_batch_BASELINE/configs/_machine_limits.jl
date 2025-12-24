# MACHINE SPECIFIC LIMITS - GENERATED AUTOMATICALLY
# GPU_ID: NVIDIA GeForce RTX 3080-9GB
# Generated: 2025-12-24T11:15:25.873
# Total VRAM: 10.0 GB
# Tested Range: 2.0M to 44.0M elements
# Crash Point: 44.0M elements
# 
# These limits were determined by stress-testing GMG solver memory allocation.
# The test simulates the full memory footprint of the geometric multigrid solver:
#   - Finest level: 6 vectors (r, p, Ap, x, z, temp) + element data
#   - Coarse levels: 3 vectors per level + element data (3-4 levels)
# 
# SAFETY MARGIN: 5% (reduced from previous 10% based on empirical stability data)
# Empirical data from production runs shows stable operation at 95% of crash point.
# 
# MEMORY ESTIMATE: ~300 bytes per element (varies with mesh size)
#   - Small meshes (5M): ~400 bytes/element
#   - Large meshes (30M+): ~200 bytes/element
#   - Fixed overhead amortized across elements
# 
# To re-test (e.g., after GPU upgrade or driver update):
#   Delete this file and restart the application.
# 
module MachineLimits
    # GMG (Geometric Multigrid) Solver Limit
    # This is the maximum element count for the memory-intensive GMG preconditioner
    const MAX_GMG_ELEMENTS = 41800000
    
    # Jacobi Preconditioner Limit
    # Jacobi uses significantly less memory than GMG (no coarse level hierarchy)
    # Empirical data shows Jacobi can handle 3-4x more elements than GMG
    # Conservative estimate: 2.5x GMG limit
    const MAX_JACOBI_ELEMENTS = 104500000
end
