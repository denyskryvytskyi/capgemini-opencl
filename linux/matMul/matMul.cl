#define TILE_SIZE 16

// Simple multiplication kenrel
__kernel void matMul(__global float* pMatA, __global float* pMatB, __global float* pMatRes, int matDimN, int matDimM, int matDimK)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < matDimN && col < matDimK) { // bounding checks
        float sum = 0.0f;
        for (int i = 0; i < matDimM; ++i) {
            sum += pMatA[row * matDimM + i] * pMatB[i * matDimK + col];
        }
        pMatRes[row * matDimK + col] = sum;
    }
}

__kernel void transpose(__global float* pMatB, __global float* pMatB_T, int MAT_DIM_M, int MAT_DIM_K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < MAT_DIM_M && col < MAT_DIM_K) {
        pMatB_T[col * MAT_DIM_M + row] = pMatB[row * MAT_DIM_K + col];
    }
}


// Main kernel
__kernel void matMulTiled(__global float* pMatA, __global float* pMatB, __global float* pMatRes, int matDimN, int matDimM, int matDimK, int tilesAmount)
{
    // Allocate local memory for tiles (shared and accessible by all work-items inside one work-group)
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    // Get global row and column indices
    const int row = get_global_id(1);
    const int col = get_global_id(0);

    // Get local row and column (work-item) within the work-group
    const int localRow = get_local_id(1);
    const int localCol = get_local_id(0);

    float sum = 0.0f;  // Accumulator for the result

    // Loop over tiles of input matrices
    for (int tileIdx = 0; tileIdx < tilesAmount; ++tileIdx) {

        // Load a tile of pMatA and pMatB into local memory
        const int elementTileAStride = tileIdx * TILE_SIZE + localCol;
        if (row < matDimN && elementTileAStride < matDimM) {
            tileA[localRow][localCol] = pMatA[row * matDimM + elementTileAStride];
        } else {
            tileA[localRow][localCol] = 0.0f; // Padding for out of bounds work-item
        }

        const int elementTileBStride = tileIdx * TILE_SIZE + localRow;
        if (col < matDimK && elementTileBStride < matDimM) {
            tileB[localRow][localCol] = pMatB[matDimM * col + elementTileBStride];
        } else {
            tileB[localRow][localCol] = 0.0f; // Padding for out of bounds work-item
        }
        
        // Synchronize to ensure all work-items have loaded the tiles
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply tiles and accumulate the result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        // Synchronize before loading new tiles
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result to global memory
    if (row < matDimN && col < matDimK) {
        pMatRes[row * matDimK + col] = sum;
    }
}