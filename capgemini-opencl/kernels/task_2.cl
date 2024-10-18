__kernel void matMul(__global float* pMatA, __global float* pMatB, __global float* pMatRes, int matDimN, int matDimM, int matDimK)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < matDimN && col < matDimK) {           // bounding checks
        float sum = 0.0f;
        for (int i = 0; i < matDimM; ++i) {
            sum += pMatA[row * matDimM + i] * pMatB[i * matDimK + col];
        }
        pMatRes[row * matDimK + col] = sum;
    }
}

#define TILE_SIZE 16

__kernel void matMulTiled(__global float* pMatA, __global float* pMatB, __global float* pMatRes, int matDimN, int matDimM, int matDimK)
{
    // Allocate local memory for tiles
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    // Get global row and column indices
    const int row = get_global_id(1);
    const int col = get_global_id(0);

    // Get local row and column within the workgroup
    const int localRow = get_local_id(1);
    const int localCol = get_local_id(0);

    const int tilesAmount = (matDimM  + TILE_SIZE - 1 ) / TILE_SIZE;

    float sum = 0.0f;  // Accumulator for the result

    // Loop over tiles of input matrices
    for (int tileIdx = 0; tileIdx < tilesAmount; ++tileIdx) {

        // Load a tile of pMatA and pMatB into local memory
        tileA[localRow][localCol] = pMatA[row * matDimM + tileIdx * TILE_SIZE + localCol];
        tileB[localRow][localCol] = pMatB[(tileIdx * TILE_SIZE + localRow) * matDimK + col];
        
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