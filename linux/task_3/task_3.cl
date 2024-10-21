#define WARP_SIZE 32

// Last warp unrolling for OpenCL
inline void warpReduce(volatile __local float4* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// Main kernel for reduction operation
__kernel void reduce(__global float4* pArr, __global float4* pArrOut, int arrSize, __local float4* sdata)
{
    const unsigned int tid = get_local_id(0);             // work-item id in the work-group
    const unsigned int localSize = get_local_size(0);     // dimension of the work-group
    const unsigned int groupId = get_group_id(0);         // id of work-group

    // Calculate the global index for the input array
    const unsigned int idx = groupId * (localSize * 2) + tid;   // index of element to process

    // Initialize shared memory for this work item
    sdata[tid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    // Load the two elements into shared memory with bounds check
    if (idx < arrSize) {
        sdata[tid] = pArr[idx];  // Load the first element
        if (idx + localSize < arrSize) {
            sdata[tid] += pArr[idx + localSize];  // Add the second element if in bounds
        }
    } else {
        return;
    }
    barrier(CLK_LOCAL_MEM_FENCE);  // Synchronization across work-items in the same work-group

    // Tree-based sum up
    for (unsigned int s = get_local_size(0) / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Warp unrolling for the final warp
    if (tid < WARP_SIZE) {
        warpReduce(sdata, tid);
    }

    // Write result of this work-group to global memory
    if (tid == 0) {
        pArrOut[get_group_id(0)] = sdata[0];
    }
}