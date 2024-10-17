__kernel void vector_add(__global const float4* bufferA, __global const float4* bufferB, __global float4* bufferRes, const unsigned int size)
{
    __const int id = get_global_id(0); // Get index into global data vector

    // Check bounds
    if (id < size) {
        bufferRes[id] = bufferA[id] + bufferB[id];
    }
}