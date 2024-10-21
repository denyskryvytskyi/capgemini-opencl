__kernel void add(__global const float4* bufferA,
                  __global const float4* bufferB,
                  __global float4* bufferRes,
                  int size)
{
    const unsigned int id = get_global_id(0);
    if (id < size) {
        bufferRes[id] = bufferA[id] + bufferB[id];
    }
}