__kernel void sort(__global int* data, int size, int stride, int stage)
{
    const int idx = get_global_id(0); // Index of element
    const int compareIdx = idx ^ stride; // Index of compared element

    if (compareIdx > idx) {
        // Ascending order if (idx & stage) == 0, otherwise descending
        if ((idx & stage) == 0) {
            if (data[idx] > data[compareIdx]) {
                const int temp = data[idx];
                data[idx] = data[compareIdx];
                data[compareIdx] = temp;
            }
        } else {
            if (data[idx] < data[compareIdx]) {
                const int temp = data[idx];
                data[idx] = data[compareIdx];
                data[compareIdx] = temp;
            }
        }
    }
}