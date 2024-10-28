#include <iostream>

#include "addVector.h"
#include "matMul.h"
#include "reduction.h"
#include "sort.h"

int main()
{
    // vector addition test
    if (true) {
        addVector::run();
    }

    // matrix multiplication test
    if (true) {
        matMul::run();
    }

    // reduction (sum) test
    if (true) {
        reduction::run();
    }

    // sorting test
    if (true) {
        sort::run();
    }

    return 0;
}
