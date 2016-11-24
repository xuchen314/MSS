   %You should set the defaul fold as Utilities
    %mex -O fastWHtrans.cpp
    mex -O -largeArrayDims updateSval1.c -lmwblas -output  updateSval1
    %mex -O -largeArrayDims updateSval.c 
    %mex -O updateSvalZw.c
    mex -O -largeArrayDims partXY.c -lmwblas -output partXY
    %mex -O -largeArrayDims partXY.c
    %mex -O proj_LR_sum.c



