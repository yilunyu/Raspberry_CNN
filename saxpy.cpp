#include <pimd.h>

int saxpy(int size, float a, float X[], float Y[], float Z[]) {
    const int opsize = 4, argsize = 4;

    PimdArg ops[opsize];
    ops[0] = OP_VLOAD;
    ops[1] = OP_SFMUL;
    ops[2] = OP_VFADD;
    ops[3] = OP_STORE;

    PimdArg args[argsize];
    args[0] = &X;
    args[1] = a;
    args[2] = &Y;
    args[3] = &Z;
    
    int mb = pimd_open();
    PimdFunction func= PimdFunction(mb, ops, opsize);
    int output = func.call(args, argsize, size, INT_MAX);

    pimd_close(mb);
    func.free();
    
    return output;
}