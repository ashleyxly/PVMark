pragma circom 2.1.1;

include "./summation.circom";


template test(N)
{
    signal input a;
    signal output b;

    component sum = Summation(N);
    var temp = 0;
    var temp2 = 1;
    sum.in[0] <== temp;
    // sum.in[1] <== temp2;
    b <== sum.sum;

}

component main = test(1);