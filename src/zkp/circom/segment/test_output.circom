pragma circom 2.1.1;


template test(N)
{
    signal input a[N];
    signal output b[N];


    var temp = 0;
    for (var i = 0; i < N; i++)
    {
        temp = a[i] * 2;
        temp += 1;
        b[i] <== temp;

        // b[i] <== a[i] * 2;
    }


}

component main = test(3);