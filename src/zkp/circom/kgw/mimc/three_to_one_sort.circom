pragma circom 2.0.0;

include "../../circomlib/circuits/mimc.circom";
include "../../circomlib/circuits/comparators.circom";

template detect(T) {
    //public input
    //signal input prompt_index;
    //signal input token_index_list[T];
    signal input threshold_list[T];
    signal input prompt_and_token_index_list[T+1];

    //private input
    signal input sk;

    //output
    signal output sG; //number of green list tokens in text

    //tmp signal
    signal seed[T];
    signal hash_this[T];
    signal isgreenlist[T];
    
    var count = 0;

    component hasher1[T];
    component lt[T];
    
    for (var i = 0; i < T; i++) {
        hasher1[i] = MultiMiMC7(3, 91);
        hasher1[i].in[0] <== sk;
        hasher1[i].in[1] <== prompt_and_token_index_list[i];
        hasher1[i].in[2] <== prompt_and_token_index_list[i + 1];
        hasher1[i].k <== 2024;
        hash_this[i] <== hasher1[i].out;

        lt[i] = LessThan(256); // note the upper bound
        lt[i].in[0] <== hash_this[i];
        lt[i].in[1] <== threshold_list[i];
        isgreenlist[i] <== lt[i].out;
        
        count += isgreenlist[i];
    }
    sG <== count;
}

// component main  = detect(200);
component main {public [threshold_list, prompt_and_token_index_list]} = detect(200);
