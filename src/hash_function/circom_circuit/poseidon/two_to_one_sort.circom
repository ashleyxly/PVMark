pragma circom 2.0.0;

include "../../zkp/circom/circomlib/circuits/poseidon.circom";
include "../../zkp/circom/circomlib/circuits/comparators.circom";

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
    component hasher2[T];
    component lt[T];
    
    for (var i = 0; i < T; i++) {
        hasher1[i] = Poseidon(2);
        hasher1[i].inputs[0] <== sk;
        hasher1[i].inputs[1] <== prompt_and_token_index_list[i];
        seed[i] <== hasher1[i].out;

        hasher2[i] = Poseidon(2);
        hasher2[i].inputs[0] <== seed[i];
        hasher2[i].inputs[1] <== prompt_and_token_index_list[i+1];
        hash_this[i] <== hasher2[i].out;

        lt[i] = LessThan(255); // note the upper bound
        lt[i].in[0] <== hash_this[i];
        lt[i].in[1] <== threshold_list[i];
        isgreenlist[i] <== lt[i].out;
        
        count += isgreenlist[i];
    }
    sG <== count;
}

// component main  = detect(200);
component main {public [threshold_list, prompt_and_token_index_list]} = detect(200);
