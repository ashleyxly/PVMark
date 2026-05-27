pragma circom 2.0.0;

include "../../zkp/circom/circomlib/circuits/poseidon.circom";
include "../../zkp/circom/circomlib/circuits/comparators.circom";

template detect(T) {
    //public input
    //signal input prompt_index;
    //signal input token_index_list[T];
    signal input fixed_threshold;
    signal input prompt_and_token_index_list[T + 1];

    //private input
    signal input sk;

    //output
    signal output sG;//number of green list tokens in text

    //tmp signal
    signal seed[T];
    signal hash_this[T];
    signal isgreenlist[T];
    
    var count = 0;

    component hasher[T];
    component lt[T];
    
    for (var i = 0; i < T; i++) {
        hasher[i] = Poseidon(3);
        hasher[i].inputs[0] <== sk;
        hasher[i].inputs[1] <== prompt_and_token_index_list[i];
        hasher[i].inputs[2] <== prompt_and_token_index_list[i+1];
        hash_this[i] <== hasher[i].out;

        lt[i] = LessThan(255);
        lt[i].in[0] <== hash_this[i];
        lt[i].in[1] <== fixed_threshold;
        isgreenlist[i] <== lt[i].out;
        
        count += isgreenlist[i]; 
    } 
    
    sG <== count;
}

// component main  = detect(200);
component main {public [fixed_threshold, prompt_and_token_index_list]} = detect(200);
