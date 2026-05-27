pragma circom 2.0.0;

include "../../circomlib/circuits/poseidon2/poseidon2_hash.circom";
include "../../circomlib/circuits/comparators.circom";

template detect(T) {
    signal input fixed_threshold;
    signal input prompt_and_token_index_list[T+1];

    signal input sk;

    signal output sG;

    signal seed[T];
    signal hash_this[T];
    signal isgreenlist[T];
    
    var count = 0;

    component hasher1[T];
    component hasher2[T];
    component lt[T];
    
    for (var i = 0; i < T; i++) {
        hasher1[i] = Poseidon2_hash(2);
        hasher1[i].inp[0] <== sk;
        hasher1[i].inp[1] <== prompt_and_token_index_list[i];
        seed[i] <== hasher1[i].out;

        hasher2[i] = Poseidon2_hash(2);
        hasher2[i].inp[0] <== seed[i];
        hasher2[i].inp[1] <== prompt_and_token_index_list[i+1];
        hash_this[i] <== hasher2[i].out;

        lt[i] = LessThan(255);
        lt[i].in[0] <== hash_this[i];
        lt[i].in[1] <== fixed_threshold;
        isgreenlist[i] <== lt[i].out;
        
        count += isgreenlist[i];
    }
    sG <== count;
}

component main {public [fixed_threshold, prompt_and_token_index_list]} = detect(400);
