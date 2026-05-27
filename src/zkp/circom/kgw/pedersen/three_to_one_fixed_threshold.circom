pragma circom 2.0.0;

include "../../circomlib/circuits/pedersen.circom";
include "../../circomlib/circuits/bitify.circom";

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
    component n2b_sk;
    component n2b_prompt_and_token_index_list[T + 1];
    component n2b[T];

    n2b_sk = Num2Bits(256);
    n2b_sk.in <== sk;
    for (var i = 0; i < T + 1; i++) {
        n2b_prompt_and_token_index_list[i] = Num2Bits(256);
        n2b_prompt_and_token_index_list[i].in <== prompt_and_token_index_list[i];
    }
    
    for (var i = 0; i < T; i++) {
        hasher1[i] = Pedersen(256 * 3);
        for (var j = 0; j < 256; j++) {
            hasher1[i].in[j] <== n2b_sk.out[j];
            hasher1[i].in[j + 256] <== n2b_prompt_and_token_index_list[i].out[j];
            hasher1[i].in[j + 256 * 2] <== n2b_prompt_and_token_index_list[i + 1].out[j];
        }

        hash_this[i] <== hasher1[i].out[1];

        lt[i] = LessThan(256);
        lt[i].in[0] <== hash_this[i];
        lt[i].in[1] <== fixed_threshold;
        isgreenlist[i] <== lt[i].out;
        
        count += isgreenlist[i];
    }
    sG <== count;
}

component main {public [fixed_threshold, prompt_and_token_index_list]} = detect(200);
