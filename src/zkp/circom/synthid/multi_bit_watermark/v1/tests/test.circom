pragma circom 2.1.1;

include "../../../circomlib/circuits/mimc.circom";
include "../../../circomlib/circuits/comparators.circom";
include "../../../circomlib/circuits/poseidon2/poseidon2_hash.circom";

// K = 1 -> sliding_window_size = 1 

template detect_directly(seq_len) {

    //private input
    signal input key;

    //public input
    signal input uni_gram[seq_len];
    signal input current_token_index[seq_len];

    //public output
    signal output output_count[2];

    //tmp signal
    var sum_temp = 0;

    // corresponding to gamma = 0.5
    var threshold = 10944121435919637611123202872628637544274182200208017171849102093287904247808;

    component hasher1[seq_len];
    component lt[seq_len];


    for (var i = 0; i < seq_len; i++) {
        hasher1[i] = Poseidon2_hash(3);
        
        hasher1[i].inp[0] <== key;
        hasher1[i].inp[1] <== uni_gram[i];
        hasher1[i].inp[2] <== current_token_index[i];

        lt[i] = LessThan(255); // note the upper bound
        lt[i].in[0] <== hasher1[i].out;
        lt[i].in[1] <== threshold;

        sum_temp += lt[i].out;
    }

    output_count[0] <== sum_temp;
    output_count[1] <== sum_temp * 2;

}

component main {public [uni_gram, current_token_index]} = detect_directly(2);


