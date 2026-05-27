pragma circom 2.1.1;

include "../../../../circomlib/circuits/mimc.circom";
include "../../../../circomlib/circuits/comparators.circom";
include "../../../../circomlib/circuits/poseidon.circom";

template detect_directly(seq_len, sliding_window_size, watermarking_depth) {

    //private input
    // signal input current_count;
    // signal input output_count;
    signal input key[watermarking_depth];

    //public input
    signal input ngrams[seq_len][sliding_window_size];
    signal input current_token_index[seq_len];

    signal output output_count;


    //tmp signal
    signal g_value[seq_len][watermarking_depth];
    signal judge_g_value[seq_len][watermarking_depth];
    // signal is_one[seq_len][watermarking_depth];
    var sum_temp = 0;

    var sum_ngrams = seq_len * sliding_window_size;

    var threshold = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
    

    component hasher1[seq_len];
    component hasher2[seq_len][watermarking_depth];
    component lt[seq_len][watermarking_depth]; // note the upper bound


    for (var i = 0; i < seq_len; i++) {
        hasher1[i] = Poseidon(sliding_window_size + 1);
        // hasher1[i].current_hash <== 1;
        for (var j = 0; j < sliding_window_size; j++) {
            hasher1[i].inputs[j] <== ngrams[i][j];
        }
        hasher1[i].inputs[sliding_window_size] <== current_token_index[i];

        for (var k = 0; k < watermarking_depth; k ++) {
            hasher2[i][k] = Poseidon(2);
            hasher2[i][k].inputs[0] <== hasher1[i].out;
            hasher2[i][k].inputs[1] <== key[k];
            judge_g_value[i][k] <== hasher2[i][k].out;
            lt[i][k] = LessThan(255); // note the upper bound
            lt[i][k].in[0] <== judge_g_value[i][k];
            lt[i][k].in[1] <== threshold;
            g_value[i][k] <== lt[i][k].out;
        }

        for (var k = 0; k < watermarking_depth; k ++) {
            sum_temp += g_value[i][k];
        }

    }

    output_count <== sum_temp;

}

component main = detect_directly(25, 4, 30);

