pragma circom 2.1.1;

include "../../../../circomlib/circuits/mimc.circom";
include "../../../../circomlib/circuits/comparators.circom";

// in order to consistent with the rust version (no two-input mimc available)
// template Multiple_MIMC7(nInputs, nRounds) {
//     signal input current_hash;
//     signal input data[nInputs];
//     signal input k;
//     signal output out;

//     component mims[nInputs];
//     for (var i = 0; i < nInputs; i++) {
//         mims[i] = MiMC7(nRounds);
//         mims[i].x_in <== data[i];
//         mims[i].k <== current_hash;
//         current_hash <== mims[i].out;
//     }
// }

template detect_recursive(seq_len, sliding_window_size, watermarking_depth) {
    //recursive input [0] count [1] ngrams [2] current_token_index
    signal input step_in[3];

    //private input (used in hash)
    signal input random_seed;

    //private input
    signal input current_count;
    signal input output_count;
    signal input key[watermarking_depth];

    //public input
    signal input ngrams[seq_len][sliding_window_size];
    signal input current_token_index[seq_len];

    //recursive output //number of green list tokens in text
    signal output step_out[3]; 

    //tmp signal
    signal g_value[seq_len][watermarking_depth];
    signal judge_g_value[seq_len][watermarking_depth];
    // signal is_one[seq_len][watermarking_depth];
    var sum_temp = current_count;

    var sum_ngrams = seq_len * sliding_window_size;

    var threshold = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
    

    // var count = current_count;
    component input_hasher = MultiMiMC7(1, 91);
    component output_hasher = MultiMiMC7(1, 91);
    component output2_hasher = MultiMiMC7(sum_ngrams + 1, 91); // +1 denotes the previous res
    component output3_hasher = MultiMiMC7(seq_len + 1, 91); // +1 denotes the previous res

    component hasher1[seq_len];
    component hasher2[seq_len][watermarking_depth];
    component lt[seq_len][watermarking_depth]; // note the upper bound

    input_hasher.in[0] <== current_count;
    input_hasher.k <== random_seed;
    step_in[0] === input_hasher.out;


    for (var i = 0; i < seq_len; i++) {
        hasher1[i] = MultiMiMC7(sliding_window_size + 1, 91);
        // hasher1[i].current_hash <== 1;
        for (var j = 0; j < sliding_window_size; j++) {
            hasher1[i].in[j] <== ngrams[i][j];
        }
        hasher1[i].in[sliding_window_size] <== current_token_index[i];
        hasher1[i].k <== 0;

        for (var k = 0; k < watermarking_depth; k ++) {
            hasher2[i][k] = MultiMiMC7(2, 91);
            hasher2[i][k].in[0] <== hasher1[i].out;
            hasher2[i][k].in[1] <== key[k];
            hasher2[i][k].k <== 0;
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

    output_count === sum_temp;

    output_hasher.in[0] <== output_count;
    output_hasher.k <== random_seed;
    step_out[0] <== output_hasher.out;

    output2_hasher.in[0] <== step_in[1];
    var output2_hasher_index = 1;
    for (var i = 0; i < seq_len; i ++) {
        for (var j = 0; j < sliding_window_size; j++) {
            output2_hasher.in[output2_hasher_index] <== ngrams[i][j];
            output2_hasher_index += 1;
        }
    }
    output2_hasher.k <== random_seed;
    step_out[1] <== output2_hasher.out;

    output3_hasher.in[0] <== step_in[2];
    var output3_hasher_index = 1;
    for (var i = 0; i < seq_len; i ++) {
        output3_hasher.in[output3_hasher_index] <== current_token_index[i];
        output3_hasher_index += 1;
    }
    output3_hasher.k <== random_seed;
    step_out[2] <== output3_hasher.out;


}

component main { public [step_in] } = detect_recursive(10, 4, 30);

