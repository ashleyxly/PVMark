pragma circom 2.1.1;

include "each_position_recursive_detect.circom";

template total_detect(seq_len, each_position_infor, nLevels, position_len, token_num_1, token_num_2)
{
    signal input key;
    // signal input position[seq_len];

    //public input
    signal input public_root;

    signal input ngrams_1[token_num_1];
    signal input current_token_index_1[token_num_1];
    signal input pathIndices_1[token_num_1][nLevels];
    signal input siblings_1[token_num_1][nLevels];

    signal input ngrams_2[token_num_2];
    signal input current_token_index_2[token_num_2];
    signal input pathIndices_2[token_num_2][nLevels];
    signal input siblings_2[token_num_2][nLevels];

    signal output out[position_len * each_position_infor];

    component each_position_detect_1 = each_position_detect(seq_len, each_position_infor, nLevels);
    component each_position_detect_2 = each_position_detect(seq_len, each_position_infor, nLevels);

    each_position_detect_1.key <== key;
    each_position_detect_1.position <== current_token_index_1;
    each_position_detect_1.ngrams <== ngrams_1;
    each_position_detect_1.current_token_index <== current_token_index_1;
    each_position_detect_1.public_root <== public_root;
    each_position_detect_1.pathIndices <== pathIndices_1;
    each_position_detect_1.siblings <== siblings_1;

    each_position_detect_2.key <== key;
    each_position_detect_2.position <== current_token_index_2;
    each_position_detect_2.ngrams <== ngrams_2;
    each_position_detect_2.current_token_index <== current_token_index_2;
    each_position_detect_2.public_root <== public_root;
    each_position_detect_2.pathIndices <== pathIndices_2;
    each_position_detect_2.siblings <== siblings_2;

    for (var i = 0; i < each_position_infor; i ++)
    {
        out[i] <== each_position_detect_1.out[i];
        out[i + each_position_infor] <== each_position_detect_2.out[i];
    }

}

component main { public [public_root, ngrams_1, current_token_index_1, pathIndices_1, siblings_1, ngrams_2, current_token_index_2, pathIndices_2, siblings_2]} = total_detect(2, 2, 2, 2, 2, 2);