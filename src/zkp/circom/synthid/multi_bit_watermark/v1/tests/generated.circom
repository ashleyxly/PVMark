pragma circom 2.1.1;
include "../each_position_detect.circom";

template total_detect(seq_len, each_position_infor, nLevels, position_len, token_num_1, token_num_2, token_num_3)
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
    signal input ngrams_3[token_num_3];
    signal input current_token_index_3[token_num_3];
    signal input pathIndices_3[token_num_3][nLevels];
    signal input siblings_3[token_num_3][nLevels];
    signal output out[position_len * each_position_infor];
    component each_position_detect_1 = each_position_detect(seq_len, each_position_infor, nLevels);
    component each_position_detect_2 = each_position_detect(seq_len, each_position_infor, nLevels);
    component each_position_detect_3 = each_position_detect(seq_len, each_position_infor, nLevels);
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
    each_position_detect_3.key <== key;
    each_position_detect_3.position <== current_token_index_3;
    each_position_detect_3.ngrams <== ngrams_3;
    each_position_detect_3.current_token_index <== current_token_index_3;
    each_position_detect_3.public_root <== public_root;
    each_position_detect_3.pathIndices <== pathIndices_3;
    each_position_detect_3.siblings <== siblings_3;
    for (var i = 0; i < each_position_infor; i ++)
    {
        out[i + 0 * each_position_infor] <== each_position_detect_1.out[i];
        out[i + 1 * each_position_infor] <== each_position_detect_2.out[i];
        out[i + 2 * each_position_infor] <== each_position_detect_3.out[i];
    }
}
component main {public [public_root, ngrams_1, current_token_index_1, pathIndices_1, siblings_1, ngrams_2, current_token_index_2, pathIndices_2, siblings_2, ngrams_3, current_token_index_3, pathIndices_3, siblings_3]} = total_detect(2, 2, 2, 3, 2, 2, 2);