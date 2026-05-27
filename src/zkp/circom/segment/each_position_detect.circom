pragma circom 2.1.1;

// include "../circomlib/circuits/mimc.circom";
include "../circomlib/circuits/comparators.circom";
// include "../circomlib/circuits/poseidon2/poseidon2_hash.circom";
include "../circomlib/circuits/poseidon.circom";
include "merkle_tree.circom";
include "summation.circom";
// include "build_merkletree.circom";

// 所有Token都对应同一个position，即该函数的作用是复原特定position上的message
template each_position_detect(seq_len, each_position_infor, nLevels) {

    //private input
    signal input key;
    signal input position[seq_len];

    //public input
    signal input ngrams[seq_len];
    signal input current_token_index[seq_len];
    signal input public_root;

    var count_temp[each_position_infor];
    // var count_temp[position_len][each_position_infor];

    for (var i = 0; i < each_position_infor; i ++)
    {
        count_temp[i] = 0;
    }
    // signal output out[position_len * each_position_infor];
    signal output out[each_position_infor];
    // count matrix + root number


    // position inclusion proof
    component hash_leaf[seq_len];
    for (var i = 0; i < seq_len; i ++)
    {
        hash_leaf[i] = Poseidon(2);
        hash_leaf[i].inputs[0] <== position[i];
        hash_leaf[i].inputs[1] <== current_token_index[i];
    }


    signal input pathIndices[seq_len][nLevels];
    signal input siblings[seq_len][nLevels];
    
    component merkle_tree[seq_len];
    signal root[seq_len];
    // signal temp_temp;
    for (var i = 0; i < seq_len; i ++)
    {
        merkle_tree[i] = MerkleTreeInclusionProof(nLevels);
        // merkle_tree[i].leaf <== position[i];
        merkle_tree[i].leaf <== hash_leaf[i].out;
        for (var j = 0; j < nLevels; j ++)
        {
            merkle_tree[i].pathIndices[j] <== pathIndices[i][j];
            merkle_tree[i].siblings[j] <== siblings[i][j];
        }
        root[i] <== merkle_tree[i].root;
    }
    for (var i = 0; i < seq_len; i ++)
    {
        root[i] === public_root;
    }
    // inclusion proof end

    var threshold = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
    

    component hasher1[seq_len][each_position_infor];
    component hasher2[seq_len][each_position_infor];
    component lt[seq_len][each_position_infor]; // note the upper bound
    // component sum[seq_len][each_position_infor];

    for (var i = 0; i < seq_len; i++) 
    {
        for (var j = 0; j <= each_position_infor - 1; j ++)
        {
            hasher1[i][j] = Poseidon(3);
            hasher1[i][j].inputs[0] <== key;
            hasher1[i][j].inputs[1] <== ngrams[i];
            hasher1[i][j].inputs[2] <== j;

            
            hasher2[i][j] = Poseidon(2);
            hasher2[i][j].inputs[0] <== hasher1[i][j].out;
            hasher2[i][j].inputs[1] <== current_token_index[i];
            
            lt[i][j] = LessThan(255); // note the upper bound
            lt[i][j].in[0] <== hasher2[i][j].out;
            lt[i][j].in[1] <== threshold;

            // var position_index = position[i];
            // count_temp[position[i]][j] += lt[i][j].out;
            // count_temp[position[i] * each_position_infor + j] += lt[i][j].out;
            count_temp[j] += lt[i][j].out;
            // count_all += lt[i][j].out;
            // count_one_d[i] += lt[i][j].out;
            
        }
    }

    for (var i = 0; i < each_position_infor; i ++)
    {
        out[i] <== count_temp[i];
    }


}

// component main = each_position_detect(1, 16, 20);