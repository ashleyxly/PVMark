pragma circom 2.1.1;

include "../../../../../circom_hash_and_detection/circomlib/circuits/mimc.circom";
include "../../../../../circom_hash_and_detection/circomlib/circuits/comparators.circom";
include "../../../../../circom_hash_and_detection/circomlib/circuits/poseidon2/poseidon2_hash.circom";
include "../../../../../circom_hash_and_detection/circomlib/circuits/poseidon.circom";
include "../../../../../circom_hash_and_detection/multi_bit_watermark/v1/merkle_tree.circom";
include "../../../../../circom_hash_and_detection/multi_bit_watermark/v1/summation.circom";
// include "../../../../../circom_hash_and_detection/multi_bit_watermark/v1/build_merkletree.circom";

// 所有Token都对应同一个position，即该函数的作用是复原特定position上的message
template each_position_detect(partial_seq_len, each_position_infor, nLevels) {
    
    //recursive input [0] count [1] ngrams_hash [2] current_token_index_hash
    // in the end, [1] [2] need to be public in order to complete consistency check
    signal input step_in[partial_seq_len * each_position_infor + 2];
    signal output step_out[partial_seq_len * each_position_infor + 2];

    signal input current_count[partial_seq_len][each_position_infor];
    signal input output_count[partial_seq_len][each_position_infor];

    //private input
    signal input key;
    signal input position[partial_seq_len];

    //public input
    signal input ngrams[partial_seq_len];
    signal input current_token_index[partial_seq_len];
    signal input public_root;


    // check step_in and current_count
    component input_hasher[partial_seq_len][each_position_infor];
    component output_hasher[partial_seq_len][each_position_infor];
    component output_hasher2;
    component output_hasher3;

    for (var i = 0; i < partial_seq_len; i ++)
    {
        for (var j = 0; j < each_position_infor; j ++)
        {
            // input_hasher[i][j] = Poseidon(2); 
            input_hasher[i][j] = Poseidon2_hash(2);
            input_hasher[i][j].inp[0] <== key;
            input_hasher[i][j].inp[1] <== current_count[i][j];
            // log("input_hasher[i][j].out", input_hasher[i][j].out);
            step_in[i * each_position_infor + j] === input_hasher[i][j].out;
        }
    }

    var count_temp[partial_seq_len][each_position_infor];

    for (var i = 0; i < partial_seq_len; i ++)
    {
        for (var j = 0; j < each_position_infor; j ++)
        {
            count_temp[i][j] = current_count[i][j];
        }
    }


    // position inclusion proof
    component hash_leaf[partial_seq_len];
    for (var i = 0; i < partial_seq_len; i ++)
    {
        // hash_leaf[i] = Poseidon(2);
        hash_leaf[i] = Poseidon2_hash(2);
        hash_leaf[i].inp[0] <== position[i];
        hash_leaf[i].inp[1] <== current_token_index[i];
    }


    signal input pathIndices[partial_seq_len][nLevels];
    signal input siblings[partial_seq_len][nLevels];
    
    component merkle_tree[partial_seq_len];
    signal root[partial_seq_len];
    // signal temp_temp;
    for (var i = 0; i < partial_seq_len; i ++)
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
    for (var i = 0; i < partial_seq_len; i ++)
    {
        root[i] === public_root;
        // log("root[i]", root[i]);
    }
    // inclusion proof end

    var threshold = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
    

    component hasher1[partial_seq_len][each_position_infor];
    component hasher2[partial_seq_len][each_position_infor];
    component lt[partial_seq_len][each_position_infor]; // note the upper bound
    // component sum[partial_seq_len][each_position_infor];

    for (var i = 0; i < partial_seq_len; i++) 
    {
        for (var j = 0; j <= each_position_infor - 1; j ++)
        {
            // hasher1[i][j] = Poseidon(3);
            hasher1[i][j] = Poseidon2_hash(3);
            hasher1[i][j].inp[0] <== key;
            hasher1[i][j].inp[1] <== ngrams[i];
            hasher1[i][j].inp[2] <== j;

            
            // hasher2[i][j] = Poseidon(2);
            hasher2[i][j] = Poseidon2_hash(2);
            hasher2[i][j].inp[0] <== hasher1[i][j].out;
            hasher2[i][j].inp[1] <== current_token_index[i];
            
            lt[i][j] = LessThan(255); // note the upper bound
            lt[i][j].in[0] <== hasher2[i][j].out;
            lt[i][j].in[1] <== threshold;


            count_temp[i][j] += lt[i][j].out;
            
        }
    }

    for (var i = 0; i < partial_seq_len; i ++)
    {
        for (var j = 0; j < each_position_infor; j ++)
        {
            output_count[i][j] === count_temp[i][j];
            // log("count_temp[i][j]", count_temp[i][j]);

            output_hasher[i][j] = Poseidon2_hash(2);
            output_hasher[i][j].inp[0] <== key;
            output_hasher[i][j].inp[1] <== output_count[i][j];
            // log("output_hasher[i].out", output_hasher[i].out);
            step_out[i * each_position_infor + j] <== output_hasher[i][j].out;
        }
    }

    output_hasher2 = MultiMiMC7(partial_seq_len + 1, 91);
    output_hasher2.in[0] <== step_in[each_position_infor];
    for (var i = 1; i <= partial_seq_len; i ++)
    {
        output_hasher2.in[i] <== ngrams[i - 1];
    }
    output_hasher2.k <== key;
    // log("output_hasher2.out", output_hasher2.out);
    step_out[partial_seq_len * each_position_infor] <== output_hasher2.out;

    output_hasher3 = MultiMiMC7(partial_seq_len + 1, 91);
    output_hasher3.in[0] <== step_in[each_position_infor + 1];
    for (var i = 1; i <= partial_seq_len; i ++)
    {
        output_hasher3.in[i] <== current_token_index[i - 1];
    }
    output_hasher3.k <== key;
    // log("output_hasher3.out", output_hasher3.out);
    step_out[partial_seq_len * each_position_infor + 1] <== output_hasher3.out;


    // log("step_out", step_out);
    log("final");

}

component main { public [step_in] } = each_position_detect(80, 16, 15);

// component main = each_position_detect(80, 16, 15);