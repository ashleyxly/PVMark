pragma circom 2.1.1;

template LCG_Generator(data_len) {
    signal input current_hash;
    signal input data[data_len];

    signal output out;
    var multiplier = 6364136223846793005;
    var increment = 1;

    var temp;
    var hash_value = current_hash;
    for (var i = 0; i <= data_len - 1; i ++) {
        hash_value = hash_value + data[i];
        temp = hash_value * multiplier + increment;
        hash_value = temp;
    }
    out <== hash_value;
}