pragma circom 2.1.1;

include "../lcg_generator.circom";

template test() {
    signal input data;
    signal output out;
    component lcg_generator = LCG_Generator(1);
    lcg_generator.current_hash <== 0;
    lcg_generator.data[0] <== data;
    out <== lcg_generator.out;
    log(out);
}

component main = test();