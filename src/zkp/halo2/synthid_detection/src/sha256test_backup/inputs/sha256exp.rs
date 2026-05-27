use halo2_gadgets::sha256::BlockWord;
use halo2_proofs::circuit::Value;

pub const INPUT_2: [BlockWord; 16 * 2] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 2];
pub const INPUT_3: [BlockWord; 16 * 3] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 3];
pub const INPUT_5: [BlockWord; 16 * 5] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 5];
pub const INPUT_9: [BlockWord; 16 * 9] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 9];
pub const INPUT_17: [BlockWord; 16 * 17] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 17];
pub const INPUT_33: [BlockWord; 16 * 33] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 33];
pub const INPUT_65: [BlockWord; 16 * 65] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 65];
pub const INPUT_129: [BlockWord; 16 * 129] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 129];
pub const INPUT_257: [BlockWord; 16 * 257] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 257];
pub const INPUT_513: [BlockWord; 16 * 513] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 513];
pub const INPUT_1025: [BlockWord; 16 * 1025] = [BlockWord(Value::known(0b01111000100000000000000000000000)); 16 * 1025];