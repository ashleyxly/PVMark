use std::convert::TryInto;

use rand::rngs::OsRng;
use rand::Rng;
use hash_rustlib::pedersen::hash::hash;
use num_bigint::BigUint;
use num_traits::Num;
use std::io::Write;
use ark_bn254::Fr;
use std::str::FromStr;
use ark_ff::fields::PrimeField;
use ark_ff::BigInteger;

fn test() {
    let mut data = [0u8; 32];
    let mut rng = OsRng::default();
    rng.fill(&mut data);
    let result = hash(&data);
    println!("result: {:?}", result);
    println!("result.len: {:?}", result.len());
    let result2 = hex::encode(result);
    println!("result2: {:?}", result2);
    let decimal_number = BigUint::parse_bytes(result2.as_bytes(), 16).unwrap();
    let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    let res = decimal_number % modulo;
    println!("res: {:?}", res);
        

}


fn generate_multi_two_to_one_hash() {
    let sk = Fr::from_str("20242024").unwrap();
    let sk_bigint = sk.into_bigint();
    let sk_bytes = sk_bigint.to_bytes_be();
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let i_fr = Fr::from(i as u64);
        let i_bigint = i_fr.into_bigint();
        let i_bytes = i_bigint.to_bytes_be();
        let mut inputs: Vec<u8> = vec![];
        inputs.append(&mut sk_bytes.clone());
        inputs.append(&mut i_bytes.clone());
        let result = hash(&inputs);
        let result2 = hex::encode(result);
        let decimal_number = BigUint::parse_bytes(result2.as_bytes(), 16).unwrap();
        let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        let res = decimal_number % modulo;
        results.push(res);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/pedersen/results.txt").unwrap();
    for i in 0..len {
        let result = results[i].clone();
        let hex_string = format!("{:x}", result);
        file.write_all(hex_string.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}


fn main() {
    // test();
    generate_multi_two_to_one_hash();
}