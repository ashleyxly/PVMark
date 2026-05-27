use ark_bn254::Fr;
use hash_rustlib::poseidon::hash::Poseidon;
use ark_ff::PrimeField;

use ark_std::str::FromStr;
use num_bigint::{BigInt,BigUint};
use std::io::Write;

fn test() {
    let b1: Fr = Fr::from_str(
        "12242166908188651009877250812424843524687801523336557272219921456462821518061",
    )
    .unwrap();
    let b2: Fr = Fr::from_str(
        "12242166908188651009877250812424843524687801523336557272219921456462821518061",
    )
    .unwrap();
    let mut big_arr: Vec<Fr> = Vec::new();
    big_arr.push(b1.clone());
    big_arr.push(b2.clone());
    let poseidon = Poseidon::new();
    let result = poseidon.hash(big_arr.clone()).unwrap();
    println!("result: {:?}", result);
    let result_bigint = result.into_bigint();
    println!("result_bigint: {:?}", result_bigint);
    let hex_string = result_bigint.to_string();
    println!("hex_string: {:?}", hex_string);
    
}

fn generate_multi_two_to_one_hash() {
    let poseidon = Poseidon::new();
    let mut sk = Fr::from_str("20242024").unwrap();
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let mut input: Vec<Fr> = vec![sk.clone(), Fr::from(i as u64)];
        let result = poseidon.hash(input.clone()).unwrap();
        results.push(result);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/poseidon/results.txt").unwrap();
    for i in 0..len {
        let result = results[i];
        let result_bigint = result.into_bigint();
        let hex_string = result_bigint.to_string();
        let decimal_bigint = BigUint::parse_bytes(hex_string.as_bytes(), 10).unwrap();
        let hex_string_hex = format!("{:x}", decimal_bigint);
        file.write_all(hex_string_hex.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}

fn generate_multi_three_to_one_hash() {
    let poseidon = Poseidon::new();
    let mut sk = Fr::from_str("20242024").unwrap();
    let pre_token_index = 14321;
    let len = 51000;
    let mut results = vec![];
    for i in 0..len {
        let mut input: Vec<Fr> = vec![sk.clone(), Fr::from(pre_token_index as u64), Fr::from(i as u64)];
        let result = poseidon.hash(input.clone()).unwrap();
        results.push(result);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/poseidon/results_2.txt").unwrap();
    for i in 0..len {
        let result = results[i];
        let result_bigint = result.into_bigint();
        let hex_string = result_bigint.to_string();
        let decimal_bigint = BigUint::parse_bytes(hex_string.as_bytes(), 10).unwrap();
        let hex_string_hex = format!("{:x}", decimal_bigint);
        file.write_all(hex_string_hex.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}


fn main() {
    // test();
    // let test = BigInt::from_str("20").unwrap();
    // let hex_string = test.to_string();
    // let temp = Fr::from_str(
    //     "21888242871839275222246405745257275088548364400416034343698204186575808495619",
    // );
    // println!("temp: {:?}", temp);
    // let temp_bigint = temp.unwrap().into_bigint();
    // println!("temp_bigint: {:?}", temp_bigint);
    // let hex_string = temp_bigint.to_string();
    // println!("hex_string: {:?}", hex_string);
    // generate_multi_two_to_one_hash();
    generate_multi_three_to_one_hash();

}