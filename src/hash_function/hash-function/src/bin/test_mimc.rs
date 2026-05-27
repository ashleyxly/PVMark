extern crate mimc_rs;
use mimc_rs::{Fr, Mimc7};
extern crate ff_2;
use ff_2::*;
use ff_2::to_hex;
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
    let mimc7 = Mimc7::new(91);
    let result = mimc7.hash(&b1, &b2);
    println!("result: {:?}", result);
}

fn test_2() {
    let b12: Fr = Fr::from_str("12").unwrap();
    let b45: Fr = Fr::from_str("45").unwrap();
    let b78: Fr = Fr::from_str("78").unwrap();
    let b41: Fr = Fr::from_str("41").unwrap();
    let mut big_arr1: Vec<Fr> = Vec::new();
    big_arr1.push(b12.clone());
    big_arr1.push(b45.clone());
    big_arr1.push(b78.clone());
    big_arr1.push(b41.clone());
    let mimc7 = Mimc7::new(91);
    let h1 = mimc7.multi_hash(big_arr1, &Fr::zero());
    let h1_1 = h1.into_repr();
    println!("h1_1: {:?}", h1_1);
    let h1_2 = h1.into_raw_repr();
    println!("h1_2: {:?}", h1_2);

    println!("h1: {:?}", h1);
}

fn generate_multi_two_to_one_hash() {
    let mimc7 = Mimc7::new(91);
    let mut sk = Fr::from_str("20242024").unwrap();
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let mut input: Vec<Fr> = vec![sk.clone(), Fr::from_str(i.to_string().as_str()).unwrap()];
        let result = mimc7.multi_hash(input.clone(), &Fr::zero());
        results.push(result);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/mimc7/results.txt").unwrap();
    for i in 0..len {
        let result = results[i];
        let result_hex = to_hex(&result);
        let decimal_bigint = BigUint::parse_bytes(result_hex.as_bytes(), 16).unwrap();
        // println!("decimal_bigint: {:?}", decimal_bigint);
        let hex_string_hex = format!("{:x}", decimal_bigint);
        // println!("hex_string_hex: {:?}", hex_string_hex);
        file.write_all(hex_string_hex.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }

}

fn main() {
    // test();
    // println!("-----------------");
    // test_2();
    // let b = Fr::from_str("1").unwrap();
    // println!("b: {:?}", b);
    generate_multi_two_to_one_hash();
}
