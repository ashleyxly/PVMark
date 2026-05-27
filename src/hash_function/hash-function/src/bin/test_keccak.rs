extern crate crypto;
use self::crypto::digest::Digest;
use self::crypto::sha3::Sha3;
use ark_bn254::Fr;
use std::str::FromStr;
use ark_ff::PrimeField;
use num_bigint::BigUint;
use ark_ff::BigInteger;
use std::io::Write;
use num_traits::Num;




fn test() {
    let mut hasher = Sha3::keccak256();

    // hasher.input_str("abc");
    let b1: Fr = Fr::from_str(
        "12242166908188651009877250812424843524687801523336557272219921456462821518061",
    )
    .unwrap();
    let string_test = b1.into_bigint().to_string();
    hasher.input_str(&string_test);

    let hex = hasher.result_str();
    println!("hex: {:?}", hex);
    let decimal_number = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap();
    println!("decimal_number: {}", decimal_number);

    let threshold = BigUint::parse_bytes(b"21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
    println!("threshold:      {:?}", threshold);

    if decimal_number > threshold {
        println!("number1 大于 number2");
    } else if decimal_number < threshold {
        println!("number1 小于 number2");
    } else {
        println!("number1 等于 number2");
    }   
}

fn generate_multi_two_to_one_hash() {
    let sk = Fr::from_str("20242024").unwrap();
    let sk_bigint = sk.into_bigint();
    let sk_bytes = sk_bigint.to_bytes_be();
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let mut hasher = Sha3::keccak256();
        let mut inputs: Vec<u8> = vec![];
        inputs.append(&mut sk_bytes.clone());
        let i_fr = Fr::from(i as u64);
        let i_bigint = i_fr.into_bigint();
        let i_bytes = i_bigint.to_bytes_be();
        inputs.append(&mut i_bytes.clone());
        hasher.input(&inputs);
        let hex = hasher.result_str();
        let decimal_number = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap();
        let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        // println!("modulo:           {}", modulo);
        // println!("decimal_number:   {}", decimal_number);
        let decimal_number_2 = decimal_number.clone() % modulo;
        // println!("decimal_number_2: {}", decimal_number_2);
        // let res = format!("{:x}", decimal_number_2);
        let res = format!("{:x}", decimal_number.clone());
        // println!("res: {}", res);
        results.push(res);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/keccak256/results_2.txt").unwrap();
    for i in 0..len {
        let result = results[i].clone();
        // let hex_string_hex = format!("{:x}", result);
        file.write_all(result.as_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}


fn main() {
    // test();
    generate_multi_two_to_one_hash();
}