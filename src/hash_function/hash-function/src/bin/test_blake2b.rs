extern crate crypto;
use self::crypto::digest::Digest;
use self::crypto::blake2b::Blake2b;
use ark_bn254::Fr;
use halo2curves::serde::SerdeObject;
// use mimc_rs::Fr;
use std::str::FromStr;
use ark_ff::PrimeField;
use num_bigint::BigUint;
use std::io::Write;
use ark_ff::BigInteger;
use num_traits::Num;
use halo2_proofs::halo2curves::bn256::Fr as Fr_bn256;


fn test() {
    let inputs: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut hasher = Blake2b::new(32);
    hasher.input(&inputs);
    let mut out = [0u8; 32];
    hasher.result(&mut out);
    let hex = hex::encode(out);
    println!("hex: {:?}", hex);

}

fn generate_multi_two_to_one_hash() {
    let sk = Fr::from_str("15485863").unwrap();
    let sk_bigint = sk.into_bigint();
    let sk_bytes = sk_bigint.to_bytes_be();
    let len = 65536;
    let mut results = vec![];
    for i in 0..len {
        let mut inputs: Vec<u8> = vec![];
        inputs.append(&mut sk_bytes.clone());
        let i_fr = Fr::from(i as u64);
        let i_bigint = i_fr.into_bigint();
        let i_bytes = i_bigint.to_bytes_be();
        inputs.append(&mut i_bytes.clone());
        let mut hasher = Blake2b::new(32);
        hasher.input(&inputs);
        let mut out = [0u8; 32];
        hasher.result(&mut out);
        let mut out_u64 = [0u64; 4];

        for i in 0..4 {
            out_u64[i] = u64::from_le_bytes([
                out[i * 8],
                out[i * 8 + 1],
                out[i * 8 + 2],
                out[i * 8 + 3],
                out[i * 8 + 4],
                out[i * 8 + 5],
                out[i * 8 + 6],
                out[i * 8 + 7],
            ]);
        }
        // println!("out: {:?}", out.clone());
        // println!("out.len: {:?}", out.len());
        // println!("out_u64: {:?}", out_u64.clone());
        // println!("out_u64.len: {:?}", out_u64.len());
        let hex_list: Vec<String> = out_u64.iter().map(|&x| format!("{:x}", x)).collect();
        // println!("hex_list: {:?}", hex_list.clone());
        let mut u64_array: [u64; 4] = [0; 4];
        for (i, hex_str) in hex_list.iter().enumerate() {
            u64_array[i] = u64::from_str_radix(hex_str, 16).expect("Failed to parse hex string to u64");
        }


        // let res = Fr_bn256::from_raw_bytes(&out.clone()).unwrap();
        // let res = Fr_bn256::from_raw(out_u64.clone());
        let res = Fr_bn256::from_raw(u64_array.clone());
        // println!("res: {:?}", res);

        // let hex = hex::encode(out);

        // let hex_biguint = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap();
        // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
        // let hex_biguint_mod = hex_biguint % modulo;

        // let hex_string = hex_biguint_mod.to_string();
        // let decimal_bigint = BigUint::parse_bytes(hex_string.as_bytes(), 10).unwrap();
        // let hex_string_hex = format!("{:x}", decimal_bigint);

        // results.push(hex_string_hex);
        results.push(res);
    }
    // write results to a txt file
    let mut file = std::fs::File::create("/mnt/disk2/username/ZKLLMWatermark/hash_uniformity_test/blake2b/results.txt").unwrap();
    for i in 0..len {
        let result = results[i].clone();
        let data = result.to_bytes();
        let hex_string: String = data.iter().map(|b| format!("{:x}", b)).collect();
        file.write_all(hex_string.as_bytes()).unwrap();
        // file.write_all(&result.to_bytes()).unwrap();
        file.write_all("\n".as_bytes()).unwrap();
    }
}


fn test_field() {
    let a = Fr_bn256::from_raw([
        0x43e1f593f0000001,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ]);
    println!("a: {:?}", a);
    let b = Fr_bn256::from_raw([
        0x43e1f593f0000010,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ]);
    println!("b: {:?}", b);
}

fn main() {
    // test();
    generate_multi_two_to_one_hash();
    // test_field();
}