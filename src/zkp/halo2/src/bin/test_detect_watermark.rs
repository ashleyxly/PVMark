use ff::PrimeField;
use halo2_proofs::circuit::Value;
// use halo2curves::pasta::Fp;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::halo2curves;

fn u64_array_to_binary(arr: [u64; 4]) -> Vec<u8> {
    let mut result = Vec::with_capacity(256);

    for &word in arr.iter() {
        for i in (0..64).rev() {
            let bit = (word >> i) & 1;
            result.push(bit as u8);
        }
    }

    result
}

fn u64_array_to_binary_2(arr: [u8; 32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(256);

    for &word in arr.iter() {
        for i in (0..8).rev() {
            let bit = (word >> i) & 1;
            result.push(bit as u8);
        }
    }

    result
}

fn bytes_to_binary(bytes: &[u8]) -> String {
    let mut result = String::new();

    for &byte in bytes.iter() {
        for i in (0..8).rev() {
            // 获取每个比特位的值
            let bit = (byte >> i) & 1;
            // 将比特位的值追加到结果字符串中
            result.push_str(&format!("{}", bit));
        }
    }

    result
}

fn bytes_to_binary_2(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    for &byte in bytes.iter() {
        let mut temp = Vec::new();
        for i in (0..8).rev() {
            // 获取每个比特位的值
            let bit = (byte >> i) & 1;
            temp.push(bit);
            // temp.reverse();
            // 将比特位的值追加到结果字符串中
            // result.push(bit);
        }
        temp.reverse();
        result.extend(temp);
    }

    result
}
use num_bigint::BigUint;
use halo2_merkle_tree::utils::*;

fn main() {
    // let mut large_number: [u64; 4] = [0x1122334455667788, 0x99aabbccddeeff00, 0x0011223344556677, 0x8899aabbccddeeff];
    let mut large_number: [u64; 4] = [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x000000000000000a];
    let mut large_number2: [u8; 32] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02];
    // Convert to little-endian if necessary
    // if cfg!(target_endian = "big") {
    //     for word in &mut large_number {
    //         *word = word.to_le();
    //     }
    // }

    // let binary_representation = u64_array_to_binary(large_number);
    
    let binary_representation = u64_array_to_binary_2(large_number2);

    println!("{:?}", binary_representation);

    // let v = Fp::from(0);
    // println!("{:?}", v);
    let f = Fp::from(550);
    println!("{:?}", f);
    let binary_f = f.to_bytes();
    println!("{:?}", binary_f);
    let binary2_f = f.to_repr();
    println!("{:?}", binary2_f);
    let binary3_f = bytes_to_binary_2(&binary2_f);
    
    // binary3_f
    println!("{:?}", binary3_f);
    // println!("{:?}", binary3_f[0]);
    let ref_binary = bytes_to_binary(&binary2_f);
    println!("{:?}", ref_binary);
    let bit_length = 254;
    let mut value: Fp = Fp::zero();
    for i in 0..bit_length {
        value += Fp::from(2u64.pow(i as u32) * binary3_f[i] as u64);
        // println!("{:?}", binary3_f[i]);
    }
    println!("{:?}", value);

    let mut test_fp = Fp::from(0);
    println!("{:?}", test_fp);
    let mut test_fp2 = Fp::from(1);
    println!("{:?}", test_fp2);
    let mut test_fp3 = test_fp - test_fp2;
    println!("{:?}", test_fp3);


    let mut test_fp4 = pow_of_two(253);
    println!("{:?}", test_fp4);

}
