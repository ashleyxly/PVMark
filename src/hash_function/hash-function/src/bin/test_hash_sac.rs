extern crate rand;
extern crate indicatif;

use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;

use rayon::prelude::*;  // Import rayon crate
use std::sync::Arc;
use std::sync::Mutex;

use hash_rustlib::two_inputs_hash_computation;
use hash_rustlib::two_inputs_hash_computation_used_only_in_sac_test;
use hash_rustlib::HashType;

// Define your hash function F(x) here
// For demonstration, let's use a simple XOR-based hash
fn hash_function(input: &[u8]) -> u64 {
    let mut hash_value: u64 = 0;
    for &byte in input {
        hash_value ^= byte as u64;
    }
    hash_value
}

// 将十六进制字符串转换为二进制字符串
fn hex_to_bin(hex_string: &str) -> String {
    let mut bin_string = String::new();

    for ch in hex_string.chars() {
        let num = ch.to_digit(16).expect("Invalid hex character");
        bin_string.push_str(&format!("{:04b}", num));
    }

    bin_string
}

// 执行位级别的异或操作
fn xor_bin_strings(bin_string1: &str, bin_string2: &str) -> String {
    let mut xor_result = String::new();

    // Assuming bin_string1 and bin_string2 have the same length
    for (ch1, ch2) in bin_string1.chars().zip(bin_string2.chars()) {
        let bit1 = ch1.to_digit(2).expect("Invalid binary character");
        let bit2 = ch2.to_digit(2).expect("Invalid binary character");
        let xor_bit = bit1 ^ bit2;
        xor_result.push_str(&format!("{}", xor_bit));
    }

    xor_result
}

fn count_set_bits(bin_string: &str) -> u32 {
    bin_string.chars().filter(|&c| c == '1').count() as u32
}

fn test(run_time: usize, hash_type: HashType, input_bits: usize) {
    // const INPUT_BITS: usize = 64; // Number of input bits
    let INPUT_BITS = input_bits;
    // const M: usize = 250000; // Number of iterations
    let M = run_time;

    let mut rng = rand::thread_rng();
    // Create a thread-safe random number generator
    // let rng = Arc::new(Mutex::new(rand::thread_rng()));
    let mut avalanche_coefficients: Vec<f64> = Vec::new();

    // Create a progress bar
    let pb = ProgressBar::new(M as u64);
    let style = ProgressStyle::default_bar()
    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})");

    // Handle potential error from template method
    let style = match style {
        Ok(style) => style,
        Err(err) => {
            eprintln!("Failed to create progress bar style: {}", err);
            return;
        }
    };

    pb.set_style(style.progress_chars("#>-"));

    for _ in 0..M {
    // (0..M).into_par_iter().for_each(|_| {
        // Step 1: Generate random number A1
        // let mut rng = rng.lock().unwrap();
        let mut A1: u128 = rng.gen();
        let mut A1_2: u128 = rng.gen();

        // Step 2: Calculate hash value H1 = F(A1)
        // let H1 = hash_function(&A1.to_le_bytes());
        let H1 = two_inputs_hash_computation_used_only_in_sac_test(A1.to_string(), A1_2.to_string(), hash_type);

        // Step 3: Toggle any bit of A1 randomly to generate A2
        let bit_to_toggle = rng.gen_range(0..INPUT_BITS);
        let A2 = A1 ^ (1 << bit_to_toggle);

        // Step 4: Calculate hash value H2 = F(A2)
        // let H2 = hash_function(&A2.to_le_bytes());
        let H2 = two_inputs_hash_computation_used_only_in_sac_test(A2.to_string(), A1_2.to_string(), hash_type);

        // Step 5: Compute X = H1 xor H2
        // 将十六进制字符串转换为二进制字符串
        let bin_string1 = hex_to_bin(&H1);
        let bin_string2 = hex_to_bin(&H2);
        // let X = H1 ^ H2;
        let X = xor_bin_strings(&bin_string1, &bin_string2);

        // Step 6: Calculate the number of set bits N in X
        let mut num_set_bits = 0;
        let mut bits = X;
        // while bits > 0.to_string() {
        //     bits &= bits - 1;
        //     num_set_bits += 1;
        // }
        num_set_bits = count_set_bits(&bits);

        // Step 7: Calculate avalanche coefficient K = N / n
        let n = INPUT_BITS as f64; // Number of bits in the output
        let K = num_set_bits as f64 / n;

        // Store the avalanche coefficient for this iteration
        avalanche_coefficients.push(K);

        // Advance the progress bar
        pb.inc(1);
    }
    // });

    // Finish the progress bar
    pb.finish();

    // Step 8: Calculate the mean of all K
    let sum: f64 = avalanche_coefficients.iter().sum();
    let mean = sum / M as f64;

    // Find the maximum value
    let max_value = *avalanche_coefficients.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&std::f64::NAN);

    // Find the minimum value
    let min_value = *avalanche_coefficients.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&std::f64::NAN);

    println!("Maximum value: {}", max_value);
    println!("Minimum value: {}", min_value);

    println!("Mean Avalanche Coefficient (K) {:?} for {:?}", mean, hash_type);


}

use num_bigint::{BigUint, ToBigUint};
use num_traits::Num;
fn main() {
    // let run_time = 1000000;
    // test(run_time, HashType::SHA256, 256);
    // test(run_time, HashType::BLAKE2b, 256);
    // test(run_time, HashType::KECCAK256, 256);
    // test(run_time, HashType::POSEIDON, 254);
    // test(run_time, HashType::POSEIDON2, 254);
    // test(run_time, HashType::MIMC, 254);
    let field_prime = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
    let big_prime_int = match BigUint::from_str_radix(&field_prime, 16) {
        Ok(v) => v,
        Err(_e) => {
            println!("Failed to parse field_prime to BigUint");
            return;
        }
    };
    println!("big_prime_int: {:?}", big_prime_int);
    let threshold = big_prime_int / BigUint::from(2u64);
    println!("threshold: {:?}", threshold);
}
