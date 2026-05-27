extern crate rand;
extern crate indicatif;

use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::vec;
use std::collections::HashSet;

use rayon::prelude::*;  // Import rayon crate
use std::sync::Arc;
use std::sync::Mutex;

use num_bigint::{BigInt, ToBigInt};
use num_traits::cast::ToPrimitive;
use std::cmp::Ordering;

use hash_rustlib::two_inputs_hash_computation;
// use hash_rustlib::two_inputs_hash_computation_used_only_in_sac_test;
use hash_rustlib::HashType;


fn test(run_time: usize, hash_type: HashType) {
    let pb = ProgressBar::new(run_time as u64);
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

    let mut results = vec![];

    // for _ in 0..run_time {
    results = (0..run_time).into_par_iter().map(|_| {

        let hash_number = 50265;
        // select secret key randomly
        let secret_key: u64 = rand::thread_rng().gen();
        // let secret_key_string = format!("{:x}", secret_key);
        let secret_key_string = secret_key.to_string();

        let hash_values: Vec<String> = (0..hash_number)
            .into_par_iter()
            .map(|i| {
                let input2 = i.to_string();
                // println!("input2: {}", input2);
                // println!("secret_key_string: {}", secret_key_string.clone());
                two_inputs_hash_computation(secret_key_string.clone(), input2, hash_type)
            })
            .collect();

        let decimal_values: Vec<BigInt> = hash_values.iter().map(|hex_str| {
            BigInt::parse_bytes(hex_str.as_bytes(), 16).unwrap()
        }).collect();

        // Define min_value and max_value
        let min_value: BigInt = 0.to_bigint().unwrap();
        let max_value: BigInt = "21888242871839275222246405745257275088548364400416034343698204186575808495617"
            .parse()
            .unwrap();

        // println!("最小值: {}", min_value);
        // println!("最大值: {}", max_value);

        // 确定频率的区间数
        let num_bins = 17;
        let bin_size = (max_value.clone() - min_value.clone()) / num_bins.to_bigint().unwrap();

        // 初始化频率数组
        let mut frequency = vec![0; num_bins];

        // 计算每个区间内的哈希值出现次数
        for value in &decimal_values {
            let bin_index = ((value - &min_value) / &bin_size).to_usize().unwrap().min(num_bins - 1);
            frequency[bin_index] += 1;
        }

        // 计算期望值
        let expected = hash_values.len() / num_bins;

        // 计算卡方值
        let mut chi_squared = BigInt::from(0);
        for &count in &frequency {
            let count_bigint = count.to_bigint().unwrap();
            chi_squared += ((count_bigint - expected.to_bigint().unwrap()).pow(2)) / expected.to_bigint().unwrap();
        }

        // println!("频率: {:?}", frequency);
        // println!("卡方分数: {}", chi_squared);

        // 判断
        let threshold: f64 = 26.296;
        let chi_squared_f64 = chi_squared.to_f64().unwrap();
        // if chi_squared_f64 <= threshold {
        //     println!("哈希函数是均匀的");
        // } else {
        //     println!("哈希函数不是均匀的");
        // }
        // results.push(chi_squared_f64);
        pb.inc(1);
        chi_squared_f64

    }).collect();
    pb.finish();
    //print max and min in results
    let max_value = results.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let min_value = results.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("Maximum value: {}", max_value);
    println!("Minimum value: {}", min_value);
    //print how many value less than 26.296 
    let mut count = 0;
    for i in results.iter() {
        if i <= &26.296 {
            count += 1;
        }
    }
    println!("{:?} values are less than 26.296 for {:?}", count, hash_type);
    let sum: f64 = results.iter().sum();
    let mean = sum / run_time as f64;
    //compute standard deviation
    let mut sum_of_squared_deviations = 0.0;
    for i in results.iter() {
        sum_of_squared_deviations += (i - mean).powi(2);
    }
    let standard_deviation = (sum_of_squared_deviations / (run_time - 1) as f64).sqrt();

    //compute standard error
    let standard_error = standard_deviation / (run_time as f64).sqrt();
    println!("Standard deviation: {:?} for {:?}", standard_deviation, hash_type);
    println!("Standard error: {:?} for {:?}", standard_error, hash_type);

    if mean <= 26.296 {
        println!("哈希函数是均匀的");
    } else {
        println!("哈希函数不是均匀的");
    }
    println!("Mean chi_squared_f64 {:?} for {:?}", mean, hash_type);
    

}

fn main() {
    let run_time = 1000;
    test(run_time, HashType::SHA256);
    test(run_time, HashType::BLAKE2b);
    test(run_time, HashType::KECCAK256);
    test(run_time, HashType::POSEIDON);
    test(run_time, HashType::POSEIDON2);
    test(run_time, HashType::MIMC);

}