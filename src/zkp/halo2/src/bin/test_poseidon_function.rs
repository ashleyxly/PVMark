use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;
use rayon::range;


use std::time::{Duration, Instant};

// use halo2_proofs::pasta::{Eq, EqAffine};
// use halo2_proofs::plonk::{
//     create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, BatchVerifier, Circuit,
//     Column, ConstraintSystem, Error, Fixed, TableColumn,
// };

use halo2_proofs::dev::VerifyFailure;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
use halo2_gadgets::poseidon::{primitives::*, Hash, Pow5Chip, Pow5Config};
use halo2_proofs::arithmetic::Field;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::halo2curves;
use halo2_proofs::{circuit::*, plonk::*};
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSlice;

use std::marker::PhantomData;

use halo2_proofs::poly::Rotation;
use halo2_proofs::dev::MockProver;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, TranscriptReadBuffer,
    TranscriptWriterBuffer,
};
use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};

use halo2_proofs::poly::commitment::CommitmentScheme;
// use halo2_proofs::poly::commitment::Params;
// use halo2_proofs::poly::commitment::ParamsProver;
// use log::info;
// use std::error::Error;
// use std::fs::File;
// use std::io::BufReader;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;
// use halo2_merkle_tree::chips::poseidon::{PoseidonChip, PoseidonConfig};
use halo2_merkle_tree::chips::poseidon_3::{PoseidonChip, PoseidonConfig};
// use halo2_merkle_tree::chips::less_than_chip::{LessThanChip, LessThanConfig, self};
use halo2_merkle_tree::chips::add_chip::{AddChip, AddConfig, self};
use halo2_merkle_tree::poseidon::spec_width_3::{POSEIDON_WIDTH, POSEIDON_RATE};
use halo2_merkle_tree::poseidon::spec_width_3::PoseidonSpec;
use halo2_merkle_tree::chips::less_than_lookup_chip::{LTConfig, LTChip};
use halo2_merkle_tree::chips::summation_chip::{SummationChip, SummationConfig, SUMMATION_NUM, self};
use halo2_merkle_tree::utils::*;

const WIDTH: usize = POSEIDON_WIDTH;
const RATE: usize = POSEIDON_RATE;

const NUM_BITS_COMPARE: usize = 253;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
//    /// File to read
//    filename: String,

    /// The value of left input
   #[arg(short, long, default_value = "1")]
   left_input: String,

   /// The value of right input
   #[arg(short, long, default_value = "2")]
   right_input: String,

//    /// Scale factor
//    #[arg(short, long, default_value_t = 4)]
//    scale: i32,



}

fn poseidon_two_inputs_hash(input1: String, input2: String) -> String
{
    let pad_left_hash = pad_string(&input1, 64);
    let pad_right_hash = pad_string(&input2, 64);

    let raw_pad_left_hash = process_hex_string(&pad_left_hash);
    let raw_pad_right_hash = process_hex_string(&pad_right_hash);

    let hash_inputs = [Fp::from_raw(raw_pad_left_hash), Fp::from_raw(raw_pad_right_hash)];
    let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    .hash(hash_inputs);


    let mut result_value = result.to_bytes();
    let hex_string: String = result_value.iter().rev().map(|&byte| format!("{:02x}", byte)).collect();

    // Ok(hex_string)
    hex_string
}

fn get_greenlist_id_and_threshold(seed: String, vocab_size: i32, green_list_size: i32) -> (Vec<usize>, String)
{
    let pad_seed = pad_string(&seed, 64);
    let raw_pad_seed = process_hex_string(&pad_seed);

    let dec_number = 4;

    let quotient = vocab_size / dec_number;
    let hash_number = if vocab_size % dec_number == 0 {
        quotient
    } else {
        quotient + 1
    };

    let mut result_number = 0;
    let mut flag = 0;

    // let each_number_bit_length = 64 / dec_number;
    let each_number_bit_length = 8;

    let mut this_round_token_hash = vec![];

    for i in 0..hash_number {
        let mut hash_inputs = [Fp::from_raw(raw_pad_seed), Fp::from(i as u64)];
        let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(hash_inputs);
        let mut result_bytes = result.to_bytes();

        
        for j in 0..dec_number {
            let begin = j * each_number_bit_length;
            let end = (j + 1) * each_number_bit_length;
            // println!("begin = {:?}", begin);
            // println!("end = {:?}", end);
            let mut res_temp = u64::from_le_bytes(result_bytes[begin as usize..end as usize].try_into().unwrap());
            // let mut res_temp = match result_bytes[begin as usize..end as usize].try_into() {
            //     Ok(bytes) => u64::from_le_bytes(bytes),
            //     Err(_) => {
            //         // 处理转换失败的情况，这里可以根据实际情况进行适当的处理
            //         // 比如打印错误信息或者返回一个默认值
            //         println!("Failed to convert slice to u64");
            //         // 这里返回一个默认值，你可以根据需求进行调整
            //         0u64
            //     }
            // };
            let mut res_temp_string = format!("{:x}", res_temp);
            this_round_token_hash.push(res_temp_string);
            result_number += 1;
            if result_number == vocab_size {
                flag = 1;
                break;
            }
        }
        if flag == 1 {
            break;
        }
    }


    let mut this_round_token_index: Vec<usize> = (0..vocab_size as usize).collect();
    this_round_token_index.sort_by(|&a, &b| this_round_token_hash[a].cmp(&this_round_token_hash[b]));

    let green_list_id = this_round_token_index[0..green_list_size as usize].to_vec();
    let this_round_threshold = poseidon_two_inputs_hash(seed, this_round_token_index[green_list_size as usize - 1].to_string());

    // Ok((green_list_id.iter().map(|&x| x as i64).collect(), this_round_threshold))
    // Ok((green_list_id, this_round_threshold))
    (green_list_id, this_round_threshold)

}

pub fn little_endian_u8_array_to_string(bytes: &[u8; 32]) -> String {
    let mut hex_string = String::new();

    for &byte in bytes.iter().rev() {
        hex_string.push_str(&format!("{:02x}", byte));
    }

    hex_string
}


fn main() {

    let args = Args::parse();
    let left_input = args.left_input;
    let right_input = args.right_input;        
    
    let pad_left_input = pad_string(&left_input, 64);
    let pad_right_input = pad_string(&right_input, 64);

    let seed = poseidon_two_inputs_hash(pad_left_input, pad_right_input);
    println!("seed = {:?}", seed);

    let (green_list_id, this_round_threshold) = get_greenlist_id_and_threshold(seed, 10, 5);

    println!("green_list_id = {:?}", green_list_id);
    println!("this_round_threshold = {:?}", this_round_threshold);

    // let test_fp = Fp::from(123561341613434643);
    let test_fp = Fp::from_raw(
        [
            0x0000000000000000,
            0x0000186530000000,
            0x0000000000000000,
            0x2000000000000000,
        ]
    );
    println!("test_fp = {:?}", test_fp);
    let test_fp_string = little_endian_u8_array_to_string(&test_fp.to_bytes());
    println!("test_fp = {:?}", test_fp_string);
    // let raw_pad_left_input = process_hex_string(&pad_left_input);
    // let raw_pad_right_input = process_hex_string(&pad_right_input);
    // // let hash_inputs = [Fp::from(1), Fp::from(2)];
    // let hash_inputs = [Fp::from_raw(raw_pad_left_input), Fp::from_raw(raw_pad_right_input)];
    // // let hash_inputs = [Fp::from(left_input_u64), Fp::from(right_input_u64)];
    // // println!("hash_inputs = {:?}", hash_inputs);

    // let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    // .hash(hash_inputs);
    // // println!("result = {:?}", result);
    // // println!("{:?}", result);
    // println!("{:?}", result);

    // let mut result_value = result.to_bytes();
    // let hex_string: String = result_value.iter().rev().map(|&byte| format!("{:02x}", byte)).collect();
    // println!("hex_string = {:?}", hex_string);




}