use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead, Write};
// use std::simd::StdFloat;

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;
use ff::PrimeField;
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

    /// The value of secret key
   #[arg(short, long, default_value = "2023")]
   secret_key: String,

   /// The path of output file including counting results
   #[arg(short, long, default_value = "1000")]
   vocab_size: String,

   /// The path of output file
   #[arg(short, long, default_value = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_poseidon_width3/hash_file.txt")]
   output_file_path: String,




}


// Compute the hash of all the vocabulary words

fn main() {
    let args = Args::parse();
    let secret_key = args.secret_key;
    let vocab_size = args.vocab_size;
    let hash_file_path = args.output_file_path;       


    let mut output_file = File::create(hash_file_path).unwrap();

    // let secret_key_int = secret_key.parse::<i128>().unwrap();
    let pad_secret_key = pad_string(&secret_key, 64);
    let raw_pad_secret_key = process_hex_string(&pad_secret_key);
    let vocab_size_int = vocab_size.parse::<i128>().unwrap();

    // Compute Hash Number
    let quotient = vocab_size_int / 4;
    let hash_number = if vocab_size_int % 4 == 0 {
        quotient
    } else {
        quotient + 1
    };

    let mut result_number = 0;

    for i in 0..hash_number {
        let mut hash_inputs = [Fp::from_raw(raw_pad_secret_key), Fp::from(i as u64)];
        let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(hash_inputs);
        let mut result_bytes = result.to_bytes();
        let result_1 = u64::from_le_bytes(result_bytes[0..8].try_into().unwrap());
        result_number += 1;
        writeln!(output_file, "0x{:X}", result_1).unwrap();
        if result_number == vocab_size_int {
            break;
        }
        let result_2 = u64::from_le_bytes(result_bytes[8..16].try_into().unwrap());
        result_number += 1;
        writeln!(output_file, "0x{:X}", result_2).unwrap();
        if result_number == vocab_size_int {
            break;
        }
        let result_3 = u64::from_le_bytes(result_bytes[16..24].try_into().unwrap());
        result_number += 1;
        writeln!(output_file, "0x{:X}", result_3).unwrap();
        if result_number == vocab_size_int {
            break;
        }
        let result_4 = u64::from_le_bytes(result_bytes[24..32].try_into().unwrap());
        result_number += 1;
        writeln!(output_file, "0x{:X}", result_4).unwrap();
        if result_number == vocab_size_int {
            break;
        }

        // writeln!(output_file, "{:?}", result).unwrap();
        // result_number += 1;
    }


    // for i in 0..vocab_size_int {
    //     let mut hash_inputs = [Fp::from_raw(raw_pad_secret_key), Fp::from(i as u64)];
    //     let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    //         .hash(hash_inputs);
    //     writeln!(output_file, "{:?}", result).unwrap();
    // }

    // println!("Finish computing the hash of all the vocabulary words!")

}