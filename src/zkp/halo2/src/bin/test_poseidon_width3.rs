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


fn main() {
    let args = Args::parse();
    let left_input = args.left_input;
    let right_input = args.right_input;        
    
    let pad_left_input = pad_string(&left_input, 64);
    let pad_right_input = pad_string(&right_input, 64);

    let raw_pad_left_input = process_hex_string(&pad_left_input);
    let raw_pad_right_input = process_hex_string(&pad_right_input);
    // let hash_inputs = [Fp::from(1), Fp::from(2)];
    let hash_inputs = [Fp::from_raw(raw_pad_left_input), Fp::from_raw(raw_pad_right_input)];
    // let hash_inputs = [Fp::from(left_input_u64), Fp::from(right_input_u64)];
    // println!("hash_inputs = {:?}", hash_inputs);

    let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    .hash(hash_inputs);
    // println!("result = {:?}", result);
    // println!("{:?}", result);
    print!("{:?}", result);




}