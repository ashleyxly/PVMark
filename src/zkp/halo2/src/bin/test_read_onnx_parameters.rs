use core::num;
use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};


use clap::Parser;
use clap::builder::Str;

// use ff::FieldElement;
// use halo2curves::ff::PrimeField;
// use halo2curves::pasta::Fp as F;

// use halo2_merkle_tree::chips::poseidon;

use halo2_merkle_tree::chips::merkle_v3::MerkleTreeV3Circuit;
use halo2_merkle_tree::poseidon::spec::PoseidonSpec;
// use halo2_gadgets::poseidon::{
//     primitives::{self as poseidon1, ConstantLength, P128Pow5T3 as OrchardNullifier, Spec},
//     Hash,
// };
// use halo2_proofs::{circuit::Value, dev::MockProver, pasta::Fp};
use std::time::{Duration, Instant};

// use halo2_proofs::pasta::{Eq, EqAffine};
// use halo2_proofs::plonk::{
//     create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, BatchVerifier, Circuit,
//     Column, ConstraintSystem, Error, Fixed, SingleVerifier, TableColumn, VerificationStrategy,
// };
// use halo2_proofs::poly::commitment::{Guard, MSM};
// use halo2_proofs::poly::{commitment::Params, Rotation};
// use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge};
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
use std::error::Error;
// use std::fs::File;
// use std::io::BufReader;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;
use std::io::{self, Write};


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
//    /// File to read
//    filename: String,

    /// The path of onnx file
   #[arg(short, long, default_value = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_onnx_parameters/network.onnx")]
   network_onnx_file_path: String,

   /// Length of characters to seek
   #[arg(short, long, default_value = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_onnx_parameters/params.txt")]
   output_file_path: String,

   /// Scale factor
   #[arg(short, long, default_value_t = 4)]
   scale: i32,

   /// The path of proof
   #[arg(short, long, default_value = "/mnt/disk2/username/kzg-halo2-merkle-tree/onnx_test/proof.bin")]
   proof_path: String,

}


fn call_python_function(onnx_file_path: String, output_file_path: String) {
    let output = Command::new("python")
        // .arg("-c")
        // .arg("from read_onnx_file import test_hello_world; print(test_hello_world())")
        .arg("/mnt/disk2/username/kzg-halo2-merkle-tree/src/read_onnx_file.py")
        .arg("-Net")
        .arg(onnx_file_path)
        .arg("-P")
        .arg(output_file_path)
        .output()
        .expect("failed to execute process");

    println!("finished running python script");
    io::stdout().flush().unwrap();
    let result = String::from_utf8_lossy(&output.stdout);
    // println!("{}", result);

    let mut file = File::create("/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_onnx_parameters/result.txt").expect("failed to create file");
    file.write_all(result.as_bytes()).expect("failed to write to file");
}



fn read_file_numbers(path: &str) -> Vec<f64> {
    println!("path: {}", path);
    let file = File::open(path).expect("failed to open file");
    let reader = BufReader::new(file);
    let mut numbers = Vec::new();

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let mut words = line.split_whitespace();

        while let Some(word) = words.next() {
            if let Ok(number) = word.parse::<f64>() {
                numbers.push(number);
            }
        }
    }

    numbers
}

fn read_file_numbers_2(path: &str) -> usize {
    println!("path: {}", path);
    let file = File::open(path).expect("failed to open file");
    let reader = BufReader::new(file);
    let mut numbers = Vec::new();

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let mut words = line.split_whitespace();

        while let Some(word) = words.next() {
            if let Ok(number) = word.parse::<f64>() {
                numbers.push(number);
            }
        }
    }

    numbers.len()
}


fn main() {
    let args = Args::parse();
    // println!("{:?}", args);
    let scale_factor = args.scale;
    let output_file_path_str = args.output_file_path.clone();
    // let proof_path_str = args.proof_path.clone();
    // call_python_function(args.network_onnx_file_path, args.output_file_path);
    let numbers = read_file_numbers_2(output_file_path_str.as_str());
    println!("numbers: {:?}", numbers);
}