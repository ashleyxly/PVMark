use std::ffi::NulError;
use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;

use std::time::{Duration, Instant};

// use halo2_proofs::pasta::{Eq, EqAffine};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, BatchVerifier, Circuit,
    Column, ConstraintSystem, Error, Fixed, TableColumn,
};
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
use halo2_proofs::halo2curves;
use halo2_proofs::{circuit::{*, self}, plonk::*};
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
// use halo2_proofs::poly::Rotation;
// use halo2_proofs::poly::commitment::Params;
// use halo2_proofs::poly::commitment::ParamsProver;
// use log::info;
// use std::error::Error;
// use std::fs::File;
// use std::io::BufReader;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;

// use halo2_merkle_tree::chips::range_check_chip::{RangeCheckChip, RangeCheckConfig};
// use halo2_merkle_tree::chips::less_than_chip::{LessThanChip, LessThanConfig, self};
use halo2_merkle_tree::chips::less_than_lookup_chip::*;

#[derive(Debug, Clone, Default)]
pub struct LessThanLookupTestCircuit {
    pub x1: Fp,
    pub x2: Fp,
    // pub range_length: usize,
}

impl Circuit<Fp> for LessThanLookupTestCircuit {
    type Config = LTConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let advice_1 = meta.advice_column();
        let advice_2 = meta.advice_column();
        let advice_3 = meta.advice_column();
        let instance = meta.instance_column();
        LTChip::configure(meta, advice_1, advice_2, advice_3, instance)
        // config
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), Error> {
        let less_than_chip = LTChip::construct(config.clone());
        // less_than_chip.is_less_than(layouter.namespace(|| "Judge X1 Less Than X2"), self.x1, self.x2, self.range_length)?;
        less_than_chip.load_lookup_table(layouter.namespace(|| "Load Lookup Table"))?;
        let lt_flag_cell = less_than_chip.is_less_than(layouter.namespace(|| "less than"), self.x1, self.x2)?;
        less_than_chip.expose_public(layouter.namespace(|| "check instance"), &lt_flag_cell, 0)?;

        Ok(())
    }
}


const NUM_LENGTH: usize = 254;
use num_bigint::BigUint;



fn process_hex_string(hex_string: &str) -> [u64; 4] {
    // 步骤 1: 将hex_string分割为4个长度为16的string
    let chunks: Vec<&str> = hex_string.as_bytes().chunks(16).map(|chunk| std::str::from_utf8(chunk).unwrap()).collect();

    // 步骤 2: 将每个长度为16的string转化为u64
    let u64_values: Vec<u64> = chunks.into_iter().map(|s| u64::from_str_radix(s, 16).unwrap()).collect();

    // 步骤 3: 将所有的string按照倒序存到[u64;4]
    let mut result_array: [u64; 4] = Default::default();
    result_array.copy_from_slice(&u64_values[..4]);
    result_array.reverse();
    
    result_array
    
}

fn pad_string(input: &str, target_length: usize) -> String {
    let current_length = input.len();
    
    if current_length >= target_length {
        return input.to_string();
    }

    let padding_length = target_length - current_length;
    let padding = "0".repeat(padding_length);

    let padded_string = format!("{}{}", padding, input);

    padded_string
}

// use halo2_merkle_tree::utils::*;

fn main() {
    // let mut rng = rand::thread_rng();
    let x1: Fp = Fp::from(10);
    let x2: Fp = Fp::from(20);
    let range_length: usize = NUM_LENGTH;

    let circuit = LessThanLookupTestCircuit {
        x1,
        x2,
        // range_length,
    };
    let mut public_inputs = vec![Fp::ONE];
    // let mut public_inputs = vec![Fp::ZERO];
    let prover = MockProver::run(10, &circuit, vec![public_inputs.clone()]).unwrap();
    // println!("{:?}", prover);
    println!("circuit is satisfied: {:?}", prover.verify());
    assert_eq!(prover.verify(), Ok(()));
    println!("success");



}