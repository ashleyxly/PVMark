use std::fmt::format;
use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;
use rayon::vec;


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
// use std::io::Result;
// use halo2_merkle_tree::chips::poseidon::{PoseidonChip, PoseidonConfig};
use halo2_merkle_tree::chips::poseidon_3::{PoseidonChip, PoseidonConfig};
// use halo2_merkle_tree::chips::less_than_chip::{LessThanChip, LessThanConfig, self};
use halo2_merkle_tree::chips::add_chip::{AddChip, AddConfig, self};
use halo2_merkle_tree::poseidon::spec_width_3::{POSEIDON_WIDTH, POSEIDON_RATE};
use halo2_merkle_tree::poseidon::spec_width_3::PoseidonSpec;
use halo2_merkle_tree::chips::less_than_lookup_chip::{LTConfig, LTChip};
use halo2_merkle_tree::chips::summation_chip::{SummationChip, SummationConfig, SUMMATION_NUM, self};
use halo2_merkle_tree::utils::*;
use halo2_merkle_tree::chips::check_dec_chip::{CheckDecChip, CheckDecConfig, DEC_NUM, N_BYTES, self};


use halo2_gadgets::{*};
use halo2_gadgets::sha256::{Sha256, BLOCK_SIZE, BlockWord};
use halo2_gadgets::sha256::{Table16Chip, Table16Config};


const WIDTH: usize = POSEIDON_WIDTH;
const RATE: usize = POSEIDON_RATE;

const NUM_BITS_COMPARE: usize = 253;

const POW_253_FP: Fp = Fp::from_raw(
    [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x2000000000000000,
            ]
);

const POW_252_FP: Fp = Fp::from_raw(
    [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x1000000000000000,
            ]
);


pub fn msg_schedule_test_input() -> [BlockWord; BLOCK_SIZE] {
    [
        BlockWord(Value::known(0b01100001011000100110001110000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000000000)),
        BlockWord(Value::known(0b00000000000000000000000000011000)),
    ]
}


#[derive(Debug)]
struct MyCircuit {}

impl Circuit<Fp> for MyCircuit {
    type Config = Table16Config;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        MyCircuit {}
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        Table16Chip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let table16_chip = Table16Chip::construct(config.clone());
        Table16Chip::load(config, &mut layouter)?;

        // Test vector: "abc"
        let test_input = msg_schedule_test_input();

        // Create a message of length 31 blocks
        let mut input = Vec::with_capacity(31 * BLOCK_SIZE);
        for _ in 0..31 {
            input.extend_from_slice(&test_input);
        }

        Sha256::digest(table16_chip, layouter.namespace(|| "'abc' * 31"), &input)?;

        Ok(())
    }
}

fn main() {
    // let test = sha256::Sha256::<Fp, sha256::Table16Chip>::default();
    let circuit = MyCircuit {};
    println!("circuit: {:?}", circuit);

}