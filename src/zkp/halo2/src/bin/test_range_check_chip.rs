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

use halo2_merkle_tree::chips::range_check_chip::{RangeCheckChip, RangeCheckConfig};


#[derive(Debug, Clone, Default)]
pub struct RangeCheckTestCircuit {
    pub target_value: Fp,
    pub range_length: usize,
    pub success_flag: usize,
}

impl Circuit<Fp> for RangeCheckTestCircuit {
    type Config = RangeCheckConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // let advice = [meta.advice_column(); 3];
        let advice_1 = meta.advice_column();
        let advice_2 = meta.advice_column();
        let advice_3 = meta.advice_column();
        let instance = meta.instance_column();
        RangeCheckChip::configure(meta, advice_1, advice_2, advice_3, instance)
        // RangeCheckChip::configure(meta, advice, instance)
        // RangeCheckChip::configure(meta)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), Error> {
        let chip = RangeCheckChip::construct(config);

        // let bit_inputs_cell = chip.assign_and_bit_check(layouter.namespace(|| "Bit Check Test in RangeCheck Chip"), self.target_value, self.range_length, self.success_flag)?;
        // let bit_inputs_cell = chip.assign_and_bit_check_2(layouter.namespace(|| "Bit Check Test in RangeCheck Chip"), self.target_value, self.range_length, self.success_flag)?;
        let bit_inputs_cell = chip.assign_and_bit_check_2(layouter.namespace(|| "Bit Check Test in RangeCheck Chip"), self.target_value, self.range_length, self.success_flag)?;
        
        // println!("-----------------------");
        // chip.assign_test(layouter.namespace(|| "AssignTestCircuit"), Fp::from(1), Fp::from(2), Fp::from(3));
        //temp = half_bit_inputs_cell
        // println!("bit_inputs_cell = {:?}", bit_inputs_cell);
        // let temp = bit_inputs_cell[bit_inputs_cell.len()/2..bit_inputs_cell.len()].to_vec();
        // println!("temp = {:?}", temp);
        // println!("temp.len() = {:?}", temp.len());
        let output_cell = chip.assign_and_summation(layouter.namespace(|| "Summation Test in RangeCheck Chip"), &bit_inputs_cell)?;
        chip.expose_public(layouter.namespace(|| "instance check"), &output_cell, 0)?;
        Ok(())
    }
}

// fn generate_range_check_test_case() {

// }




fn main() {

    let target_value = Fp::from(13);
    let range_length = 4;
    let success_flag = 1;

    let circuit = RangeCheckTestCircuit {
        target_value,
        range_length,
        success_flag,
    };

    let mut public_inputs = vec![target_value];

    // let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(4, &circuit);
    // println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    

    let prover = MockProver::run(6, &circuit, vec![public_inputs]).unwrap();
    // println!("{:?}", prover);
    println!("{:?}", prover.verify());
    assert_eq!(prover.verify(), Ok(()));
    println!("success");




}