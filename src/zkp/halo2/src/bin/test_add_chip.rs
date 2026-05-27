use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;

// use ff::FieldElement;
// use halo2curves::ff::PrimeField;
// use halo2curves::pasta::Fp as F;

// use halo2_merkle_tree::chips::poseidon;

use halo2_merkle_tree::chips::merkle_width_9::MerkleTreeV3Circuit;
use halo2_merkle_tree::poseidon::spec_width_9::PoseidonSpec;
// use halo2_proofs::dev::metadata::Column;
// use syn::token::Colon;
// use halo2_gadgets::poseidon::{
//     primitives::{self as poseidon1, ConstantLength, P128Pow5T3 as OrchardNullifier, Spec},
//     Hash,
// };
// use halo2_proofs::{circuit::Value, dev::MockProver, pasta::Fp};
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
use halo2_merkle_tree::chips::add_chip::{AddChip, AddConfig};


#[derive(Debug, Default, Clone)]
pub struct AddTestCircuit {
    /*Simple add test */
    // pub add_input1: Vec<Fp>,
    // pub add_input2: Vec<Fp>,
    /*************** */

    /*Summation test */
    pub summation_inputs: Vec<Fp>,
    /*************** */

}

impl Circuit<Fp> for AddTestCircuit {
    type Config = AddConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let advice_1 = meta.advice_column();
        let advice_2 = meta.advice_column();
        let advice_3 = meta.advice_column();
        let instance = meta.instance_column();
        
        // AddChip::configure(meta)
        AddChip::configure(meta, advice_1, advice_2, advice_3, instance)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        let chip = AddChip::construct(config);

        /* test simple add */
        // for i in 0..self.add_input1.len() {

        //     let (add_input1_cell, add_input2_cell) = chip.load_private(layouter.namespace(|| format!("No. {} -- Load Add Input", i)), self.add_input1[i], self.add_input2[i])?;
        //     let (output_cell) = chip.simple_cell_add(layouter.namespace(|| format!("Add {}", i)), add_input1_cell, add_input2_cell)?;
        //     chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_cell, i)?;
        // }
        /*******************/

        /* test summation */
        let summation_inputs_number = self.summation_inputs.len();
        let summation_output_cell = chip.assign_multiple_value_and_summation(layouter.namespace(|| "summation"), &self.summation_inputs, summation_inputs_number)?;
        let _ = chip.expose_public(layouter.namespace(|| "summation result check"), &summation_output_cell, 0);
        /*******************/


        Ok(())
    }
}

fn generate_add_chip_test_case(len: usize) -> (Vec<Fp>, Vec<Fp>, Vec<Fp>) {
    let mut rng = rand::thread_rng();
    let mut input1 = Vec::new();
    let mut input2 = Vec::new();
    let mut output = Vec::new();
    for _ in 0..len {
        let a = Fp::from(rng.gen_range(0..30));
        let b = Fp::from(rng.gen_range(0..20));
        let c = a + b;
        input1.push(a);
        input2.push(b);
        output.push(c);
    }
    (input1, input2, output)
}

fn generate_summation_test_case(len: usize) -> (Vec<Fp>, Fp) {
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..len {
        let a = Fp::from(rng.gen_range(0..30));
        input.push(a);
    }
    let mut sum = Fp::zero();
    for i in 0..len {
        sum += input[i];
        // output.push(sum);
    }
    (input, sum)
}




fn main() {
    // let (input_add_1, input_add_2, output_add) = generate_add_chip_test_case(1);
    let (summation_inputs, result) = generate_summation_test_case(4);
    /*simple add test circuit */
    // let circuit = AddTestCircuit {
    //     add_input1: input_add_1.clone(),
    //     add_input2: input_add_2.clone(),
    // };
    /*Summation test circuit */
    let circuit = AddTestCircuit {
        summation_inputs: summation_inputs.clone(),
    };
    let mut public_inputs = vec![];
    public_inputs.push(result);
    // for i in 0..output_add.len() {
    //     public_inputs.push(output_add[i]);
    // }
    println!("summation_inputs: {:?}", summation_inputs);
    println!("result: {:?}", result);
    // println!("input_add_1: {:?}", input_add_1);
    // println!("input_add_2: {:?}", input_add_2);
    // println!("public_inputs: {:?}", public_inputs);
    // println!("circuit: {:?}", circuit);
    // let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(4, &circuit);
    // println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    
    let prover = MockProver::run(4, &circuit, vec![public_inputs.clone()]).unwrap();
    println!("{:?}", prover);
    assert_eq!(prover.verify(), Ok(()));
    println!("success");
    

    //test wrong public inputs
    public_inputs[0] = Fp::from(100);
    let prover = MockProver::run(4, &circuit, vec![public_inputs.clone()]).unwrap();
    assert_eq!(!prover.verify().is_ok(), true);
    // println!("{:?}", prover.verify());
    println!("Wrong public inputs test success");


}