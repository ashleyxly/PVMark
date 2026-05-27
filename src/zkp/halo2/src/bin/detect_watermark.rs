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


// use halo2_proofs::{circuit::Value, dev::MockProver};
const WIDTH: usize = 9;
const RATE: usize = 8;

const NUM_LENGTH: usize = 4;


fn u64_array_to_binary(arr: [u64; 4]) -> Vec<u8> {
    let mut result = Vec::with_capacity(256);

    for &word in arr.iter() {
        for i in (0..64).rev() {
            let bit = (word >> i) & 1;
            result.push(bit as u8);
        }
    }

    result
}

fn u64_array_to_binary_2(arr: [u8; 32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(256);

    for &word in arr.iter() {
        for i in (0..8).rev() {
            let bit = (word >> i) & 1;
            result.push(bit as u8);
        }
    }

    result
}


#[derive(Debug, Clone)]
pub struct OneOrZeroTestConfig {
    pub bit_input: Column<Advice>,
    pub green_list_flag: Column<Advice>,
    pub pow_two: Column<Advice>,
    pub instance: Column<Instance>,
    pub selector: Selector,
}

#[derive(Debug, Clone)]
pub struct OneOrZeroTestChip {
    config: OneOrZeroTestConfig,
    _marker: PhantomData<Fp>,
}

impl OneOrZeroTestChip {
    pub fn construct(config: OneOrZeroTestConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        // bit_input: Column<Advice>,
        // green_list_flag: Column<Advice>,
        // pow_two: Column<Advice>,
        // instance: Column<Instance>,
        // selector: Selector,

    ) -> OneOrZeroTestConfig {
        // meta.enable_equality(advice);
        // for column in &advice {
        //     meta.enable_equality(*column);
        // }
        let bit_input = meta.advice_column();
        let green_list_flag = meta.advice_column();
        let pow_two = meta.advice_column();
        let instance = meta.instance_column();
        let selector = meta.selector();

        meta.enable_equality(bit_input);
        meta.enable_equality(green_list_flag);
        meta.enable_equality(pow_two);
        meta.enable_equality(instance);

        
        meta.create_gate("OneOrZero_Test", |meta| {
            let s = meta.query_selector(selector);
            let input = meta.query_advice(bit_input, Rotation::cur());
            // let input2 = meta.query_advice(, Rotation::cur());
            // let input2 = meta.query_instance(instance, Rotation::cur());
            let input2 = meta.query_advice(pow_two, Rotation::cur());
            let flag = meta.query_advice(green_list_flag, Rotation::cur());
            // let instance = meta.query_instance(instance, Rotation::cur());
            // let temp = instance - input.clone();
            let temp = input2 - input.clone();
            // let s_lessthan = meta.query_selector(selector);
            vec![s * flag * input * temp]
            // vec![input * temp]
        });
        
        OneOrZeroTestConfig {
            // inputs: advice,
            bit_input,
            green_list_flag,
            pow_two,
            instance,
            selector,
        }
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fp>,
        bit_input_value: Value<Fp>,
        green_list_flag_value: Value<Fp>,
        instance_row: usize,
        // instance_value: Value<Fp>,
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        // let config = self.config();
        layouter.assign_region(
            || "OneOrZero_Test",
            |mut region| {
                self.config.selector.enable(&mut region, 0)?;
                let bit_input_cell = region.assign_advice(
                    || "bit_input",
                    self.config.bit_input,
                    0,
                    || bit_input_value
                )?;
                let green_list_flag_cell = region.assign_advice(
                    || "green_list_flag",
                    self.config.green_list_flag,
                    0,
                    || green_list_flag_value
                )?;
                let pow_two_cell = region.assign_advice_from_instance(
                    || "pow_two", 
                    self.config.instance, 
                    // 0, 
                    instance_row,
                    self.config.pow_two,
                    0)?;
                
                Ok((bit_input_cell, green_list_flag_cell, pow_two_cell))
            },
        )
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }

    pub fn assign_multi(
        &self,
        mut layouter: impl Layouter<Fp>,
        bit_input_value: [Value<Fp>; NUM_LENGTH],
        green_list_flag_value: Value<Fp>,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        for i in 0..NUM_LENGTH {
            let (bit_input_cell, green_list_flag_cell, pow_two_cell) = self.assign(layouter.namespace(|| format!("assign {}", i)), bit_input_value[i], green_list_flag_value, i)?;
            self.expose_public(layouter.namespace(|| format!("pow_two instance check {}", i)), &pow_two_cell, 0)?;
        }
        Ok(())
    }



    
}




#[derive(Debug, Clone)]
pub struct BinaryTestConfig {
    // pub bit_inputs: [Column<Advice>; NUM_LENGTH],
    // pub green_list_flag: Column<Advice>,
    // pub pow_two_inputs: [Column<Advice>; NUM_LENGTH],
    // pub instance: [Column<Instance>; NUM_LENGTH],
    // pub selector: Selector,
    pub one_or_zero_test_config: OneOrZeroTestConfig,
    pub add_config: AddConfig,
}

#[derive(Debug, Clone)]
pub struct BinaryTestChip {
    config: BinaryTestConfig,
    _marker: PhantomData<Fp>,
}


impl BinaryTestChip {
    pub fn construct(config: BinaryTestConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        // advice: [Column<Advice>; NUM_LENGTH],
        // instance: [Column<Instance>; NUM_LENGTH],
        // selector: Selector,
    
    ) -> BinaryTestConfig {
        let one_or_zero_test_config = OneOrZeroTestChip::configure(meta);
        let add_config = AddChip::configure(meta);

        BinaryTestConfig {
            // bit_inputs,
            // green_list_flag,
            // pow_two_inputs,
            // instance,
            // selector,
            one_or_zero_test_config,
            add_config,
        }
    }

    pub fn binary_test(
        &self,
        mut layouter: impl Layouter<Fp>,
        bit_inputs_value: [Fp; NUM_LENGTH],
        green_list_flag_value: Fp,
        // instance_row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        let one_or_zero_chip = OneOrZeroTestChip::construct(self.config.one_or_zero_test_config.clone());
        let add_chip = AddChip::construct(self.config.add_config.clone());
        for i in 0..NUM_LENGTH {
            let (bit_input_cell, green_list_flag_cell, pow_two_cell) = one_or_zero_chip.assign(layouter.namespace(|| format!("assign {}", i)), Value::known(bit_inputs_value[i]), Value::known(green_list_flag_value), i)?;

            one_or_zero_chip.expose_public(layouter.namespace(|| format!("pow_two instance check {}", i)), &pow_two_cell, i)?;
        }
        // let mut output = add_chip.simple_add(layouter.namespace(|| "add"), bit_inputs_value[0], bit_inputs_value[1])?;
        // for i in 0..NUM_LENGTH-3 {
        //     let output_temp = add_chip.simple_add(layouter.namespace(|| format!("add {}", i)), output, bit_inputs_value[i+2])?;
        //     output = output_temp;
        // }


        


        Ok(())
    }


}


#[derive(Debug, Default, Clone)]
pub struct DetectWatermarkCircuit {
    // pub bit_inputs: [Value<Fp>; NUM_LENGTH],
    // pub green_list_flag: Value<Fp>,
    // pub pow_two_inputs: [Value<Fp>; NUM_LENGTH],

    pub bit_inputs: [Fp; NUM_LENGTH],
    pub green_list_flag: Fp,
    pub pow_two_inputs: [Fp; NUM_LENGTH],

}

impl Circuit<Fp> for DetectWatermarkCircuit {
    type Config = BinaryTestConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // let inputs = meta.advice_column();
        // let inputs2 = meta.advice_column();
        // let advice = [meta.advice_column(), meta.advice_column(), meta.advice_column()];

        // let advice = [meta.advice_column(); 3];
        // let instance = meta.instance_column();
        // let selector = meta.selector();

        // OneOrZeroTestChip::configure(meta, advice[0], advice[1], advice[2], instance, selector)
        // let one_or_zero_chip = OneOrZeroTestChip::configure(meta);
        // let add_chip = AddChip::configure(meta);


        // OneOrZeroTestChip::configure(meta)
        BinaryTestChip::configure(meta)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        // let one_or_zero_chip = OneOrZeroTestChip::construct(config);
        // one_or_zero_chip.assign_multi(layouter.namespace(|| "check"), self.bit_inputs, self.green_list_flag)?;
        // let (bit_input_cell, green_list_flag_cell, pow_two_cell) = one_or_zero_chip.assign(layouter.namespace(|| "assign value"), self.bit_inputs[0], self.green_list_flag)?;
        // one_or_zero_chip.expose_public(layouter.namespace(|| "pow_two instance check"), &pow_two_cell, 0)?;
        // for i in 0..NUM_LENGTH {
        //     let (bit_input_cell, green_list_flag_cell, pow_two_cell) = one_or_zero_chip.assign(layouter.namespace(|| format!("assign {}", i)), self.bit_inputs[i], self.green_list_flag, i)?;
        //     one_or_zero_chip.expose_public(layouter.namespace(|| format!("pow_two instance check {}", i)), &pow_two_cell, i)?;
        // }

        let binary_test_chip = BinaryTestChip::construct(config);
        let output = binary_test_chip.binary_test(layouter.namespace(|| "BinaryTest"), self.bit_inputs, self.green_list_flag);
        println!("output = {:?}", output);
        Ok(())
    }
}


fn generate_muti_oneorzero_test_case() -> ([Value<Fp>; NUM_LENGTH], Value<Fp>, [Value<Fp>; NUM_LENGTH], [Fp; NUM_LENGTH]) {
    let mut rng = rand::thread_rng();
    let mut bit_input_temp = [Value::known(Fp::from(0)); NUM_LENGTH];
    let mut green_list_flag_temp = Value::known(Fp::from(1));
    let mut pow_two_value: [Value<Fp>; NUM_LENGTH] = [Value::known(Fp::from(0)); NUM_LENGTH];
    let mut instance_temp: [Fp; NUM_LENGTH] = [Fp::from(0); NUM_LENGTH];
    // let mut selector_temp = Value::known(Fp::from(0));

    

    for i in 0..NUM_LENGTH {
        let mul = rng.gen_range(0..10);
        let temp = rng.gen_range(0..=1);
        let temp2 = 2u64.pow(mul as u32);
        bit_input_temp[i] = Value::known(Fp::from(temp * temp2 as u64));

        
        pow_two_value[i] = Value::known(Fp::from(temp2));
        // instance_temp[i] = pow_two_value[i].clone();
        instance_temp[i] = Fp::from(temp2);
    }



    (bit_input_temp, green_list_flag_temp, pow_two_value, instance_temp)
}

fn generate_muti_oneorzero_test_fp_case() -> ([Fp; NUM_LENGTH], Fp, [Fp; NUM_LENGTH], [Fp; NUM_LENGTH]) {
    let mut rng = rand::thread_rng();
    let mut bit_input_temp = [Fp::from(0); NUM_LENGTH];
    let mut green_list_flag_temp = Fp::from(1);
    let mut pow_two_value: [Fp; NUM_LENGTH] = [Fp::from(0); NUM_LENGTH];
    let mut instance_temp: [Fp; NUM_LENGTH] = [Fp::from(0); NUM_LENGTH];
    // let mut selector_temp = Value::known(Fp::from(0));

    

    for i in 0..NUM_LENGTH {
        let mul = rng.gen_range(0..10);
        let temp = rng.gen_range(0..=1);
        let temp2 = 2u64.pow(mul as u32);
        bit_input_temp[i] = Fp::from(temp * temp2 as u64);

        
        pow_two_value[i] = Fp::from(temp2);
        // instance_temp[i] = pow_two_value[i].clone();
        instance_temp[i] = Fp::from(temp2);
    }



    (bit_input_temp, green_list_flag_temp, pow_two_value, instance_temp)
}


fn main()
{
    // let input_temp = [Value::known(Fp::from(0)), Value::known(Fp::from(1)), Value::known(Fp::from(8))];
    // let instance_temp = Value::known(Fp::from(8));
    // let selector_temp = Value::known(Fp::from(1));
    let (input_temp, green_list_flag, pow_two_value, instance_value) = generate_muti_oneorzero_test_fp_case();
    // let input_temp = [Value::known(Fp::from(4)), Value::known(Fp::from(2))];
    // let green_list_flag = Value::known(Fp::from(1));
    // let pow_two_value = [Value::known(Fp::from(4)), Value::known(Fp::from(4))];
    // let instance_value = [Fp::from(4), Fp::from(4)];
    // println!("input_temp = {:?}, green_list_flag = {:?}, pow_two_value = {:?}, instance_value = {:?}", input_temp, green_list_flag, pow_two_value, instance_value);
    println!("input_temp = {:?}", input_temp);
    println!("green_list_flag = {:?}", green_list_flag);
    println!("pow_two_value = {:?}", pow_two_value);
    println!("instance_value = {:?}", instance_value);

    let circuit = DetectWatermarkCircuit {
        bit_inputs: input_temp,
        green_list_flag,
        pow_two_inputs: pow_two_value,
        // instance: instance_value,
        // selector: selector_temp,
    };

    let mut public_input = vec![];
    for i in 0..NUM_LENGTH {
        public_input.push(instance_value[i]);
    }
    // let mut public_input = vec![Fp::from(8)];


    // let mut k = 21;
    // println!("k = {}", k);


    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(4, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    // println!("{:?}, {:?}", circuit, public_input);
    // // std::process::exit(0);
    // let mut new_k = 1;
    // let mut pow_new_k = 2;
    // while pow_new_k < total_row {
    //     pow_new_k <<= 1;
    //     new_k += 1;
    // }
    // println!("change k = {} to k = {}", k, new_k);
    // k = new_k;


    // use plotters::prelude::*;
    // let root = BitMapBackend::new("./detect_watermark_test_data/detect_watermark_layout_test_1.png", (1024, 768)).into_drawing_area();
    // root.fill(&WHITE).unwrap();
    // let root = root
    //     .titled("Example Circuit Layout", ("sans-serif", 60))
    //     .unwrap();


    // halo2_proofs::dev::CircuitLayout::default()
    // .show_labels(true)
    // .render(k, &circuit, &root).unwrap();
    // Generate the DOT graph string.
    let dot_string = halo2_proofs::dev::circuit_dot_graph(&circuit);

    // Now you can either handle it in Rust, or just
    // print it out to use with command-line tools.
    println!("{}", dot_string);
    

    let prover = MockProver::run(4, &circuit, vec![public_input.clone()]).unwrap();
    println!("{:?}", prover);
    assert_eq!(prover.verify(), Ok(()));
    println!("success");
    


}