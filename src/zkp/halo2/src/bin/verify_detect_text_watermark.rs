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
// use halo2_merkle_tree::chips::poseidon::{PoseidonChip, PoseidonConfig};
use halo2_merkle_tree::chips::poseidon_2::{PoseidonChip, PoseidonConfig};
use halo2_merkle_tree::chips::less_than_chip::{LessThanChip, LessThanConfig, self};
use halo2_merkle_tree::chips::add_chip::{AddChip, AddConfig, self};
use halo2_merkle_tree::poseidon::spec_width_3::{POSEIDON_WIDTH, POSEIDON_RATE};
use halo2_merkle_tree::poseidon::spec_width_3::PoseidonSpec;


const WIDTH: usize = POSEIDON_WIDTH;
const RATE: usize = POSEIDON_RATE;

const NUM_BITS_COMPARE: usize = 255;

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

pub fn sub_fp(value: Fp) -> Fp {
    // if value < POW_253_FP {
    //     return value;
    // }
    // let result = value - POW_253_FP;
    // result
    value
}

#[derive(Debug, Clone)]
pub struct DetectTextWatermarkConfig {
    pub advice_1: Column<Advice>,
    pub advice_2: Column<Advice>,
    pub advice_3: Column<Advice>,
    pub instance_1: Column<Instance>,
    
    pub poseidon_config: PoseidonConfig<WIDTH, RATE, RATE>,
    pub less_than_config: LessThanConfig,
    pub add_config: AddConfig,
}

#[derive(Debug, Clone, Default)]
pub struct DetectTextWatermarkCircuit {
    pub secret_key: Fp,

    pub last_prompt_index: usize,
    pub text_token_index_list: Vec<usize>, // length = text_token_value_list_length
    pub text_token_value_list_length: usize,

    // pub text_token_hash_threshold: Vec<usize>,
    pub text_token_hash_threshold: Vec<Fp>,

    pub compare_range_length: usize, // Bit_length of X3=X1-X2
}

impl Circuit<Fp> for DetectTextWatermarkCircuit {
    type Config = DetectTextWatermarkConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> DetectTextWatermarkConfig {
        let advice_1 = meta.advice_column();
        let advice_2 = meta.advice_column();
        let advice_3 = meta.advice_column();
        let instance_1 = meta.instance_column();
        // let poseidon_config = PoseidonChip::<PoseidonSpec<WIDTH, RATE, RATE>::configure(meta);
        // let less_than_config = LessThanChip::configure(meta, advice_1, advice_2, advice_3, instance_1);
        
        DetectTextWatermarkConfig {
            advice_1: advice_1,
            advice_2: advice_2,
            advice_3: advice_3,
            instance_1: instance_1,
            poseidon_config: PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::configure(meta, instance_1),
            less_than_config: LessThanChip::configure(meta, advice_1, advice_2, advice_3, instance_1),
            add_config: AddChip::configure(meta, advice_1, advice_2, advice_3, instance_1),
        }
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        let poseidon_chip = PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::construct(config.poseidon_config);
        let less_than_chip = LessThanChip::construct(config.less_than_config);
        let add_chip = AddChip::construct(config.add_config);

        // Compute First Random Seed
        let mut hash_input1 = [Fp::from(self.secret_key), Fp::from(self.last_prompt_index as u64)];
        let mut seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input1);
        // seed = sub_fp(seed.clone());

        // Compute Hash_this_token in text
        let mut hash_input2 = [seed, Fp::from(self.text_token_index_list[0] as u64)];
        let mut seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input2);
        // seed2 = sub_fp(seed2.clone());

        // Judge if Hash_this_token < text_token_hash_threshold[0]
        let mut less_than_flag: usize = 0;
        // let mut threshold_this = Fp::from(self.text_token_hash_threshold[0] as u64);
        let mut threshold_this = self.text_token_hash_threshold[0];
        if seed2 < threshold_this {
            less_than_flag = 1;
        }

        // Verify
        let mut green_flag_cell_vec = vec![];
        let mut threshold_cell_vec = vec![];

        let mut hash_input1_cell = poseidon_chip.load_private_inputs(layouter.namespace(|| "Load Poseidon Hash Function Inputs"), [Value::known(hash_input1[0]), Value::known(hash_input1[1])])?;
        let mut seed_cell = poseidon_chip.hash(layouter.namespace(|| "Poseidon Hash -- Compute Random Seed"), &hash_input1_cell)?;

        let mut hash_input2_cell = poseidon_chip.load_private_inputs(layouter.namespace(|| "Load Poseidon Hash Function Inputs"), [Value::known(hash_input2[0]), Value::known(hash_input2[1])])?;
        let mut seed2_cell = poseidon_chip.hash(layouter.namespace(|| "Poseidon Hash -- Compute Hash_this_token"), &hash_input2_cell)?;
        // println!("-------------------------------------enter less_than_flag_cell");
        // println!("seed2: {:?}", seed2);
        // println!("threshold_this: {:?}", threshold_this);
        let (mut less_than_flag_cell, mut input_x2_cell) = less_than_chip.is_less_than_2(
            layouter.namespace(|| "Judge if Hash_this_token < text_token_hash_threshold[0]"), 
            &seed2_cell, 
            seed2, 
            threshold_this,
            self.compare_range_length,
        )?;
        green_flag_cell_vec.push(less_than_flag_cell.clone());
        threshold_cell_vec.push(input_x2_cell.clone());

        // verify greater_than
        less_than_chip.is_less_than_3(layouter.namespace(
            || "Judge greater than"), 
            &input_x2_cell, 
            &seed2_cell, 
            threshold_this, 
            seed2, 
            self.compare_range_length,
        )?;    

        // println!("-------------------------------------enter for loop(round:  text_token_value_list_length-1)");
        for round in 0..self.text_token_value_list_length-1 {
            println!("---------------------------------------------------------round -> {:?}", round);
            // Compute Random Seed
            hash_input1 = [Fp::from(self.secret_key), Fp::from(self.text_token_index_list[round] as u64)];
            seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(hash_input1);
            // seed = sub_fp(seed.clone());

            // Compute Hash_this_token
            hash_input2 = [seed, Fp::from(self.text_token_index_list[round+1] as u64)];
            seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(hash_input2);
            // seed2 = sub_fp(seed2.clone());
            
            // Judge if Hash_this_token < text_token_hash_threshold[round+1]
            // threshold_this = Fp::from(self.text_token_hash_threshold[round+1] as u64);
            threshold_this = self.text_token_hash_threshold[round+1];
            if seed2 < threshold_this {
                less_than_flag = 1;
            } else {
                less_than_flag = 0;
            }

            // Verify
            hash_input1_cell = poseidon_chip.load_private_inputs(layouter.namespace(|| "Load Poseidon Hash Function Inputs"), [Value::known(hash_input1[0]), Value::known(hash_input1[1])])?;
            seed_cell = poseidon_chip.hash(layouter.namespace(|| "Poseidon Hash -- Compute Random Seed"), &hash_input1_cell)?;

            hash_input2_cell = poseidon_chip.load_private_inputs(layouter.namespace(|| "Load Poseidon Hash Function Inputs"), [Value::known(hash_input2[0]), Value::known(hash_input2[1])])?;
            seed2_cell = poseidon_chip.hash(layouter.namespace(|| "Poseidon Hash -- Compute Hash_this_token"), &hash_input2_cell)?;

            // println!("begin is_less_than_2--------------------------------------------");
            (less_than_flag_cell, input_x2_cell) = less_than_chip.is_less_than_2(
                layouter.namespace(|| format!("Judge if Hash_this_token < text_token_hash_threshold[{:?}]", round+1)), 
                &seed2_cell, 
                seed2, 
                threshold_this,
                self.compare_range_length,
            )?;
            green_flag_cell_vec.push(less_than_flag_cell.clone());
            threshold_cell_vec.push(input_x2_cell.clone());

            // verify greater_than
            less_than_chip.is_less_than_3(layouter.namespace(
                || format!("Judge greater than[{:?}]", round+1)), 
                &input_x2_cell, 
                &seed2_cell, 
                threshold_this, 
                seed2, 
                self.compare_range_length,
            )?;

        }

        // Count the number of green flags
        println!("green_flag_cell_vec:{:?}", green_flag_cell_vec);
        let count_result_cell = add_chip.assign_multiple_value_and_summation_3(
            layouter.namespace(|| "Count the number of green flags"), 
            &green_flag_cell_vec, 
            green_flag_cell_vec.len(),
        )?;

        // Expose Public Inputs
        add_chip.expose_public(layouter.namespace(|| "Expose Count Result"), &count_result_cell, 0)?;
        println!("count_result_cell: {:?}", count_result_cell);

        // Expose Threshold
        for round in 0..threshold_cell_vec.len() {
            add_chip.expose_public(layouter.namespace(|| format!("Expose Threshold[{:?}]", round)), &threshold_cell_vec[round], round + 1)?;
        }



        Ok(())
    }
}


use rand::seq::SliceRandom;
use rand::thread_rng;

fn generate_unique_random_numbers(min: usize, max: usize, count: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut unique_numbers: Vec<usize> = (min..=max).collect();
    unique_numbers.shuffle(&mut rng);
    unique_numbers.truncate(count);
    unique_numbers
}

// secret_key, last_prompt_index, text_token_index_list, text_token_value_list_length, text_token_hash_threshold, compare_range_length, count
fn generate_test_case(vocabulary_size: usize) -> (Fp, usize, Vec<usize>, usize, Vec<Fp>, usize, usize) {
    let mut rng = rand::thread_rng();
    // let mut test_case_list = vec![];
    let serect_key: Fp = Fp::from(2023);
    let last_prompt_index: usize = rng.gen_range(0..vocabulary_size);
    
    let compare_range_length = NUM_BITS_COMPARE;
    let text_token_value_list_length: usize = vocabulary_size;
    let text_token_index_list: Vec<usize> = generate_unique_random_numbers(0, vocabulary_size-1, vocabulary_size);
    let mut text_token_hash_threshold: Vec<Fp> = vec![];
    for _ in 0..text_token_value_list_length {
        let mut temp = Fp::from(rng.gen_range(0..100) as u64);
        let mut temp2 = Fp::from(rng.gen_range(0..100) as u64);
        // temp = sub_fp(temp.clone());
        let mut threshold_temp = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash([temp, temp2]);
        text_token_hash_threshold.push(threshold_temp);
    }

    let mut count = 0;
    let mut hash_input1 = [Fp::from(serect_key), Fp::from(last_prompt_index as u64)];
    let mut seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    .hash(hash_input1);
    seed = sub_fp(seed.clone());

    let mut hash_input2 = [seed, Fp::from(text_token_index_list[0] as u64)];
    let mut seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    .hash(hash_input2);
    seed2 = sub_fp(seed2.clone());

    let mut less_than_flag: usize = 0;
    let mut threshold_this = text_token_hash_threshold[0];
    if seed2 < threshold_this {
        less_than_flag = 1;
    }
    count += less_than_flag;
    for i in 0..text_token_value_list_length-1 {
        hash_input1 = [Fp::from(serect_key), Fp::from(text_token_index_list[i] as u64)];
        seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input1);
        seed = sub_fp(seed.clone());

        hash_input2 = [seed, Fp::from(text_token_index_list[i+1] as u64)];
        seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input2);
        seed2 = sub_fp(seed2.clone());

        if seed2 < text_token_hash_threshold[i+1] {
            less_than_flag = 1;
        } else {
            less_than_flag = 0;
        }
        count += less_than_flag;
    }

    (serect_key, last_prompt_index, text_token_index_list, text_token_value_list_length, text_token_hash_threshold, compare_range_length, count)

}


fn main() {
    let vocabulary_size: usize = 10;
    let (serect_key, last_prompt_index, text_token_index_list, text_token_value_list_length, text_token_hash_threshold, compare_range_length, count) = generate_test_case(vocabulary_size);
    println!("serect_key = {:?}", serect_key);
    println!("last_prompt_index = {:?}", last_prompt_index);
    println!("text_token_index_list = {:?}", text_token_index_list);
    println!("text_token_value_list_length = {:?}", text_token_value_list_length);
    println!("text_token_hash_threshold = {:?}", text_token_hash_threshold);
    println!("compare_range_length = {:?}", compare_range_length);
    println!("count = {:?}", count);


    let circuit = DetectTextWatermarkCircuit {
        secret_key: serect_key,
        last_prompt_index: last_prompt_index,
        text_token_index_list: text_token_index_list,
        text_token_value_list_length: text_token_value_list_length,
        text_token_hash_threshold: text_token_hash_threshold.clone(),
        compare_range_length: compare_range_length,
    };

    println!("{:?}", circuit);

    let mut public_inputs = vec![];
    public_inputs.push(Fp::from(count as u64));
    for i in 0..text_token_value_list_length {
        public_inputs.push(text_token_hash_threshold[i]);
    }

    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(14, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    

    println!("start proving");
    // let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(count as u64)], public_inputs.clone()]).unwrap();
    // let prover = MockProver::run(11, &circuit, vec![vec![Fp::from(count as u64)]]).unwrap();
    let prover = MockProver::run(15, &circuit, vec![public_inputs.clone()]).unwrap();
    // println!("{:?}", prover);
    assert_eq!(prover.verify(), Ok(()));
    println!("success")


}