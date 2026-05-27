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
use num_bigint::{BigUint, ToBigUint};
use num_traits::Num;

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

const BIG_PRIME: &str = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
//    /// File to read
//    filename: String,
// /mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/input.txt

    /// The path of input file including public inputs
   #[arg(short, long, default_value = "/mnt/disk2/username/lm-watermarking/test_result/halo2_circuit_public_inputs_fixed.txt")]
   input_file_path: String,

   /// The path of output file including counting results
   #[arg(short, long, default_value = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/output_fixed.txt")]
   output_file_path: String,

//    /// Scale factor
//    #[arg(short, long, default_value_t = 4)]
//    scale: i32,

   /// The path of proof
   #[arg(short, long, default_value = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/proof.bin")]
   proof_path: String,

   /// The secret key value
   #[arg(short, long, default_value_t = 2024)]
   secret_key: usize,

    /// The number of token
    #[arg(short, long, default_value_t = 200)]
    max_token_num: usize,

    /// Gamma (green list size)
    #[arg(short, long, default_value_t = 0.25)]
    gamma: f64,

}





pub fn sub_fp(value: Fp) -> Fp {
    if value < POW_253_FP {
        return value;
    }
    let result = value - POW_253_FP;
    result
    // value
}

#[derive(Debug, Clone)]
pub struct DetectTextWatermarkConfig {
    pub advice_1: Column<Advice>,
    pub advice_2: Column<Advice>,
    pub advice_3: Column<Advice>,

    //
    // pub advice_4: Column<Advice>,
    // pub advice_5: Column<Advice>,
    pub range_u8: Column<Fixed>,
    //

    pub advice_many: [Column<Advice>; SUMMATION_NUM],

    pub instance_1: Column<Instance>,
    
    pub poseidon_config: PoseidonConfig<WIDTH, RATE, RATE>,
    pub less_than_config: LTConfig,
    // pub add_config: AddConfig,
    pub summation_config: SummationConfig,
    // pub check_dec_config: CheckDecConfig,
}

#[derive(Debug, Clone, Default)]
pub struct DetectTextWatermarkCircuit {
    pub secret_key: Fp,

    pub last_prompt_index: usize,
    pub text_token_index_list: Vec<usize>, // length = text_token_value_list_length
    pub text_token_value_list_length: usize,

    // pub text_token_hash_threshold: Vec<usize>,
    // pub text_token_hash_threshold: Vec<Fp>,
    pub text_token_hash_threshold: Fp,

    // pub compare_range_length: usize, // Bit_length of X3=X1-X2
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

        //
        let advice_4 = meta.advice_column();
        let advice_5 = meta.advice_column();
        //

        let advice_many = [(); SUMMATION_NUM].map(|_| meta.advice_column());

        let range_u8 = meta.fixed_column();

        let rc_a = [(); WIDTH].map(|_| meta.fixed_column());
        let rc_b = [(); WIDTH].map(|_| meta.fixed_column());

        let instance_1 = meta.instance_column();
        // let poseidon_config = PoseidonChip::<PoseidonSpec<WIDTH, RATE, RATE>::configure(meta);
        // let less_than_config = LessThanChip::configure(meta, advice_1, advice_2, advice_3, instance_1);
        
        DetectTextWatermarkConfig {
            advice_1: advice_1,
            advice_2: advice_2,
            advice_3: advice_3,
            // advice_4: advice_4,
            // advice_5: advice_5,
            range_u8: range_u8,
            advice_many: advice_many,
            instance_1: instance_1,
            // poseidon_config: PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::configure(meta, [advice_1, advice_2, advice_3], advice_many[0], instance_1),
            // less_than_config: LTChip::configure(meta, advice_1, advice_2, advice_3, advice_many, instance_1),
            less_than_config: LTChip::configure(meta, advice_1, advice_2, advice_3, range_u8, advice_many, instance_1),
            /*poseidon_3 */
            poseidon_config: PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::configure(meta, [advice_1, advice_2, advice_3], advice_many[0], rc_a, rc_b, instance_1),
            // less_than_config: LTChip::configure(meta, advice_1, advice_2, advice_3, rc_a[0], advice_many, instance_1),
            /*poseidon_3 */
            // less_than_config: LTChip::configure(meta, advice_1, advice_2, advice_3, instance_1),
            summation_config: SummationChip::configure(meta, advice_many, advice_1, instance_1),
            // add_config: AddChip::configure(meta, advice_1, advice_2, advice_3, instance_1),

            // check_dec_config: CheckDecChip::configure(meta, [advice_1, advice_2, advice_3, advice_4], advice_5, advice_many, instance_1, range_u8),

        }
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        let poseidon_chip = PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::construct(config.poseidon_config);
        let less_than_chip = LTChip::construct(config.less_than_config);
        // let add_chip = AddChip::construct(config.add_config);
        let summation_chip = SummationChip::construct(config.summation_config);
        // let check_dec_chip = CheckDecChip::construct(config.check_dec_config);

        less_than_chip.load_lookup_table(layouter.namespace(|| "load lookup table"))?;

        // Compute First Random Seed
        let mut hash_input1 = [Fp::from(self.secret_key), Fp::from(self.last_prompt_index as u64)];
        let mut seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input1);
        seed = sub_fp(seed.clone());

        // Compute Hash_this_token in text
        // let mut real_index = (self.text_token_index_list[0] / DEC_NUM) as usize;
        // let mut mod_index = (self.text_token_index_list[0] % DEC_NUM) as usize;
        let mut hash_input2 = [seed, Fp::from(self.text_token_index_list[0] as u64)];
        // let mut hash_input2 = [seed, Fp::from(real_index as u64)];
        let mut seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input2);
        seed2 = sub_fp(seed2.clone());
        // let mut org_seed2 = seed2.clone();
        // let mut dec_inputs_array: [Fp; DEC_NUM] = [Fp::zero(); DEC_NUM];
        // (seed2, dec_inputs_array) = dec_num_fp(org_seed2, mod_index, DEC_NUM);
        // println!("dec_inputs_array: {:?}", dec_inputs_array);
        // println!("seed2: {:?}", seed2);
        

        // Judge if Hash_this_token < text_token_hash_threshold[0]
        let mut less_than_flag: usize = 0;
        // let mut threshold_this = Fp::from(self.text_token_hash_threshold[0] as u64);
        let mut threshold_this = self.text_token_hash_threshold;
        if seed2 < threshold_this {
            less_than_flag = 1;
        }

        // Verify
        let mut green_flag_cell_vec = vec![];
        // let mut threshold_cell_vec = vec![];
        let mut token_index_cell_vec = vec![];
        
        let mut hash_input1_cell = poseidon_chip.load_private_inputs(layouter.namespace(|| "Load Poseidon Hash Function Inputs"), [Value::known(hash_input1[0]), Value::known(hash_input1[1])])?;
        let mut seed_cell = poseidon_chip.hash(layouter.namespace(|| "Poseidon Hash -- Compute Random Seed"), &hash_input1_cell)?;

        let mut hash_input2_cell = poseidon_chip.load_private_inputs(layouter.namespace(|| "Load Poseidon Hash Function Inputs"), [Value::known(hash_input2[0]), Value::known(hash_input2[1])])?;
        let mut seed2_cell = poseidon_chip.hash(layouter.namespace(|| "Poseidon Hash -- Compute Hash_this_token"), &hash_input2_cell)?;
        // println!("-------------------------------------enter less_than_flag_cell");
        // println!("seed2: {:?}", seed2);
        // println!("threshold_this: {:?}", threshold_this);
        // let (mut less_than_input1_cell, mut less_than_input2_cell, mut less_than_flag_cell) = less_than_chip.is_less_than(layouter.namespace(|| "Judge if Hash_this_token < text_token_hash_threshold[0]"), seed2, threshold_this)?;
        // check_dec_chip.assign_value_and_check(layouter.namespace(|| "check dec inputs"), dec_inputs_array, org_seed2)?;
        let (mut less_than_input1_cell, mut less_than_input2_cell, mut less_than_flag_cell) = less_than_chip.assign_value_and_is_less_than(layouter.namespace(|| "Judge if Hash_this_token < text_token_hash_threshold[0]"), seed2, threshold_this)?;
        

        let mut last_prompt_index_cell = hash_input1_cell[1].clone();
        token_index_cell_vec.push(hash_input2_cell[1].clone());
        green_flag_cell_vec.push(less_than_flag_cell.clone());
        // threshold_cell_vec.push(less_than_input2_cell.clone());
        let fixed_threshold_cell = less_than_input2_cell.clone();

        // verify greater_than
        // less_than_chip.is_less_than(layouter.namespace(|| "greater than"), threshold_this, seed2)?;
        less_than_chip.is_less_than_expr_greater_than(layouter.namespace(|| "greater than"), &less_than_input2_cell, &less_than_input1_cell, &less_than_flag_cell, threshold_this, seed2)?;

        // println!("-------------------------------------enter for loop(round:  text_token_value_list_length-1)");
        for round in 0..self.text_token_value_list_length-1 {
            // println!("---------------------------------------------------------round -> {:?}", round);
            // Compute Random Seed
            hash_input1 = [Fp::from(self.secret_key), Fp::from(self.text_token_index_list[round] as u64)];
            seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(hash_input1);
            seed = sub_fp(seed.clone());

            // Compute Hash_this_token
            // real_index = (self.text_token_index_list[round+1] / DEC_NUM) as usize;
            // mod_index = (self.text_token_index_list[round+1] % DEC_NUM) as usize;
            hash_input2 = [seed, Fp::from(self.text_token_index_list[round+1] as u64)];
            // hash_input2 = [seed, Fp::from(real_index as u64)];
            seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(hash_input2);
            // org_seed2 = seed2.clone();
            // (seed2, dec_inputs_array) = dec_num_fp(org_seed2, mod_index, DEC_NUM);
            seed2 = sub_fp(seed2.clone());
            
            // Judge if Hash_this_token < text_token_hash_threshold[round+1]
            // threshold_this = Fp::from(self.text_token_hash_threshold[round+1] as u64);
            threshold_this = self.text_token_hash_threshold;
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

            // check_dec_chip.assign_value_and_check(layouter.namespace(|| "check dec inputs"), dec_inputs_array, org_seed2)?;
            // println!("begin is_less_than_2--------------------------------------------");
            // (less_than_input1_cell, less_than_input2_cell, less_than_flag_cell) = less_than_chip.is_less_than(layouter.namespace(|| format!("Judge less than[{:?}]", round+1)), seed2, threshold_this)?;
            (less_than_input1_cell, less_than_input2_cell, less_than_flag_cell) = less_than_chip.assign_value_and_is_less_than(layouter.namespace(|| format!("Judge less than[{:?}]", round+1)), seed2, threshold_this)?;
            

            token_index_cell_vec.push(hash_input2_cell[1].clone());
            green_flag_cell_vec.push(less_than_flag_cell.clone());
            // threshold_cell_vec.push(less_than_input2_cell.clone());

            // verify greater_than
            // less_than_chip.is_less_than(layouter.namespace(|| format!("greater than[{:?}]", round+1)), threshold_this, seed2)?;
            less_than_chip.is_less_than_expr_greater_than(layouter.namespace(|| format!("greater than[{:?}]", round+1)), &less_than_input2_cell, &less_than_input1_cell, &less_than_flag_cell, threshold_this, seed2)?;

        }

        // Count the number of green flags
        // println!("green_flag_cell_vec:{:?}", green_flag_cell_vec);
        // let count_result_cell = add_chip.assign_multiple_value_and_summation_3(
        //     layouter.namespace(|| "Count the number of green flags"), 
        //     &green_flag_cell_vec, 
        //     green_flag_cell_vec.len(),
        // )?;
        let count_result_cell = summation_chip.assign_multiple_value_and_summation(
            layouter.namespace(|| "Count the number of green flags"), 
            &green_flag_cell_vec, 
            green_flag_cell_vec.len(),
        )?;
        // println!("count_result_cell: {:?}", count_result_cell);
        
        let mut instance_index: usize = 0;
        // Expose Public Inputs
        // add_chip.expose_public(layouter.namespace(|| "Expose Count Result"), &count_result_cell, 0)?;
        summation_chip.expose_public(layouter.namespace(|| "Expose Count Result"), &count_result_cell, 0)?;
        instance_index = instance_index + 1;
        // println!("count_result_cell: {:?}", count_result_cell);

        // Expose Last Prompt index
        summation_chip.expose_public(layouter.namespace(|| "Expose Last Prompt Index"), &last_prompt_index_cell, instance_index)?;
        instance_index = instance_index + 1;

        // Expose Token Index
        for round in 0..token_index_cell_vec.len() {
            summation_chip.expose_public(layouter.namespace(|| format!("Expose Token Index[{:?}]", round)), &token_index_cell_vec[round], instance_index)?;
            instance_index = instance_index + 1;
        }
        // println!("token_index_cell_vec: {:?}", token_index_cell_vec);

        // Expose Threshold
        summation_chip.expose_public(layouter.namespace(|| format!("Expose Fixed Threshold")), &fixed_threshold_cell, instance_index)?;
        // for round in 0..threshold_cell_vec.len() {
        //     summation_chip.expose_public(layouter.namespace(|| format!("Expose Threshold[{:?}]", round)), &threshold_cell_vec[round], instance_index)?;
        //     instance_index = instance_index + 1;
        // }
        // println!("threshold_cell_vec: {:?}", threshold_cell_vec);



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
fn generate_test_case(vocabulary_size: usize) -> (Fp, usize, Vec<usize>, usize, Vec<Fp>, usize) {
    let mut rng = rand::thread_rng();
    // let mut test_case_list = vec![];
    let serect_key: Fp = Fp::from(2023);
    let last_prompt_index: usize = rng.gen_range(0..vocabulary_size);
    
    // let compare_range_length = NUM_BITS_COMPARE;
    let text_token_value_list_length: usize = vocabulary_size;
    let text_token_index_list: Vec<usize> = generate_unique_random_numbers(0, vocabulary_size-1, vocabulary_size);
    let mut text_token_hash_threshold: Vec<Fp> = vec![];
    for _ in 0..text_token_value_list_length {
        let mut temp = Fp::from(rng.gen_range(0..1000) as u64);
        let mut temp2 = Fp::from(rng.gen_range(0..1000) as u64);
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

    let mut real_index = (text_token_index_list[0] / DEC_NUM) as usize;
    let mut mod_index = (text_token_index_list[0] % DEC_NUM) as usize;
    // println!("real_index = {:?}", real_index);
    // println!("mod_index = {:?}", mod_index);
    // let mut hash_input2 = [seed, Fp::from(text_token_index_list[0] as u64)];
    let mut hash_input2 = [seed, Fp::from(real_index as u64)];
    let mut seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    .hash(hash_input2);
    (seed2, _) = dec_num_fp(seed2, mod_index, DEC_NUM);
    // seed2 = sub_fp(seed2.clone());

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

        real_index = (text_token_index_list[i+1] / DEC_NUM) as usize;
        mod_index = (text_token_index_list[i+1] % DEC_NUM) as usize;
        // hash_input2 = [seed, Fp::from(text_token_index_list[i+1] as u64)];
        hash_input2 = [seed, Fp::from(real_index as u64)];
        seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        .hash(hash_input2);
        (seed2, _) = dec_num_fp(seed2, mod_index, DEC_NUM);
        // seed2 = sub_fp(seed2.clone());

        if seed2 < text_token_hash_threshold[i+1] {
            less_than_flag = 1;
        } else {
            less_than_flag = 0;
        }
        count += less_than_flag;
    }

    (serect_key, last_prompt_index, text_token_index_list, text_token_value_list_length, text_token_hash_threshold, count)

}


// fn get_public_inputs_from_file(file_path: &str) -> std::io::Result<(Vec<usize>, Vec<String>)> {
fn get_public_inputs_from_file_old(file_path: &str) -> (Vec<usize>, Vec<String>) {
    // 打开文件
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);

    // 初始化向量
    let mut usize_values = Vec::new();
    let mut string_values = Vec::new();

    // 遍历文件的每一行
    for line in reader.lines() {
        // 解析每行的数据
        if let Ok(line) = line {
            let mut parts = line.split_whitespace();

            // 读取usize值
            if let Some(usize_value_str) = parts.next() {
                if let Ok(usize_value) = usize_value_str.parse::<usize>() {
                    // 读取String值
                    if let Some(string_value) = parts.next() {
                        // 将值存入向量
                        usize_values.push(usize_value);
                        string_values.push(string_value.to_string());
                    }
                }
            }
        }
    }

    // Ok((usize_values, string_values))
    ((usize_values, string_values))
}


fn get_public_inputs_from_file(filename: &str, max_token: usize) -> Result<(Vec<usize>), std::io::Error> {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    // let mut string_vec: Vec<String> = Vec::new();
    let mut usize_vec: Vec<usize> = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        // println!("index = {:?}", index);
        if index < max_token + 1 {
            if let Ok(num) = line.unwrap().trim().parse::<usize>() {
                usize_vec.push(num);
            }
            else {
                println!("Failed to parse line as usize");
            }
        }
        else {
            break;
        }

    }

    Ok((usize_vec))
}

fn count_greenlist_num(max_token_num: usize, secret_key: usize, text_token_index_list: &Vec<usize>, fixed_threshold: &String) -> usize {
    let mut count = 0;
    let secret_key_fp = Fp::from(secret_key as u64);
    // println!("text_token_index_list_len = {:?}", text_token_index_list.len());
    // println!("threshold_list_len = {:?}", threshold_list.len());
    for i in 0..max_token_num {
        let seed_hash_inputs = [secret_key_fp, Fp::from(text_token_index_list[i] as u64)];
        let mut seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(seed_hash_inputs);
        seed = sub_fp(seed.clone());
        let this_token_hash_inputs = [seed, Fp::from(text_token_index_list[i + 1] as u64)];
        // println!("text_token_index_list[i+1] = {:?}", text_token_index_list[i+1]);
        let mut this_token_hash = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(this_token_hash_inputs);
        this_token_hash = sub_fp(this_token_hash.clone());
        let pad_threshold = pad_string(&fixed_threshold, 64);
        let raw_threshold = process_hex_string(&pad_threshold);
        let threshold = Fp::from_raw(raw_threshold);
        // println!("this_token_hash = {:?}", this_token_hash);
        // println!("threshold = {:?}", threshold);
        if this_token_hash <= threshold {
            // println!("----------------------------------------------------------------------------this_token_hash <= threshold");
            count += 1;
        }
    }

    count
}

fn compute_z_score(greenlist_count: usize, gamma: f64, total_token_num: usize) -> f64 {
    let expected_count = gamma;
    let numer = greenlist_count as f64 - expected_count * total_token_num as f64;
    let denom = (total_token_num as f64 * expected_count * (1.0 - expected_count)).sqrt();
    let z = numer / denom;
    z
}


fn main() {
    let args = Args::parse();
    let cli_input_file_path = args.input_file_path;
    let cli_output_file_path = args.output_file_path;
    let cli_proof_path = args.proof_path;
    let cli_secret_key = args.secret_key;
    let cli_max_token_num = args.max_token_num;
    let cli_gamma = args.gamma;

    let max_token_num: usize = cli_max_token_num;
    // let gamma: f64 = 0.25;
    let secret_key: usize = cli_secret_key.clone();

    let gamma_u64 = gamma_to_u64(cli_gamma).unwrap();
    // let big_prime_int = BigUint::from_str_radix(&BIG_PRIME, 16).unwrap();
    let big_prime_int = match BigUint::from_str_radix(&BIG_PRIME, 16) {
        Ok(val) => val,
        Err(_) => {
            println!("Error parsing big_prime");
            return
        }
    };

    let fixed_threshold = big_prime_int / gamma_u64;
    let fixed_threshold_hex_string = fixed_threshold.to_str_radix(16);
    
    // 删除0x前缀
    let fixed_threshold_hex_string_without_prefix = fixed_threshold_hex_string.trim_start_matches("0x").to_string();
    println!("fixed_threshold_hex_string_without_prefix = {:?}", fixed_threshold_hex_string_without_prefix);

    // let (usize_values, string_values) = get_public_inputs_from_file(&cli_input_file_path).unwrap();
    let (usize_values) = get_public_inputs_from_file(&cli_input_file_path, max_token_num).unwrap();
    // println!("string_values_len = {:?}", string_values.len());
    println!("usize_values_len = {:?}", usize_values.len());
    // println!("string_values = {:?}", string_values);
    // println!("usize_values = {:?}", usize_values);
    // let first_five: [usize; 5] = usize_values[..5].try_into().unwrap();
    // let first_five_2: Vec<&str> = string_values.iter().take(5).map(String::as_str).collect();

    // pezgenezkl -rintln!("usize_values: {:?}", first_five);
    // println!("string_values: {:?}", first_five_2);
    let greenlist_count = count_greenlist_num(max_token_num, secret_key, &usize_values, &fixed_threshold_hex_string_without_prefix);
    println!("greenlist_count: {:?}", greenlist_count);
    let z_score = compute_z_score(greenlist_count, cli_gamma, max_token_num);
    println!("z_score: {:?}", z_score);
    let z_threshold = 4.0;
    if z_score > z_threshold {
        println!("The text contains watermark");
    } else {
        println!("The text does not contain watermark");
    }

    // assert_eq!(1, 2);
    // let vocabulary_size: usize = 1000;
    // let (serect_key, last_prompt_index, text_token_index_list, text_token_value_list_length, text_token_hash_threshold, count) = generate_test_case(vocabulary_size);
    // println!("serect_key = {:?}", serect_key);
    // println!("last_prompt_index = {:?}", last_prompt_index);
    // println!("text_token_index_list = {:?}", text_token_index_list);
    // println!("text_token_value_list_length = {:?}", text_token_value_list_length);
    // println!("text_token_hash_threshold = {:?}", text_token_hash_threshold);
    // println!("compare_range_length = {:?}", compare_range_length);
    // println!("count = {:?}", count);
    let last_prompt_index = usize_values[0];
    let text_token_index_list = usize_values[1..].to_vec();
    let text_token_value_list_length = cli_max_token_num;
    // let text_token_hash_threshold: Vec<Fr> = string_values.iter().map(|x| Fp::from_raw(process_hex_string(&pad_string(x, 64)))).collect();
    let text_token_hash_fixed_threshold = Fp::from_raw(process_hex_string(&pad_string(&fixed_threshold_hex_string_without_prefix, 64)));


    let circuit = DetectTextWatermarkCircuit {
        secret_key: Fp::from(cli_secret_key.clone() as u64),
        last_prompt_index: last_prompt_index.clone(),
        text_token_index_list: text_token_index_list.clone(),
        text_token_value_list_length: text_token_value_list_length.clone(),
        // text_token_hash_threshold: text_token_hash_threshold.clone(),
        text_token_hash_threshold: text_token_hash_fixed_threshold.clone(),
        // compare_range_length: compare_range_length,
    };

    // println!("{:?}", circuit);

    let mut public_inputs = vec![];
    public_inputs.push(Fp::from(greenlist_count as u64));
    public_inputs.push(Fp::from(last_prompt_index as u64));
    for i in 0..text_token_value_list_length {
        public_inputs.push(Fp::from(text_token_index_list[i] as u64));
    }
    public_inputs.push(text_token_hash_fixed_threshold.clone());
    // for i in 0..text_token_value_list_length {
    //     public_inputs.push(text_token_hash_threshold[i]);
    // }
    // println!("public_inputs = {:?}", public_inputs);

    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(25, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    let mut k = select_suitable_k_value(total_row as i32);
    println!("k = {:?}", k);
    // k = k + 1;

    let mock_start_time = Instant::now(); // Start time
    let correct_prover = MockProver::run(
        // 15,
        k,
        &circuit,
        vec![public_inputs.clone()],
        // vec![correct_public_input.clone()],
    )
    .unwrap();
    // println!("MockProver:{:?}", correct_prover);
    let mock_elapsed_time = mock_start_time.elapsed();
    println!("Running Mock took {:.4} seconds.", mock_elapsed_time.as_millis() as f64 / 1000.0);

    correct_prover.assert_satisfied();
    println!("success");


    let srs_path = "/mnt/disk2/username/onnx_test/srs_params/perpetual-powers-of-tau-raw-".to_string() + &k.to_string();
    // let params: Params<Bn256> = Params::new(k);
    let params = load_params_cmd(srs_path.into(), k).expect("load_params_cmd should not fail");
    // let mut params: ParamsKZG<Bn256> = load_srs::<KZGCommitmentScheme<Bn256>>(srs_path.into())?;
    // let params: Params<G1Affine> = halo2_proofs::poly::commitment::Params::new(k);
    // Initialize the proving key
    let setup_start_time = Instant::now(); // Start time

    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");
    // let pk = keygen_pk(params, vk, &circuit).expect("keygen_pk should not fail");

    let setup_elapsed_time = setup_start_time.elapsed(); 

    // println!("Running Setup took {} seconds.", setup_elapsed_time.as_secs());
    println!("Running Setup took {:.4} seconds.", setup_elapsed_time.as_millis() as f64 / 1000.0);


    // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let mut transcript = TranscriptWriterBuffer::<_, G1Affine, _>::init(Vec::new());

    let instance_temp = vec![public_inputs.clone()];
    
    let temp_inner = instance_temp.iter().map(|inner| inner.as_slice()).collect::<Vec<_>>();
    let temp_inner2: &[&[&[Fp]]] = &[&temp_inner];
    let prove_start_time = Instant::now(); // Start time
    // create_proof(
    create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, Blake2bWrite<_, _, _>, _>(
        &params,
        &pk,
        // &[circuit.clone(), circuit.clone()],
        &[circuit.clone()],
        temp_inner2,
        // &[&[&[leaf.clone()]], &[&[digest.clone()]]],
        // &[&[&[digest.clone()]]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let prove_elapsed_time = prove_start_time.elapsed(); 

    // println!("Running Prove took {} seconds.", prove_elapsed_time.as_secs());
    println!("Running Prove took {:.4} seconds.", prove_elapsed_time.as_millis() as f64 / 1000.0);

    let proof: Vec<u8> = transcript.finalize();

    let proof_path = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_verify_detect_text_watermark/proof.bin";

    // std::fs::write("plonk_api_proof.bin", &proof[..])
    //     .expect("should succeed to write new proof");
    std::fs::write(proof_path, &proof[..])
        .expect("should succeed to write new proof");


    // let strategy = SingleVerifier::new(&params);
    let strategy = KZGSingleStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    
    let verify_start_time = Instant::now(); // Start time
    assert!(verify_proof::<KZGCommitmentScheme<_>, VerifierGWC<'_, Bn256>, _, Blake2bRead<_, _, _>, _>(
        &params,
        pk.get_vk(),
        strategy,
        temp_inner2,
        &mut transcript,
    )
    .is_ok());
    let verify_elapsed_time = verify_start_time.elapsed();
    // println!("Running Verify took {} seconds.", verify_elapsed_time.as_secs());
    println!("Running Verify took {:.4} seconds.", verify_elapsed_time.as_millis() as f64 / 1000.0);

    println!("success");


    

}