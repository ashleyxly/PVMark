use std::fmt::format;
use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod super::fieldutils;
use clap::Parser;
use clap::builder::Str;
// use halo2_proofs::dev::metadata::Column;
// use halo2_proofs::plonk::{Advice, Instance};
use rayon::vec;


use std::time::{Duration, Instant};

// use halo2_proofs::pasta::{Eq, EqAffine};
// use halo2_proofs::plonk::{
//     create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, BatchVerifier, Circuit,
//     Column, ConstraintSystem, Error, Fixed, TableColumn,
// };

use halo2_proofs::{circuit::*, plonk::*};
use halo2_proofs::poly::Rotation;
use halo2_proofs::arithmetic::*;

use halo2_proofs::dev::VerifyFailure;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2curves::bn256::{Bn256, Fr as Fp, G1Affine};


use std::marker::PhantomData;

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


use sha256test::chips::add_chip::{AddChip, AddConfig, self};
use sha256test::chips::less_than_lookup_chip::{LTConfig, LTChip};
use sha256test::chips::summation_chip::{SummationChip, SummationConfig, SUMMATION_NUM, self};
// use sha256test::chips::summation_used_in_mimc_chip::{SummationChip as Summation_MiMC_Chip, SummationConfig as Summation_MiMC_Config, SUMMATION_NUM as SUMMATION_NUM_MiMC, self};
use sha256test::utils::*;
// use sha256test::chips::check_dec_chip::{CheckDecChip, CheckDecConfig, DEC_NUM, N_BYTES, self};


use halo2_gadgets::sha256::{Sha256, Sha256Digest, BlockWord, BLOCK_SIZE};
use halo2_gadgets::sha256::{Table16Chip, Table16Config};

extern crate crypto;
use crypto::sha2::Sha256 as Sha256Crypto;
use crypto::digest::Digest;
use ark_bn254::Fr as Sha256_Fr;
use sha256test::utils::convert_to_decimal_str;
use sha256test::utils::{process_hex_string, process_hex_string_3};
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct Sha256DetectionConfig {
    pub advice_1: Column<Advice>,
    pub advice_2: Column<Advice>,
    pub advice_3: Column<Advice>,

    pub range_u8: Column<Fixed>,

    pub advice_many: [Column<Advice>; SUMMATION_NUM],

    pub instance_1: Column<Instance>,

    pub table16_config: Table16Config,

    pub less_than_config: LTConfig,
    pub summation_config: SummationConfig,

}

#[derive(Clone, Debug, Default)]
pub struct Sha256DetectionCircuit {
    pub secret_key: Fp,

    pub last_prompt_index: usize,
    pub text_token_index_list: Vec<usize>,
    pub text_token_value_list_length: usize,

    pub text_token_hash_threshold: Vec<Fp>,
}

impl Circuit<Fp> for Sha256DetectionCircuit {
    type Config = Sha256DetectionConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Sha256DetectionConfig {
        let advice_1 = meta.advice_column();
        let advice_2 = meta.advice_column();
        let advice_3 = meta.advice_column();

        // let state_1 = meta.advice_column();
        // let key_1 = meta.advice_column();

        let advice_many = [(); SUMMATION_NUM].map(|_| meta.advice_column());

        // let round_constants = meta.fixed_column();
        let range_u8 = meta.fixed_column();

        let instance_1 = meta.instance_column();

        // let selector = meta.selector();

        Sha256DetectionConfig {
            advice_1: advice_1,
            advice_2: advice_2,
            advice_3: advice_3,
            advice_many: advice_many,
            range_u8: range_u8,
            instance_1: instance_1,
            // table16_config: Table16Chip::configure(meta, advice_1, [advice_many[0], advice_many[1], advice_many[2], advice_many[3], advice_many[4], advice_many[5]], advice_2, advice_3, advice_many[6]),
            table16_config: Table16Chip::configure(meta),
            less_than_config: LTChip::configure(meta, advice_1, advice_2, advice_3, range_u8, advice_many, instance_1),
            summation_config: SummationChip::configure(meta, advice_many, advice_1, instance_1),
        }

    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        Table16Chip::load(config.table16_config.clone(), &mut layouter)?;
        let table16_chip = Table16Chip::construct(config.table16_config.clone());
        let less_than_chip = LTChip::construct(config.less_than_config);
        let summation_chip = SummationChip::construct(config.summation_config);
        
        // Table16Chip::load(config.table16_config.clone(), layouter.namespace(|| "load sha256 lookup table"))?;
        
        
        less_than_chip.load_lookup_table(layouter.namespace(|| "load lookup table"))?;
        // table16_chip.load(config.table16_config.clone(), layouter.namespace(|| "load sha256 lookup table"))?;

        // Compute First Random Seed
        let secret_key_bytes = self.secret_key.to_bytes();
        let last_prompt_index_bytes = Fp::from(self.last_prompt_index as u64).to_bytes();
        let mut text_token_index_list_bytes = Fp::from(self.text_token_index_list[0] as u64).to_bytes();
        // let secret_key_str = convert_to_decimal_str(&secret_key_bytes);
        // let mut hash_input1 = [Sha256_Fr::from_str(&secret_key_str).unwrap(), Sha256_Fr::from(self.last_prompt_index as u64)];
        // let mut hash_input1: Vec<u8> = Vec::new();
        // hash_input1.extend_from_slice(&secret_key_bytes);
        // hash_input1.extend_from_slice(&last_prompt_index_bytes);
        // hash_input1.extend_from_slice(&text_token_index_list_bytes);
        // let mut hasher1 = Sha256Crypto::new();
        // hasher1.input(&hash_input1);
        // let mut seed_string = hasher1.result_str();
        // note seed  --> already hex string

        // let mut seed_u64 = process_hex_string(&pad_string(&seed_string, 64));
        // let mut seed = Fp::from_raw(seed_u64);
        // let mut seed_bytes = seed.to_bytes();
        // let mut text_token_index_list_bytes = Fp::from(self.text_token_index_list[0] as u64).to_bytes();

        let mut hash_input2: Vec<u8> = Vec::new();
        // hash_input2.extend_from_slice(&seed_bytes);
        hash_input2.extend_from_slice(&secret_key_bytes);
        hash_input2.extend_from_slice(&last_prompt_index_bytes);
        hash_input2.extend_from_slice(&text_token_index_list_bytes);
        let mut hasher2 = Sha256Crypto::new();
        hasher2.input(&hash_input2);
        let mut seed2_string = hasher2.result_str();
        // note seed_2  --> already hex string

        let mut seed2_u64 = process_hex_string(&pad_string(&seed2_string, 64));
        let mut seed2 = Fp::from_raw(seed2_u64);

        let mut less_than_flag: usize = 0;
        
        let mut threshold_this = self.text_token_hash_threshold[0];

        if seed2 < threshold_this {
            // println!("seed2 < threshold_this");
            less_than_flag = 1;
        }


        // Verify
        let mut green_flag_cell_vec = vec![];
        let mut threshold_cell_vec = vec![];
        let mut token_index_cell_vec = vec![];

        //##########################################################
        let mut hash1_input1_part1 = [BlockWord(Value::known(0)); 8];
        let mut secret_key_str = convert_to_hex_str(&secret_key_bytes);
        let mut secret_key_u32 = process_hex_string_3(&pad_string(&secret_key_str, 64));
        let mut hash1_input1_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(secret_key_u32[i])));
        let mut hash1_input1 = [hash1_input1_part1, hash1_input1_part2].concat();

        let mut hash1_input2_part1 = [BlockWord(Value::known(0)); 8];
        let mut last_prompt_index_str = convert_to_hex_str(&last_prompt_index_bytes);
        let mut last_prompt_index_u32 = process_hex_string_3(&pad_string(&last_prompt_index_str, 64));
        let mut hash1_input2_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(last_prompt_index_u32[i])));
        let mut hash1_input2 = [hash1_input2_part1, hash1_input2_part2].concat();

        let mut hash1_input = [hash1_input1, hash1_input2].concat();
        // Sha256::digest(table16_chip.clone(), layouter.namespace(|| "sha256 hash1"), &hash1_input);


        // let mut hash2_input1_part1 = [BlockWord(Value::known(0)); 8];
        // let mut seed_u32 = process_hex_string_3(&pad_string(&seed_string, 64));
        // let mut hash2_input1_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(seed_u32[i])));
        // let mut hash2_input1 = [hash2_input1_part1, hash2_input1_part2].concat();

        let mut hash2_input2_part1 = [BlockWord(Value::known(0)); 8];
        let mut text_token_index_list_str = convert_to_hex_str(&text_token_index_list_bytes);
        let mut text_token_index_list_u32 = process_hex_string_3(&pad_string(&text_token_index_list_str, 64));
        let mut hash2_input2_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(text_token_index_list_u32[i])));
        let mut hash2_input2 = [hash2_input2_part1, hash2_input2_part2].concat();

        let mut hash2_input = [hash1_input, hash2_input2].concat();
        Sha256::digest(table16_chip.clone(), layouter.namespace(|| "sha256 hash2"), &hash2_input);

        let (mut less_than_input1_cell, mut less_than_input2_cell, mut less_than_flag_cell) = less_than_chip.assign_value_and_is_less_than(layouter.namespace(|| "Judge if Hash_this_token < text_token_hash_threshold[0]"), seed2, threshold_this)?;
        let mut last_prompt_index_cell = layouter.assign_region(
            || "load last prompt index", 
            |mut region| {
                region.assign_advice(
                    || "laod last prompt index",
                    config.advice_1, 
                    0, 
                || Value::known(Fp::from(self.last_prompt_index as u64)))
            }
        )?;
        let mut token_index_cell = layouter.assign_region(
            || "load text token index", 
            |mut region| {
                region.assign_advice(
                    || "laod text token index",
                    config.advice_2, 
                    0, 
                || Value::known(Fp::from(self.text_token_index_list[0] as u64)))
            }
        )?;

        token_index_cell_vec.push(token_index_cell.clone());
        green_flag_cell_vec.push(less_than_flag_cell.clone());
        threshold_cell_vec.push(less_than_input2_cell.clone());

        less_than_chip.is_less_than_expr_greater_than(layouter.namespace(|| "greater than"), &less_than_input2_cell, &less_than_input1_cell, &less_than_flag_cell, threshold_this, seed2)?;

        for round in 0..self.text_token_value_list_length-1 {
            // let mut hash_input1: Vec<u8> = Vec::new();
            // hash_input1.extend_from_slice(&secret_key_bytes);

            let mut text_token_index_list_bytes = Fp::from(self.text_token_index_list[round] as u64).to_bytes();
            // hash_input1.extend_from_slice(&text_token_index_list_bytes);
            // let mut hasher1 = Sha256Crypto::new();
            // hasher1.input(&hash_input1);
            // seed_string = hasher1.result_str();
            // seed_u64 = process_hex_string(&pad_string(&seed_string, 64));
            // seed = Fp::from_raw(seed_u64);
            // seed_bytes = seed.to_bytes();

            let mut hash_input2: Vec<u8> = Vec::new();
            // hash_input2.extend_from_slice(&seed_bytes);
            let mut text_token_index_list_2_bytes = Fp::from(self.text_token_index_list[round+1] as u64).to_bytes();
            hash_input2.extend_from_slice(&secret_key_bytes);
            hash_input2.extend_from_slice(&text_token_index_list_bytes);
            hash_input2.extend_from_slice(&text_token_index_list_2_bytes);
            let mut hasher2 = Sha256Crypto::new();
            hasher2.input(&hash_input2);
            seed2_string = hasher2.result_str();
            seed2_u64 = process_hex_string(&pad_string(&seed2_string, 64));
            seed2 = Fp::from_raw(seed2_u64);

            threshold_this = self.text_token_hash_threshold[round+1];
            if seed2 < threshold_this {
                // println!("seed2 < threshold_this");
                less_than_flag = 1;
            } else {
                less_than_flag = 0;
            }




            hash1_input1_part1 = [BlockWord(Value::known(0)); 8];
            secret_key_str = convert_to_hex_str(&secret_key_bytes);
            secret_key_u32 = process_hex_string_3(&pad_string(&secret_key_str, 64));
            hash1_input1_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(secret_key_u32[i])));
            hash1_input1 = [hash1_input1_part1, hash1_input1_part2].concat();

            hash1_input2_part1 = [BlockWord(Value::known(0)); 8];
            let mut text_token_index_list_str = convert_to_hex_str(&text_token_index_list_bytes);
            let mut text_token_index_list_u32 = process_hex_string_3(&pad_string(&text_token_index_list_str, 64));
            hash1_input2_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(text_token_index_list_u32[i])));
            hash1_input2 = [hash1_input2_part1, hash1_input2_part2].concat();

            hash1_input = [hash1_input1, hash1_input2].concat();
            // Sha256::digest(table16_chip.clone(), layouter.namespace(|| format!("sha256 hash1_{}", round)), &hash1_input);

            // hash2_input1_part1 = [BlockWord(Value::known(0)); 8];
            // seed_u32 = process_hex_string_3(&pad_string(&seed_string, 64));
            // hash2_input1_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(seed_u32[i])));
            // hash2_input1 = [hash2_input1_part1, hash2_input1_part2].concat();

            hash2_input2_part1 = [BlockWord(Value::known(0)); 8];
            let mut text_token_index_list_2_bytes = Fp::from(self.text_token_index_list[round+1] as u64).to_bytes();
            let mut text_token_index_list_2_str = convert_to_hex_str(&text_token_index_list_2_bytes);
            let mut text_token_index_list_2_u32 = process_hex_string_3(&pad_string(&text_token_index_list_2_str, 64));
            hash2_input2_part2 = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| BlockWord(Value::known(text_token_index_list_2_u32[i])));
            hash2_input2 = [hash2_input2_part1, hash2_input2_part2].concat();

            hash2_input = [hash1_input, hash2_input2].concat();
            Sha256::digest(table16_chip.clone(), layouter.namespace(|| format!("sha256 hash2_{}", round)), &hash2_input);

            (less_than_input1_cell, less_than_input2_cell, less_than_flag_cell) = less_than_chip.assign_value_and_is_less_than(layouter.namespace(|| format!("Judge if Hash_this_token < text_token_hash_threshold[{}]", round+1)), seed2, threshold_this)?;
            let mut token_index_cell = layouter.assign_region(
                || "load text token index", 
                |mut region| {
                    region.assign_advice(
                        || "load text token index",
                        config.advice_1, 
                        0, 
                    || Value::known(Fp::from(self.text_token_index_list[round+1] as u64)))
                }
            )?;
            token_index_cell_vec.push(token_index_cell.clone());
            green_flag_cell_vec.push(less_than_flag_cell.clone());
            threshold_cell_vec.push(less_than_input2_cell.clone());

            less_than_chip.is_less_than_expr_greater_than(layouter.namespace(|| format!("greater than_{}", round)), &less_than_input2_cell, &less_than_input1_cell, &less_than_flag_cell, threshold_this, seed2)?;
        
        }

        let count_result_cell = summation_chip.assign_multiple_value_and_summation(
            layouter.namespace(|| "Count the number of green flags"), 
            &green_flag_cell_vec, 
            green_flag_cell_vec.len(),
        )?;

        let mut instance_index: usize = 0;
        // Expose Public Inputs
        // add_chip.expose_public(layouter.namespace(|| "Expose Count Result"), &count_result_cell, 0)?;
        summation_chip.expose_public(layouter.namespace(|| "Expose Count Result"), &count_result_cell, 0)?;
        instance_index = instance_index + 1;
        println!("count_result_cell: {:?}", count_result_cell);

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
        for round in 0..threshold_cell_vec.len() {
            summation_chip.expose_public(layouter.namespace(|| format!("Expose Threshold[{:?}]", round)), &threshold_cell_vec[round], instance_index)?;
            instance_index = instance_index + 1;
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
        // let mut temp = Fp::from(rng.gen_range(0..1000) as u64);
        // let mut temp2 = Fp::from(rng.gen_range(0..1000) as u64);
        let mut temp = Fp::from(1234 as u64);
        let mut temp2 = Fp::from(4567 as u64);
        let mut temp_bytes = temp.to_bytes();
        let mut temp2_bytes = temp2.to_bytes();
        let mut hash_input: Vec<u8> = Vec::new();
        hash_input.extend_from_slice(&temp_bytes);
        hash_input.extend_from_slice(&temp2_bytes);
        // temp = sub_fp(temp.clone());
        // let mut threshold_temp = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        // .hash([temp, temp2]);
        let mut hasher = Sha256Crypto::new();
        hasher.input(&hash_input);
        let mut hash_string = hasher.result_str();
        let mut hash_u64 = process_hex_string(&pad_string(&hash_string, 64));
        let mut threshold_temp = Fp::from_raw(hash_u64);
        // let mut threshold_temp = multi_mimc7_hash_bn256_new(vec![temp, temp2]);
        text_token_hash_threshold.push(threshold_temp);
    }

    let mut count = 0;
    // let mut hash_input1 = [Fp::from(serect_key), Fp::from(last_prompt_index as u64)].to_vec();
    let serect_key_bytes = Fp::from(serect_key).to_bytes();
    let last_prompt_index_bytes = Fp::from(last_prompt_index as u64).to_bytes();
    // let mut hash_input1: Vec<u8> = Vec::new();
    // hash_input1.extend_from_slice(&serect_key_bytes);
    // hash_input1.extend_from_slice(&last_prompt_index_bytes);
    // let mut hasher1 = Sha256Crypto::new();
    // hasher1.input(&hash_input1);
    // let mut seed_string = hasher1.result_str();
    // let mut seed_u64 = process_hex_string(&pad_string(&seed_string, 64));
    // let mut seed = Fp::from_raw(seed_u64);

    // let mut real_index = (text_token_index_list[0] / DEC_NUM) as usize;
    // let mut mod_index = (text_token_index_list[0] % DEC_NUM) as usize;
    // println!("real_index = {:?}", real_index);
    // println!("mod_index = {:?}", mod_index);
    let mut hash_input2: Vec<u8> = Vec::new();
    hash_input2.extend_from_slice(&serect_key_bytes);
    hash_input2.extend_from_slice(&last_prompt_index_bytes);
    // hash_input2.extend_from_slice(&seed.to_bytes());
    let mut text_token_index_list_bytes = Fp::from(text_token_index_list[0] as u64).to_bytes();
    hash_input2.extend_from_slice(&text_token_index_list_bytes);
    let mut hasher2 = Sha256Crypto::new();
    hasher2.input(&hash_input2);
    let mut seed2_string = hasher2.result_str();
    let mut seed2_u64 = process_hex_string(&pad_string(&seed2_string, 64));
    let mut seed2 = Fp::from_raw(seed2_u64);

    // let mut hash_input2 = [seed, Fp::from(text_token_index_list[0] as u64)].to_vec();
    // let mut hash_input2 = [seed, Fp::from(real_index as u64)].to_vec();
    // let mut seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
    // .hash(hash_input2);
    // let mut seed2 = multi_mimc7_hash_bn256_new(hash_input2);
    // (seed2, _) = dec_num_fp(seed2, mod_index, DEC_NUM);
    // seed2 = sub_fp(seed2.clone());

    let mut less_than_flag: usize = 0;
    let mut threshold_this = text_token_hash_threshold[0];
    if seed2 < threshold_this {
        less_than_flag = 1;
    }
    count += less_than_flag;
    for i in 0..text_token_value_list_length-1 {
        let mut text_token_index_list_bytes = Fp::from(text_token_index_list[i] as u64).to_bytes();
        // let mut hash_input1: Vec<u8> = Vec::new();
        // hash_input1.extend_from_slice(&serect_key_bytes);
        // hash_input1.extend_from_slice(&text_token_index_list_bytes);
        // let mut hasher1 = Sha256Crypto::new();
        // hasher1.input(&hash_input1);
        // let mut seed_string = hasher1.result_str();
        // let mut seed_u64 = process_hex_string(&pad_string(&seed_string, 64));
        // let mut seed = Fp::from_raw(seed_u64);
        // hash_input1 = [Fp::from(serect_key), Fp::from(text_token_index_list[i] as u64)].to_vec();
        // seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        // .hash(hash_input1);
        // seed = sub_fp(seed.clone());
        // seed = multi_mimc7_hash_bn256_new(hash_input1);

        let mut hash_input2: Vec<u8> = Vec::new();
        hash_input2.extend_from_slice(&serect_key_bytes);
        hash_input2.extend_from_slice(&text_token_index_list_bytes);
        // hash_input2.extend_from_slice(&seed.to_bytes());
        let mut text_token_index_list_2_bytes = Fp::from(text_token_index_list[i+1] as u64).to_bytes();
        hash_input2.extend_from_slice(&text_token_index_list_2_bytes);
        let mut hasher2 = Sha256Crypto::new();
        hasher2.input(&hash_input2);
        let mut seed2_string = hasher2.result_str();
        let mut seed2_u64 = process_hex_string(&pad_string(&seed2_string, 64));
        let mut seed2 = Fp::from_raw(seed2_u64);

        // real_index = (text_token_index_list[i+1] / DEC_NUM) as usize;
        // mod_index = (text_token_index_list[i+1] % DEC_NUM) as usize;
        // hash_input2 = [seed, Fp::from(text_token_index_list[i+1] as u64)].to_vec();
        // hash_input2 = [seed, Fp::from(real_index as u64)].to_vec();
        // seed2 = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
        // .hash(hash_input2);
        // seed2 = multi_mimc7_hash_bn256_new(hash_input2);
        // (seed2, _) = dec_num_fp(seed2, mod_index, DEC_NUM);
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

// fn count_greenlist_num(max_token_num: usize, secret_key: usize, text_token_index_list: &Vec<usize>, threshold_list: &Vec<String>) -> usize {
//     let mut count = 0;
//     let secret_key_fp = Fp::from(secret_key as u64);
//     // println!("text_token_index_list_len = {:?}", text_token_index_list.len());
//     // println!("threshold_list_len = {:?}", threshold_list.len());
//     for i in 0..max_token_num {
//         let seed_hash_inputs = [secret_key_fp, Fp::from(text_token_index_list[i] as u64)];
//         let mut seed = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
//             .hash(seed_hash_inputs);
//         seed = sub_fp(seed.clone());
//         let this_token_hash_inputs = [seed, Fp::from(text_token_index_list[i + 1] as u64)];
//         // println!("text_token_index_list[i+1] = {:?}", text_token_index_list[i+1]);
//         let mut this_token_hash = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
//             .hash(this_token_hash_inputs);
//         this_token_hash = sub_fp(this_token_hash.clone());
//         let pad_threshold = pad_string(&threshold_list[i], 64);
//         let raw_threshold = process_hex_string(&pad_threshold);
//         let threshold = Fp::from_raw(raw_threshold);
//         // println!("this_token_hash = {:?}", this_token_hash);
//         // println!("threshold = {:?}", threshold);
//         if this_token_hash <= threshold {
//             // println!("----------------------------------------------------------------------------this_token_hash <= threshold");
//             count += 1;
//         }
//     }

//     count
// }

fn compute_z_score(greenlist_count: usize, gamma: f64, total_token_num: usize) -> f64 {
    let expected_count = gamma;
    let numer = greenlist_count as f64 - expected_count * total_token_num as f64;
    let denom = (total_token_num as f64 * expected_count * (1.0 - expected_count)).sqrt();
    let z = numer / denom;
    z
}

use ark_std::start_timer;
use std::path::Path;
use ark_std::end_timer;
fn main() {
    let vocabulary_size: usize = 200;
    let (serect_key, last_prompt_index, text_token_index_list, text_token_value_list_length, text_token_hash_threshold, count) = generate_test_case(vocabulary_size);
    // println!("serect_key = {:?}", serect_key);
    // println!("last_prompt_index = {:?}", last_prompt_index);
    // println!("text_token_index_list = {:?}", text_token_index_list);
    // println!("text_token_value_list_length = {:?}", text_token_value_list_length);
    // println!("text_token_hash_threshold = {:?}", text_token_hash_threshold);
    // println!("compare_range_length = {:?}", compare_range_length);
    println!("count = {:?}", count);

    let circuit = Sha256DetectionCircuit {
        secret_key: serect_key,
        last_prompt_index: last_prompt_index,
        text_token_index_list: text_token_index_list.clone(),
        text_token_value_list_length: text_token_value_list_length,
        text_token_hash_threshold: text_token_hash_threshold.clone(),
    };

    let mut public_inputs = vec![];
    public_inputs.push(Fp::from(count as u64));
    public_inputs.push(Fp::from(last_prompt_index as u64));
    for i in 0..text_token_value_list_length {
        public_inputs.push(Fp::from(text_token_index_list[i] as u64));
    }
    for i in 0..text_token_value_list_length {
        public_inputs.push(text_token_hash_threshold[i]);
    }
    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(25, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    let mut k = select_suitable_k_value(total_row as i32);
    println!("k = {:?}", k);
    // k = k + 1;

    // let mock_start_time = Instant::now(); // Start time
    // let correct_prover = MockProver::run(
    //     // 15,
    //     k,
    //     &circuit,
    //     vec![public_inputs.clone()],
    //     // vec![correct_public_input.clone()],
    // )
    // .unwrap();
    // // println!("MockProver:{:?}", correct_prover);
    // let mock_elapsed_time = mock_start_time.elapsed();
    

    // correct_prover.assert_satisfied();
    // println!("success");
    // println!("Running Mock took {:.4} seconds.", mock_elapsed_time.as_millis() as f64 / 1000.0);
    let params_path_str = format!("./sha256_params_k_{}", k);
    let params_path = Path::new(params_path_str.as_str());
    if File::open(params_path).is_err() {
        let timer_get_param = start_timer!(|| "get param");
        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        let mut buf = Vec::new();

        params.write(&mut buf).expect("Failed to write params");
        let mut file = File::create(params_path).expect("Failed to create sha256_params");

        file.write_all(&buf[..])
            .expect("Failed to write params to file");
        end_timer!(timer_get_param);
    }

    // let srs_path = "${PVMark_EXTERNAL_DATA_ROOT:-external}".to_string() + &k.to_string();
    // let params: Params<Bn256> = Params::new(k);
    let params = load_params_cmd_2(params_path.into()).expect("load_params_cmd should not fail");
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
    // create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, Blake2bWrite<_, _, _>, _>(
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverGWC<'_, Bn256>,
        Challenge255<G1Affine>,
        _,
        Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
        _,
    >(
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

    let proof_path = "${PVMark_EXTERNAL_DATA_ROOT:-external}";

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