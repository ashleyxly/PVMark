use ff::PrimeField;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};
use halo2_proofs::poly::Rotation;
use halo2_proofs::arithmetic::*;
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::range;
use rayon::slice::ParallelSlice;

use std::marker::PhantomData;
use super::add_chip::{AddChip, AddConfig, self};

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



fn bytes_to_binary(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    for &byte in bytes.iter() {
        let mut temp = Vec::new();
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            temp.push(bit);
        }
        temp.reverse();
        result.extend(temp);
    }

    result
}



#[derive(Debug, Clone)]
pub struct RangeCheckConfig {
    pub bit_input: Column<Advice>, // z_i * 2^i
    pub green_list_flag: Column<Advice>,
    pub pow_two: Column<Advice>,
    pub instance: Column<Instance>,
    pub selector: Selector,
    pub add_config: AddConfig,
}

#[derive(Debug, Clone)]
pub struct RangeCheckChip {
    config: RangeCheckConfig,
    _marker: PhantomData<Fp>,
}

impl RangeCheckChip {
    pub fn construct(config: RangeCheckConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }


    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        // advice: [Column<Advice>; 3],
        advice_1: Column<Advice>,
        advice_2: Column<Advice>,
        advice_3: Column<Advice>,
        instance: Column<Instance>,
    ) -> RangeCheckConfig {
        // let bit_input = meta.advice_column();
        // let green_list_flag = meta.advice_column();
        // let pow_two = meta.advice_column();
        // let instance = meta.instance_column();
        // let bit_input = advice[0];
        // let green_list_flag = advice[1];
        // let pow_two = advice[2];
        // let instance = instance;

        let selector = meta.selector();

        meta.enable_equality(instance);
        // meta.enable_equality(pow_two);
        // meta.enable_equality(green_list_flag);
        // meta.enable_equality(bit_input);
        meta.enable_equality(advice_1);
        meta.enable_equality(advice_2);
        meta.enable_equality(advice_3);

        let add_config = AddChip::configure(meta, advice_1, advice_2, advice_3, instance);
        // let add_config = AddChip::configure(meta);
        // let add_config = AddChip::configure(meta, bit_input, green_list_flag, pow_two, instance);
        // let add_config = AddChip::configure(meta, advice[0], advice[1], advice[2], instance);
        // let add_config = AddChip::configure(meta, 
        //     |meta| meta.query_advice(bit_input), 
        //     |meta| meta.query_advice(green_list_flag), 
        //     |meta| meta.query_advice(pow_two), 
        //     instance);

        meta.create_gate("OneOrZero_Test in RangeCheckChip", |meta| {
            // let bit_input = meta.query_advice(bit_input, Rotation::cur());
            // let green_list_flag = meta.query_advice(green_list_flag, Rotation::cur());
            // let pow_two = meta.query_advice(pow_two, Rotation::cur());
            let bit_input = meta.query_advice(advice_1, Rotation::cur());
            let green_list_flag = meta.query_advice(advice_2, Rotation::cur());
            let pow_two = meta.query_advice(advice_3, Rotation::cur());
            // let instance = meta.query_instance(instance, Rotation::cur());
            let s = meta.query_selector(selector);
            let temp = pow_two - bit_input.clone();

            vec![s * green_list_flag * bit_input * temp]
        });

        RangeCheckConfig {
            // bit_input,
            // green_list_flag,
            // pow_two,
            // advice: [advice[0], advice[1], advice[2]],
            bit_input: advice_1,
            green_list_flag: advice_2,
            pow_two: advice_3,
            instance,
            selector,
            add_config,
        }
    }

    pub fn assign_test(
        &self,
        mut layouter: impl Layouter<Fp>,
        test_value1: Fp,
        test_value2: Fp,
        test_value3: Fp,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "assign test",
        |mut region| {
            let test_value1_cell = region.assign_advice(
                || "test_value1", 
                self.config.bit_input, 
                0,
                || Value::known(test_value1)
            )?;
            let test_value2_cell = region.assign_advice(
                || "test_value2", 
                self.config.green_list_flag, 
                0,
                || Value::known(test_value2)
            )?;
            let test_value3_cell = region.assign_advice(
                || "test_value3", 
                self.config.pow_two, 
                0,
                || Value::known(test_value3)
            )?;
            Ok(())
        }
        )
    }

    pub fn assign_and_bit_check(
        &self,
        mut layouter: impl Layouter<Fp>,
        target_value: Fp,
        range_length: usize,
        check_success_flag: usize, // 0 or 1
    ) -> Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error> {

        println!("Function assign_and_bit_check is called");

        let target_value_bytes = target_value.to_repr();
        let target_value_binary = bytes_to_binary(&target_value_bytes);
        let mut target_value_bit_input = Vec::new(); // z_i * 2^i
        for i in 0..range_length {
            let temp = Fp::from((target_value_binary[i] as u64 * 2u64.pow(i as u32)) as u64);
            target_value_bit_input.push(temp);
        }
        println!("target_value_bit_input = {:?}", target_value_bit_input);
        let mut bit_input_cell_vec = Vec::new();

        layouter.assign_region(
            || "Range Check",
            |mut region| {
                println!("Function assign_region is called");
                println!("range_length = {}", range_length);
                for i in 0..range_length {
                    println!("i = {}", i);
                    self.config.selector.enable(&mut region, i)?;
                    let mut bit_input_cell = region.assign_advice(
                        || "bit_input", 
                        self.config.bit_input, 
                        i,
                        || Value::known(target_value_bit_input[i])
                    )?;
                    // );
                    bit_input_cell_vec.push(bit_input_cell);
                    println!("bit_input_cell_vec.len() = {}", bit_input_cell_vec.len());

                    let mut green_list_flag_cell = region.assign_advice(
                        || "green_list_flag", 
                        self.config.green_list_flag, 
                        i,
                        || Value::known(Fp::from(check_success_flag as u64))
                    )?;
                    // );
                    let mut pow_two_cell = region.assign_advice(
                        || "pow_two", 
                        self.config.pow_two, 
                        i,
                        || Value::known(Fp::from(2u64.pow(i as u32)))
                    )?;
                    // );
                }
                
                println!("Finish Loop --- bit_input_cell_vec.len() = {}", bit_input_cell_vec.len());
                // self.assign_and_summation(&mut region, &bit_input_cell_vec);
                // Ok(())

                Ok(bit_input_cell_vec.clone())
            }
        )
        // Ok(bit_input_cell_vec)
        // layouter.assign_region(
        //     || "Range Check",
        //     |mut region|)


       
    }

    pub fn assign_and_bit_check_2(
        &self,
        mut layouter: impl Layouter<Fp>,
        target_value: Fp,
        range_length: usize,
        check_success_flag: usize, // 0 or 1
    // ) -> Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error> {
    ) -> Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error> {

        // println!("Function assign_and_bit_check is called");

        let target_value_bytes = target_value.to_repr();
        println!("target_value_bytes = {:?}", target_value_bytes);
        let target_value_binary = bytes_to_binary(&target_value_bytes);
        println!("target_value_binary = {:?}", target_value_binary);
        let mut target_value_bit_input = Vec::new(); // z_i * 2^i
        for i in 0..range_length {
            let temp_pow_two = BigUint::from(2u64).pow(i as u32);
            let hex_string = format!("{:X}", temp_pow_two);
            let pad_hex_string = pad_string(&hex_string, 64);
            let temp_pow_two_fp = Fp::from_raw(process_hex_string(&pad_hex_string));


            // let temp = Fp::from((target_value_binary[i] as u64 * 2u64.pow(i as u32)) as u64);
            let temp = Fp::from(target_value_binary[i] as u64) * temp_pow_two_fp;
            target_value_bit_input.push(temp);
        }
        println!("target_value_bit_input.len() = {:?}", target_value_bit_input.len());
        // println!("target_value_bit_input = {:?}", target_value_bit_input);
        // let mut bit_input_cell_vec = Vec::new();

        let bit_inputs_cell_vec_res =  layouter.assign_region(
            || "Range Check Function 2", 
            |mut region| -> Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error> {
                // println!("Function assign_and_bit_check_2 ----- assign_region is called");
                let bit_input_cell_res = target_value_bit_input
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    region.assign_advice(
                        || "bit_input", 
                        self.config.bit_input, 
                        i,
                        || Value::known(target_value_bit_input[i])
                    )
                })
                .collect::<Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error>>();

                for k in 0..range_length {
                    self.config.selector.enable(&mut region, k)?;
                    let mut green_list_flag_cell = region.assign_advice(
                        || "green_list_flag", 
                        self.config.green_list_flag, 
                        k,
                        || Value::known(Fp::from(check_success_flag as u64))
                    )?;
                    let temp_pow_two = BigUint::from(2u64).pow(k as u32);
                    let hex_string = format!("{:X}", temp_pow_two);
                    let pad_hex_string = pad_string(&hex_string, 64);
                    let temp_pow_two_fp = Fp::from_raw(process_hex_string(&pad_hex_string));
                    let mut pow_two_cell = region.assign_advice(
                        || "pow_two", 
                        self.config.pow_two, 
                        k,
                        // || Value::known(Fp::from(2u64.pow(k as u32)))
                        || Value::known(temp_pow_two_fp)
                    )?;
                }

                // self.assign_and_summation(&mut region, bit_input_cell_res);

                Ok(bit_input_cell_res?.try_into().unwrap())       
            }
        );
        Ok(bit_inputs_cell_vec_res?)

        // let output_cell = self.assign_and_summation(layouter.namespace(|| "Summation in RangeCheck Chip"), &bit_inputs_cell_vec_res?)?;
        // let mut temp = output_cell.value().map(|x| x.to_owned());
        // let mut temp_value = temp.as_mut().unwrap();
        // self.expose_public(layouter.namespace(|| "Check Instance Equality"), &output_cell, 0)?;
        // Ok(())

        // layouter.assign_region(
        //     || "Range Check",
        //     |mut region|)


       
    }



    pub fn assign_and_bit_check_3(
        &self,
        mut layouter: impl Layouter<Fp>,
        target_value: Fp,
        range_length: usize,
        check_success_flag: usize, // 0 or 1
    // ) -> Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error> {
    ) -> Result<(Vec<AssignedCell<Fp, Fp>>, Vec<AssignedCell<Fp, Fp>>), halo2_proofs::plonk::Error> {

        // println!("Function assign_and_bit_check is called");
        // println!("target_value = {:?}", target_value);

        let target_value_bytes = target_value.to_repr();
        // println!("target_value_bytes = {:?}", target_value_bytes);
        let target_value_binary = bytes_to_binary(&target_value_bytes);
        // println!("target_value_binary = {:?}", target_value_binary);
        let mut target_value_bit_input = Vec::new(); // z_i * 2^i
        let mut flag_vec = Vec::new();

        for i in 0..range_length {
            let temp_pow_two = BigUint::from(2u64).pow(i as u32);
            let hex_string = format!("{:X}", temp_pow_two);
            let pad_hex_string = pad_string(&hex_string, 64);
            let temp_pow_two_fp = Fp::from_raw(process_hex_string(&pad_hex_string));


            // let temp = Fp::from((target_value_binary[i] as u64 * 2u64.pow(i as u32)) as u64);
            let temp = Fp::from(target_value_binary[i] as u64) * temp_pow_two_fp;
            target_value_bit_input.push(temp);

            flag_vec.push(check_success_flag);
        }
        // println!("target_value_bit_input.len() = {:?}", target_value_bit_input.len());
        // println!("target_value_bit_input = {:?}", target_value_bit_input);
        // let mut bit_input_cell_vec = Vec::new();

        // let bit_inputs_cell_vec_res = layouter.assign_region(
        layouter.assign_region(   
            || "Range Check Function 2", 
            |mut region| {
                // println!("Function assign_and_bit_check_2 ----- assign_region is called");
                let bit_input_cell_res = target_value_bit_input
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    region.assign_advice(
                        || "bit_input", 
                        self.config.bit_input, 
                        i,
                        || Value::known(target_value_bit_input[i])
                    )
                })
                .collect::<Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error>>();

                // temp init
                // let mut green_list_flag_cell: AssignedCell<Fp, Fp> = bit_input_cell_res?[0].clone();
                let green_list_flag_cell_res = flag_vec
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    region.assign_advice(
                        || "green_list_flag", 
                        self.config.green_list_flag, 
                        i,
                        || Value::known(Fp::from(check_success_flag as u64))
                    )
                })
                .collect::<Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error>>();

                for k in 0..range_length {
                    self.config.selector.enable(&mut region, k)?;
                    // let mut green_list_flag_cell = region.assign_advice(
                    //     || "green_list_flag", 
                    //     self.config.green_list_flag, 
                    //     k,
                    //     || Value::known(Fp::from(check_success_flag as u64))
                    // )?;
                    // if k == 0 {
                    //     bit_input_cell_res?.push(green_list_flag_cell);
                    // }
                    // green_list_flag_cell = region.assign_advice(
                    //     || "green_list_flag", 
                    //     self.config.green_list_flag, 
                    //     k,
                    //     || Value::known(Fp::from(check_success_flag as u64))
                    // )?;
                    let temp_pow_two = BigUint::from(2u64).pow(k as u32);
                    let hex_string = format!("{:X}", temp_pow_two);
                    let pad_hex_string = pad_string(&hex_string, 64);
                    let temp_pow_two_fp = Fp::from_raw(process_hex_string(&pad_hex_string));
                    let mut pow_two_cell = region.assign_advice(
                        || "pow_two", 
                        self.config.pow_two, 
                        k,
                        // || Value::known(Fp::from(2u64.pow(k as u32)))
                        || Value::known(temp_pow_two_fp)
                    )?;
                }

                Ok((bit_input_cell_res?, green_list_flag_cell_res?).try_into().unwrap())

                // self.assign_and_summation(&mut region, bit_input_cell_res);
                // let green_list_flag_cell = green_list_flag_cell_res?[0].clone();
                // let clone_bit_input_cell_res = bit_input_cell_res?.clone();
                // Ok((bit_input_cell_res?.try_into().unwrap()), green_list_flag_cell)) 
                // Ok((bit_input_cell_res, green_list_flag_cell_res)?.try_into().unwrap())      
            },
        )
       
    }

    // used in greater_than_chip  check_success_flag_cell input
    // pub fn assign_and_bit_check_4(
    //     &self,
    //     mut layouter: impl Layouter<Fp>,
    //     target_value: Fp,
    //     range_length: usize,
    //     // check_success_flag: usize, // 0 or 1
    //     check_success_flag_cell: &AssignedCell<Fp, Fp>,
    // // ) -> Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error> {
    // ) -> Result<(Vec<AssignedCell<Fp, Fp>>, Vec<AssignedCell<Fp, Fp>>), halo2_proofs::plonk::Error> {

    //     // println!("Function assign_and_bit_check is called");

    //     let target_value_bytes = target_value.to_repr();
    //     println!("target_value_bytes = {:?}", target_value_bytes);
    //     let target_value_binary = bytes_to_binary(&target_value_bytes);
    //     println!("target_value_binary = {:?}", target_value_binary);
    //     let mut target_value_bit_input = Vec::new(); // z_i * 2^i
    //     let mut flag_vec = Vec::new();

    //     for i in 0..range_length {
    //         let temp_pow_two = BigUint::from(2u64).pow(i as u32);
    //         let hex_string = format!("{:X}", temp_pow_two);
    //         let pad_hex_string = pad_string(&hex_string, 64);
    //         let temp_pow_two_fp = Fp::from_raw(process_hex_string(&pad_hex_string));


    //         // let temp = Fp::from((target_value_binary[i] as u64 * 2u64.pow(i as u32)) as u64);
    //         let temp = Fp::from(target_value_binary[i] as u64) * temp_pow_two_fp;
    //         target_value_bit_input.push(temp);

    //         // flag_vec.push(check_success_flag);
    //     }
    //     println!("target_value_bit_input.len() = {:?}", target_value_bit_input.len());
    //     // println!("target_value_bit_input = {:?}", target_value_bit_input);
    //     // let mut bit_input_cell_vec = Vec::new();

    //     // let bit_inputs_cell_vec_res = layouter.assign_region(
    //     layouter.assign_region(   
    //         || "Range Check Function 2", 
    //         |mut region| {
    //             // println!("Function assign_and_bit_check_2 ----- assign_region is called");
    //             let bit_input_cell_res = target_value_bit_input
    //             .iter()
    //             .enumerate()
    //             .map(|(i, x)| {
    //                 region.assign_advice(
    //                     || "bit_input", 
    //                     self.config.bit_input, 
    //                     i,
    //                     || Value::known(target_value_bit_input[i])
    //                 )
    //             })
    //             .collect::<Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error>>();

    //             // temp init
    //             // let mut green_list_flag_cell: AssignedCell<Fp, Fp> = bit_input_cell_res?[0].clone();
    //             let green_list_flag_cell_res = flag_vec
    //             .iter()
    //             .enumerate()
    //             .map(|(i, x)| {
    //                 region.assign_advice(
    //                     || "green_list_flag", 
    //                     self.config.green_list_flag, 
    //                     i,
    //                     || Value::known(Fp::from(check_success_flag as u64))
    //                 )
    //             })
    //             .collect::<Result<Vec<AssignedCell<Fp, Fp>>, halo2_proofs::plonk::Error>>();

    //             for k in 0..range_length {
    //                 self.config.selector.enable(&mut region, k)?;
    //                 // let mut green_list_flag_cell = region.assign_advice(
    //                 //     || "green_list_flag", 
    //                 //     self.config.green_list_flag, 
    //                 //     k,
    //                 //     || Value::known(Fp::from(check_success_flag as u64))
    //                 // )?;
    //                 // if k == 0 {
    //                 //     bit_input_cell_res?.push(green_list_flag_cell);
    //                 // }
    //                 // green_list_flag_cell = region.assign_advice(
    //                 //     || "green_list_flag", 
    //                 //     self.config.green_list_flag, 
    //                 //     k,
    //                 //     || Value::known(Fp::from(check_success_flag as u64))
    //                 // )?;
    //                 let temp_pow_two = BigUint::from(2u64).pow(k as u32);
    //                 let hex_string = format!("{:X}", temp_pow_two);
    //                 let pad_hex_string = pad_string(&hex_string, 64);
    //                 let temp_pow_two_fp = Fp::from_raw(process_hex_string(&pad_hex_string));
    //                 let mut pow_two_cell = region.assign_advice(
    //                     || "pow_two", 
    //                     self.config.pow_two, 
    //                     k,
    //                     // || Value::known(Fp::from(2u64.pow(k as u32)))
    //                     || Value::known(temp_pow_two_fp)
    //                 )?;
    //             }

    //             Ok((bit_input_cell_res?, green_list_flag_cell_res?).try_into().unwrap())

    //             // self.assign_and_summation(&mut region, bit_input_cell_res);
    //             // let green_list_flag_cell = green_list_flag_cell_res?[0].clone();
    //             // let clone_bit_input_cell_res = bit_input_cell_res?.clone();
    //             // Ok((bit_input_cell_res?.try_into().unwrap()), green_list_flag_cell)) 
    //             // Ok((bit_input_cell_res, green_list_flag_cell_res)?.try_into().unwrap())      
    //         },
    //     )
       
    // }


    pub fn assign_and_summation(
        &self,
        mut layouter: impl Layouter<Fp>,
        bit_inputs: &Vec<AssignedCell<Fp, Fp>>,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        // println!("Function assign_and_summation is called");
        let add_chip = AddChip::construct(self.config.add_config.clone());
        // println!("assign_and_summation function ----- bit_inputs.len() = {}", bit_inputs.len());
        let add_result_cell = add_chip.assign_multiple_value_and_summation_3(
            layouter,
            bit_inputs,
            bit_inputs.len(),
        )?;
        Ok(add_result_cell)
        
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }
    
    
}