use clap::builder::styling::Color;
use ff::PrimeField;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};
use halo2_proofs::poly::Rotation;
use halo2_proofs::arithmetic::*;
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::range;
use rayon::slice::ParallelSlice;

use std::ffi::NulError;
use std::marker::PhantomData;
use super::range_check_chip::{RangeCheckChip, RangeCheckConfig};
// use super::add_chip::{AddChip, AddConfig};



const NUM_LENGTH: usize = 254;

//Judge X1 < X2
#[derive(Debug, Clone)]
pub struct LessThanConfig {
    pub compare_input1: Column<Advice>, // X1
    pub compare_input2: Column<Advice>, // X2
    pub compare_input3: Column<Advice>, // X3
    pub instance: Column<Instance>, //public
    pub selector: Selector,
    pub range_check_config: RangeCheckConfig,
}

#[derive(Debug, Clone)]
pub struct LessThanChip {
    config: LessThanConfig,
    _marker: PhantomData<Fp>,
    
}

impl LessThanChip {
    pub fn construct(config: LessThanConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        advice_input1: Column<Advice>,
        advice_input2: Column<Advice>,
        advice_input3: Column<Advice>,
        instance: Column<Instance>,
    ) -> LessThanConfig {
        meta.enable_equality(advice_input1);
        meta.enable_equality(advice_input2);
        meta.enable_equality(advice_input3);
        meta.enable_equality(instance);
        
        
        let selector = meta.selector();
        // let advice_input3 = meta.advice_column();
        // let add_config = AddChip::configure(meta, advice_input1, advice_input2, advice_input3, instance);
        let range_check_config = RangeCheckChip::configure(meta, advice_input1, advice_input2, advice_input3, instance);

        let mod_value_fp = Fp::from_raw([
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);
        let pow_254_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x4000000000000000,
            ]
        );
        let pow_253_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x2000000000000000,
            ]
        );
        let pow_252_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x1000000000000000,
            ]
        );
        meta.create_gate("X3=X1-X2", |meta| {
            let x1 = meta.query_advice(advice_input1, Rotation::cur());
            let x2 = meta.query_advice(advice_input2, Rotation::cur());
            let x3 = meta.query_advice(advice_input3, Rotation::cur());
            // let instance = meta.query_instance(instance, Rotation::cur());
            let s = meta.query_selector(selector);
            
            vec![s * (x3 - (x1 - x2 - Expression::Constant(mod_value_fp - pow_254_fp)))]
        });
        
    
    
        
        LessThanConfig {
            compare_input1: advice_input1,
            compare_input2: advice_input2,
            compare_input3: advice_input3,
            instance,
            selector,
            range_check_config,
        }
        


        
    }

    // return less_than_flag and x2(threshold)
    pub fn is_less_than(
        &self,
        mut layouter: impl Layouter<Fp>,
        x1: Fp,
        x2: Fp,
        num_length: usize,
    // ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
    ) -> Result<(), halo2_proofs::plonk::Error> {
        let range_check_chip = RangeCheckChip::construct(self.config.range_check_config.clone());
        // if x1 < x2, then less_than_flag = 1, else less_than_flag = 0
        let mut less_than_flag: usize = 0;
        if x1 < x2 {
            less_than_flag = 1;
        }
        println!("less_than_flag = {:?}", less_than_flag);
        let mod_value_fp = Fp::from_raw([
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);
        let pow_253_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x2000000000000000,
            ]
        );
        let mut x3 = x1 - x2 - (mod_value_fp - pow_253_fp);
        println!("x1 = {:?}", x1);
        println!("x2 = {:?}", x2);
        println!("x3 = {:?}", x3);

        let bit_inputs_cell = range_check_chip.assign_and_bit_check_2(layouter.namespace(|| "Bit RangeCheck Test in LessThan Chip"), x3, num_length - 1, less_than_flag)?;
        let output_cell = range_check_chip.assign_and_summation(layouter.namespace(|| "Summation in LessThan Chip"), &bit_inputs_cell)?;
        println!("output_cell = {:?}", output_cell);

        // let temp = Value::known((Fp::MODULUS - Fp::from(2u64.pow(num_length - 1))));
        let x2_cell = layouter.assign_region(
            || "less than", 
            |mut region| -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
                
                // self.config.selector.enable(&mut region, 0)?;
                if less_than_flag == 1 {
                    self.config.selector.enable(&mut region, 0)?;
                }
                let x1_cell = region.assign_advice(
                    || "load x1",
                    self.config.compare_input1,
                    0,
                    || Value::known(x1)
                )?;
                let x2_cell = region.assign_advice(
                    || "load x2",
                    self.config.compare_input2,
                    0,
                    || Value::known(x2)
                )?;
                // let x3_cell = region.assign_advice(
                //     || "load x3",
                //     self.config.compare_input3,
                //     0,
                //     || {
                //         let x1_value = x1_cell.value();
                //         let x2_value = x2_cell.value();
                //         // let x3_value = x1_value - x2_value - (Value::known((Fp::MODULUS - Fp::from(2u64.pow(num_length - 1))))).as_ref();
                //         let x3_value = x1_value - x2_value - (mod_value_fp - Fp::from(2u64.pow(num_length - 1)));
                //         x3_value
                //     }
                //     // || x1_cell.value() - x2_cell.value()
                // )?;
                let x3_cell = output_cell.copy_advice(
                    || "Load X3",
                    &mut region,
                    self.config.compare_input3,
                    0,
                )?;
                // self.expose_public(&mut region, &x2_cell, 0)?;
                // // range_check_chip.expose_public(layouter.namespace(|| "instance check"), &output_cell, 0)?;
                Ok(x2_cell)

            }
        );
        self.expose_public(layouter.namespace(|| "revealing X2(Threshold)"), &x2_cell?, 0)?;
        Ok(())
    }
    


    pub fn is_less_than_2(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1_cell: &AssignedCell<Fp, Fp>,
        // input_x2_cell: &AssignedCell<Fp, Fp>,
        x1: Fp,
        x2: Fp,
        num_length: usize,
    // ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        let range_check_chip = RangeCheckChip::construct(self.config.range_check_config.clone());
        // if x1 < x2, then less_than_flag = 1, else less_than_flag = 0
        let mut less_than_flag: usize = 0;
        if x1 < x2 {
            less_than_flag = 1;
        }
        // println!("less_than_flag = {:?}", less_than_flag);
        let mod_value_fp = Fp::from_raw([
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);
        let pow_254_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x4000000000000000,
            ]
        );
        let pow_253_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x2000000000000000,
            ]
        );
        let pow_252_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x1000000000000000,
            ]
        );
        let mut x3 = x1 - x2 - (mod_value_fp - pow_254_fp);
        // println!("x1 = {:?}", x1);
        // println!("x2 = {:?}", x2);
        // println!("x3 = {:?}", x3);

        let (bit_inputs_cell, less_than_flag_cell) = range_check_chip.assign_and_bit_check_3(layouter.namespace(|| "Bit RangeCheck Test in LessThan Chip"), x3, num_length - 1, less_than_flag)?;
        // println!("bit_inputs_cell.len() -> {:?}", bit_inputs_cell.clone().len());
        let output_cell = range_check_chip.assign_and_summation(layouter.namespace(|| "Summation in LessThan Chip"), &bit_inputs_cell)?;
        // println!("output_cell = {:?}", output_cell);

        // let temp = Value::known((Fp::MODULUS - Fp::from(2u64.pow(num_length - 1))));
        let x2_cell = layouter.assign_region(
            || "less than", 
            |mut region| -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
                
                // self.config.selector.enable(&mut region, 0)?;
                if less_than_flag == 1 {
                    self.config.selector.enable(&mut region, 0)?;
                }
                // let x1_cell = region.assign_advice(
                //     || "load x1",
                //     self.config.compare_input1,
                //     0,
                //     || Value::known(x1)
                // )?;
                let x1_cell = input_x1_cell.copy_advice(
                    || "Load X1",
                    &mut region,
                    self.config.compare_input1,
                    0,
                )?;
                let x2_cell = region.assign_advice(
                    || "load x2",
                    self.config.compare_input2,
                    0,
                    || Value::known(x2)
                )?;
                // let x2_cell = input_x2_cell.copy_advice(
                //     || "Load X2",
                //     &mut region,
                //     self.config.compare_input2,
                //     0,
                // )?;
                // let x3_cell = region.assign_advice(
                //     || "load x3",
                //     self.config.compare_input3,
                //     0,
                //     || {
                //         let x1_value = x1_cell.value();
                //         let x2_value = x2_cell.value();
                //         // let x3_value = x1_value - x2_value - (Value::known((Fp::MODULUS - Fp::from(2u64.pow(num_length - 1))))).as_ref();
                //         let x3_value = x1_value - x2_value - (mod_value_fp - Fp::from(2u64.pow(num_length - 1)));
                //         x3_value
                //     }
                //     // || x1_cell.value() - x2_cell.value()
                // )?;
                let x3_cell = output_cell.copy_advice(
                    || "Load X3",
                    &mut region,
                    self.config.compare_input3,
                    0,
                )?;
                // self.expose_public(&mut region, &x2_cell, 0)?;
                // // range_check_chip.expose_public(layouter.namespace(|| "instance check"), &output_cell, 0)?;
                Ok(x2_cell)

            }
        );
        // self.expose_public(layouter.namespace(|| "revealing X2(Threshold)"), &x2_cell?, 0)?;
        Ok((less_than_flag_cell[0].clone(), x2_cell?))
    }
    

    //used in greater_than_chip test
    pub fn is_less_than_3(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1_cell: &AssignedCell<Fp, Fp>,
        input_x2_cell: &AssignedCell<Fp, Fp>,
        x1: Fp,
        x2: Fp,
        num_length: usize,
    // ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        let range_check_chip = RangeCheckChip::construct(self.config.range_check_config.clone());
        // if x1 < x2, then less_than_flag = 1, else less_than_flag = 0
        let mut less_than_flag: usize = 0;
        if x1 < x2 {
            less_than_flag = 1;
        }
        // println!("less_than_flag = {:?}", less_than_flag);
        let mod_value_fp = Fp::from_raw([
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);
        let pow_254_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x4000000000000000,
            ]
        );
        let pow_253_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x2000000000000000,
            ]
        );
        let pow_252_fp = Fp::from_raw(
            [
                0x0000000000000000,
                0x0000000000000000,
                0x0000000000000000,
                0x1000000000000000,
            ]
        );
        let mut x3 = x1 - x2 - (mod_value_fp - pow_254_fp);
        // println!("Greater than chip -----------------------------------is_less_than_3 function is called");
        // println!("less_than_flag = {:?}", less_than_flag);
        // println!("x1 = {:?}", x1);
        // println!("x2 = {:?}", x2);
        // println!("x3 = {:?}", x3);

        let (bit_inputs_cell, less_than_flag_cell) = range_check_chip.assign_and_bit_check_3(layouter.namespace(|| "Bit RangeCheck Test in LessThan Chip"), x3, num_length - 1, less_than_flag)?;
        // println!("bit_inputs_cell.len() -> {:?}", bit_inputs_cell.clone().len());
        let output_cell = range_check_chip.assign_and_summation(layouter.namespace(|| "Summation in LessThan Chip"), &bit_inputs_cell)?;
        // println!("output_cell = {:?}", output_cell);

        // let temp = Value::known((Fp::MODULUS - Fp::from(2u64.pow(num_length - 1))));
        let x2_cell = layouter.assign_region(
            || "less than", 
            |mut region| -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
                
                // self.config.selector.enable(&mut region, 0)?;
                if less_than_flag == 1 {
                    self.config.selector.enable(&mut region, 0)?;
                }
                // let x1_cell = region.assign_advice(
                //     || "load x1",
                //     self.config.compare_input1,
                //     0,
                //     || Value::known(x1)
                // )?;
                let x1_cell = input_x1_cell.copy_advice(
                    || "Load X1",
                    &mut region,
                    self.config.compare_input1,
                    0,
                )?;
                // let x2_cell = region.assign_advice(
                //     || "load x2",
                //     self.config.compare_input2,
                //     0,
                //     || Value::known(x2)
                // )?;
                let x2_cell = input_x2_cell.copy_advice(
                    || "Load X2",
                    &mut region,
                    self.config.compare_input2,
                    0,
                )?;
                // let x3_cell = region.assign_advice(
                //     || "load x3",
                //     self.config.compare_input3,
                //     0,
                //     || {
                //         let x1_value = x1_cell.value();
                //         let x2_value = x2_cell.value();
                //         // let x3_value = x1_value - x2_value - (Value::known((Fp::MODULUS - Fp::from(2u64.pow(num_length - 1))))).as_ref();
                //         let x3_value = x1_value - x2_value - (mod_value_fp - Fp::from(2u64.pow(num_length - 1)));
                //         x3_value
                //     }
                //     // || x1_cell.value() - x2_cell.value()
                // )?;
                let x3_cell = output_cell.copy_advice(
                    || "Load X3",
                    &mut region,
                    self.config.compare_input3,
                    0,
                )?;
                // self.expose_public(&mut region, &x2_cell, 0)?;
                // // range_check_chip.expose_public(layouter.namespace(|| "instance check"), &output_cell, 0)?;
                Ok(x2_cell)

            }
        );
        // self.expose_public(layouter.namespace(|| "revealing X2(Threshold)"), &x2_cell?, 0)?;
        Ok((less_than_flag_cell[0].clone(), x2_cell?))
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