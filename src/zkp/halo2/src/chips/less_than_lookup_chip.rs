use clap::builder::styling::Color;
use ff::PrimeField;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};
use halo2_proofs::poly::Rotation;
use halo2_proofs::arithmetic::*;
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::{range, vec};
use rayon::slice::ParallelSlice;

use std::ffi::NulError;
use std::marker::PhantomData;
use crate::utils::pow_of_two;

use super::super::utils::*;

const N_BYTES: usize = 32;

// pub fn pow_of_two(by: usize) -> Fp {
//     Fp::from(2u64).pow([by as u64, 0, 0, 0])
// }

#[derive(Debug, Clone, Copy)]
pub struct LTConfig {
    pub compare_input1: Column<Advice>,
    pub compare_input2: Column<Advice>,

    pub lt_flag: Column<Advice>,
    pub diff: [Column<Advice>; N_BYTES],
    pub range_u8: Column<Fixed>,
    
    pub selector: Selector,
    pub instance: Column<Instance>,
    pub lookup_selector: Selector,
    // pub range
    
}

#[derive(Clone, Debug)]
pub struct LTChip {
    config: LTConfig,
    _marker: PhantomData<Fp>,
}

impl LTChip {
    pub fn construct(config: LTConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        advice_1: Column<Advice>,
        advice_2: Column<Advice>,
        advice_3: Column<Advice>,
        range_u8: Column<Fixed>,
        diff: [Column<Advice>; N_BYTES],
        instance: Column<Instance>,
    ) -> LTConfig {
        meta.enable_equality(advice_1);
        meta.enable_equality(advice_2);
        meta.enable_equality(advice_3);
        meta.enable_equality(instance);
        for i in 0..N_BYTES {
            meta.enable_equality(diff[i]);
        }

        // let diff = [(); N_BYTES].map(|_| meta.advice_column());
        let range = pow_of_two(253);
        // let range_u8 = meta.fixed_column();
        let selector = meta.selector();
        let lookup_selector = meta.complex_selector();

        meta.create_gate("less than gate", |meta| {
            let s = meta.query_selector(selector);
            let x1 = meta.query_advice(advice_1, Rotation::cur());
            let x2 = meta.query_advice(advice_2, Rotation::cur());
            let lt_flag = meta.query_advice(advice_3, Rotation::cur());
            
            let diff_bytes = diff
                .iter()
                .map(|c| meta.query_advice(*c, Rotation::cur()))
                .collect::<Vec<Expression<Fp>>>();

            let check_a = x1 - x2 - expr_from_bytes(&diff_bytes) + ((Expression::Constant(Fp::ONE) - lt_flag.clone()) * Expression::Constant(range));
            let check_b = bool_check(lt_flag.clone());

            [check_a, check_b]
                .into_iter()
                .map(move |poly| s.clone() * poly)

        });

        
        meta.annotate_lookup_any_column(range_u8, || "Lookup range_u8");

        diff[0..N_BYTES].iter().for_each(|column| {
            meta.lookup_any("Range Check for u8", |meta| {
                let lookup_s = meta.query_selector(lookup_selector);
                let range_u8_cell = meta.query_advice(*column, Rotation::cur());
                let lookup_u8 = meta.query_fixed(range_u8, Rotation::cur());
                vec![(lookup_s * range_u8_cell, lookup_u8)]
            });
        });

    

        LTConfig {
            compare_input1: advice_1,
            compare_input2: advice_2,
            lt_flag: advice_3,
            diff,
            range_u8,
            selector,
            lookup_selector,
            instance,
        }

    }

    pub fn load_lookup_table(
        &self,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        const RANGE_LOOKUP: usize = 256;

        layouter.assign_region(
            || "load lookup table",
            |mut region| {
                for i in 0..RANGE_LOOKUP {
                    region.assign_fixed(
                        || "assign lookup table cell in fixed column",
                        self.config.range_u8,
                        i,
                        || Value::known(Fp::from(i as u64)),
                    )?;
                }
                Ok(())
            }
        )
    }

    pub fn assign_value(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1_cell: &AssignedCell<Fp, Fp>,
        input_x2_cell: &AssignedCell<Fp, Fp>,
        input_x1: Fp,
        input_x2: Fp,
    ) -> Result<(AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        let mut lt_flag: usize = 0;
        if input_x1 < input_x2 {
            lt_flag = 1;
        }

        let diff = input_x1 - input_x2;
        let diff_bytes = diff.to_bytes();

        let lt_flag_cell = layouter.assign_region(
            || "less than",
            |mut region| {
                self.config.lookup_selector.enable(&mut region, 0)?;
                if lt_flag == 1 {
                    self.config.selector.enable(&mut region, 0)?;
                }
                

                let _ = input_x1_cell.copy_advice(
                    || "copy input_x1",
                    &mut region,
                    self.config.compare_input1,
                    0,
                )?;

                let _ = input_x2_cell.copy_advice(
                    || "copy input_x2",
                    &mut region,
                    self.config.compare_input2,
                    0,
                )?;
                
                let lt_flag_cell = region.assign_advice(
                    || "assign lt_flag",
                    self.config.lt_flag,
                    0,
                    || Value::known(Fp::from(lt_flag as u64)),
                )?;

                for i in 0..N_BYTES {
                    let _ = region.assign_advice(
                        || "assign_bytes",
                        self.config.diff[i],
                        0,
                        || Value::known(Fp::from(diff_bytes[i] as u64)),
                    );
                }
                Ok((lt_flag_cell))

            }
        )?;

        Ok((lt_flag_cell))

    }


    pub fn load_inputs(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1: Fp,
        input_x2: Fp,
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Load Less Than Chip Inputs",
            |mut region| {
                // self.config.selector.enable(&mut region, 0)?;
                let compare_input1_cell = region.assign_advice(
                    || "load compare_input1",
                    self.config.compare_input1,
                    0,
                    || Value::known(input_x1)
                )?;
                let compare_input2_cell = region.assign_advice(
                    || "load compare_input2",
                    self.config.compare_input2,
                    0,
                    || Value::known(input_x2)
                )?;
                Ok((compare_input1_cell, compare_input2_cell))
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


    pub fn is_less_than(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1: Fp,
        input_x2: Fp,

    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {

        let (input_x1_cell, input_x2_cell) = self.load_inputs(layouter.namespace(|| "load inputs"), input_x1, input_x2)?;
        let lt_flag_cell = self.assign_value(layouter.namespace(|| "assign value"), &input_x1_cell, &input_x2_cell, input_x1, input_x2)?;
        Ok((input_x1_cell, input_x2_cell, lt_flag_cell))
    }


    pub fn assign_value_and_is_less_than(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1: Fp,
        input_x2: Fp,
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        let mut lt_flag: usize = 0;
        if input_x1 < input_x2 {
            lt_flag = 1;
        }

        let diff = input_x1 - input_x2;
        let diff_bytes = diff.to_bytes();

        let (input_x1_cell, input_x2_cell, lt_flag_cell) = layouter.assign_region(
            || "less than",
            |mut region| {
                self.config.lookup_selector.enable(&mut region, 0)?;
                if lt_flag == 1 {
                    self.config.selector.enable(&mut region, 0)?;
                }
                

                // let _ = input_x1_cell.copy_advice(
                //     || "copy input_x1",
                //     &mut region,
                //     self.config.compare_input1,
                //     0,
                // )?;

                // let _ = input_x2_cell.copy_advice(
                //     || "copy input_x2",
                //     &mut region,
                //     self.config.compare_input2,
                //     0,
                // )?;
                let input_x1_cell = region.assign_advice(
                    || "load compare_input1",
                    self.config.compare_input1,
                    0,
                    || Value::known(input_x1)
                )?;
                let input_x2_cell = region.assign_advice(
                    || "load compare_input2",
                    self.config.compare_input2,
                    0,
                    || Value::known(input_x2)
                )?;
                
                let lt_flag_cell = region.assign_advice(
                    || "assign lt_flag",
                    self.config.lt_flag,
                    0,
                    || Value::known(Fp::from(lt_flag as u64)),
                )?;

                for i in 0..N_BYTES {
                    let _ = region.assign_advice(
                        || "assign_bytes",
                        self.config.diff[i],
                        0,
                        || Value::known(Fp::from(diff_bytes[i] as u64)),
                    );
                }
                Ok((input_x1_cell, input_x2_cell, lt_flag_cell))

            }
        )?;

        Ok((input_x1_cell, input_x2_cell, lt_flag_cell))

    }

    pub fn is_less_than_expr_greater_than(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1_cell: &AssignedCell<Fp, Fp>,
        input_x2_cell: &AssignedCell<Fp, Fp>,
        less_than_flag_cell: &AssignedCell<Fp, Fp>,
        input_x1: Fp,
        input_x2: Fp,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        let mut lt_flag: usize = 0;
        if input_x1 < input_x2 {
            lt_flag = 1;
        }
        let diff = input_x1 - input_x2;
        let diff_bytes = diff.to_bytes();
        layouter.assign_region(
            || "greater than",
            |mut region| {
                self.config.lookup_selector.enable(&mut region, 0)?;
                if lt_flag == 1 {
                    self.config.selector.enable(&mut region, 0)?;
                }

                let _ = input_x1_cell.copy_advice(
                    || "copy input_x1",
                    &mut region,
                    self.config.compare_input1,
                    0,
                )?;

                let _ = input_x2_cell.copy_advice(
                    || "copy input_x2",
                    &mut region,
                    self.config.compare_input2,
                    0,
                )?;

                let _ = region.assign_advice(
                    || "assign lt_flag",
                    self.config.lt_flag,
                    0,
                    || Value::known(Fp::from(1)) - less_than_flag_cell.value().cloned(),
                )?;

                for i in 0..N_BYTES {
                    let _ = region.assign_advice(
                        || "assign_bytes",
                        self.config.diff[i],
                        0,
                        || Value::known(Fp::from(diff_bytes[i] as u64)),
                    );
                }

                Ok(())

            }
        )



        // Ok(())
    }


    

}