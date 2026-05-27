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
// use crate::utils::pow_of_two;

use super::super::utils::*;

pub const DEC_NUM: usize = 4;
pub const N_BYTES: usize = 32;

#[derive(Clone, Debug, Copy)]
pub struct CheckDecConfig {
    pub dec_inputs: [Column<Advice>; DEC_NUM],
    pub org_value: Column<Advice>,
    pub each_u8: [Column<Advice>; N_BYTES],

    // pub success_flag: Column<Advice>,
    // pub range_u64: Column<Fixed>,
    pub range_u8: Column<Fixed>,
    pub selector: Selector,
    pub instance: Column<Instance>,
    pub lookup_selector: Selector,

}

#[derive(Clone, Debug)]
pub struct CheckDecChip {
    config: CheckDecConfig,
    _marker: PhantomData<Fp>,
}

impl CheckDecChip {
    pub fn construct(config: CheckDecConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        dec_inputs: [Column<Advice>; DEC_NUM],
        org_value: Column<Advice>,
        each_u8: [Column<Advice>; N_BYTES],
        // success_flag: Column<Advice>,
        instance: Column<Instance>,
        range_u8: Column<Fixed>,
    ) -> CheckDecConfig {
        meta.enable_equality(org_value);
        for i in 0..DEC_NUM {
            meta.enable_equality(dec_inputs[i])
        }
        for i in 0..N_BYTES {
            meta.enable_equality(each_u8[i])
        }
        // meta.enable_equality(success_flag);
        meta.enable_equality(instance);


        let selector = meta.selector();
        let lookup_selector = meta.complex_selector();

        meta.create_gate("NumDec gate", |meta| {
            let s = meta.query_selector(selector);
            let all_u8 = each_u8
                .iter()
                .map(|c| meta.query_advice(*c, Rotation::cur()))
                .collect::<Vec<Expression<Fp>>>();
            let org_value = meta.query_advice(org_value, Rotation::cur());
            let check_org_value = org_value - expr_from_bytes(&all_u8);

            let dec_inputs = dec_inputs
                .iter()
                .map(|c| meta.query_advice(*c, Rotation::cur()))
                .collect::<Vec<Expression<Fp>>>();

            let value_a = expr_from_bytes(&all_u8[0..8]);
            let value_b = expr_from_bytes(&all_u8[8..16]);
            let value_c = expr_from_bytes(&all_u8[16..24]);
            let value_d = expr_from_bytes(&all_u8[24..32]);

            let check_value_a = dec_inputs[0].clone() - value_a;
            let check_value_b = dec_inputs[1].clone() - value_b;
            let check_value_c = dec_inputs[2].clone() - value_c;
            let check_value_d = dec_inputs[3].clone() - value_d;


            // [check_org_value, check_value_a, check_value_b, check_value_c, check_value_d]
            //     .into_iter()
            //     .map(move |poly| s.clone() * poly)
            vec![s.clone() * (check_org_value + check_value_a + check_value_b + check_value_c + check_value_d)]

    
        });

        meta.annotate_lookup_any_column(range_u8, || "Lookup range_u8");

        each_u8[0..N_BYTES].iter().for_each(|column| {
            meta.lookup_any("Range Check for u8", |meta| {
                let lookup_s = meta.query_selector(lookup_selector);
                let range_u8_cell = meta.query_advice(*column, Rotation::cur());
                let lookup_u8 = meta.query_fixed(range_u8, Rotation::cur());
                vec![(lookup_s * range_u8_cell, lookup_u8)]
            });
        });

        CheckDecConfig {
            dec_inputs,
            org_value,
            each_u8,
            // success_flag,
            range_u8,
            selector,
            instance,
            lookup_selector,
        }

        
    }

    pub fn assign_value_and_check(
        &self,
        mut layouter: impl Layouter<Fp>,
        dec_inputs: [Fp; DEC_NUM],
        org_value: Fp,
        // dec_inputs_cell: &[AssignedCell<Fp, Fp>; DEC_NUM],
        // org_value_cell: &AssignedCell<Fp, Fp>,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        let value_a_bytes = dec_inputs[0].clone().to_bytes();
        let value_b_bytes = dec_inputs[1].clone().to_bytes();
        let value_c_bytes = dec_inputs[2].clone().to_bytes();
        let value_d_bytes = dec_inputs[3].clone().to_bytes();

        // println!("value_a_bytes: {:?}", value_a_bytes);
        // println!("value_b_bytes: {:?}", value_b_bytes);
        // println!("value_c_bytes: {:?}", value_c_bytes);
        // println!("value_d_bytes: {:?}", value_d_bytes);

        layouter.assign_region(
            || "Check Dec Chip",
        |mut region| {
            self.config.selector.enable(&mut region, 0)?;
            self.config.lookup_selector.enable(&mut region, 0)?;

            let _ = region.assign_advice(
                || "assign org_value",
                self.config.org_value,
                0,
                || Value::known(org_value),
            );

            let _ = region.assign_advice(
                || "assign value_a",
                self.config.dec_inputs[0],
                0,
                || Value::known(dec_inputs[0].clone()),
            );

            let _ = region.assign_advice(
                || "assign value_b",
                self.config.dec_inputs[1],
                0,
                || Value::known(dec_inputs[1].clone()),
            );

            let _ = region.assign_advice(
                || "assign value_c",
                self.config.dec_inputs[2],
                0,
                || Value::known(dec_inputs[2].clone()),
            );

            let _ = region.assign_advice(
                || "assign value_d",
                self.config.dec_inputs[3],
                0,
                || Value::known(dec_inputs[3].clone()),
            );

            for i in 0..8 {
                let _ = region.assign_advice(
                    || "assign value_a",
                    self.config.each_u8[i],
                    0,
                    || Value::known(Fp::from(value_a_bytes[i] as u64)),
                );
            }

            for i in 0..8 {
                let _ = region.assign_advice(
                    || "assign value_b",
                    self.config.each_u8[i+8],
                    0,
                    || Value::known(Fp::from(value_b_bytes[i] as u64)),
                );
            }

            for i in 0..8 {
                let _ = region.assign_advice(
                    || "assign value_c",
                    self.config.each_u8[i+16],
                    0,
                    || Value::known(Fp::from(value_c_bytes[i] as u64)),
                );
            }

            for i in 0..8 {
                let _ = region.assign_advice(
                    || "assign value_d",
                    self.config.each_u8[i+24],
                    0,
                    || Value::known(Fp::from(value_d_bytes[i] as u64)),
                );
            }

            Ok(())

        })




        // Ok(())
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
}