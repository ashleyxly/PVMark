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


use super::less_than_chip::{LessThanChip, LessThanConfig, self};

#[derive(Debug, Clone)]
pub struct GreaterThanConfig {
    pub less_than_config: LessThanConfig,
}

#[derive(Debug, Clone)]
pub struct GreaterThanChip {
    config: GreaterThanConfig,
    _marker: PhantomData<Fp>,
}

impl GreaterThanChip {
    pub fn construct(config: GreaterThanConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        advice_input1: Column<Advice>,
        advice_input2: Column<Advice>,
        advice_output: Column<Advice>,
        instance_input1: Column<Instance>,
    ) -> GreaterThanConfig {
        meta.enable_equality(advice_input1);
        meta.enable_equality(advice_input2);
        meta.enable_equality(advice_output);
        meta.enable_equality(instance_input1);

        let selector = meta.selector();

        let less_than_config = LessThanChip::configure(meta, advice_input1, advice_input2, advice_output, instance_input1);
        GreaterThanConfig {
            less_than_config,
        }
    }

    pub fn is_greater_than(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_x1_cell: &AssignedCell<Fp, Fp>,
        input_x2_cell: &AssignedCell<Fp, Fp>,
        x1: Fp,
        x2: Fp,
        num_length: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        let less_than_chip = LessThanChip::construct(self.config.less_than_config.clone());
        less_than_chip.is_less_than_3(layouter.namespace(|| "is_greater_than"), input_x2_cell, input_x1_cell, x2, x1, num_length)?;
        Ok(())
    }
}