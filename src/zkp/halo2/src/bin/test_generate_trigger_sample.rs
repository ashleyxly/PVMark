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
use halo2_proofs::dev::metadata::Constraint;
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
use halo2_merkle_tree::utils::*;


// pub const TRIGGER_SIZE: usize = 3*3;
// pub const TRIGGER_SIZE: usize = 14*14;
// pub const TRIGGER_SIZE: usize = 32*32;
pub const TRIGGER_SIZE: usize = 0;
// pub const TOTAL_PIX_NUM: usize = 32 * 32;
pub const TOTAL_PIX_NUM: usize = 26 * 26;


#[derive(Debug, Clone)]
pub struct PixAddConfig {
    pub col_input_image_r: Column<Advice>,
    pub col_input_image_g: Column<Advice>,
    pub col_input_image_b: Column<Advice>,
    pub col_pattern_r: Column<Advice>,
    pub col_pattern_g: Column<Advice>,
    pub col_pattern_b: Column<Advice>,
    pub col_output_image_r: Column<Advice>,
    pub col_output_image_g: Column<Advice>,
    pub col_output_image_b: Column<Advice>,
    pub col_instance: Column<Instance>,
    pub col_selector: Selector,
}


#[derive(Debug, Clone)]
pub struct PixAddChip {
    pub config: PixAddConfig,
    pub _marker: PhantomData<Fp>,
}

impl PixAddChip {
    pub fn construct(config: PixAddConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        col_input_image_r: Column<Advice>,
        col_input_image_g: Column<Advice>,
        col_input_image_b: Column<Advice>,
        col_pattern_r: Column<Advice>,
        col_pattern_g: Column<Advice>,
        col_pattern_b: Column<Advice>,
        col_output_image_r: Column<Advice>,
        col_output_image_g: Column<Advice>,
        col_output_image_b: Column<Advice>,
        col_instance: Column<Instance>,
    ) -> PixAddConfig {
        let col_selector = meta.selector();
        meta.enable_equality(col_input_image_r);
        meta.enable_equality(col_input_image_g);
        meta.enable_equality(col_input_image_b);
        meta.enable_equality(col_pattern_r);
        meta.enable_equality(col_pattern_g);
        meta.enable_equality(col_pattern_b);
        meta.enable_equality(col_output_image_r);
        meta.enable_equality(col_output_image_g);
        meta.enable_equality(col_output_image_b);
        meta.enable_equality(col_instance);

        meta.create_gate("pix add gate", |meta| {
            let s = meta.query_selector(col_selector);
            let input_image_r = meta.query_advice(col_input_image_r, Rotation::cur());
            let input_image_g = meta.query_advice(col_input_image_g, Rotation::cur());
            let input_image_b = meta.query_advice(col_input_image_b, Rotation::cur());
            let pattern_r = meta.query_advice(col_pattern_r, Rotation::cur());
            let pattern_g = meta.query_advice(col_pattern_g, Rotation::cur());
            let pattern_b = meta.query_advice(col_pattern_b, Rotation::cur());
            let output_image_r = meta.query_advice(col_output_image_r, Rotation::cur());
            let output_image_g = meta.query_advice(col_output_image_g, Rotation::cur());
            let output_image_b = meta.query_advice(col_output_image_b, Rotation::cur());

            vec![
                s.clone() * (input_image_r + pattern_r - output_image_r),
                s.clone() * (input_image_g + pattern_g - output_image_g),
                s * (input_image_b + pattern_b - output_image_b),
            ]
        });

        PixAddConfig {
            col_input_image_r,
            col_input_image_g,
            col_input_image_b,
            col_pattern_r,
            col_pattern_g,
            col_pattern_b,
            col_output_image_r,
            col_output_image_g,
            col_output_image_b,
            col_instance,
            col_selector,
        }
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.col_instance, row)
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fp>,
        // input_image_r: Value<Fp>,
        // input_image_g: Value<Fp>,
        // input_image_b: Value<Fp>,
        // pattern_r: Value<Fp>,
        // pattern_g: Value<Fp>,
        // pattern_b: Value<Fp>,
        input_image_r: Fp,
        input_image_g: Fp,
        input_image_b: Fp,
        pattern_r: Fp,
        pattern_g: Fp,
        pattern_b: Fp,

    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Pix Add Operation",
            |mut region| {
                self.config.col_selector.enable(&mut region, 0)?;
                let input_image_r_cell = region.assign_advice(
                    || "input_image_r",
                    self.config.col_input_image_r,
                    0,
                    || Value::known(input_image_r)
                )?;
                let input_image_g_cell = region.assign_advice(
                    || "input_image_g",
                    self.config.col_input_image_g,
                    0,
                    || Value::known(input_image_g)
                )?;
                let input_image_b_cell = region.assign_advice(
                    || "input_image_b",
                    self.config.col_input_image_b,
                    0,
                    || Value::known(input_image_b)
                )?;
                let pattern_r_cell = region.assign_advice(
                    || "pattern_r",
                    self.config.col_pattern_r,
                    0,
                    || Value::known(pattern_r)
                )?;
                let pattern_g_cell = region.assign_advice(
                    || "pattern_g",
                    self.config.col_pattern_g,
                    0,
                    || Value::known(pattern_g)
                )?;
                let pattern_b_cell = region.assign_advice(
                    || "pattern_b",
                    self.config.col_pattern_b,
                    0,
                    || Value::known(pattern_b)
                )?;
                let output_image_r_cell = region.assign_advice(
                    || "output_image_r",
                    self.config.col_output_image_r,
                    0,
                    || Value::known(input_image_r + pattern_r)
                )?;
                let output_image_g_cell = region.assign_advice(
                    || "output_image_g",
                    self.config.col_output_image_g,
                    0,
                    || Value::known(input_image_g + pattern_g)
                )?;
                let output_image_b_cell = region.assign_advice(
                    || "output_image_b",
                    self.config.col_output_image_b,
                    0,
                    || Value::known(input_image_b + pattern_b)
                )?;
                Ok((output_image_r_cell, output_image_g_cell, output_image_b_cell))

            }
        )
    }



}



#[derive(Debug, Clone)]
pub struct PixEqCheckConfig {
    pub col_input_image_r: Column<Advice>,
    pub col_input_image_g: Column<Advice>,
    pub col_input_image_b: Column<Advice>,
    // pub col_pattern_r: Column<Advice>,
    // pub col_pattern_g: Column<Advice>,
    // pub col_pattern_b: Column<Advice>,
    pub col_output_image_r: Column<Advice>,
    pub col_output_image_g: Column<Advice>,
    pub col_output_image_b: Column<Advice>,
    pub col_instance: Column<Instance>,
    pub col_selector: Selector,

}

#[derive(Debug, Clone)]
pub struct PixEqCheckChip {
    pub config: PixEqCheckConfig,
    pub _marker: PhantomData<Fp>,
}

impl PixEqCheckChip {
    pub fn construct(config: PixEqCheckConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        col_input_image_r: Column<Advice>,
        col_input_image_g: Column<Advice>,
        col_input_image_b: Column<Advice>,
        col_output_image_r: Column<Advice>,
        col_output_image_g: Column<Advice>,
        col_output_image_b: Column<Advice>,
        col_instance: Column<Instance>,
    ) -> PixEqCheckConfig {
        let col_selector = meta.selector();
        meta.enable_equality(col_input_image_r);
        meta.enable_equality(col_input_image_g);
        meta.enable_equality(col_input_image_b);
        // meta.enable_equality(col_pattern_r);
        // meta.enable_equality(col_pattern_g);
        // meta.enable_equality(col_pattern_b);
        meta.enable_equality(col_output_image_r);
        meta.enable_equality(col_output_image_g);
        meta.enable_equality(col_output_image_b);
        meta.enable_equality(col_instance);

        meta.create_gate("pix eq check gate", |meta| {
            let s = meta.query_selector(col_selector);
            let input_image_r = meta.query_advice(col_input_image_r, Rotation::cur());
            let input_image_g = meta.query_advice(col_input_image_g, Rotation::cur());
            let input_image_b = meta.query_advice(col_input_image_b, Rotation::cur());
            // let pattern_r = meta.query_advice(col_pattern_r, Rotation::cur());
            // let pattern_g = meta.query_advice(col_pattern_g, Rotation::cur());
            // let pattern_b = meta.query_advice(col_pattern_b, Rotation::cur());
            let output_image_r = meta.query_advice(col_output_image_r, Rotation::cur());
            let output_image_g = meta.query_advice(col_output_image_g, Rotation::cur());
            let output_image_b = meta.query_advice(col_output_image_b, Rotation::cur());

            vec![
                s.clone() * (input_image_r - output_image_r),
                s.clone() * (input_image_g - output_image_g),
                s * (input_image_b - output_image_b),]
        });

        PixEqCheckConfig {
            col_input_image_r,
            col_input_image_g,
            col_input_image_b,
            // col_pattern_r,
            // col_pattern_g,
            // col_pattern_b,
            col_output_image_r,
            col_output_image_g,
            col_output_image_b,
            col_instance,
            col_selector,
        }
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.col_instance, row)
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_image_r: Fp,
        input_image_g: Fp,
        input_image_b: Fp,
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Assign Eq",
            |mut region| {
                self.config.col_selector.enable(&mut region, 0)?;
                let input_image_r_cell = region.assign_advice(
                    || "input_image_r",
                    self.config.col_input_image_r,
                    0,
                    || Value::known(input_image_r)
                )?;
                let input_image_g_cell = region.assign_advice(
                    || "input_image_g",
                    self.config.col_input_image_g,
                    0,
                    || Value::known(input_image_g)
                )?;
                let input_image_b_cell = region.assign_advice(
                    || "input_image_b",
                    self.config.col_input_image_b,
                    0,
                    || Value::known(input_image_b)
                )?;
                let output_image_r_cell = region.assign_advice(
                    || "output_image_r",
                    self.config.col_output_image_r,
                    0,
                    || Value::known(input_image_r)
                )?;
                let output_image_g_cell = region.assign_advice(
                    || "output_image_g",
                    self.config.col_output_image_g,
                    0,
                    || Value::known(input_image_g)
                )?;
                let output_image_b_cell = region.assign_advice(
                    || "output_image_b",
                    self.config.col_output_image_b,
                    0,
                    || Value::known(input_image_b)
                )?;
                Ok((output_image_r_cell, output_image_g_cell, output_image_b_cell))

            }

        )
    }
}




#[derive(Clone, Debug)]
pub struct GenerateInWhiteBoxConfig {
    pub col_input_image_r: Column<Advice>,
    pub col_input_image_g: Column<Advice>,
    pub col_input_image_b: Column<Advice>,
    pub col_pattern_r: Column<Advice>,
    pub col_pattern_g: Column<Advice>,
    pub col_pattern_b: Column<Advice>,
    pub col_output_image_r: Column<Advice>,
    pub col_output_image_g: Column<Advice>,
    pub col_output_image_b: Column<Advice>,
    pub col_instance: Column<Instance>,
    // pub col_selector: Selector,

    pub pix_add_config: PixAddConfig,
    pub pix_eq_check_config: PixEqCheckConfig,
}

#[derive(Clone, Debug, Default)]
pub struct GenerateInWhiteBoxCircuit {
    pub input_image_r: Vec<Fp>,
    pub input_image_g: Vec<Fp>,
    pub input_image_b: Vec<Fp>,
    pub pattern_r: Vec<Fp>,
    pub pattern_g: Vec<Fp>,
    pub pattern_b: Vec<Fp>,
}

impl Circuit<Fp> for GenerateInWhiteBoxCircuit {
    type Config = GenerateInWhiteBoxConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> GenerateInWhiteBoxConfig {
        let col_input_image_r = meta.advice_column();
        let col_input_image_g = meta.advice_column();
        let col_input_image_b = meta.advice_column();
        let col_pattern_r = meta.advice_column();
        let col_pattern_g = meta.advice_column();
        let col_pattern_b = meta.advice_column();
        let col_output_image_r = meta.advice_column();
        let col_output_image_g = meta.advice_column();
        let col_output_image_b = meta.advice_column();
        let col_instance = meta.instance_column();
        let col_selector = meta.selector();

        GenerateInWhiteBoxConfig {
            col_input_image_r,
            col_input_image_g,
            col_input_image_b,
            col_pattern_r,
            col_pattern_g,
            col_pattern_b,
            col_output_image_r,
            col_output_image_g,
            col_output_image_b,
            col_instance,
            // col_selector,
            pix_add_config: PixAddChip::configure(
                meta,
                col_input_image_r,
                col_input_image_g,
                col_input_image_b,
                col_pattern_r,
                col_pattern_g,
                col_pattern_b,
                col_output_image_r,
                col_output_image_g,
                col_output_image_b,
                col_instance,
            ),
            pix_eq_check_config: PixEqCheckChip::configure(
                meta,
                col_input_image_r,
                col_input_image_g,
                col_input_image_b,
                col_output_image_r,
                col_output_image_g,
                col_output_image_b,
                col_instance,
            ),
        }

        
    }


    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        let pix_add_chip = PixAddChip::construct(config.pix_add_config);
        let pix_check_eq_chip = PixEqCheckChip::construct(config.pix_eq_check_config);

        let mut instance_num = 0;

        for i in 0..TOTAL_PIX_NUM {
            if i < TRIGGER_SIZE {
                let (output_image_r_cell, output_image_g_cell, output_image_b_cell) = pix_add_chip.assign(
                    layouter.namespace(|| format!("No. {} -- Pix Add", i)),
                    self.input_image_r[i],
                    self.input_image_g[i],
                    self.input_image_b[i],
                    self.pattern_r[i],
                    self.pattern_g[i],
                    self.pattern_b[i],
                )?;
                pix_add_chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_image_r_cell, instance_num)?;
                instance_num += 1;
                pix_add_chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_image_g_cell, instance_num)?;
                instance_num += 1;
                pix_add_chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_image_b_cell, instance_num)?;
                instance_num += 1;

            } else {
                let (output_image_r_cell, output_image_g_cell, output_image_b_cell) = pix_check_eq_chip.assign(
                    layouter.namespace(|| format!("No. {} -- Pix Eq Check", i)),
                    self.input_image_r[i],
                    self.input_image_g[i],
                    self.input_image_b[i],
                )?;
                pix_check_eq_chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_image_r_cell, instance_num)?;
                instance_num += 1;
                pix_check_eq_chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_image_g_cell, instance_num)?;
                instance_num += 1;
                pix_check_eq_chip.expose_public(layouter.namespace(|| format!("expose output {}", i)), &output_image_b_cell, instance_num)?;
                instance_num += 1;
            }
        }

        Ok(())


    }



}

fn generate_test_case() -> (Vec<Fp>, Vec<Fp>, Vec<Fp>, Vec<Fp>, Vec<Fp>, Vec<Fp>, Vec<Fp>, Vec<Fp>, Vec<Fp>) {
    let mut rng = rand::thread_rng();
    let mut input_image_r = Vec::new();
    let mut input_image_g = Vec::new();
    let mut input_image_b = Vec::new();
    let mut pattern_r = Vec::new();
    let mut pattern_g = Vec::new();
    let mut pattern_b = Vec::new();
    let mut output_image_r = Vec::new();
    let mut output_image_g = Vec::new();
    let mut output_image_b = Vec::new();
    for i in 0..TOTAL_PIX_NUM {
        let a = Fp::from(rng.gen_range(0..30));
        let b = Fp::from(rng.gen_range(0..20));
        if i < TRIGGER_SIZE {
            input_image_r.push(a);
            input_image_g.push(a);
            input_image_b.push(a);
            pattern_r.push(b);
            pattern_g.push(b);
            pattern_b.push(b);
            output_image_r.push(a + b);
            output_image_g.push(a + b);
            output_image_b.push(a + b);
        } else {
            input_image_r.push(a);
            input_image_g.push(a);
            input_image_b.push(a);
            // pattern_r.push(0);
            // pattern_g.push(0);
            // pattern_b.push(0);
            output_image_r.push(a);
            output_image_g.push(a);
            output_image_b.push(a);
        }
    }
    (input_image_r, input_image_g, input_image_b, pattern_r, pattern_g, pattern_b, output_image_r, output_image_g, output_image_b)
    

}



fn main() {
    let (input_image_r, input_image_g, input_image_b, pattern_r, pattern_g, pattern_b, output_image_r, output_image_g, output_image_b) = generate_test_case();

    
    let circuit = GenerateInWhiteBoxCircuit {
        input_image_r: input_image_r.clone(),
        input_image_g: input_image_g.clone(),
        input_image_b: input_image_b.clone(),
        pattern_r: pattern_r.clone(),
        pattern_g: pattern_g.clone(),
        pattern_b: pattern_b.clone(),
    };

    let mut public_inputs = vec![];
    for i in 0..TOTAL_PIX_NUM {
        public_inputs.push(output_image_r[i]);
        public_inputs.push(output_image_g[i]);
        public_inputs.push(output_image_b[i]);
    }

    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(25, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    let mut k = select_suitable_k_value(total_row as i32);
    println!("k = {:?}", k);
    k = k + 2;
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

    let proof_path = "/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_generate_trigger_sample/proof.bin";

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