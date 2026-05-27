use std::ffi::NulError;
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
// use halo2_proofs::poly::Rotation;
// use halo2_proofs::poly::commitment::Params;
// use halo2_proofs::poly::commitment::ParamsProver;
// use log::info;
// use std::error::Error;
// use std::fs::File;
// use std::io::BufReader;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;

// use halo2_merkle_tree::chips::range_check_chip::{RangeCheckChip, RangeCheckConfig};
// use halo2_merkle_tree::chips::less_than_chip::{LessThanChip, LessThanConfig, self};
// use halo2_merkle_tree::chips::less_than_lookup_chip::*;
use halo2_merkle_tree::chips::check_dec_chip::{CheckDecChip, CheckDecConfig, DEC_NUM, N_BYTES};

#[derive(Debug, Clone, Default)]
pub struct CheckDecTestCircuit {
    pub dec_inputs: [Fp; DEC_NUM],
    pub org_value: Fp,
}

impl Circuit<Fp> for CheckDecTestCircuit {
    type Config = CheckDecConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let dec_inputs = [(); DEC_NUM].map(|_| meta.advice_column());
        let org_value = meta.advice_column();
        let each_u8 = [(); N_BYTES].map(|_| meta.advice_column());
        let instance = meta.instance_column();
        let range_u8 = meta.fixed_column();

        CheckDecChip::configure(
            meta,
            dec_inputs,
            org_value,
            each_u8,
            instance,
            range_u8,
            // selector,
            // lookup_selector,
        )
        
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), Error> {
        let check_dec_chip = CheckDecChip::construct(config.clone());
        //load lookup table
        check_dec_chip.load_lookup_table(layouter.namespace(|| "Load Lookup Table"))?;
        check_dec_chip.assign_value_and_check(layouter.namespace(|| "assign"), self.dec_inputs, self.org_value)?;

        Ok(())
    }
}



fn main() {
    let org_value = Fp::from(123141356);
    let org_value_bytes = org_value.clone().to_bytes();
    let dec_input_1 = u64::from_le_bytes(org_value_bytes[0..8].try_into().unwrap());
    let dec_input_2 = u64::from_le_bytes(org_value_bytes[8..16].try_into().unwrap());
    let dec_input_3 = u64::from_le_bytes(org_value_bytes[16..24].try_into().unwrap());
    let dec_input_4 = u64::from_le_bytes(org_value_bytes[24..32].try_into().unwrap());

    let dec_inputs = [Fp::from(dec_input_1), Fp::from(dec_input_2), Fp::from(dec_input_3), Fp::from(dec_input_4)];
    println!("dec_inputs: {:?}", dec_inputs);

    // let wrong_inputs = [Fp::from(1), Fp::from(2), Fp::from(3), Fp::from(1234)];
    let wrong_inputs = [Fp::from(dec_input_1), Fp::from(dec_input_2), Fp::from(dec_input_3), Fp::from(dec_input_4 + 1)];
    println!("wrong_inputs: {:?}", wrong_inputs);

    let circuit = CheckDecTestCircuit {
        dec_inputs,
        org_value,
    };

    let mut public_inputs: Vec<Fp> = vec![];
    let prover = MockProver::run(9, &circuit, vec![public_inputs.clone()]).unwrap();
    // println!("prover: {:?}", prover);
    println!("circuit is satisfied: {:?}", prover.verify());
    assert_eq!(prover.verify(), Ok(()));
    println!("success");

    let wrong_circuit = CheckDecTestCircuit {
        dec_inputs: wrong_inputs,
        org_value,
    };
    let wrong_prover = MockProver::run(9, &wrong_circuit, vec![public_inputs.clone()]).unwrap();
    // println!("wrong_prover: {:?}", wrong_prover);
    println!("circuit is satisfied: {:?}", wrong_prover.verify());
    println!("wrong circuit should not be satisfied");
    // assert_eq!(prover.verify(), Ok(()));

        


}