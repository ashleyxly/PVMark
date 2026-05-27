use ark_ff::{FpParameters, PrimeField};
use ark_r1cs_std::{fields::fp::FpVar, prelude::Boolean, uint8::UInt8, ToBitsGadget};
use ark_relations::r1cs::SynthesisError;
use ark_std::vec::Vec;

pub fn to_field_elements<F: PrimeField>(bytes: &[u8]) -> Vec<F> {
    let max_size = (<F::Params as FpParameters>::CAPACITY / 8) as usize;
    bytes
        .chunks(max_size + 1)
        .map(|chunk| F::from_le_bytes_mod_order(chunk))
        .collect::<Vec<_>>()
}

pub fn to_field_elements_r1cs<F: PrimeField>(
    bytes: &[UInt8<F>],
) -> Result<Vec<FpVar<F>>, SynthesisError> {
    let max_size = (<F::Params as FpParameters>::CAPACITY / 8) as usize;
    bytes
        .chunks(max_size + 1)
        .map(|chunk| Boolean::le_bits_to_fp_var(&chunk.to_bits_le()?))
        .collect::<Result<Vec<_>, _>>()
}



use std::{error::Error, str::FromStr};

use ark_bn254::Fr;
use ark_crypto_primitives::crh::TwoToOneCRH;
use ark_ff::{to_bytes, One, Zero};

use crate::{params::round_keys_contants_to_vec, MiMC, MiMCFeistelCRH, MiMCNonFeistelCRH};
use ark_crypto_primitives::CRH;


pub fn mimc_hash_non_feistel(input1: String, input2: String, input3: Option<String>) -> String {
    use crate::params::mimc_7_91_bn254::{MIMC_7_91_BN254_PARAMS, MIMC_7_91_BN254_ROUND_KEYS};

    let param = MiMC::<Fr, MIMC_7_91_BN254_PARAMS>::new(
        1,
        Fr::zero(),
        round_keys_contants_to_vec(&MIMC_7_91_BN254_ROUND_KEYS),
    );

    let mut final_result = String::new();

    match input3 {
        Some(input3) => {
            let mut inputs = Vec::new();
            // let input1 = to_bytes!(Fr::from_str(&input1).unwrap()).unwrap();
            // let input2 = to_bytes!(Fr::from_str(&input2).unwrap()).unwrap();
            // let input3 = to_bytes!(Fr::from_str(&input3).unwrap()).unwrap();
            // inputs.concat();

            inputs.push(to_bytes!(Fr::from_str(&input1).unwrap()).unwrap());
            inputs.push(to_bytes!(Fr::from_str(&input2).unwrap()).unwrap());
            inputs.push(to_bytes!(Fr::from_str(&input3).unwrap()).unwrap());
            let mut merged_inputs = inputs.concat();

            let result = <MiMCNonFeistelCRH<Fr, MIMC_7_91_BN254_PARAMS> as CRH>::evaluate(
                &param,
                &merged_inputs,
            ).unwrap();

            // println!("{result}");

            final_result = result.into_repr().to_string();
        },
        None => {
            let mut inputs = Vec::new();
            // let input1 = to_bytes!(Fr::from_str(&input1).unwrap()).unwrap();
            // let input2 = to_bytes!(Fr::from_str(&input2).unwrap()).unwrap();
            // let input3 = to_bytes!(Fr::from_str(&input3).unwrap()).unwrap();
            // inputs.concat();

            inputs.push(to_bytes!(Fr::from_str(&input1).unwrap()).unwrap());
            inputs.push(to_bytes!(Fr::from_str(&input2).unwrap()).unwrap());

            let mut merged_inputs = inputs.concat();

            let result = <MiMCNonFeistelCRH<Fr, MIMC_7_91_BN254_PARAMS> as CRH>::evaluate(
                &param,
                &merged_inputs,
            ).unwrap();

            // println!("{result}");

            final_result = result.into_repr().to_string();
        }
    }
    final_result

}