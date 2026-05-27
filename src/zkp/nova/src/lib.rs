use std::{
    collections::HashMap,
    env::current_dir,
    fs,
    path::{Path, PathBuf},
};

use crate::circom::reader::generate_witness_from_bin;
use circom::circuit::{CircomCircuit, R1CS};
use ff::Field;
use nova_snark::{
    traits::{circuit::TrivialTestCircuit, Group},
    PublicParams, RecursiveSNARK,
};
use num_bigint::BigInt;
use num_traits::Num;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(not(target_family = "wasm"))]
use crate::circom::reader::generate_witness_from_wasm;

#[cfg(target_family = "wasm")]
use crate::circom::wasm::generate_witness_from_wasm;

pub mod circom;

pub type F<G> = <G as Group>::Scalar;
pub type EE<G> = nova_snark::provider::ipa_pc::EvaluationEngine<G>;
pub type S<G> = nova_snark::spartan::snark::RelaxedR1CSSNARK<G, EE<G>>;
pub type C1<G> = CircomCircuit<<G as Group>::Scalar>;
pub type C2<G> = TrivialTestCircuit<<G as Group>::Scalar>;

#[derive(Clone)]
pub enum FileLocation {
    PathBuf(PathBuf),
    URL(String),
}

pub fn create_public_params<G1, G2>(r1cs: R1CS<F<G1>>) -> PublicParams<G1, G2, C1<G1>, C2<G2>>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{
    let circuit_primary = CircomCircuit {
        r1cs,
        witness: None,
    };
    let circuit_secondary = TrivialTestCircuit::default();

    PublicParams::setup(circuit_primary.clone(), circuit_secondary.clone())
}

#[derive(Serialize, Deserialize)]
struct CircomInput {
    step_in: Vec<String>,

    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[cfg(not(target_family = "wasm"))]
fn compute_witness<G1, G2>(
    current_public_input: Vec<String>,
    private_input: HashMap<String, Value>,
    witness_generator_file: FileLocation,
    witness_generator_output: &Path,
) -> Vec<<G1 as Group>::Scalar>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{
    let decimal_stringified_input: Vec<String> = current_public_input
        .iter()
        .map(|x| BigInt::from_str_radix(x, 16).unwrap().to_str_radix(10))
        .collect();

    let input = CircomInput {
        step_in: decimal_stringified_input.clone(),
        extra: private_input.clone(),
    };

    let is_wasm = match &witness_generator_file {
        FileLocation::PathBuf(path) => path.extension().unwrap_or_default() == "wasm",
        FileLocation::URL(_) => true,
    };
    let input_json = serde_json::to_string(&input).unwrap();

    if is_wasm {
        generate_witness_from_wasm::<F<G1>>(
            &witness_generator_file,
            &input_json,
            &witness_generator_output,
        )
    } else {
        let witness_generator_file = match &witness_generator_file {
            FileLocation::PathBuf(path) => path,
            FileLocation::URL(_) => panic!("unreachable"),
        };
        generate_witness_from_bin::<F<G1>>(
            &witness_generator_file,
            &input_json,
            &witness_generator_output,
        )
    }
}

#[cfg(target_family = "wasm")]
async fn compute_witness<G1, G2>(
    current_public_input: Vec<String>,
    private_input: HashMap<String, Value>,
    witness_generator_file: FileLocation,
) -> Vec<<G1 as Group>::Scalar>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{
    let decimal_stringified_input: Vec<String> = current_public_input
        .iter()
        .map(|x| BigInt::from_str_radix(x, 16).unwrap().to_str_radix(10))
        .collect();

    let input = CircomInput {
        step_in: decimal_stringified_input.clone(),
        extra: private_input.clone(),
    };

    let is_wasm = match &witness_generator_file {
        FileLocation::PathBuf(path) => path.extension().unwrap_or_default() == "wasm",
        FileLocation::URL(_) => true,
    };
    let input_json = serde_json::to_string(&input).unwrap();

    if is_wasm {
        generate_witness_from_wasm::<F<G1>>(
            &witness_generator_file,
            &input_json,
        )
        .await
    } else {
        let root = current_dir().unwrap(); // compute path only when generating witness from a binary
        let witness_generator_output = root.join("circom_witness.wtns");
        let witness_generator_file = match &witness_generator_file {
            FileLocation::PathBuf(path) => path,
            FileLocation::URL(_) => panic!("unreachable"),
        };
        generate_witness_from_bin::<F<G1>>(
            &witness_generator_file,
            &input_json,
            &witness_generator_output,
        )
    }
}

#[cfg(not(target_family = "wasm"))]
pub fn create_recursive_circuit<G1, G2>(
    witness_generator_file: FileLocation,
    r1cs: R1CS<F<G1>>,
    private_inputs: Vec<HashMap<String, Value>>,
    start_public_input: Vec<F<G1>>,
    pp: &PublicParams<G1, G2, C1<G1>, C2<G2>>,
) -> Result<RecursiveSNARK<G1, G2, C1<G1>, C2<G2>>, std::io::Error>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{
    let root = current_dir().unwrap();
    let witness_generator_output = root.join("circom_witness.wtns");

    let iteration_count = private_inputs.len();

    let start_public_input_hex = start_public_input
        .iter()
        .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
        .collect::<Vec<String>>();
    let mut current_public_input = start_public_input_hex.clone();

    let witness_0 = compute_witness::<G1, G2>(
        current_public_input.clone(),
        private_inputs[0].clone(),
        witness_generator_file.clone(),
        &witness_generator_output,
    );

    let circuit_0 = CircomCircuit {
        r1cs: r1cs.clone(),
        witness: Some(witness_0),
    };
    let circuit_secondary = TrivialTestCircuit::default();
    let z0_secondary = vec![G2::Scalar::ZERO];

    let mut recursive_snark = RecursiveSNARK::<G1, G2, C1<G1>, C2<G2>>::new(
        &pp,
        &circuit_0,
        &circuit_secondary,
        start_public_input.clone(),
        z0_secondary.clone(),
    );

    for i in 0..iteration_count {
        let witness = compute_witness::<G1, G2>(
            current_public_input.clone(),
            private_inputs[i].clone(),
            witness_generator_file.clone(),
            &witness_generator_output,
        );

        let circuit = CircomCircuit {
            r1cs: r1cs.clone(),
            witness: Some(witness),
        };

        let current_public_output = circuit.get_public_outputs();
        current_public_input = current_public_output
            .iter()
            .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
            .collect();

        let res = recursive_snark.prove_step(
            &pp,
            &circuit,
            &circuit_secondary,
            start_public_input.clone(),
            z0_secondary.clone(),
        );
        assert!(res.is_ok());
    }
    fs::remove_file(witness_generator_output)?;

    Ok(recursive_snark)
}

#[cfg(target_family = "wasm")]
pub async fn create_recursive_circuit<G1, G2>(
    witness_generator_file: FileLocation,
    r1cs: R1CS<F<G1>>,
    private_inputs: Vec<HashMap<String, Value>>,
    start_public_input: Vec<F<G1>>,
    pp: &PublicParams<G1, G2, C1<G1>, C2<G2>>,
) -> Result<RecursiveSNARK<G1, G2, C1<G1>, C2<G2>>, std::io::Error>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{

    let iteration_count = private_inputs.len();

    let start_public_input_hex = start_public_input
        .iter()
        .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
        .collect::<Vec<String>>();
    let mut current_public_input = start_public_input_hex.clone();

    let witness_0 = compute_witness::<G1, G2>(
        current_public_input.clone(),
        private_inputs[0].clone(),
        witness_generator_file.clone(),
    )
    .await;

    let circuit_0 = CircomCircuit {
        r1cs: r1cs.clone(),
        witness: Some(witness_0),
    };
    let circuit_secondary = TrivialTestCircuit::default();
    let z0_secondary = vec![G2::Scalar::ZERO];

    let mut recursive_snark = RecursiveSNARK::<G1, G2, C1<G1>, C2<G2>>::new(
        &pp,
        &circuit_0,
        &circuit_secondary,
        start_public_input.clone(),
        z0_secondary.clone(),
    );

    for i in 0..iteration_count {
        let witness = compute_witness::<G1, G2>(
            current_public_input.clone(),
            private_inputs[i].clone(),
            witness_generator_file.clone(),
        )
        .await;

        let circuit = CircomCircuit {
            r1cs: r1cs.clone(),
            witness: Some(witness),
        };

        let current_public_output = circuit.get_public_outputs();
        current_public_input = current_public_output
            .iter()
            .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
            .collect();

        let res = recursive_snark.prove_step(
            &pp,
            &circuit,
            &circuit_secondary,
            start_public_input.clone(),
            z0_secondary.clone(),
        );
        assert!(res.is_ok());
    }

    Ok(recursive_snark)
}

#[cfg(not(target_family = "wasm"))]
pub fn continue_recursive_circuit<G1, G2>(
    recursive_snark: &mut RecursiveSNARK<G1, G2, C1<G1>, C2<G2>>,
    last_zi: Vec<F<G1>>,
    witness_generator_file: FileLocation,
    r1cs: R1CS<F<G1>>,
    private_inputs: Vec<HashMap<String, Value>>,
    start_public_input: Vec<F<G1>>,
    pp: &PublicParams<G1, G2, C1<G1>, C2<G2>>,
) -> Result<(), std::io::Error>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{
    let root = current_dir().unwrap();
    let witness_generator_output = root.join("circom_witness.wtns");

    let iteration_count = private_inputs.len();

    let mut current_public_input = last_zi
        .iter()
        .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
        .collect::<Vec<String>>();

    let circuit_secondary = TrivialTestCircuit::default();
    let z0_secondary = vec![G2::Scalar::ZERO];

    for i in 0..iteration_count {
        let witness = compute_witness::<G1, G2>(
            current_public_input.clone(),
            private_inputs[i].clone(),
            witness_generator_file.clone(),
            &witness_generator_output,
        );

        let circuit = CircomCircuit {
            r1cs: r1cs.clone(),
            witness: Some(witness),
        };

        let current_public_output = circuit.get_public_outputs();
        current_public_input = current_public_output
            .iter()
            .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
            .collect();

        let res = recursive_snark.prove_step(
            pp,
            &circuit,
            &circuit_secondary,
            start_public_input.clone(),
            z0_secondary.clone(),
        );

        assert!(res.is_ok());
    }

    fs::remove_file(witness_generator_output)?;

    Ok(())
}

#[cfg(target_family = "wasm")]
pub async fn continue_recursive_circuit<G1, G2>(
    recursive_snark: &mut RecursiveSNARK<G1, G2, C1<G1>, C2<G2>>,
    last_zi: Vec<F<G1>>,
    witness_generator_file: FileLocation,
    r1cs: R1CS<F<G1>>,
    private_inputs: Vec<HashMap<String, Value>>,
    start_public_input: Vec<F<G1>>,
    pp: &PublicParams<G1, G2, C1<G1>, C2<G2>>,
) -> Result<(), std::io::Error>
where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
{
    let root = current_dir().unwrap();
    let witness_generator_output = root.join("circom_witness.wtns");

    let iteration_count = private_inputs.len();

    let mut current_public_input = last_zi
        .iter()
        .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
        .collect::<Vec<String>>();

    let circuit_secondary = TrivialTestCircuit::default();
    let z0_secondary = vec![G2::Scalar::ZERO];

    for i in 0..iteration_count {
        let witness = compute_witness::<G1, G2>(
            current_public_input.clone(),
            private_inputs[i].clone(),
            witness_generator_file.clone(),
        )
        .await;

        let circuit = CircomCircuit {
            r1cs: r1cs.clone(),
            witness: Some(witness),
        };

        let current_public_output = circuit.get_public_outputs();
        current_public_input = current_public_output
            .iter()
            .map(|&x| format!("{:?}", x).strip_prefix("0x").unwrap().to_string())
            .collect();

        let res = recursive_snark.prove_step(
            pp,
            &circuit,
            &circuit_secondary,
            start_public_input.clone(),
            z0_secondary.clone(),
        );

        assert!(res.is_ok());
    }

    fs::remove_file(witness_generator_output)?;

    Ok(())
}


// pub fn two_inputs_hash_computation_decimal(input1: String, input2: String, hash_type: HashType) -> String {
//     let mut final_result = String::new();
//     match hash_type {
//         HashType::SHA256 => {
//             // let mut hasher = Sha256::new();
//             // let mut inputs: Vec<u8> = vec![];
//             // let input1_bytes = input1.as_bytes();
//             // let input2_bytes = input2.as_bytes();
//             // inputs.append(&mut input1_bytes.to_vec());
//             // inputs.append(&mut input2_bytes.to_vec());
//             // hasher.input(&inputs);

//             // let result = hasher.result_str();
//             // let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
//             // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
//             // let result2 = decimal_number % modulo;
//             // // let hex = format!("{:x}", result2);
//             // final_result = result2.to_string();

//         }
//         HashType::BLAKE2b => {
//             // let mut hasher = Blake2b::new(32);
//             // let mut inputs: Vec<u8> = vec![];
//             // let input1_bytes = input1.as_bytes();
//             // let input2_bytes = input2.as_bytes();
//             // inputs.append(&mut input1_bytes.to_vec());
//             // inputs.append(&mut input2_bytes.to_vec());
//             // hasher.input(&inputs);
//             // // let mut out = [0u8; 32];
//             // // hasher.result(&mut out);
//             // let result = hasher.result_str();
//             // let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
//             // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
//             // let result2 = decimal_number % modulo;
//             // // let hex = format!("{:x}", result2);
//             // final_result = result2.to_string();
            
//         }
//         HashType::KECCAK256 => {
//             // let mut hasher = Sha3::keccak256();
//             // let mut inputs: Vec<u8> = vec![];
//             // let input1_bytes = input1.as_bytes();
//             // let input2_bytes = input2.as_bytes();
//             // inputs.append(&mut input1_bytes.to_vec());
//             // inputs.append(&mut input2_bytes.to_vec());
//             // hasher.input(&inputs);
//             // // let mut out = [0u8; 32];
//             // // hasher.result(&mut out);
//             // let result = hasher.result_str();
//             // let decimal_number = BigUint::parse_bytes(result.as_bytes(), 16).unwrap();
//             // let modulo = BigUint::from_str_radix("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10).unwrap();
//             // let result2 = decimal_number % modulo;
//             // // let hex = format!("{:x}", result2);
//             // final_result = result2.to_string();
//         }

//         HashType::POSEIDON => {
//             let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
//             let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
//                 .hash(input);
//             // let res_string = little_endian_u8_array_to_string(&result.to_bytes());
//             let res_string = little_endian_u8_array_to_decimal_string(&result.to_bytes());
//             final_result = res_string.clone();
//         }
//         HashType::POSEIDON2 => {
//             // let poseidon = Poseidon2::new(&POSEIDON_2_BN256_PARAMS_T_2);
//             // let mut sk = Poseidon2_Fr::from_str(&input1).unwrap();
//             // let input: Vec<Poseidon2_Fr> = vec![sk.clone(), Poseidon2_Fr::from_str(&input2).unwrap()];
//             // let perm = poseidon.permutation(&input);
//             // let result = perm[0];
//             // let result_bigint = result.into_bigint();
//             // let result_bigint_string = result_bigint.to_string();
//             // let decimal_bigint = BigUint::parse_bytes(result_bigint_string.as_bytes(), 10).unwrap();
//             // // let hex_string = format!("{:x}", decimal_bigint);
//             // final_result = decimal_bigint.to_string();
//         }
//         HashType::MIMC => {
//         //     let res = mimc_hash_non_feistel(input1, input2, None);
//         //     let decimal_number = BigUint::from_str_radix(res.as_str(), 16).unwrap();
//         //     final_result = decimal_number.to_string();
//         // }
//     }
//     final_result
// }

use num_bigint::BigUint;
// use hash_rustlib::poseidon_fast::spec_width_3::PoseidonSpec as PoseidonSpec3;
// use hash_rustlib::poseidon_fast::spec_width_4::PoseidonSpec as PoseidonSpec4;
// use hash_rustlib::poseidon_fast::poseidon_params_width_3;
use poseidon_fast::spec_width_3::PoseidonSpec as PoseidonSpec3;
use poseidon_fast::spec_width_4::PoseidonSpec as PoseidonSpec4;
use poseidon_fast::poseidon_params_width_3;
use halo2_gadgets::poseidon::primitives::ConstantLength;

pub mod poseidon_fast;

pub fn from_str_to_fr(s: &str) -> Result<halo2curves::bn256::Fr, ()> {
    if s.is_empty() {
        return Err(());
    }

    if s == "0" {
        return Ok(halo2curves::bn256::Fr::zero());
    }

    let mut res = halo2curves::bn256::Fr::zero();

    let ten = halo2curves::bn256::Fr::from(10u64);

    let mut first_digit = true;

    for c in s.chars() {
        match c.to_digit(10) {
            Some(c) => {
                if first_digit {
                    if c == 0 {
                        return Err(());
                    }

                    first_digit = false;
                }

                // res.mul_assign(&ten);
                res = res * ten;
                let digit = halo2curves::bn256::Fr::from(u64::from(c));
                // res.add_assign(&digit);
                res = res + digit;
            },
            None => {
                return Err(());
            },
        }
    }
    // if res.is_geq_modulus() {
    //     Err(())
    // } else {
    //     Ok(res)
    // }
    Ok(res)
}

pub fn little_endian_u8_array_to_decimal_string(bytes: &[u8; 32]) -> String {
    let mut hex_string = String::new();

    for &byte in bytes.iter().rev() {
        hex_string.push_str(&format!("{:02x}", byte));
    }

    BigUint::parse_bytes(hex_string.as_bytes(), 16).unwrap().to_string()
}


pub fn two_inputs_hash_computation_decimal(input1: String, input2: String) -> String {
    let mut final_result = String::new();


    let mut input = [from_str_to_fr(&input1).unwrap(), from_str_to_fr(&input2).unwrap()];
    let mut result = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec3, ConstantLength<2>, 3, 2>::init()
        .hash(input);
    // let res_string = little_endian_u8_array_to_string(&result.to_bytes());
    let res_string = little_endian_u8_array_to_decimal_string(&result.to_bytes());
    final_result = res_string.clone();
        

    final_result
}