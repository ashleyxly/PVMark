use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

pub mod fieldutils;

use clap::Parser;
use clap::builder::Str;

// use ff::FieldElement;
// use halo2curves::ff::PrimeField;
// use halo2curves::pasta::Fp as F;

// use halo2_merkle_tree::chips::poseidon;

use halo2_merkle_tree::chips::merkle_v3::MerkleTreeV3Circuit;
use halo2_merkle_tree::poseidon::spec::PoseidonSpec;
// use halo2_gadgets::poseidon::{
//     primitives::{self as poseidon1, ConstantLength, P128Pow5T3 as OrchardNullifier, Spec},
//     Hash,
// };
// use halo2_proofs::{circuit::Value, dev::MockProver, pasta::Fp};
use std::time::{Duration, Instant};

// use halo2_proofs::pasta::{Eq, EqAffine};
// use halo2_proofs::plonk::{
//     create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, BatchVerifier, Circuit,
//     Column, ConstraintSystem, Error, Fixed, SingleVerifier, TableColumn, VerificationStrategy,
// };
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
use halo2_proofs::{circuit::*, plonk::*};
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
use std::error::Error;
// use std::fs::File;
// use std::io::BufReader;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;


// use halo2_proofs::{circuit::Value, dev::MockProver};
const WIDTH: usize = 5;
const RATE: usize = 4;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
//    /// File to read
//    filename: String,

    /// The path of onnx file
   #[arg(short, long, default_value = "/home/username/Codes/halo2-merkle-tree/onnx_test/network.onnx")]
   network_onnx_file_path: String,

   /// Length of characters to seek
   #[arg(short, long, default_value = "/home/username/Codes/halo2-merkle-tree/params.txt")]
   output_file_path: String,

   /// Scale factor
   #[arg(short, long, default_value_t = 4)]
   scale: i32,

   /// The path of proof
   #[arg(short, long, default_value = "/home/username/Codes/halo2-merkle-tree/onnx_test/proof.bin")]
   proof_path: String,

}



fn read_file_numbers(path: &str) -> Vec<f64> {
    let file = File::open(path).expect("failed to open file");
    let reader = BufReader::new(file);
    let mut numbers = Vec::new();

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let mut words = line.split_whitespace();

        while let Some(word) = words.next() {
            if let Ok(number) = word.parse::<f64>() {
                numbers.push(number);
            }
        }
    }

    numbers
}


fn call_python_function(onnx_file_path: String, output_file_path: String) {
    let output = Command::new("python")
        // .arg("-c")
        // .arg("from read_onnx_file import test_hello_world; print(test_hello_world())")
        .arg("/home/username/Codes/halo2-merkle-tree/src/read_onnx_file.py")
        .arg("-Net")
        .arg(onnx_file_path)
        .arg("-P")
        .arg(output_file_path)
        .output()
        .expect("failed to execute process");

    // println!("finished running python script");
    // io::stdout().flush().unwrap();
    // let result = String::from_utf8_lossy(&output.stdout);
    // println!("{}", result);

    // let mut file = File::create("result.txt").expect("failed to create file");
    // file.write_all(result.as_bytes()).expect("failed to write to file");
}

fn compute_merkle_root(leaf: &u64, elements: &Vec<u64>, indices: &Vec<u64>) -> Fp {
    let remainder = elements.len() % 3;
    let flag = if remainder == 0 {
        0
    } else {
        1
    };
    let layers = if elements.len() % 3 == 0 {
        elements.len() / 3
    } else {
        elements.len() / 3 + 1
    };
    
    // let k = elements.len();
    let mut digest = Fp::from(leaf.clone());
    // let mut message: [Fp; 2];
    let mut message: [Fp; 4] = [Fp::from(0), Fp::from(0), Fp::from(0), Fp::from(0)];
    let mut index = 0;
    for i in 0..layers {
        if i == 0 {
            message = [digest, Fp::from(elements[index]), Fp::from(elements[index+1]), Fp::from(elements[index+2])];
            index = index + 3;

        }
        else if i == layers - 1 {
            if flag == 0 {
                message = [digest, Fp::from(elements[index]), Fp::from(elements[index + 1]), Fp::from(elements[index + 2])];
            }
            else {
                if remainder == 1 {
                    message = [digest, Fp::from(elements[index]), Fp::from(0), Fp::from(0)];
                }
                else if remainder == 2 {
                    message = [digest, Fp::from(elements[index]), Fp::from(elements[index + 1]), Fp::from(0)];
                }
                else {
                    println!("error");
                }
            }
        }
        else {
            message = [digest, Fp::from(elements[index]), Fp::from(elements[index + 1]), Fp::from(elements[index + 2])];
            index = index + 3;
        }
        // digest = poseidon1::Hash::<_, OrchardNullifier, ConstantLength<4>, 5, 4>::init()
        //     .hash(message);
        digest = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(message);
    }

    return digest;

}

fn compute_merkle_root_fp(leaf: &Fp, elements: &Vec<Fp>, indices: &Vec<u64>) -> Fp {
    let remainder = elements.len() % 3;
    let flag = if remainder == 0 {
        0
    } else {
        1
    };
    let layers = if elements.len() % 3 == 0 {
        elements.len() / 3
    } else {
        elements.len() / 3 + 1
    };
    
    // let k = elements.len();
    let mut digest = Fp::from(leaf.clone());
    // let mut message: [Fp; 2];
    let mut message: [Fp; 4] = [Fp::from(0), Fp::from(0), Fp::from(0), Fp::from(0)];
    let mut index = 0;
    for i in 0..layers {
        if i == 0 {
            message = [digest, Fp::from(elements[index]), Fp::from(elements[index+1]), Fp::from(elements[index+2])];
            index = index + 3;

        }
        else if i == layers - 1 {
            if flag == 0 {
                message = [digest, Fp::from(elements[index]), Fp::from(elements[index + 1]), Fp::from(elements[index + 2])];
            }
            else {
                if remainder == 1 {
                    message = [digest, Fp::from(elements[index]), Fp::from(0), Fp::from(0)];
                }
                else if remainder == 2 {
                    message = [digest, Fp::from(elements[index]), Fp::from(elements[index + 1]), Fp::from(0)];
                }
                else {
                    println!("error");
                }
            }
        }
        else {
            message = [digest, Fp::from(elements[index]), Fp::from(elements[index + 1]), Fp::from(elements[index + 2])];
            index = index + 3;
        }
        digest = halo2_gadgets::poseidon::primitives::Hash::<_, PoseidonSpec, ConstantLength<RATE>, WIDTH, RATE>::init()
            .hash(message);
    }

    return digest;
}

pub fn gen_srs<Scheme: CommitmentScheme>(k: u32) -> Scheme::ParamsProver {
    Scheme::ParamsProver::new(k)
}

pub fn load_srs<Scheme: CommitmentScheme>(
    path: PathBuf,
) -> Result<Scheme::ParamsVerifier, Box<dyn Error>> {
    println!("loading srs from {:?}", path);
    let f = File::open(path.clone())
        .map_err(|_| format!("failed to load srs at {}", path.display()))?;
    let mut reader = BufReader::new(f);
    Params::<'_, Scheme::Curve>::read(&mut reader).map_err(Box::<dyn Error>::from)
}

/// helper function for load_params
pub(crate) fn load_params_cmd(
    srs_path: PathBuf,
    logrows: u32,
// ) -> ParamsKZG<Bn256> {
) -> Result<ParamsKZG<Bn256>, Box<dyn Error>> {
    let mut params: ParamsKZG<Bn256> = load_srs::<KZGCommitmentScheme<Bn256>>(srs_path)?;
    println!("downsizing params to {} logrows", logrows);
    if logrows < params.k() {
        params.downsize(logrows);
    }
    Ok(params)
    // params
}

/// Creates a [VerifyingKey] and [ProvingKey] for a [crate::graph::GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
// pub fn create_keys<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
//     circuit: &C,
//     params: &'_ Scheme::ParamsProver,
// ) -> Result<ProvingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
// where
//     C: Circuit<Scheme::Scalar>,
//     <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
// {
//     //	Real proof
//     let empty_circuit = <C as Circuit<F>>::without_witnesses(circuit);

//     // Initialize verifying key
//     let now = Instant::now();
//     trace!("preparing VK");
//     let vk = keygen_vk(params, &empty_circuit)?;
//     let elapsed = now.elapsed();
//     info!("VK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());

//     // Initialize the proving key
//     let now = Instant::now();
//     let pk = keygen_pk(params, vk, &empty_circuit)?;
//     let elapsed = now.elapsed();
//     info!("PK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());
//     Ok(pk)
// }

// fn test_field_element() {

// }

fn merkle_tree_poseidon_hash(leaf: Fp, elements: Vec<Fp>, indices: Vec<u64>, proof_path: String)  {
    // let leaf = 99u64;
    // let elements = vec![1u64, 5u64, 6u64, 9u64, 9u64];
    // let indices = vec![0u64, 0u64, 0u64, 0u64, 0u64];
    let digest = compute_merkle_root_fp(&leaf, &elements, &indices);

    let leaf_fp = Value::known(leaf);
    let elements_fp: Vec<Value<Fp>> = elements
            .iter()
            .map(|x| Value::known(x.to_owned()))
            .collect();
    let indices_fp: Vec<Value<Fp>> = indices
        .iter()
        .map(|x| Value::known(Fp::from(x.to_owned())))
        .collect();


    let circuit = MerkleTreeV3Circuit {
        leaf: leaf_fp,
        elements: elements_fp,
        // indices: indices_fp,
    };

    // let correct_public_input = vec![Fp::from(leaf), Fp::from(digest)];
    // let correct_public_input = vec![digest];
    let correct_public_input = vec![digest];

    let mut k = 21;
    println!("k = {}", k);

    use plotters::prelude::*;
    let root = BitMapBackend::new("layout_22.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root = root
        .titled("Example Circuit Layout", ("sans-serif", 60))
        .unwrap();


    // halo2_proofs::dev::CircuitLayout::default().render(k, &circuit, &root).unwrap();
    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(k, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    // std::process::exit(0);
    let mut new_k = 1;
    let mut pow_new_k = 2;
    while pow_new_k < total_row {
        pow_new_k <<= 1;
        new_k += 1;
    }
    println!("change k = {} to k = {}", k, new_k);
    k = new_k;

    
    let mock_start_time = Instant::now(); // Start time
    let correct_prover = MockProver::run(
        // 15,
        k,
        &circuit,
        vec![correct_public_input.clone(), correct_public_input.clone()],
        // vec![correct_public_input.clone()],
    )
    .unwrap();
    let mock_elapsed_time = mock_start_time.elapsed();
    println!("Running Mock took {:.4} seconds.", mock_elapsed_time.as_millis() as f64 / 1000.0);

    correct_prover.assert_satisfied();
    println!("success");

    


    // let k = 18;
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
    // Create a proof
    let instance_temp = vec![vec![digest.clone()], vec![digest.clone()]];
    // let temp_inner = instance_temp
    // .iter()
    // .map(|e| e.deref())
    // .collect::<Vec<&[Scheme::Scalar]>>();
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


fn main() {
    let args = Args::parse();
    // println!("{:?}", args);
    let scale_factor = args.scale;
    let output_file_path_str = args.output_file_path.clone();
    let proof_path_str = args.proof_path.clone();
    call_python_function(args.network_onnx_file_path, args.output_file_path);
    
    /*test2023.9.4 */
    // let numbers = read_file_numbers(output_file_path_str.as_str());
    let num = 140000; // 指定生成元素的数量
    let mut rng = rand::thread_rng(); // 创建随机数生成器

    let numbers: Vec<f64> = (0..num).map(|_| rng.gen()).collect();


    /*test2023.8.4 */
    println!("numbers: {:?}", numbers.len());

    let mut model_params = Vec::new();
    // let test_fp = Fp::from((-3.1415 * 10.0_f64.powi(scale_factor)).abs() as u64);
    // println!("test_fp: {:?}", test_fp);
    // println!("test_fp: {:?}", test_fp.neg());
    // println!("after--test_fp: {:?}", test_fp);
    

    for num in numbers {
        let scaled_num = num * 10.0_f64.powi(scale_factor);
        let mut fp_temp = Fp::from(scaled_num.abs() as u64);
        // fp_temp.neg();
        // let i128_num = scaled_num as i128;
        // let fe: Fp = fieldutils::i128_to_felt(i128_num);
        if (scaled_num) < 0.0 {
            fp_temp = fp_temp.neg();
        }
        model_params.push(fp_temp);
    }
    let model_params_fp: Vec<Fp> = model_params.clone();
    
    let len_model_params = model_params.len();

    let mut file = File::create("numbers.txt").expect("failed to create file");

    for num in model_params {
        writeln!(file, "{:?}", num).expect("failed to write to file");
    }

    let leaf = 9u64;
    let indices = vec![0u64; len_model_params];
    
    // test
    // let leaf = 99u64;
    // let elements = vec![1u64, 5u64, 6u64, 9u64, 9u64];
    // let indices = vec![0u64, 0u64, 0u64, 0u64, 0u64];
    // let mut elements_fp = Vec::new();
    // for element in elements {
    //     let ele_temp = Fp::from(element);
    //     elements_fp.push(ele_temp);
    // }
    
    println!("Start running merkle_tree_poseidon_hash()...");

    let start_time = Instant::now(); // Start time

    merkle_tree_poseidon_hash(Fp::from(leaf), model_params_fp, indices, proof_path_str);

    // let end_time = Instant::now(); // 
    let elapsed_time = start_time.elapsed(); 

    println!("Running merkle_tree_poseidon_hash() took {} seconds.", elapsed_time.as_secs());

    // let mut file = File::create("result.txt").expect("failed to create file");
    // file.write_all(number in numbers_u64).expect("failed to write to file");




}





