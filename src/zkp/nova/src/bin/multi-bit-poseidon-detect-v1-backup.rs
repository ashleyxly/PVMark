use std::{collections::HashMap, env::current_dir, time::Instant};

use ff::derive::bitvec::vec;
use nova_scotia::{
    circom::reader::load_r1cs, continue_recursive_circuit, create_public_params,
    create_recursive_circuit, FileLocation, F, S,
};
use nova_snark::{provider, CompressedSNARK, PublicParams};
use serde_json::json;
use primitive_types::U256;
use rand::seq::SliceRandom;
use rand::thread_rng;

use clap::Parser;

use std::process::{Command, exit};
use std::env;
use std::io::{self, Write};
use flate2::{write::ZlibEncoder, Compression};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {

    /// The type of sampling
    #[arg(short, long, default_value = "multi-bit-watermark")]
    type_sampling: String,
    /// The number of token per sub-circuit
    #[arg(short, long, default_value_t = 1)]
    partial_seq_len: usize,

    /// The depth of merkle hash tree
    #[arg(short, long, default_value_t = 1)]
    merkle_tree_nlevels: usize,

    /// The information length per position
    #[arg(short, long, default_value_t = 2)]
    each_position_infor: usize,

    /// Folding number
    #[arg(short, long, default_value_t = 3)]
    folding_num: usize,

}



// Function to change the current working directory and handle errors
fn change_directory(dir: &str) -> io::Result<()> {
    println!("Changing directory to: {}", dir);
    std::env::set_current_dir(dir)
}

// Function to run a shell command and handle success/failure
fn run_command(command: &str, args: &[&str]) -> Result<(), String> {
    let output = Command::new(command)
        .args(args)
        .output();

    match output {
        Ok(output) if output.status.success() => {
            println!("Command '{}' executed successfully.", command);
            Ok(())
        },
        Ok(output) => {
            let error_message = String::from_utf8_lossy(&output.stderr).to_string();
            Err(format!("Command '{}' failed with exit code: {}\n{}", command, output.status, error_message))
        },
        Err(e) => Err(format!("Failed to execute '{}': {}", command, e)),
    }
}


// Function to execute the full workflow with parameters
fn execute_workflow(mimc_dir: &str, detect_cpp_dir: &str, file_name: &str) -> Result<(), String> {
    // Step 1: Change directory to the mimc directory
    // change_directory(mimc_dir).map_err(|e| format!("Failed to change directory to '{}': {}", mimc_dir, e))?;
    let args_1 = &[mimc_dir];
    println!("args_1 = {:?}", args_1);
    run_command("cd", args_1)?;

    // Step 2: Run the 'circom' command with the given arguments
    let circom_args = &[file_name, "--r1cs", "--sym", "--c"];
    println!("circom_args = {:?}", circom_args);
    run_command("circom", circom_args)?;

    // Step 3: Change directory to the detect_recursive_cpp directory
    // change_directory(detect_cpp_dir).map_err(|e| format!("Failed to change directory to '{}': {}", detect_cpp_dir, e))?;
    let args_2 = &[detect_cpp_dir];
    run_command("cd", args_2)?;

    // Step 4: Run the 'make' command
    run_command("make", &[])?;

    Ok(())
}



fn run_test(circuit_filepath: String, witness_gen_filepath: String, partial_seq_len: usize, each_position_infor: usize, merkle_tree_nlevels: usize, folding_num: usize) {
    type G1 = provider::bn256_grumpkin::bn256::Point;
    type G2 = provider::bn256_grumpkin::grumpkin::Point;

    println!(
        "Running test with witness generator: {} and group: {}",
        witness_gen_filepath,
        std::any::type_name::<G1>()
    );
    // let root = current_dir().unwrap();

    // let circuit_file = root.join(circuit_filepath);
    let circuit_file = circuit_filepath.clone();
    let r1cs = load_r1cs::<G1, G2>(&FileLocation::PathBuf(circuit_file.into()));
    // let witness_generator_file = root.join(witness_gen_filepath);
    let witness_generator_file = witness_gen_filepath.clone();

    println!("r1cs_path = {:?}", circuit_filepath.clone());
    println!("witness_gen_path = {:?}", witness_gen_filepath);

    let mut private_inputs = Vec::new();

    // generate test cases
    let mut key = 123;
    let mut ngrams: Vec<u64> = Vec::new();
    let mut current_token_index = vec![];
    let mut position: Vec<u64> = Vec::new();

    let mut public_root = "7573692340688590766492649878680688150352897113713566229392554765340228435899";
    let mut path_indices: Vec<Vec<u64>> = Vec::new();
    let mut siblings: Vec<Vec<u64>> = Vec::new();

    // let random_seed = 2025;


    for i in 0..partial_seq_len {
        ngrams.push(50);
    }

    for i in 0..partial_seq_len {
        current_token_index.push(100);
    }

    for i in 0..partial_seq_len {
        position.push(0);
    }

    for i in 0..partial_seq_len {
        let mut path_index: Vec<u64> = Vec::new();
        for j in 0..merkle_tree_nlevels {
            path_index.push(0);
        }
        path_indices.push(path_index);
    }

    for i in 0..partial_seq_len {
        let mut sibling: Vec<u64> = Vec::new();
        for j in 0..merkle_tree_nlevels {
            sibling.push(0);
        }
        siblings.push(sibling);
    }

    // let mut current_count_temp = 0;
    let mut current_count_temp: Vec<Vec<u64>> = Vec::new();
    for i in 0..each_position_infor {
        let mut current_count: Vec<u64> = Vec::new();
        current_count.push(0);
        current_count_temp.push(current_count);
    }


    for i in 0..folding_num {
        let mut private_input = HashMap::new();
        // private_input.insert("adder".to_string(), json!(i));
        private_input.insert("current_count".to_string(), json!(current_count_temp.clone()));
        private_input.insert("position".to_string(), json!(position.clone()));
        // depends on the test data
        // current_count_temp += 1;
        for j in 0..each_position_infor {
            current_count_temp[j][0] += 1;
        }
        private_input.insert("output_count".to_string(), json!(current_count_temp.clone()));
        private_input.insert("key".to_string(), json!(key.clone()));
        private_input.insert("ngrams".to_string(), json!(ngrams.clone()));
        private_input.insert("current_token_index".to_string(), json!(current_token_index.clone()));
        private_input.insert("public_root".to_string(), json!(public_root.clone()));
        private_input.insert("pathIndices".to_string(), json!(path_indices.clone()));
        private_input.insert("siblings".to_string(), json!(siblings.clone()));
        private_inputs.push(private_input);
    }

    // let start_public_input = [F::<G1>::from(10), F::<G1>::from(10)];
    let test_string = "20429117652372020057839316435756885130308478474159872910416124230460920935074";
    let test_raw = U256::from_dec_str(test_string).unwrap();
    let start_public_input = [F::<G1>::from_raw(test_raw.0.clone()), F::<G1>::from_raw(test_raw.0.clone()), F::<G1>::from(0u64), F::<G1>::from(0u64)];

    let start = Instant::now();
    let pp: PublicParams<G1, G2, _, _> = create_public_params(r1cs.clone());
    let setup_time = start.elapsed();
    println!("PublicParams creation took {:?}", setup_time);

    println!(
        "Number of constraints per step (primary circuit): {}",
        pp.num_constraints().0
    );
    println!(
        "Number of constraints per step (secondary circuit): {}",
        pp.num_constraints().1
    );

    println!(
        "Number of variables per step (primary circuit): {}",
        pp.num_variables().0
    );
    println!(
        "Number of variables per step (secondary circuit): {}",
        pp.num_variables().1
    );

    println!("Creating a RecursiveSNARK...");
    let start = Instant::now();
    let mut recursive_snark = create_recursive_circuit(
        FileLocation::PathBuf(witness_generator_file.clone().into()),
        r1cs.clone(),
        private_inputs,
        start_public_input.to_vec(),
        &pp,
    )
    .unwrap();
    let create_prove_time = start.elapsed();
    // println!("RecursiveSNARK creation took {:?}", start.elapsed());
    println!("RecursiveSNARK creation took {:?}", create_prove_time);

    // TODO: empty?
    let z0_secondary = [F::<G2>::from(0)];

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let start = Instant::now();
    let res = recursive_snark.verify(&pp, folding_num, &start_public_input, &z0_secondary);
    let verify_recursive_time = start.elapsed();
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res,
        // start.elapsed()
        verify_recursive_time
    );
    assert!(res.is_ok());

    // let z_last = res.unwrap().0;
    // println!("z_last = {:?}", z_last);

    // assert_eq!(z_last[0], F::<G1>::from(20));
    // assert_eq!(z_last[1], F::<G1>::from(70));

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with IPA-PC...");
    let start = Instant::now();
    let (pk, vk) = CompressedSNARK::<_, _, _, _, S<G1>, S<G2>>::setup(&pp).unwrap();
    let setup_time_2 = start.elapsed();
    println!("CompressedSNARK setup took {:?}", setup_time_2);

    let start = Instant::now();
    let res = CompressedSNARK::<_, _, _, _, S<G1>, S<G2>>::prove(&pp, &pk, &recursive_snark);
    let compressed_prove_time = start.elapsed();
    println!(
        "CompressedSNARK::prove: {:?}, took {:?}",
        res.is_ok(),
        // start.elapsed()
        compressed_prove_time
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
    let compressed_snark_encoded = encoder.finish().unwrap();
    println!(
      "CompressedSNARK::len {:?} bytes",
      compressed_snark_encoded.len()
    );

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let start = Instant::now();
    let res = compressed_snark.verify(
        &vk,
        folding_num,
        start_public_input.to_vec(),
        z0_secondary.to_vec(),
    );
    let verify_compressed_time = start.elapsed();
    println!(
        "CompressedSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        // start.elapsed()
        verify_compressed_time
    );
    assert!(res.is_ok());

    let total_setup_time = setup_time + setup_time_2;
    let total_prove_time = create_prove_time + compressed_prove_time;
    let total_verify_time = verify_recursive_time + verify_compressed_time;
    println!("Total setup time: {:?}, consists of {:?} and {:?}", total_setup_time, setup_time, setup_time_2);
    println!("Total prove time: {:?}, consists of {:?} and {:?}", total_prove_time, create_prove_time, compressed_prove_time);
    println!("Total verify time: {:?}, consists of {:?} and {:?}", total_verify_time, verify_recursive_time, verify_compressed_time);

}


fn run_all_position_test(circuit_filepath: String, witness_gen_filepath: String, position_total_num: usize, partial_seq_len: Vec<usize>, each_position_infor: usize, merkle_tree_nlevels: usize, folding_num: Vec<usize>) {
    type G1 = provider::bn256_grumpkin::bn256::Point;
    type G2 = provider::bn256_grumpkin::grumpkin::Point;

    println!(
        "Running test with witness generator: {} and group: {}",
        witness_gen_filepath,
        std::any::type_name::<G1>()
    );
    // let root = current_dir().unwrap();

    // let circuit_file = root.join(circuit_filepath);
    let circuit_file = circuit_filepath.clone();
    let r1cs = load_r1cs::<G1, G2>(&FileLocation::PathBuf(circuit_file.into()));
    // let witness_generator_file = root.join(witness_gen_filepath);
    let witness_generator_file = witness_gen_filepath.clone();

    println!("r1cs_path = {:?}", circuit_filepath.clone());
    println!("witness_gen_path = {:?}", witness_gen_filepath);

    let mut private_inputs = Vec::new();

    // generate test cases
    let mut key = 123;
    let mut ngrams: Vec<u64> = Vec::new();
    let mut current_token_index = vec![];
    let mut position: Vec<u64> = Vec::new();

    let mut public_root = "7573692340688590766492649878680688150352897113713566229392554765340228435899";
    let mut path_indices: Vec<Vec<u64>> = Vec::new();
    let mut siblings: Vec<Vec<u64>> = Vec::new();

    // let random_seed = 2025;


    for i in 0..partial_seq_len {
        ngrams.push(50);
    }

    for i in 0..partial_seq_len {
        current_token_index.push(100);
    }

    for i in 0..partial_seq_len {
        position.push(0);
    }

    for i in 0..partial_seq_len {
        let mut path_index: Vec<u64> = Vec::new();
        for j in 0..merkle_tree_nlevels {
            path_index.push(0);
        }
        path_indices.push(path_index);
    }

    for i in 0..partial_seq_len {
        let mut sibling: Vec<u64> = Vec::new();
        for j in 0..merkle_tree_nlevels {
            sibling.push(0);
        }
        siblings.push(sibling);
    }

    // let mut current_count_temp = 0;
    let mut current_count_temp: Vec<Vec<u64>> = Vec::new();
    for i in 0..each_position_infor {
        let mut current_count: Vec<u64> = Vec::new();
        current_count.push(0);
        current_count_temp.push(current_count);
    }


    for i in 0..folding_num {
        let mut private_input = HashMap::new();
        // private_input.insert("adder".to_string(), json!(i));
        private_input.insert("current_count".to_string(), json!(current_count_temp.clone()));
        private_input.insert("position".to_string(), json!(position.clone()));
        // depends on the test data
        // current_count_temp += 1;
        for j in 0..each_position_infor {
            current_count_temp[j][0] += 1;
        }
        private_input.insert("output_count".to_string(), json!(current_count_temp.clone()));
        private_input.insert("key".to_string(), json!(key.clone()));
        private_input.insert("ngrams".to_string(), json!(ngrams.clone()));
        private_input.insert("current_token_index".to_string(), json!(current_token_index.clone()));
        private_input.insert("public_root".to_string(), json!(public_root.clone()));
        private_input.insert("pathIndices".to_string(), json!(path_indices.clone()));
        private_input.insert("siblings".to_string(), json!(siblings.clone()));
        private_inputs.push(private_input);
    }

    // let start_public_input = [F::<G1>::from(10), F::<G1>::from(10)];
    let test_string = "20429117652372020057839316435756885130308478474159872910416124230460920935074";
    let test_raw = U256::from_dec_str(test_string).unwrap();
    let start_public_input = [F::<G1>::from_raw(test_raw.0.clone()), F::<G1>::from_raw(test_raw.0.clone()), F::<G1>::from(0u64), F::<G1>::from(0u64)];

    let start = Instant::now();
    let pp: PublicParams<G1, G2, _, _> = create_public_params(r1cs.clone());
    let setup_time = start.elapsed();
    println!("PublicParams creation took {:?}", setup_time);

    println!(
        "Number of constraints per step (primary circuit): {}",
        pp.num_constraints().0
    );
    println!(
        "Number of constraints per step (secondary circuit): {}",
        pp.num_constraints().1
    );

    println!(
        "Number of variables per step (primary circuit): {}",
        pp.num_variables().0
    );
    println!(
        "Number of variables per step (secondary circuit): {}",
        pp.num_variables().1
    );

    println!("Creating a RecursiveSNARK...");
    let start = Instant::now();
    let mut recursive_snark = create_recursive_circuit(
        FileLocation::PathBuf(witness_generator_file.clone().into()),
        r1cs.clone(),
        private_inputs,
        start_public_input.to_vec(),
        &pp,
    )
    .unwrap();
    let create_prove_time = start.elapsed();
    // println!("RecursiveSNARK creation took {:?}", start.elapsed());
    println!("RecursiveSNARK creation took {:?}", create_prove_time);

    // TODO: empty?
    let z0_secondary = [F::<G2>::from(0)];

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let start = Instant::now();
    let res = recursive_snark.verify(&pp, folding_num, &start_public_input, &z0_secondary);
    let verify_recursive_time = start.elapsed();
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res,
        // start.elapsed()
        verify_recursive_time
    );
    assert!(res.is_ok());

    // let z_last = res.unwrap().0;
    // println!("z_last = {:?}", z_last);

    // assert_eq!(z_last[0], F::<G1>::from(20));
    // assert_eq!(z_last[1], F::<G1>::from(70));

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with IPA-PC...");
    let start = Instant::now();
    let (pk, vk) = CompressedSNARK::<_, _, _, _, S<G1>, S<G2>>::setup(&pp).unwrap();
    let setup_time_2 = start.elapsed();
    println!("CompressedSNARK setup took {:?}", setup_time_2);

    let start = Instant::now();
    let res = CompressedSNARK::<_, _, _, _, S<G1>, S<G2>>::prove(&pp, &pk, &recursive_snark);
    let compressed_prove_time = start.elapsed();
    println!(
        "CompressedSNARK::prove: {:?}, took {:?}",
        res.is_ok(),
        // start.elapsed()
        compressed_prove_time
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
    let compressed_snark_encoded = encoder.finish().unwrap();
    println!(
      "CompressedSNARK::len {:?} bytes",
      compressed_snark_encoded.len()
    );

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let start = Instant::now();
    let res = compressed_snark.verify(
        &vk,
        folding_num,
        start_public_input.to_vec(),
        z0_secondary.to_vec(),
    );
    let verify_compressed_time = start.elapsed();
    println!(
        "CompressedSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        // start.elapsed()
        verify_compressed_time
    );
    assert!(res.is_ok());

    let total_setup_time = setup_time + setup_time_2;
    let total_prove_time = create_prove_time + compressed_prove_time;
    let total_verify_time = verify_recursive_time + verify_compressed_time;
    println!("Total setup time: {:?}, consists of {:?} and {:?}", total_setup_time, setup_time, setup_time_2);
    println!("Total prove time: {:?}, consists of {:?} and {:?}", total_prove_time, create_prove_time, compressed_prove_time);
    println!("Total verify time: {:?}, consists of {:?} and {:?}", total_verify_time, verify_recursive_time, verify_compressed_time);

}



fn generate_unique_random_numbers(min: usize, max: usize, count: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut unique_numbers: Vec<usize> = (min..=max).collect();
    unique_numbers.shuffle(&mut rng);
    unique_numbers.truncate(count);
    unique_numbers
}

fn compute_z_score(greenlist_count: usize, gamma: f64, total_token_num: usize) -> f64 {
    let expected_count = gamma;
    let numer = greenlist_count as f64 - expected_count * total_token_num as f64;
    let denom = (total_token_num as f64 * expected_count * (1.0 - expected_count)).sqrt();
    let z = numer / denom;
    z
}


fn main() {
    let args = Args::parse();
    let type_sampling = args.type_sampling;
    let partial_seq_len = args.partial_seq_len;
    let merkle_tree_nlevels = args.merkle_tree_nlevels;
    let each_position_infor = args.each_position_infor;
    let folding_num = args.folding_num;

    // let group_name = "bn254";
    let hash_name = "poseidon";
    let file_name = "detect_recursive";

    // let circuit_filepath = format!("examples/toy/{}/toy.r1cs", group_name);

    let dir_path = format!("external/Nova-Scotia/src/{}-detect-circom/v1/{}", type_sampling, hash_name);
    let dir_cpp_path = format!("external/Nova-Scotia/src/{}-detect-circom/v1/{}/{}_cpp", type_sampling, hash_name, file_name);
    let r1cs_path = format!("{}.r1cs", file_name);
    let circuit_filepath = format!("external/Nova-Scotia/src/{}-detect-circom/v1/{}/{}.r1cs", type_sampling, hash_name, file_name);
    
    // // Execute the workflow and handle any errors
    // if let Err(e) = execute_workflow(&dir_path, &dir_cpp_path, &r1cs_path) {
    //     eprintln!("Error: {}", e);
    //     exit(1);
    // } else {
    //     println!("Workflow completed successfully.");
    // }

    let witness_gen_filepath = format!("external/Nova-Scotia/src/{}-detect-circom/v1/{}/{}_cpp/{}", type_sampling, hash_name, file_name, file_name);
    run_test(circuit_filepath.clone(), witness_gen_filepath, partial_seq_len, each_position_infor, merkle_tree_nlevels, folding_num);
    
}