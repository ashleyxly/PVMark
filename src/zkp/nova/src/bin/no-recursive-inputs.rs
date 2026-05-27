use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use clap::Parser;

use std::process::{Command, exit};
use std::env;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The number of token per sub-circuit
    #[arg(short, long, default_value_t = 100)]
    seq_len: usize,

    /// The sliding window size
    #[arg(short, long, default_value_t = 4)]
    window_size: usize,

    /// The watermarking depth per sub-circuit
    #[arg(short, long, default_value_t = 30)]
    depth: usize,


}

fn main() {
    // let seq_len = 100;
    // // let folding_num = 1;
    // let window_size = 4;
    // let depth = 30;
    let args = Args::parse();
    let seq_len = args.seq_len;
    let window_size = args.window_size;
    let depth = args.depth;

    let mut key = Vec::new();
    let mut ngrams = Vec::new();
    let mut current_token_index = Vec::new();
    // let mut private_inputs = Vec::new();
    let random_seed = 2025;

    // Generate key
    for _ in 0..depth {
        key.push(123);
    }

    // Generate ngrams
    for _ in 0..seq_len {
        let mut temp = Vec::new();
        for j in 0..window_size {
            temp.push(50 + (j * 5));
        }
        ngrams.push(temp);
    }

    // Generate current_token_index
    for _ in 0..seq_len {
        current_token_index.push(100);
    }

    // Generate private_inputs
    let mut private_input = HashMap::new();
    // private_input.insert("random_seed".to_string(), json!(random_seed));
    // private_input.insert("current_count".to_string(), json!(i));
    // private_input.insert("output_count".to_string(), json!(i + seq_len));
    private_input.insert("key".to_string(), json!(key.clone()));
    private_input.insert("ngrams".to_string(), json!(ngrams.clone()));
    private_input.insert("current_token_index".to_string(), json!(current_token_index.clone()));
    // private_inputs.push(private_input);

    // Serialize the result to JSON
    let json_output = serde_json::to_string_pretty(&private_input).unwrap();

    // Save to input.json file
    let mut file = File::create("external/ZKLLMWatermark_Codes/hash_function/circom_circuit/synthid_circom/Non-Recursive-Hash/mimc/input.json").expect("Unable to create file");
    file.write_all(json_output.as_bytes()).expect("Unable to write data");


    let mut file = File::create("external/ZKLLMWatermark_Codes/hash_function/circom_circuit/synthid_circom/Non-Recursive-Hash/poseidon/input.json").expect("Unable to create file");
    file.write_all(json_output.as_bytes()).expect("Unable to write data");

    let mut file = File::create("external/ZKLLMWatermark_Codes/hash_function/circom_circuit/synthid_circom/Non-Recursive-Hash/poseidon2/input.json").expect("Unable to create file");
    file.write_all(json_output.as_bytes()).expect("Unable to write data");
    println!("JSON data saved to input.json");
}
