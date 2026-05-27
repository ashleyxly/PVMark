use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

fn main() {
    let seq_len = 1;
    let folding_num = 1;
    let window_size = 4;
    let depth = 1;

    let mut key = Vec::new();
    let mut ngrams = Vec::new();
    let mut current_token_index = Vec::new();
    let mut private_inputs = Vec::new();
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
    for i in 0..folding_num {
        let mut private_input = HashMap::new();
        private_input.insert("random_seed".to_string(), json!(random_seed));
        private_input.insert("current_count".to_string(), json!(i));
        private_input.insert("output_count".to_string(), json!(i + seq_len));
        private_input.insert("key".to_string(), json!(key.clone()));
        private_input.insert("ngrams".to_string(), json!(ngrams.clone()));
        private_input.insert("current_token_index".to_string(), json!(current_token_index.clone()));
        private_inputs.push(private_input);
    }

    // Serialize the result to JSON
    let json_output = serde_json::to_string_pretty(&private_inputs).unwrap();

    // Save to input.json file
    let mut file = File::create("external/Nova-Scotia/src/synthid-detect-circom/hash-rand/mimc/input.json").expect("Unable to create file");
    file.write_all(json_output.as_bytes()).expect("Unable to write data");

    println!("JSON data saved to input.json");
}
