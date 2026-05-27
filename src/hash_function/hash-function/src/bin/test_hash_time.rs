use hash_rustlib::two_inputs_hash_computation;
use hash_rustlib::HashType;
// use std::time::Instant;

// fn test_run_time(run_time: usize, hash_type: HashType) {
//     let input1 = "123".to_string();
//     let input2 = "456".to_string();
//     // let mut total_time = 0;
//     let start_time = Instant::now();
//     for i in 0..run_time {
//         two_inputs_hash_computation(input1.clone(), input2.clone(), hash_type);
//     }
//     let elapsed_time = start_time.elapsed();
//     println!("Time taken for {:?} iterations: {:?}", hash_type, elapsed_time);
// }

use std::time::{Instant, Duration};
use indicatif::{ProgressBar, ProgressStyle};

fn test_run_time(run_time: usize, hash_type: HashType) {
    let input1 = "123".to_string();
    let input2 = "456".to_string();

    // Create a progress bar
    let pb = ProgressBar::new(run_time as u64);
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta})");

    // Handle potential error from template method
    let style = match style {
        Ok(style) => style,
        Err(err) => {
            eprintln!("Failed to create progress bar style: {}", err);
            return;
        }
    };

    pb.set_style(style.progress_chars("#>-"));

    let start_time = Instant::now();
    for _ in 0..run_time {
        two_inputs_hash_computation(input1.clone(), input2.clone(), hash_type);
        pb.inc(1); // Increment the progress bar
    }
    pb.finish_and_clear(); // Clear the progress bar

    let elapsed_time = start_time.elapsed();
    println!("Time taken for {:?} iterations: {:?}", hash_type, elapsed_time);
}


fn main() {
    let run_time = 100000;
    test_run_time(run_time, HashType::SHA256);
    test_run_time(run_time, HashType::BLAKE2b);
    test_run_time(run_time, HashType::KECCAK256);
    test_run_time(run_time, HashType::POSEIDON);
    test_run_time(run_time, HashType::POSEIDON2);
    test_run_time(run_time, HashType::MIMC);

}