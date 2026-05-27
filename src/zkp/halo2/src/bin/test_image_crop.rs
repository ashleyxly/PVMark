use std::process::Command;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufRead};

// pub mod fieldutils;

use clap::Parser;
use clap::builder::Str;

use std::time::{Duration, Instant};


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
// use halo2_proofs::plonk::Error;
// use std::fs::File;
// use std::io::BufReader;
use std::path::PathBuf;

use rand_core::OsRng;
use rand::Rng;
use rand::rngs::StdRng;



#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
//    /// File to read
//    filename: String,

    /// The path of onnx file
   #[arg(short, long, default_value = "/home/username/Codes/halo2-merkle-tree/crop_test/input.png")]
   input_image_file_path: String,

   /// Length of characters to seek
   #[arg(short, long, default_value = "/home/username/Codes/halo2-merkle-tree/crop_test/output.png")]
   output_image_file_path: String,

//    /// Scale factor
//    #[arg(short, long, default_value_t = 4)]
//    scale: i32,

   /// The path of proof
   #[arg(short, long, default_value = "/home/username/Codes/halo2-merkle-tree/crop_test/proof.bin")]
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




// ANCHOR: instructions
#[derive(Debug, Clone)]
pub struct ImageCropConfig {
    // pub advice: [Column<Advice>; 2],
    pub col_a: Column<Advice>,
    pub col_b: Column<Advice>,
    // c: Column<Advice>,
    pub instance: Column<Instance>,
    pub selector: Selector,
}

#[derive(Debug, Clone)]
pub struct ImageCropChip {
    config: ImageCropConfig,
    _marker: PhantomData<Fp>,
}

impl ImageCropChip {
    pub fn construct(config: ImageCropConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure (
        meta: &mut ConstraintSystem<Fp>,
        advice: [Column<Advice>; 2],
        instance: Column<Instance>,
    ) -> ImageCropConfig {
        let col_a = advice[0];
        let col_b = advice[1];
        let crop_selector = meta.selector();
        meta.enable_equality(col_a);
        meta.enable_equality(col_b);
        meta.enable_equality(instance);

        meta.create_gate("crop_equal", |meta| {
            let a = meta.query_advice(col_a, Rotation::cur());
            let b = meta.query_advice(col_b, Rotation::cur());
            // let i = meta.query_instance(instance, Rotation::cur());
            let s = meta.query_selector(crop_selector);
            vec![s * (a - b)]
            // vec![s * (a - i)]
        });


        ImageCropConfig { col_a: col_a, col_b: col_b, instance: instance, selector: crop_selector }

    }

    pub fn load_private(
        &self,
        mut layouter: impl Layouter<Fp>,
        input: Value<Fp>,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "load private",
            |mut region| {
                region.assign_advice(|| "private input", self.config.col_a, 0, || input)
            },
        )
    }

    // pub fn load_constant(
    //     &self,
    //     mut layouter: impl Layouter<Fp>,
    //     constant: Fp,
    // ) -> Result<AssignedCell<Fp, Fp>, Error> {
    //     layouter.assign_region(
    //         || "load constant",
    //         |mut region| {
    //             region.assign_advice_from_constant(
    //                 || "constant value",
    //                 self.config.a,
    //                 0,
    //                 constant,
    //             )
    //         },
    //     )
    // }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }

    pub fn assign_row(
        &self,
        mut layouter: impl Layouter<Fp>,
        input_image: Value<Fp>,
        output_image: Value<Fp>,
        // input_image: &AssignedCell<Fp, Fp>,
        row: usize,
        // value: Option<Fp>,
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {

        layouter.assign_region(|| "row",
        |mut region| {
            self.config.selector.enable(&mut region, 0)?;

            let a_cell = region.assign_advice(
                || "Input", 
                self.config.col_a, 
                0, 
                || input_image)?;

            // let b_cell = region.assign_advice_from_instance(
            //     || "Output", 
            //     self.config.instance.to_owned(),
            //     row, 
            //     self.config.col_b,
            //     0)?;
            let b_cell = region.assign_advice(
                || "Output", 
                self.config.col_b, 
                0, 
                || output_image)?;
            
            Ok((a_cell, b_cell))
            
        }
        
        )
        

        
    }

}

#[derive(Default, Clone)]
pub struct ImageCropCircuit {
    // input: Option<Fp>,
    // output: Option<Fp>,
    pub input: Vec<Value<Fp>>,
    pub output: Vec<Value<Fp>>,
}

impl Circuit<Fp> for ImageCropCircuit {
    type Config = ImageCropConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // let advice: [Column<Advice>; 2] = [
        //     meta.advice_column(),
        //     // meta.advice_column(),
        //     // meta.advice_column(),
        // ];
        let col_a = meta.advice_column();
        let col_b = meta.advice_column();
        let instance = meta.instance_column();

        ImageCropChip::configure(meta, [col_a, col_b], instance)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), halo2_proofs::plonk::Error> {
        let chip = ImageCropChip::construct(config);
        let k = self.input.len();
        for i in 0..k {
            let (input, output) = chip.assign_row(
                layouter.namespace(||  format!("prove_row_{}", i)),
                self.input[i],
                self.output[i],
                i,
            )?;
            // let input = chip.load_private(layouter.namespace(|| "load input image"), self.input[i])?;
            // let output = chip.load_private(layouter.namespace(|| "load output image"), self.output[i])?;
            // chip.expose_public(layouter, &input, i)?;
            chip.expose_public(layouter.namespace(|| "output"), &output, i)?;
        }

        // let input = chip.load_private(layouter.namespace(|| "load input image"), self.input)?;
        // // let output = chip.load_private(layouter, self.output)?;

        // chip.expose_public(layouter, &input, 0)?;
        // chip.expose_public(layouter, &output, 1)?;

        Ok(())
    }
}

fn generate_random_vector(n: usize) -> Vec<u64> {
    // let mut rng = StdRng::from_entropy();
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
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


fn main() {

    // let input_image = vec![1u64, 5u64, 6u64, 9u64, 9u64];
    // let output_image = vec![1u64, 5u64, 6u64, 9u64, 9u64];
    let original_size = 32;
    let cropped_size = 3;
    let channel = 3;
    let new_size = original_size - 2 * cropped_size;
    // let path = "proof_".to_owned() + &original_size.to_string() + "_" + cropped_size.to_string() + ".bin";
    let input_image = generate_random_vector(new_size * new_size * channel);
    let output_image = input_image.clone();



    let input_image_fp: Vec<Value<Fp>> = input_image
            .iter()
            .map(|x| Value::known(Fp::from(x.to_owned())))
            .collect();
    let output_image_fp: Vec<Value<Fp>> = output_image
        .iter()
        .map(|x| Value::known(Fp::from(x.to_owned())))
        .collect();

    let circuit = ImageCropCircuit {
        input: input_image_fp,
        output: output_image_fp,
    };

    let mut correct_public_input = Vec::new();
    for i in 0..output_image.len() {
        let temp = output_image[i];
        correct_public_input.push(Fp::from(temp));

    }

    let mut k = 21;
    println!("k = {}", k);

    // use plotters::prelude::*;
    // let root = BitMapBackend::new("layout_22.png", (1024, 768)).into_drawing_area();
    // root.fill(&WHITE).unwrap();
    // let root = root
    //     .titled("Example Circuit Layout", ("sans-serif", 60))
    //     .unwrap();


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
    // let correct_public_input = output_image_fp.clone();
    let correct_prover = MockProver::run(
        k,
        &circuit,
        vec![correct_public_input.clone()],
    )
    .unwrap();
    let mock_elapsed_time = mock_start_time.elapsed(); 
    println!("Running Mock took {:.4} seconds.", mock_elapsed_time.as_millis() as f64 / 1000.0);

    // println!("{:?}", correct_prover);
    correct_prover.assert_satisfied();
    println!("The proof is correct!");


    let srs_path = "/mnt/disk2/username/onnx_test/srs_params/perpetual-powers-of-tau-raw-".to_string() + &k.to_string();
    // let params: Params<Bn256> = Params::new(k);
    let params = load_params_cmd(srs_path.into(), k).expect("load_params_cmd should not fail");
    // let mut params: ParamsKZG<Bn256> = load_srs::<KZGCommitmentScheme<Bn256>>(srs_path.into())?;
    // let params: Params<G1Affine> = halo2_proofs::poly::commitment::Params::new(k);
    // Initialize the proving key
    let setup_start_time = Instant::now(); // Start time

    let vk = keygen_vk(&params, &circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &circuit).expect("keygen_pk should not fail");

    let setup_elapsed_time = setup_start_time.elapsed(); 

    // println!("Running Setup took {} seconds.", setup_elapsed_time.as_secs());
    println!("Running Setup took {:.4} seconds.", setup_elapsed_time.as_millis() as f64 / 1000.0);

    let instance_temp = vec![correct_public_input.clone()];
    let temp_inner = instance_temp.iter().map(|inner| inner.as_slice()).collect::<Vec<_>>();
    let temp_inner2: &[&[&[Fp]]] = &[&temp_inner];
    // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    let mut transcript = TranscriptWriterBuffer::<_, G1Affine, _>::init(Vec::new());
    
    // Create a proof
    let prove_start_time = Instant::now(); // Start time
    // create_proof(
    //     &params,
    //     &pk,
    //     // &[circuit.clone(), circuit.clone()],
    //     &[circuit.clone()],
    //     temp_inner2,
    //     // &[&[&[digest.clone()]]],
    //     OsRng,
    //     &mut transcript,
    // )
    // .expect("proof generation should not fail");
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

    std::fs::write("plonk_api_proof.bin", &proof[..])
        .expect("should succeed to write new proof");


    // let strategy = SingleVerifier::new(&params);
    // let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    
    // let verify_start_time = Instant::now(); // Start time
    // assert!(verify_proof(
    //     &params,
    //     pk.get_vk(),
    //     strategy,
    //     temp_inner2,
    //     &mut transcript,
    // )
    // .is_ok());
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