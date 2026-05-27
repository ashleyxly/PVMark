use ark_std::{end_timer, start_timer};
use std::{env, fs::File, io::{BufReader, BufWriter, Write}};
use std::path::Path;

use ff::Field;
use halo2_gadgets::sha256::{BlockWord, Sha256, Table16Chip, Table16Config};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Fixed, Instance, ProvingKey,
    },
    poly::{
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::SingleStrategy,
        },
        Rotation,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2_proofs::poly::commitment::Params;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use rand_core::OsRng;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
use sha256test::inputs::sha256exp::{INPUT_1025, INPUT_129, INPUT_17, INPUT_2, INPUT_257, INPUT_3, INPUT_33, INPUT_5, INPUT_513, INPUT_65, INPUT_9};
use sha256test::chips::add_chip::{AddChip, AddConfig};
// extern crate halo2_detection;
// use halo2_detection::chips::add_chip::{AddChip, AddConfig};



#[derive(Default)]
struct MyCircuit {
    sha_count: u64,
}

#[derive(Clone, Debug)]
pub struct CircuitConfig {
    pub advice_1: Column<Advice>,
    pub advice_2: Column<Advice>,
    pub advice_3: Column<Advice>,

    pub table_16_config: Table16Config,
    pub add_config: AddConfig,

    pub instance_1: Column<Instance>,

    // pub range_u8: Column<Fixed>,

    // pub advice_many: [Column<Advice>; SUMMATION_NUM],

    // pub instance_1: Column<Instance>,

    // pub table16_config: Table16Config,

    // pub less_than_config: LTConfig,
    // pub summation_config: SummationConfig,

}


impl Circuit<Fr> for MyCircuit {
    type Config = CircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let advice_1 = meta.advice_column();
        let advice_2 = meta.advice_column();
        let advice_3 = meta.advice_column();

        let instance_1 = meta.instance_column();

        CircuitConfig {
            advice_1: advice_1,
            advice_2: advice_2,
            advice_3: advice_3,
            table_16_config: Table16Chip::configure(meta),
            instance_1: instance_1,
            add_config: AddChip::configure(meta, advice_1, advice_2, advice_3, instance_1),
            
        }
        // Table16Chip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        Table16Chip::load(config.table_16_config.clone(), &mut layouter)?;
        let table16_chip = Table16Chip::construct(config.table_16_config.clone());
        let add_chip = AddChip::construct(config.add_config.clone());
        match self.sha_count {
            2 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                &INPUT_2)?,
            3 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                &INPUT_3)?,
            5 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                &INPUT_5)?,
            9 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                &INPUT_9)?,
            17 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                 &INPUT_17)?,
            33 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                 &INPUT_33)?,
            65 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                 &INPUT_65)?,
            129 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                  &INPUT_129)?,
            257 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                  &INPUT_257)?,
            513 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                  &INPUT_513)?,
            1025 => Sha256::digest(table16_chip.clone(), layouter.namespace(|| "'sha one'"),
                                   &INPUT_1025)?,
            _ => panic!("unexpected sha count: {}", self.sha_count),
        };
        let add_input1 = Fr::from(1 as u64);
        let add_input2 = Fr::from(2 as u64);
        let (add_input1_cell, add_input2_cell) = add_chip.load_private(layouter.namespace(|| "load"), add_input1, add_input2)?;
        let output_cell = add_chip.simple_cell_add(layouter.namespace(||"simple add"), &add_input1_cell, &add_input2_cell)?;
        add_chip.expose_public(layouter.namespace(|| "expose"), &output_cell, 0)?;


        Ok(())
    }
}

use std::time::Instant;
use halo2_proofs::dev::MockProver;
fn main() {
    // let args: Vec<String> = env::args().collect();
    // let k: u32 = args[1].parse().unwrap();
    // let sha_block: u64 = args[2].parse().unwrap();
    // process_one(k, sha_block).unwrap();
    let sha_count: u64 = 2;
    let circuit: MyCircuit = MyCircuit {sha_count};
    let (total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns) = halo2_proofs::dev::CircuitLayout::default().compute_num_rows(25, &circuit);
    println!("total_row = {}, total_col = {}, num_instance_columns = {}, num_advice_columns = {}, num_fixed_columns = {}", total_row, total_col, num_instance_columns, num_advice_columns, num_fixed_columns);
    
    let k = 17;
    let mut public_inputs = vec![];
    public_inputs.push(Fr::from(3 as u64));
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

}
