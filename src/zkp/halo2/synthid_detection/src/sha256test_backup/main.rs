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

#[derive(Default)]
struct MyCircuit {
    sha_count: u64,
}

impl Circuit<Fr> for MyCircuit {
    type Config = Table16Config;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        Table16Chip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        Table16Chip::load(config.clone(), &mut layouter)?;
        let table16_chip = Table16Chip::construct(config);
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

        Ok(())
    }
}

fn process_one(k: u32, sha_count: u64) -> Result<(), Error> {
    println!("start process, k={}, sha count={}", k, sha_count);
    // Initialize the polynomial commitment parameters
    let params_path_str = format!("./sha256_params_k_{}", k);
    let params_path = Path::new(params_path_str.as_str());
    if File::open(params_path).is_err() {
        let timer_get_param = start_timer!(|| "get param");
        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        let mut buf = Vec::new();

        params.write(&mut buf).expect("Failed to write params");
        let mut file = File::create(params_path).expect("Failed to create sha256_params");

        file.write_all(&buf[..])
            .expect("Failed to write params to file");
        end_timer!(timer_get_param);
    }

    let params_fs = File::open(params_path).expect("couldn't load sha256_params");
    let params: ParamsKZG::<Bn256> =
        Params::read::<_>(&mut BufReader::new(params_fs)).expect("Failed to read params");

    let empty_circuit: MyCircuit = MyCircuit {sha_count};

    // Initialize the proving key
    let timer_get_pk_vk = start_timer!(|| "get pk vk");
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");
    end_timer!(timer_get_pk_vk);

    let circuit: MyCircuit = MyCircuit {sha_count};

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let timer_create_proof = start_timer!(|| "create proof");
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverGWC<'_, Bn256>,
        Challenge255<G1Affine>,
        _,
        Blake2bWrite<Vec<u8>, G1Affine, Challenge255<_>>,
        _,
    >(
        &params,
        &pk,
        &[circuit],
        &[&[]],
        OsRng,
        &mut transcript,
    )
        .expect("prover should not fail");
    end_timer!(timer_create_proof);
    let proof = transcript.finalize();

    let proof_path_str = format!("./sha256_proof_k_{}_count_{}", k, sha_count);
    let proof_path = Path::new(proof_path_str.as_str());
    let mut file = File::create(&proof_path).expect("Failed to create sha256_proof");
    file.write_all(&proof[..]).expect("Failed to write proof");

    println!("proof len: {}", proof.len());

    use halo2_proofs::poly::VerificationStrategy;
    let timer_verify = start_timer!(|| "verify");
    let strategy = AccumulatorStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let strategy = verify_proof::<KZGCommitmentScheme<_>, VerifierGWC<_>, _, _, _>(
        &params,
        pk.get_vk(),
        strategy,
        &[&[]],
        &mut transcript,
    ).unwrap();
    end_timer!(timer_verify);

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let k: u32 = args[1].parse().unwrap();
    let sha_block: u64 = args[2].parse().unwrap();
    process_one(k, sha_block).unwrap();
}
