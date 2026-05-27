use super::poseidon::{PoseidonChip, PoseidonConfig};
use super::super::poseidon::spec::PoseidonSpec;
// use halo2_gadgets::poseidon::{
//     primitives::{self as poseidon, ConstantLength, P128Pow5T3 as OrchardNullifier, Spec},
//     Hash,
// };
// use halo2_proofs::{
//     arithmetic::{Field, FieldExt},
//     circuit::*,
//     pasta::Fp,
//     plonk::*,
//     poly::Rotation,
// };

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

use super::super::poseidon::spec::{POSEIDON_WIDTH, POSEIDON_RATE};

const WIDTH: usize = POSEIDON_WIDTH;
const RATE: usize = POSEIDON_RATE;

#[derive(Debug, Clone)]
pub struct MerkleTreeV3Config {
    pub advice: [Column<Advice>; WIDTH - 1],
    // pub bool_selector: Selector,
    // pub swap_selector: Selector,
    pub instance: Column<Instance>,
    // pub poseidon_config: PoseidonConfig<3, 2, 2>,
    pub poseidon_config: PoseidonConfig<WIDTH, RATE, RATE>, 
    // pub poseidon_config: PoseidonConfig<2, 1, 2>,
}

pub fn vec_to_arr(vec: Vec<Value<Fp>>, indices: &[usize]) -> [Value<Fp>; RATE - 1] {
    let mut arr: [Value<Fp>; RATE - 1] = [Value::known(Fp::zero()); RATE - 1];
    for i in 0..RATE - 1 {
        arr[i] = vec[indices[i]].clone();
    }
    arr
}


#[derive(Debug, Clone)]
pub struct MerkleTreeV3Chip {
    config: MerkleTreeV3Config,
}

impl MerkleTreeV3Chip {
    pub fn construct(config: MerkleTreeV3Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        advice: [Column<Advice>; WIDTH - 1],
        instance: Column<Instance>,
    ) -> MerkleTreeV3Config {

        // let mut cols: [Column<Advice>; WIDTH - 1];
        // for i in 0..WIDTH - 1 {
        //     cols[i] = advice[i];
        // }
        let cols: [Column<Advice>; WIDTH - 1] = (0..WIDTH-1)
            .map(|i| Column::from(advice[i]))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // let bool_selector = meta.selector();
        // let swap_selector = meta.selector();
        for i in 0..WIDTH - 1 {
            meta.enable_equality(cols[i]);
        }

        // meta.enable_equality(col_a);
        // meta.enable_equality(col_b);
        // meta.enable_equality(col_c);
        meta.enable_equality(instance);

        

        MerkleTreeV3Config {
            advice: cols,
            // advice: [col_a, col_b, col_c],
            // bool_selector: bool_selector,
            // swap_selector: swap_selector,
            instance: instance,
            poseidon_config: PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::configure(meta),
            // poseidon_config: PoseidonChip::<OrchardNullifier, 3, 2, 5>::configure(meta),
            // poseidon_config: PoseidonChip::<OrchardNullifier, 2, 1, 2>::configure(meta),
        }
    }

    pub fn load_private(
        &self,
        mut layouter: impl Layouter<Fp>,
        input: Value<Fp>,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        layouter.assign_region(
            || "load private",
            |mut region| {
                region.assign_advice(|| "private input", self.config.advice[0], 0, || input)
            },
        )
    }

    pub fn load_constant(
        &self,
        mut layouter: impl Layouter<Fp>,
        constant: Fp,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        layouter.assign_region(
            || "load constant",
            |mut region| {
                region.assign_advice_from_constant(
                    || "constant value",
                    self.config.advice[0],
                    0,
                    constant,
                )
            },
        )
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }

    pub fn merkle_prove_layer(
        &self,
        mut layouter: impl Layouter<Fp>,
        digest: &AssignedCell<Fp, Fp>,
        element_input: [Value<Fp>; RATE - 1],
        // element: Value<Fp>,
        // index: Value<Fp>,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        let (hash_input_0, hash_input_1, hash_input_2, hash_input_3) = layouter.assign_region(
            || "merkle_prove_leaf",
            |mut region| {
                // let mut cols: [Column<Advice>; RATE - 1];
                // Row 0
                // let mut hash_input: [AssignedCell<Fp, Fp>; RATE] = [digest.clone(); RATE];

                // hash_input[0] = digest.copy_advice(|| "digest", &mut region, self.config.advice[0], 0)?;
                // for i in 1..RATE - 1 {
                //     if i == 0 {
                //         continue;
                //     }
                //     else
                //     {
                //         hash_input[i] = region.assign_advice(|| "element_".to_string() + &i.to_string(), self.config.advice[i], 0, || element_input[i - 1])?;
                //     }

                // }
                let hash_input_0 = digest.copy_advice(|| "digest", &mut region, self.config.advice[0], 0)?;
                let hash_input_1 = region.assign_advice(|| "element_1", self.config.advice[1], 0, || element_input[0])?;
                let hash_input_2 = region.assign_advice(|| "element_2", self.config.advice[2], 0, || element_input[1])?;
                let hash_input_3 = region.assign_advice(|| "element_3", self.config.advice[3], 0, || element_input[2])?;
                



                Ok((hash_input_0, hash_input_1, hash_input_2, hash_input_3))
            },
        )?;

        let poseidon_chip = PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::construct(
            self.config.poseidon_config.clone(),
        );
        // let poseidon_chip = PoseidonChip::<OrchardNullifier, 3, 2, 5>::construct(
        //     self.config.poseidon_config.clone(),
        // );
        let digest = poseidon_chip.hash(layouter.namespace(|| "poseidon"), &[hash_input_0, hash_input_1, hash_input_2, hash_input_3])?;
        Ok(digest)
    }

    pub fn merkle_prove(
        &self,
        mut layouter: impl Layouter<Fp>,
        leaf: &AssignedCell<Fp, Fp>,
        elements: &Vec<Value<Fp>>,
        // indices: &Vec<Value<Fp>>,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        let temp = RATE - 1; // Note!
        let remainder = elements.len() % temp;
        let flag = if remainder == 0 {
            0
        } else {
            1
        };
        let layers = if elements.len() % temp == 0 {
            elements.len() / temp
        } else {
            elements.len() / temp + 1
        };

        let mut indices_arr: [usize; RATE - 1] = [0; RATE - 1];

        // 0 layer
        for i in 0..RATE - 1 {
            indices_arr[i] = i;
        }

        let element_arr = vec_to_arr(elements.clone(), &indices_arr);

        let mut leaf_or_digest = self.merkle_prove_layer(
            layouter.namespace(|| "merkle_prove_layer_0"),
            leaf,
            element_arr,
            // indices[0],
        )?;

        let mut index = RATE - 1;

        // println!("layers: {}", layers);
        // println!("flag: {}", flag);
        // println!("remainder: {}", remainder);
        // println!("elements.len() {}", elements.len());

        for i in 1..layers {
            
            // println!("i: {}", i);
            // println!("index: {}", index);

            if i == layers - 1 {

                if flag == 0 {
                    for j in 0..RATE - 1 {
                        indices_arr[j] = index + j;
                    }
                    let element_temp = vec_to_arr(elements.clone(), &indices_arr);

                    leaf_or_digest = self.merkle_prove_layer(
                        layouter.namespace(|| format!("merkle_prove_layer_{}", i)),
                        &leaf_or_digest,
                        element_temp,
                        // elements[index + 1],
                        // elements[index + 2],
                    )?;
                    index = index + RATE - 1;
                }
                else {
                    let mut element_temp2: [Value<Fp>; RATE - 1] = [Value::known(Fp::from(0)); RATE - 1];
                    for j in 0..remainder {
                        element_temp2[j] = elements[index + j];
                    }
                    for j in remainder..RATE - 1 {
                        element_temp2[j] = Value::known(Fp::from(0));
                    }

                    leaf_or_digest = self.merkle_prove_layer(
                        layouter.namespace(|| format!("merkle_prove_layer_{}", i)),
                        &leaf_or_digest,
                        element_temp2,
                    )?;
                }

            }
            else {
                for j in 0..RATE - 1 {
                    indices_arr[j] = index + j;
                }
                let element_temp3 = vec_to_arr(elements.clone(), &indices_arr);


                leaf_or_digest = self.merkle_prove_layer(
                    layouter.namespace(|| format!("merkle_prove_layer_{}", i)),
                    &leaf_or_digest,
                    element_temp3,
                )?;
                index = index + 3;
            }
            
        }
            // leaf_or_digest = self.merkle_prove_layer(
            //     layouter.namespace(|| format!("merkle_prove_layer_{}", i)),
            //     &leaf_or_digest,
            //     elements[i],
            //     indices[i],
            // )?;
        
        Ok(leaf_or_digest)

    }
}

#[derive(Default, Clone)]
pub struct MerkleTreeV3Circuit {
    pub leaf: Value<Fp>,
    pub elements: Vec<Value<Fp>>,
    // pub indices: Vec<Value<Fp>>,
}

impl Circuit<Fp> for MerkleTreeV3Circuit {
    type Config = MerkleTreeV3Config;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        // let mut cols: [Column<Advice>; WIDTH - 1];
        // for i in 0..WIDTH - 1 {
        //     cols[i] = meta.advice_column();
        // }
        let col_a = meta.advice_column();
        let col_b = meta.advice_column();
        let col_c = meta.advice_column();
        let col_d = meta.advice_column();
        let instance = meta.instance_column();
        MerkleTreeV3Chip::configure(meta, [col_a, col_b, col_c, col_d], instance)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = MerkleTreeV3Chip::construct(config);
        let leaf_cell = chip.load_private(layouter.namespace(|| "load leaf"), self.leaf)?;
        // chip.expose_public(layouter.namespace(|| "public leaf"), &leaf_cell, 0)?;
        let digest = chip.merkle_prove(
            layouter.namespace(|| "merkle_prove"),
            &leaf_cell,
            &self.elements,
            // &self.indices,
        )?;
        // chip.expose_public(layouter.namespace(|| "leaf"), &leaf_cell, 0)?;
        // chip.expose_public(layouter.namespace(|| "public root"), &digest, 1)?;
        chip.expose_public(layouter.namespace(|| "public root"), &digest, 0)?;
        Ok(())
    }
}

// mod tests {
//     // use std::arch::aarch64::vreinterpret_f32_p64;

//     use crate::chips::poseidon;

//     use super::MerkleTreeV3Circuit;
//     use halo2_gadgets::poseidon::{
//         primitives::{self as poseidon1, ConstantLength, P128Pow5T3 as OrchardNullifier, Spec},
//         Hash,
//     };
//     use halo2_proofs::{circuit::Value, dev::MockProver};
//     use halo2_proofs::halo2curves::bn256::Fr as Fp;

//     fn compute_merkle_root(leaf: &u64, elements: &Vec<u64>, indices: &Vec<u64>) -> Fp {
//         let k = elements.len();
//         let mut digest = Fp::from(leaf.clone());
//         let mut message: [Fp; 2];
//         for i in 0..k {
//             if indices[i] == 0 {
//                 message = [digest, Fp::from(elements[i])];
//             } else {
//                 message = [Fp::from(elements[i]), digest];
//             }

//             digest = poseidon1::Hash::<_, OrchardNullifier, ConstantLength<2>, 3, 2>::init()
//                 .hash(message);
//             // digest = poseidon1::Hash::<_, OrchardNullifier, ConstantLength<2>, 2, 1>::init()
//             //     .hash(message);
//         }
//         return digest;
//     }

//     #[test]
//     fn test_merkle_tree_v3() {
//         let leaf = 99u64;
//         let elements = vec![1u64, 5u64, 6u64, 9u64, 9u64];
//         let indices = vec![0u64, 0u64, 0u64, 0u64, 0u64];
//         let digest = compute_merkle_root(&leaf, &elements, &indices);

//         let leaf_fp = Value::known(Fp::from(leaf));
//         let elements_fp: Vec<Value<Fp>> = elements
//             .iter()
//             .map(|x| Value::known(Fp::from(x.to_owned())))
//             .collect();
//         let indices_fp: Vec<Value<Fp>> = indices
//             .iter()
//             .map(|x| Value::known(Fp::from(x.to_owned())))
//             .collect();

//         let circuit = MerkleTreeV3Circuit {
//             leaf: leaf_fp,
//             elements: elements_fp,
//             indices: indices_fp,
//         };

//         // let correct_public_input = vec![Fp::from(leaf), Fp::from(digest)];
//         let correct_public_input = vec![Fp::from(digest)];
//         let correct_prover = MockProver::run(
//             10,
//             &circuit,
//             vec![correct_public_input.clone(), correct_public_input.clone()],
//         )
//         .unwrap();
//         correct_prover.assert_satisfied();

//         let wrong_public_input = vec![Fp::from(432058235)];
//         let wrong_prover = MockProver::run(
//             10,
//             &circuit,
//             vec![wrong_public_input.clone(), wrong_public_input.clone()],
//         )
//         .unwrap();

//         let result = wrong_prover.verify();
//         match result {
//             Ok(res) => panic!("shouldve not proved correctly but did"),
//             Err(error) => true,
//         };
//     }
// }
