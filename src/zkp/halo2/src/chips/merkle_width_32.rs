use super::poseidon::{PoseidonChip, PoseidonConfig};
use super::super::poseidon::spec_width_32::PoseidonSpec;
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

use super::super::poseidon::spec_width_32::{POSEIDON_WIDTH, POSEIDON_RATE};

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
        let (hash_input_0, hash_input_1, hash_input_2, hash_input_3, hash_input_4, hash_input_5, hash_input_6, hash_input_7, hash_input_8,
            hash_input_9, hash_input_10, hash_input_11, hash_input_12, hash_input_13, hash_input_14, hash_input_15, hash_input_16, hash_input_17, hash_input_18, hash_input_19,
            hash_input_20, hash_input_21, hash_input_22, hash_input_23, hash_input_24, hash_input_25, hash_input_26, hash_input_27, hash_input_28, hash_input_29, hash_input_30) = layouter.assign_region(
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
                let hash_input_4 = region.assign_advice(|| "element_4", self.config.advice[4], 0, || element_input[3])?;
                let hash_input_5 = region.assign_advice(|| "element_5", self.config.advice[5], 0, || element_input[4])?;
                let hash_input_6 = region.assign_advice(|| "element_6", self.config.advice[6], 0, || element_input[5])?;
                let hash_input_7 = region.assign_advice(|| "element_7", self.config.advice[7], 0, || element_input[6])?;
                let hash_input_8 = region.assign_advice(|| "element_8", self.config.advice[8], 0, || element_input[7])?;
                let hash_input_9 = region.assign_advice(|| "element_9", self.config.advice[9], 0, || element_input[8])?;
                let hash_input_10 = region.assign_advice(|| "element_10", self.config.advice[10], 0, || element_input[9])?;
                let hash_input_11 = region.assign_advice(|| "element_11", self.config.advice[11], 0, || element_input[10])?;
                let hash_input_12 = region.assign_advice(|| "element_12", self.config.advice[12], 0, || element_input[11])?;
                let hash_input_13 = region.assign_advice(|| "element_13", self.config.advice[13], 0, || element_input[12])?;
                let hash_input_14 = region.assign_advice(|| "element_14", self.config.advice[14], 0, || element_input[13])?;
                let hash_input_15 = region.assign_advice(|| "element_15", self.config.advice[15], 0, || element_input[14])?;
                let hash_input_16 = region.assign_advice(|| "element_16", self.config.advice[16], 0, || element_input[15])?;
                let hash_input_17 = region.assign_advice(|| "element_17", self.config.advice[17], 0, || element_input[16])?;
                let hash_input_18 = region.assign_advice(|| "element_18", self.config.advice[18], 0, || element_input[17])?;
                let hash_input_19 = region.assign_advice(|| "element_19", self.config.advice[19], 0, || element_input[18])?;
                let hash_input_20 = region.assign_advice(|| "element_20", self.config.advice[20], 0, || element_input[19])?;
                let hash_input_21 = region.assign_advice(|| "element_21", self.config.advice[21], 0, || element_input[20])?;
                let hash_input_22 = region.assign_advice(|| "element_22", self.config.advice[22], 0, || element_input[21])?;
                let hash_input_23 = region.assign_advice(|| "element_23", self.config.advice[23], 0, || element_input[22])?;
                let hash_input_24 = region.assign_advice(|| "element_24", self.config.advice[24], 0, || element_input[23])?;
                let hash_input_25 = region.assign_advice(|| "element_25", self.config.advice[25], 0, || element_input[24])?;
                let hash_input_26 = region.assign_advice(|| "element_26", self.config.advice[26], 0, || element_input[25])?;
                let hash_input_27 = region.assign_advice(|| "element_27", self.config.advice[27], 0, || element_input[26])?;
                let hash_input_28 = region.assign_advice(|| "element_28", self.config.advice[28], 0, || element_input[27])?;
                let hash_input_29 = region.assign_advice(|| "element_29", self.config.advice[29], 0, || element_input[28])?;
                let hash_input_30 = region.assign_advice(|| "element_30", self.config.advice[30], 0, || element_input[29])?;
                // let hash_input_31 = region.assign_advice(|| "element_31", self.config.advice[31], 0, || element_input[30])?;

                


                Ok((hash_input_0, hash_input_1, hash_input_2, hash_input_3, hash_input_4, hash_input_5, hash_input_6, hash_input_7, hash_input_8,
                    hash_input_9, hash_input_10, hash_input_11, hash_input_12, hash_input_13, hash_input_14, hash_input_15, hash_input_16, hash_input_17, hash_input_18, hash_input_19,
                    hash_input_20, hash_input_21, hash_input_22, hash_input_23, hash_input_24, hash_input_25, hash_input_26, hash_input_27, hash_input_28, hash_input_29, hash_input_30))
            },
        )?;

        let poseidon_chip = PoseidonChip::<PoseidonSpec, WIDTH, RATE, RATE>::construct(
            self.config.poseidon_config.clone(),
        );
        // let poseidon_chip = PoseidonChip::<OrchardNullifier, 3, 2, 5>::construct(
        //     self.config.poseidon_config.clone(),
        // );
        let digest = poseidon_chip.hash(layouter.namespace(|| "poseidon"), &[hash_input_0, hash_input_1, hash_input_2, hash_input_3, hash_input_4, hash_input_5, hash_input_6, hash_input_7, hash_input_8,
        hash_input_9, hash_input_10, hash_input_11, hash_input_12, hash_input_13, hash_input_14, hash_input_15, hash_input_16, hash_input_17, hash_input_18, hash_input_19,
        hash_input_20, hash_input_21, hash_input_22, hash_input_23, hash_input_24, hash_input_25, hash_input_26, hash_input_27, hash_input_28, hash_input_29, hash_input_30])?;
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
                index = index + RATE - 1;
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
        let col_1 = meta.advice_column();
        let col_2 = meta.advice_column();
        let col_3 = meta.advice_column();
        let col_4 = meta.advice_column();
        let col_5 = meta.advice_column();
        let col_6 = meta.advice_column();
        let col_7 = meta.advice_column();
        let col_8 = meta.advice_column();
        let col_9 = meta.advice_column();
        let col_10 = meta.advice_column();
        let col_11 = meta.advice_column();
        let col_12 = meta.advice_column();
        let col_13 = meta.advice_column();
        let col_14 = meta.advice_column();
        let col_15 = meta.advice_column();
        let col_16 = meta.advice_column();
        let col_17 = meta.advice_column();
        let col_18 = meta.advice_column();
        let col_19 = meta.advice_column();
        let col_20 = meta.advice_column();
        let col_21 = meta.advice_column();
        let col_22 = meta.advice_column();
        let col_23 = meta.advice_column();
        let col_24 = meta.advice_column();
        let col_25 = meta.advice_column();
        let col_26 = meta.advice_column();
        let col_27 = meta.advice_column();
        let col_28 = meta.advice_column();
        let col_29 = meta.advice_column();
        let col_30 = meta.advice_column();
        let col_31 = meta.advice_column();

        let instance = meta.instance_column();
        MerkleTreeV3Chip::configure(meta, [col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13, col_14, col_15, col_16, col_17, col_18, col_19, col_20, col_21, col_22, col_23, col_24, col_25, col_26, col_27, col_28, col_29, col_30, col_31], instance)
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

