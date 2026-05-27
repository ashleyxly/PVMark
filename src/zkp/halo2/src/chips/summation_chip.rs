use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};
use halo2_proofs::poly::Rotation;
use halo2_proofs::arithmetic::*;
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSlice;

use std::marker::PhantomData;
use std::ops::Add;

pub const SUMMATION_NUM: usize = 32;

#[derive(Debug, Clone)]
pub struct SummationConfig {
    pub add_inputs: [Column<Advice>; SUMMATION_NUM],
    pub summation_output: Column<Advice>,
    pub instance: Column<Instance>,
    pub selector: Selector,
}

#[derive(Debug, Clone)]
pub struct SummationChip {
    config: SummationConfig,
    _marker: PhantomData<Fp>,
}

impl SummationChip {
    pub fn construct(config: SummationConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        add_inputs: [Column<Advice>; SUMMATION_NUM],
        summation_output: Column<Advice>,
        instance: Column<Instance>,
        // selector: Selector,
    
    ) -> SummationConfig {
        let selector = meta.selector();
        meta.enable_equality(instance);
        for i in 0..SUMMATION_NUM {
            meta.enable_equality(add_inputs[i]);
        }
        meta.enable_equality(summation_output);

        
        meta.create_gate("summation gate", |meta| {
            let s = meta.query_selector(selector);

            let summation_output = meta.query_advice(summation_output, Rotation::cur());

            let summation_output_temp = add_inputs
                .iter()
                .map(|c| meta.query_advice(*c, Rotation::cur()))
                .collect::<Vec<Expression<Fp>>>();

            let mut output = Expression::Constant(Fp::ZERO);
            for i in 0..SUMMATION_NUM {
                output = output + summation_output_temp[i].clone();
            }


            vec![s * (output - summation_output)]
        });
        
        SummationConfig {
            // inputs: advice,
            add_inputs,
            summation_output,
            instance,
            selector,
        }
    }

    pub fn load_private(
        &self,
        mut layouter: impl Layouter<Fp>,
        // inputs: Vec<Fp>,
        input: Fp,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        let input_cell = layouter.assign_region(
            || "Load private",
            |mut region| {
                let input_cell = region.assign_advice(
                    || "input",
                    self.config.add_inputs[0],
                    0,
                    || Value::known(input)
                )?;
                Ok(input_cell)
            }
        )?;
        Ok(input_cell)
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }

    pub fn assign_multiple_value_and_summation(
        &self,
        mut layouter: impl Layouter<Fp>,
        summation_input: &Vec<AssignedCell<Fp, Fp>>,
        input_number: usize,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        // println!("Function assign_multiple_value_and_summation_3 is called");
        // println!("summation_inputs.len() -> {:?}", summation_input.len());
        // println!("input_number -> {:?}", input_number);
        // println!("summation_inputs -> {:?}", summation_input);
        let mut vector_temp = vec![];
        layouter.assign_region(
            || "Summation Operation",
            |mut region| {
                // first row
                self.config.selector.enable(&mut region, 0)?;
                let mut round = 0;
                if input_number <= SUMMATION_NUM {
                    round = 1;
                }
                else {
                    round = 1;
                    let temp_quotient = (input_number - SUMMATION_NUM) / (SUMMATION_NUM - 1);
                    round += if (input_number - SUMMATION_NUM) % (SUMMATION_NUM - 1) != 0 {
                        temp_quotient + 1
                    } else {
                        temp_quotient
                    };
                }
                
                // println!("round -> {:?}", round);
                let mut output_value = Value::known(Fp::ZERO);
                let mut index = 0;
                for i in 0..round {
                    if i == 0 {
                        region.assign_advice(
                            || "summation_input",
                            self.config.add_inputs[0],
                            i,
                            || summation_input[index].value().cloned()
                        )?;
                        output_value = output_value + summation_input[index].value().cloned();
                        index = index + 1;
                    }
                    else {
                        region.assign_advice(
                            || "summation_input",
                            self.config.add_inputs[0],
                            i,
                            || output_value.clone()
                        )?;
                    }
                    for j in 1..SUMMATION_NUM {
                        if index < input_number {
                            region.assign_advice(
                                || "summation_input",
                                self.config.add_inputs[j],
                                i,
                                || summation_input[index].value().cloned()
                            )?;
                            output_value = output_value + summation_input[index].value().cloned();
                            index = index + 1;
                        }
                        else {
                            region.assign_advice(
                                || "summation_input",
                                self.config.add_inputs[j],
                                i,
                                || Value::known(Fp::ZERO)
                            )?;
                            // output_value = output_value + summation_input[index].value().cloned();
                            index = index + 1;
                        }
                    }
                    let summation_output_cell = region.assign_advice(
                        || "summation_output",
                        self.config.summation_output,
                        i,
                        || output_value.clone()
                    )?;
                    vector_temp.push(summation_output_cell.clone());
                    // Ok(())
            }
            // println!("output_value -> {:?}", output_value);

            Ok(vector_temp[vector_temp.len() - 1].clone())
        
        })
        
        
    }


    
}
