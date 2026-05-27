use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};
use halo2_proofs::poly::Rotation;
use halo2_proofs::arithmetic::*;
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSlice;

use std::marker::PhantomData;
use std::ops::Add;



#[derive(Debug, Clone)]
pub struct AddConfig {
    pub add_input1: Column<Advice>,
    pub add_input2: Column<Advice>,
    pub add_output: Column<Advice>,
    pub instance: Column<Instance>,
    pub selector: Selector,
}

#[derive(Debug, Clone)]
pub struct AddChip {
    config: AddConfig,
    _marker: PhantomData<Fp>,
}

impl AddChip {
    pub fn construct(config: AddConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        add_input1: Column<Advice>,
        add_input2: Column<Advice>,
        add_output: Column<Advice>,
        instance: Column<Instance>,
        // selector: Selector,
    
    ) -> AddConfig {
        
        // let add_input1 = meta.advice_column();
        // let add_input2 = meta.advice_column();
        // let add_output = meta.advice_column();
        // let instance = meta.instance_column();
        let add_input1 = add_input1;
        let add_input2 = add_input2;
        let add_output = add_output;
        let instance = instance;

        let selector = meta.selector();

        meta.enable_equality(add_input1);
        meta.enable_equality(add_input2);
        meta.enable_equality(add_output);
        meta.enable_equality(instance);

        
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(selector);
            let input1 = meta.query_advice(add_input1, Rotation::cur());
            let input2 = meta.query_advice(add_input2, Rotation::cur());
            let output = meta.query_advice(add_output, Rotation::cur());
            // let instance = meta.query_instance(instance, Rotation::cur());
            // let temp = instance - input1.clone() - input2.clone();
            // let s_lessthan = meta.query_selector(selector);
            vec![s * (input1 + input2 - output)]
        });
        
        AddConfig {
            // inputs: advice,
            add_input1,
            add_input2,
            add_output,
            instance,
            selector,
        }
    }

    // Optional?
    pub fn load_private(
        &self,
        mut layouter: impl Layouter<Fp>,
        // private_input: Fp,
        add_input1: Fp,
        add_input2: Fp,
    ) -> Result<(AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>), halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Load Private Inputs",
            |mut region| {
                // self.config.selector.enable(&mut region, 0)?;
                let add_input1_cell = region.assign_advice(
                    || "load add_input1",
                    self.config.add_input1,
                    0,
                    || Value::known(add_input1)
                )?;
                let add_input2_cell = region.assign_advice(
                    || "load add_input2",
                    self.config.add_input2,
                    0,
                    || Value::known(add_input2)
                )?;
                Ok((add_input1_cell, add_input2_cell))
            },
        )
    }

    // pub fn load_instance(
    //     &self,
    //     mut layouter: impl Layouter<Fp>,
    //     public_input: Fp,
    //     row: usize,
    // )
    


    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<Fp>,
        cell: &AssignedCell<Fp, Fp>,
        row: usize,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }


    //Use this method with function load private!
    pub fn simple_cell_add(
        &self,
        mut layouter: impl Layouter<Fp>,
        add_input1_cell: &AssignedCell<Fp, Fp>,
        add_input2_cell: &AssignedCell<Fp, Fp>,
        // add_input1: Fp,
        // add_input2: Fp,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Simple Add Operation",
            |mut region| {
                self.config.selector.enable(&mut region, 0)?;
                // let add_input1_cell = region.assign_advice(
                //     || "add_input1",
                //     self.config.add_input1,
                //     0,
                //     || Value::known(add_input1)
                // )?;
                // let add_input2_cell = region.assign_advice(
                //     || "add_input2",
                //     self.config.add_input2,
                //     0,
                //     || Value::known(add_input2)
                // )?;
                add_input1_cell.copy_advice(|| "add_input1", &mut region, self.config.add_input1, 0)?;
                add_input2_cell.copy_advice(|| "add_input2", &mut region, self.config.add_input2, 0)?;
                // let output = add_input1 + add_input2;
                // let output = add_input1_cell.value();
                // let add_output_cell = region.assign_advice(
                //     || "add_output",
                //     self.config.add_output,
                //     0,
                //     || Value::known(output)
                // )?;
                let add_output_cell = region.assign_advice(
                    || "add_output", 
                    self.config.add_output,
                    0,
                    || {
                        let input1 = add_input1_cell.value();
                        let input2 = add_input2_cell.value();
                        let output = input1 + input2;
                        output
                    },
                )?;
                Ok(add_output_cell)
            },
        )
    }

    // load private and summation
    // eg. Goal get a b c summation  using eq a + b = c, c + d = e  instead of a + b + c = d
    // more rows and less columns
    pub fn assign_multiple_value_and_summation(
        &self,
        mut layouter: impl Layouter<Fp>,
        summation_inpus: &Vec<Fp>,
        input_number: usize,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Summation Operation",
            |mut region| {
                // println!("Function assign_multiple_value_and_summation is called");
                // first row
                self.config.selector.enable(&mut region, 0)?;
                let mut add_input_1_cell = region.assign_advice(
                    || "add_input1",
                    self.config.add_input1,
                    0,
                    || Value::known(summation_inpus[0])
                )?;
                let mut add_input_2_cell = region.assign_advice(
                    || "add_input2",
                    self.config.add_input2,
                    0,
                    || Value::known(summation_inpus[1])
                )?;
                let mut add_output_cell = region.assign_advice(
                    || "add_output",
                    self.config.add_output,
                    0,
                    || {
                        let input1 = add_input_1_cell.value();
                        let input2 = add_input_2_cell.value();
                        let output = input1 + input2;
                        output
                    },
                )?;
                // a1       a2 res_a1a2
                // res_a1a2 a3 res_a1a2a3
                // other rows
                for i in 1..input_number-1 {
                    self.config.selector.enable(&mut region, i)?;
                    add_input_1_cell = region.assign_advice(
                        || "add_input1",
                        self.config.add_input1,
                        i,
                        || add_output_cell.value().cloned()
                    )?;
                    // add_input_1_cell.copy_advice(|| "add_input1", 
                    // &mut region, 
                    // self.config.add_output, 
                    // 0)?;
                    add_input_2_cell = region.assign_advice(
                        || "add_input2",
                        self.config.add_input2,
                        i,
                        || Value::known(summation_inpus[i + 1])
                    )?;
                    add_output_cell = region.assign_advice(
                        || "add_output",
                        self.config.add_output,
                        i,
                        || {
                            let input1 = add_input_1_cell.value();
                            let input2 = add_input_2_cell.value();
                            let output = input1 + input2;
                            output
                        },
                    )?;
                }
                Ok(add_output_cell)

            }
        
        )
        
        
    }

    // input: Vec<Value<Fp>>
    pub fn assign_multiple_value_and_summation_2(
        &self,
        mut layouter: impl Layouter<Fp>,
        summation_inpus: &Vec<Value<Fp>>,
        input_number: usize,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Summation Operation",
            |mut region| {
                // first row
                self.config.selector.enable(&mut region, 0)?;
                let mut add_input_1_cell = region.assign_advice(
                    || "add_input1",
                    self.config.add_input1,
                    0,
                    || summation_inpus[0]
                )?;
                let mut add_input_2_cell = region.assign_advice(
                    || "add_input2",
                    self.config.add_input2,
                    0,
                    || summation_inpus[1]
                )?;
                let mut add_output_cell = region.assign_advice(
                    || "add_output",
                    self.config.add_output,
                    0,
                    || {
                        let input1 = add_input_1_cell.value();
                        let input2 = add_input_2_cell.value();
                        let output = input1 + input2;
                        output
                    },
                )?;
                // other rows
                for i in 1..input_number-1 {
                    self.config.selector.enable(&mut region, i)?;
                    add_input_1_cell = region.assign_advice(
                        || "add_input1",
                        self.config.add_input1,
                        i,
                        || add_output_cell.value().cloned()
                    )?;
                    // add_input_1_cell.copy_advice(|| "add_input1", 
                    // &mut region, 
                    // self.config.add_output, 
                    // 0)?;
                    add_input_2_cell = region.assign_advice(
                        || "add_input2",
                        self.config.add_input2,
                        i,
                        || summation_inpus[i + 1]
                    )?;
                    add_output_cell = region.assign_advice(
                        || "add_output",
                        self.config.add_output,
                        i,
                        || {
                            let input1 = add_input_1_cell.value();
                            let input2 = add_input_2_cell.value();
                            let output = input1 + input2;
                            output
                        },
                    )?;
                }
                Ok(add_output_cell)

            }
        
        )
        
        
    }

    pub fn assign_multiple_value_and_summation_3(
        &self,
        mut layouter: impl Layouter<Fp>,
        summation_inpus: &Vec<AssignedCell<Fp, Fp>>,
        input_number: usize,
    ) -> Result<AssignedCell<Fp, Fp>, halo2_proofs::plonk::Error> {
        // println!("Function assign_multiple_value_and_summation_3 is called");
        // println!("summation_inputs.len() -> {:?}", summation_inpus.len());
        // println!("input_number -> {:?}", input_number);
        // println!("summation_inputs -> {:?}", summation_inpus);
        layouter.assign_region(
            || "Summation Operation",
            |mut region| {
                // first row
                self.config.selector.enable(&mut region, 0)?;
                let mut add_input_1_cell = region.assign_advice(
                    || "add_input1",
                    self.config.add_input1,
                    0,
                    || summation_inpus[0].value().cloned()
                )?;
                let mut add_input_2_cell = region.assign_advice(
                    || "add_input2",
                    self.config.add_input2,
                    0,
                    || summation_inpus[1].value().cloned()
                )?;
                let mut add_output_cell = region.assign_advice(
                    || "add_output",
                    self.config.add_output,
                    0,
                    || {
                        let input1 = add_input_1_cell.value();
                        let input2 = add_input_2_cell.value();
                        let output = input1 + input2;
                        output
                    },
                )?;
                // other rows
                for i in 1..input_number-1 {
                    self.config.selector.enable(&mut region, i)?;
                    add_input_1_cell = region.assign_advice(
                        || "add_input1",
                        self.config.add_input1,
                        i,
                        || add_output_cell.value().cloned()
                    )?;
                    // add_input_1_cell.copy_advice(|| "add_input1", 
                    // &mut region, 
                    // self.config.add_output, 
                    // 0)?;
                    add_input_2_cell = region.assign_advice(
                        || "add_input2",
                        self.config.add_input2,
                        i,
                        || summation_inpus[i + 1].value().cloned()
                    )?;
                    add_output_cell = region.assign_advice(
                        || "add_output",
                        self.config.add_output,
                        i,
                        || {
                            let input1 = add_input_1_cell.value();
                            let input2 = add_input_2_cell.value();
                            let output = input1 + input2;
                            output
                        },
                    )?;
                }
                Ok(add_output_cell)

            }
        
        )
        
        
    }


    
}
