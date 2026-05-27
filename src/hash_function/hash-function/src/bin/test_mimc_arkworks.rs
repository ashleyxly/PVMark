extern crate arkworks_mimc;
// use arkworks_mimc::{MiMC, params::mimc_7_91_bn254::{MIMC_7_91_BN254_PARAMS, MIMC_7_91_BN254_ROUND_KEYS}};
// // use arkworks_mimc::utils::*;
// // use halo2curves::bn256::Fr;
// use ark_bn254_2::Fr;
// use ark_ff_2::fields::PrimeField;
// use arkworks_mimc::MiMCNonFeistelCRH;
// use ark_crypto_primitives::crh::TwoToOneCRH;
// use arkworks_mimc::params::round_keys_contants_to_vec;
// use std::{error::Error, str::FromStr};
// use ark_ff_2::{Zero, One, to_bytes};
// use ark_ff_2::BigInteger;
// use regex::Regex;
use arkworks_mimc::utils::mimc_hash_non_feistel;

fn test() {
    let input1 = "1".to_string();
    let input2 = "2".to_string();
    let input3 = Some("3".to_string());
    let res = mimc_hash_non_feistel(input1, input2, input3);
    println!("{:?}", res);
}

// fn correct_hash_result_params_non_feistel() -> Result<(), Box<dyn Error>> {

//     let param = MiMC::<Fr, MIMC_7_91_BN254_PARAMS>::new(
//         1,
//         Fr::zero(),
//         round_keys_contants_to_vec(&MIMC_7_91_BN254_ROUND_KEYS),
//     );
//     let left_input = Fr::one();
//     let left_input_string = left_input.to_string();
//     println!("left_input_string: {:?}", left_input_string);
//     let re = Regex::new(r"\((\d+)\)").unwrap();
//     // 在字符串中搜索匹配的部分
//     if let Some(captures) = re.captures(left_input_string) {
//         // 获取第一个捕获组的内容，即括号内的数字
//         if let Some(number_str) = captures.get(1) {
//             // 将捕获到的字符串转换为整数
//             let number: u64 = number_str.as_str().parse().unwrap();
//             println!("括号内的数字是: {}", number);
//         }
//     }
    

    // let result = <MiMCNonFeistelCRH<Fr, MIMC_7_91_BN254_PARAMS> as TwoToOneCRH>::evaluate(
    //     &param,
    //     &to_bytes!(Fr::one())?,
    //     &to_bytes!(Fr::zero())?,
    // )?;

    // println!("{result}");

    // assert_eq!(
    //     result,
    //     Fr::from_str(
    //         "21581643069407877618298966131175370729897531221281133974758693417099906058024"
    //     )
    //     .unwrap()
    // );

//     Ok(())
// }


fn main() {
    test();
    // correct_hash_result_params_non_feistel().unwrap();
    
}