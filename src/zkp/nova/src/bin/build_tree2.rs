use serde::Deserialize;
use sha2::{Sha256, Digest};
use core::hash;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
// use hash_rustlib::two_inputs_hash_computation_decimal;
use nova_scotia::two_inputs_hash_computation_decimal;
// use hash_rustlib::HashType;
use num_bigint::BigUint;
use std::str::FromStr;

// 添加Cargo.toml依赖
/*
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde-pickle = "0.8"
sha2 = "0.10"
hex = "0.4"
num-bigint = "0.4"
*/

struct MerkleTree {
    layers: Vec<Vec<String>>,  // 将Vec<u8>改为String
    original_size: usize,
}


impl MerkleTree {
    // 空哈希改为Poseidon(0,0)的哈希值
    fn empty_hash() -> String {
        let hash_decimal = two_inputs_hash_computation_decimal(
            "0".to_string(),
            "0".to_string(),
            // HashType::POSEIDON
        );
        hash_decimal
    }

    pub fn new(mappings: &[(u64, u64)]) -> Self {
        let mut leaves: Vec<String> = mappings.iter()
            .map(|(x, y)| Self::hash_numbers((*x).to_string(), (*y).to_string()))
            .collect();

        let original_size = leaves.len();
        let target_size = leaves.len().next_power_of_two();
        // 使用自定义的空哈希填充
        leaves.resize_with(target_size, || Self::empty_hash());

        let mut layers = vec![leaves];
        
        while layers.last().unwrap().len() > 1 {
            let current_layer = layers.last().unwrap();
            let mut next_layer = Vec::new();
            
            for i in (0..current_layer.len()).step_by(2) {
                let left = &current_layer[i];
                let right = &current_layer[i + 1];
                next_layer.push(Self::hash_pair(left.to_string(), right.to_string()));
            }
            
            layers.push(next_layer);
        }
        
        MerkleTree { layers, original_size }
    }

    fn hash_numbers(x: String, y: String) -> String {
        let hash_decimal = two_inputs_hash_computation_decimal(
            x.to_string(),
            y.to_string(),
            // HashType::POSEIDON
        );
        hash_decimal
    }

    fn hash_pair(left: String, right: String) -> String {
        // 将字节数组还原为大整数字符串
        // let left_num = BigUint::from_bytes_be(left);
        // let right_num = BigUint::from_bytes_be(right);
        let hash_decimal = two_inputs_hash_computation_decimal(
            left.to_string(),
            right.to_string(),
            // HashType::POSEIDON
        );
        hash_decimal
    }

    pub fn root(&self) -> Option<&String> {
        self.layers.last().unwrap().first()
    }

    pub fn get_proof(&self, x: u64, y: u64) -> Option<(Vec<u8>, Vec<String>)> {
        let target = Self::hash_numbers(x.to_string(), y.to_string());
        let leaf_layer = &self.layers[0];
        let mut index = leaf_layer[..self.original_size]
            .iter()
            .position(|h| h == &target)?;

        let mut path = Vec::new();
        let mut siblings = Vec::new();
        
        for layer in self.layers.iter().take(self.layers.len() - 1) {
            let is_right = index % 2;
            let sibling_index = if is_right == 1 {
                index - 1
            } else {
                index + 1
            };
            
            siblings.push(layer[sibling_index].clone());
            path.push(is_right as u8);
            
            index /= 2;
        }
        
        Some((path, siblings))
    }
}

// read_pkl和main函数保持不变，同原代码


fn read_pkl<P: AsRef<Path>>(path: P) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    // 直接反序列化为HashMap<u64, u64>
    let data: HashMap<u64, u64> = serde_pickle::from_reader(file, serde_pickle::DeOptions::new())?;
    
    Ok(data.into_iter().collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 从pkl文件读取数据
    let mappings = read_pkl("external/segment-watermark/balance_hash/map_freq.pkl")?;
    println!("Mappings: {:?}", &mappings[..10]); 

    // 构建Merkle树
    let mt = MerkleTree::new(&mappings);
    
    // 输出根哈希
    println!("Merkle Root: {}", hex::encode(mt.root().unwrap()));


    let test_x = 100u64;
    let test_y = 2u64;

    // 验证第一个数字对的证明
    if let Some((path, siblings)) = mt.get_proof(test_x, test_y) {
        println!("\nProof for (789, 101):");
        println!("Path directions: {:?}", path);
        println!("Sibling hashes:");
        for (i, sib) in siblings.iter().enumerate() {
            // println!(" Level {}: {}", i, hex::encode(sib));
            println!("{}", hex::encode(sib));
        }
        
        // 验证路径正确性
        // let mut current_hash = MerkleTree::hash_numbers(123, 456);
        let mut current_hash = MerkleTree::hash_numbers(test_x.to_string(), test_y.to_string());
        for (level, (&direction, sibling)) in path.iter().zip(siblings).enumerate() {
            let (left, right) = if direction == 0 {
                (&current_hash, &sibling)
            } else {
                (&sibling, &current_hash)
            };
            current_hash = MerkleTree::hash_pair(left.to_string(), right.to_string());
            println!(" Level {} combined hash: {}", level, hex::encode(&current_hash));
        }
        println!("Final root: {}", hex::encode(mt.root().unwrap()));
    }
    
    Ok(())
}
