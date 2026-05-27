use sha2::{Sha256, Digest};
use std::fmt;

struct MerkleTree {
    layers: Vec<Vec<Vec<u8>>>,
    original_size: usize,
}

impl MerkleTree {
    const EMPTY_HASH: [u8; 32] = [0u8; 32];

    // 接受数字类型的键值对
    pub fn new(mappings: &[(u64, u64)]) -> Self {
        let mut leaves: Vec<Vec<u8>> = mappings.iter()
            .map(|(x, y)| Self::hash_numbers(*x, *y))
            .collect();

        let original_size = leaves.len();
        let target_size = leaves.len().next_power_of_two();
        leaves.resize_with(target_size, || Self::EMPTY_HASH.to_vec());

        let mut layers = vec![leaves];
        
        while layers.last().unwrap().len() > 1 {
            let current_layer = layers.last().unwrap();
            let mut next_layer = Vec::new();
            
            for i in (0..current_layer.len()).step_by(2) {
                let left = &current_layer[i];
                let right = &current_layer[i + 1];
                next_layer.push(Self::hash_pair(left, right));
            }
            
            layers.push(next_layer);
        }
        
        MerkleTree { layers, original_size }
    }

    // 专门处理数字的哈希方法
    fn hash_numbers(x: u64, y: u64) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(x.to_be_bytes()); // 大端字节序
        hasher.update(y.to_be_bytes());
        hasher.finalize().to_vec()
    }

    fn hash_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().to_vec()
    }

    pub fn root(&self) -> Option<&Vec<u8>> {
        self.layers.last().unwrap().first()
    }

    pub fn get_proof(&self, x: u64, y: u64) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
        let target = Self::hash_numbers(x, y);
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

impl fmt::Debug for MerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "MerkleTree ({} layers)", self.layers.len())?;
        writeln!(f, "Original leaves: {}", self.original_size)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "Layer {} ({} nodes):", i, layer.len())?;
            for (j, hash) in layer.iter().enumerate() {
                let status = if i == 0 && j >= self.original_size {
                    "[PADDING]"
                } else {
                    ""
                };
                writeln!(f, "  {}: {}{}", j, hex::encode(hash), status)?;
            }
        }
        Ok(())
    }
}

fn main() {
    // 测试数字类型的键值对
    let data = vec![
        (123u64, 456u64),
        (789u64, 101u64),
        (102u64, 345u64),
    ];

    let mt = MerkleTree::new(&data);
    println!("{:#?}", mt);

    let test_x = 102u64;
    let test_y = 345u64;

    // 验证第一个数字对的证明
    if let Some((path, siblings)) = mt.get_proof(test_x, test_y) {
        println!("\nProof for (789, 101):");
        println!("Path directions: {:?}", path);
        println!("Sibling hashes:");
        for (i, sib) in siblings.iter().enumerate() {
            println!(" Level {}: {}", i, hex::encode(sib));
        }
        
        // 验证路径正确性
        // let mut current_hash = MerkleTree::hash_numbers(123, 456);
        let mut current_hash = MerkleTree::hash_numbers(test_x, test_y);
        for (level, (&direction, sibling)) in path.iter().zip(siblings).enumerate() {
            let (left, right) = if direction == 0 {
                (&current_hash, &sibling)
            } else {
                (&sibling, &current_hash)
            };
            current_hash = MerkleTree::hash_pair(left, right);
            println!(" Level {} combined hash: {}", level, hex::encode(&current_hash));
        }
        println!("Final root: {}", hex::encode(mt.root().unwrap()));
    }
}
