use sha2::{Sha256, Digest};
use rand::{Rng, thread_rng};
use std::collections::BTreeMap;
use hex;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Mapping {
    x: u32,
    y: u32,
    hash: Vec<u8>,
}

impl Mapping {
    fn new(x: u32, y: u32) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(x.to_be_bytes());
        hasher.update(y.to_be_bytes());
        let hash = hasher.finalize().to_vec();
        Self { x, y, hash }
    }
}

fn build_merkle_tree(mappings: Vec<Mapping>) -> Vec<u8> {
    let mut leaves = mappings.into_iter().map(|m| m.hash).collect::<Vec<_>>();
    let next_power = leaves.len().next_power_of_two();
    if leaves.len() < next_power {
        let zero_hash = Sha256::digest(&[]).to_vec();
        leaves.extend(vec![zero_hash; next_power - leaves.len()]);
    }

    let mut layers = vec![leaves];
    while layers.last().unwrap().len() > 1 {
        let current = &layers.last().unwrap();
        let mut next = Vec::with_capacity(current.len() / 2);
        for i in 0..current.len() / 2 {
            let mut hasher = Sha256::new();
            hasher.update(&current[2*i]);
            hasher.update(&current[2*i + 1]);
            next.push(hasher.finalize().to_vec());
        }
        layers.push(next);
    }
    layers.first().unwrap().last().cloned().unwrap_or_default()
}

// fn build_merkle_tree(mappings: Vec<Mapping>) -> Vec<u8> {
//     let mut leaves = mappings.into_iter()
//         .map(|m| m.hash.clone()) // 克隆避免所有权问题
//         .collect::<Vec<_>>();

//     // 1. 补零策略改为「右填充」，并使用带位置的虚拟节点（修复哈希碰撞）
//     let len = leaves.len();
//     let next_power = len.next_power_of_two();
//     if len < next_power {
//         // 虚拟节点格式：[0x00, 位置索引（u32）的大端字节]
//         for i in len..next_power {
//             let mut data = [0u8; 4];
//             data.copy_from_slice(&(i as u32).to_be_bytes());
//             let zero_hash = Sha256::digest(&data).to_vec();
//             leaves.push(zero_hash);
//         }
//     }

//     // // 2. 标准Merkle树合并逻辑（逐层两两合并，右填充节点参与计算）
//     // let mut layers = vec![leaves];
//     // while layers.last().unwrap().len() > 1 {
//     //     let current = &layers.last().unwrap();
//     //     let mut next = Vec::with_capacity(current.len() / 2);
        
//     //     for i in 0..current.len() / 2 {
//     //         let left = &current[2 * i];
//     //         let right = &current[2 * i + 1];
//     //         let mut hasher = Sha256::new();
//     //         hasher.update(left).update(right);
//     //         next.push(hasher.finalize().to_vec());
//     //     }
//     //     // 处理奇数节点（标准Merkle树会复制最后一个节点）
//     //     if current.len() % 2 != 0 {
//     //         next.push(current.last().unwrap().clone());
//     //     }
//     //     layers.push(next);
//     // }

//     // // 3. 确保根节点正确（取最后一层的第一个元素）
//     // layers.last().unwrap().first().cloned().unwrap_or_default()
//     let mut layers = vec![leaves];
//     while layers.last().unwrap().len() > 1 {
//         let current = &layers.last().unwrap();
//         let mut next = Vec::with_capacity((current.len() + 1) / 2); // 支持奇数节点
        
//         for i in 0..current.len() / 2 * 2 { // 处理偶数对
//             let left = &current[i];
//             let right = &current[i + 1];
//             let mut hasher = Sha256::new();
            
//             // ✅ 修复：分开展示更新过程（非链式调用）
//             hasher.update(left);  // 明确更新左节点
//             hasher.update(right); // 明确更新右节点
            
//             next.push(hasher.finalize().to_vec());
//         }
        
//         // 处理奇数剩余节点（标准Merkle树复制最后一个节点）
//         if current.len() % 2 == 1 {
//             next.push(current.last().unwrap().clone());
//         }
        
//         layers.push(next);
//     }

//     // 根节点取最后一层第一个元素（修复层数索引错误）
//     layers.last().unwrap().first().cloned().unwrap_or_default()
// }

// fn generate_mappings(x_range: (u32, u32), y_range: (u32, u32), count: usize) -> Vec<Mapping> {
//     let mut rng = thread_rng();
//     let mut map = BTreeMap::new();
    
//     while map.len() < count {
//         let x = rng.gen_range(x_range.0..=x_range.1);
//         let y = rng.gen_range(y_range.0..=y_range.1);
//         if !map.contains_key(&x) {
//             map.insert(x, y);
//         }
//     }
    
//     // map.into_iter()
//     //     .map(|(x, y)| Mapping::new(x, y))
//     //     .collect::<Vec<_>>()
//     //     .into_iter()
//     //     .sorted_by(|a, b| a.y.cmp(&b.y))
//     //     .collect()
//     let mut mappings: Vec<_> = map.into_iter()
//         .map(|(x, y)| Mapping::new(x, y))
//         .collect();
//     mappings.sort_by(|a, b| a.y.cmp(&b.y)); // 直接排序Vec
//     mappings
// }

fn generate_mappings(x_range: (u32, u32), y_range: (u32, u32)) -> Vec<Mapping> {
    let (x_start, x_end) = if x_range.0 <= x_range.1 { // 自动处理范围顺序
        x_range
    } else {
        (x_range.1, x_range.0)
    };
    let mut rng = thread_rng();
    
    // (x_start..=x_end) // 遍历所有x值
    //     .map(|x| {
    //         let y = rng.gen_range(y_range.0..=y_range.1); // 为每个x生成随机y
    //         Mapping::new(x, y)
    //     })
    //     .collect::<Vec<_>>()
    //     .into_iter()
    //     .sorted_by(|a, b| a.y.cmp(&b.y)) // 保持按y排序
    //     .collect()
    let mut mappings = (x_start..=x_end)
        .map(|x| Mapping::new(x, rng.gen_range(y_range.0..=y_range.1)))
        .collect::<Vec<_>>();


    mappings.sort_by_key(|m| m.y);

    mappings
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_merkle_construction() {
//         let mappings = generate_mappings((1, 100), (100, 200), 5);
//         let root = build_merkle_tree(mappings);
//         assert_eq!(root.len(), 32); // SHA256 digest length
//     }
// }


fn main() {
    // 配置参数
    let x_range = (1, 10);   // x取值范围
    let y_range = (0, 3); // y取值范围
    // let mapping_count = 13;    // 生成13组映射
    
    // 生成并排序映射
    let mappings = generate_mappings(x_range, y_range);
    println!("生成的映射（按y排序）:");
    for m in &mappings {
        println!("x: {} → y: {}", m.x, m.y);
    }

    // 构建Merkle树
    let root_hash = build_merkle_tree(mappings);
    println!("\nMerkle树根哈希:");
    println!("{}", hex::encode(root_hash));
}
