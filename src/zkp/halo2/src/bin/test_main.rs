use pyo3::prelude::*;
use pyo3::types::PyTuple;

// fn main() -> PyResult<()> {
//     // 初始化 Python 解释器
//     Python::with_gil(|py| {
//         // 导入 Python 模块
//         let module = py.import("read_onnx_file")?;

//         // 获取 Python 函数
//         let add_func = module.getattr("add")?;

//         // 调用 Python 函数并获取执行结果
//         let args = PyTuple::new(py, &[1i32.into(), 2i32.into()]);
//         let result = add_func.call(args, None)?;

//         // 将 Python 结果转换为 Rust 类型并进行处理
//         let sum: i32 = result.extract()?;
//         println!("1 + 2 = {}", sum);

//         Ok(())
//     })
// }

// use pyo3::prelude::*;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let module = py.import("example_module")?;
        let add_func = module.getattr("add")?;
        let args = PyTuple::new(py, &[1i32.to_object(py), 2i32.to_object(py)]);
        let result = add_func.call(args, None)?;
        let sum: i32 = result.extract()?;
        println!("1 + 2 = {}", sum);

        Ok(())
    })
}