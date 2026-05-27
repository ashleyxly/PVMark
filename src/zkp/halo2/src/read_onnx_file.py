import onnx
import numpy as np
from onnx import numpy_helper

def get_onnx_model_weights(model_path):
    # 加载ONNX模型
    model = onnx.load(model_path)
    # print(model.graph)

    # 获取模型的所有权重和偏置
    INTIALIZERS  = model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        print(initializer.name)
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = W
    print("onnx_weights_type:", type(onnx_weights))
    return onnx_weights


def save_dict_values(dictionary, file_path):
    with open(file_path, 'w') as f:
        values = list(dictionary.values())
        for value in values:
            f.write(str(value) + '\n')
            
# def save_dict_array_values(dictionary, file_path):
#     with open(file_path, 'w') as f:
#         values = list(dictionary.values())
#         if value.size == 0:
#             continue
#         for value in values:
#             for element in np.nditer(value):
#                 f.write(str(element) + " ")
#                 # f.write(str(value) + '\n')
#             f.write("\n")  

def save_dict_array_values(dictionary, file_path):
    with open(file_path, 'w') as f:
        for value in dictionary.values():
            if value.size == 0:
                continue  # Skip zero-sized arrays
            for element in np.nditer(value):
                f.write(str(element) + " ")
            f.write("\n")

# 示例用法
# my_dict = {'a': 1, 'b': 2, 'c': 3}
# save_dict_values(my_dict, 'my_dict_values.txt')



def read_onnx_params(file_path):
    model = onnx.load(file_path)
    params = {}
    for initializer in model.graph.initializer:
        params[initializer.name] = initializer.float_data
    return params

def test_hello_world():
    print("Hello World!")
    return

import argparse

def read_command_line_args():
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Read Onnx File')

    # 添加参数
    parser.add_argument('-Net', '--network', default="/mnt/disk2/username/kzg-halo2-merkle-tree/onnx_test/network.onnx", help='The path of onnx file')
    parser.add_argument('-P', '--params', default="/mnt/disk2/username/kzg-halo2-merkle-tree/test_result/test_onnx_parameters/params.txt", help='The path of params file')
    # parser.add_argument('-Net', '--network', help='The path of onnx file')
    # parser.add_argument('-P', '--params', help='The path of params file')

    # 解析命令行参数
    args = parser.parse_args()

    # 返回参数值
    return args

if __name__ == "__main__":
    test_hello_world()
    Args = read_command_line_args()
    params = get_onnx_model_weights(Args.network)
    save_dict_array_values(params, Args.params)
    # onnx_file_path = "/home/username/Codes/BackdoorBox/onnx_test/convtranspose_test/network.onnx"
    # params = get_onnx_model_weights(onnx_file_path)
    
    # output_file_path = "/home/username/Codes/halo2-merkle-tree/params.txt"
    # save_dict_values(params, output_file_path)
    # print(params)