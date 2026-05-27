import numpy as np
import logging

import matplotlib.pyplot as plt

# 确定支持中文的字体路径
chinese_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

# 设置Matplotlib使用中文字体
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['font.size'] = 12


# 配置日志记录
logging.basicConfig(filename='./uniformity_test.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')

def uniformity_test(file_name):
    # 从输出文件中读取数据，并过滤掉非有限值
    hash_values = []
    with open(file_name, 'r') as file:
        for line in file:
            # 将十六进制字符串转换为整数，并将其添加到列表中
            hash_values.append(int(line.strip(), 16))

    # 计算范围内的最小值和最大值
    # min_value = min(hash_values)
    # max_value = max(hash_values)
    min_value = 0
    max_value = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    logging.info("最小值: " + str(min_value))
    logging.info("最大值: " + str(max_value))
    
    # 确定频率的区间数 1+log2(n) = 1+log2(65536) = 17
    num_bins = 17 
    bin_size = (max_value - min_value) / num_bins

    # 初始化频率数组以统计每个区间内的出现次数
    frequency = np.zeros(num_bins, dtype=int)

    # 计算每个区间内的哈希值出现次数
    for value in hash_values:
        bin_index = int((value - min_value) // bin_size)
        if bin_index == num_bins:
            bin_index -= 1
        frequency[bin_index] += 1
    
    expected = len(hash_values) / num_bins

    # 计算卡方值
    # chi_squared = sum((frequency - len(hash_values) / num_bins) ** 2) / (len(hash_values) / num_bins)
    chi_squared = sum((frequency - expected) ** 2) / expected

    logging.info("频率: " + str(frequency))
    logging.info("卡方分数: " + str(chi_squared))

    # 判断
    if chi_squared <= 26.296:
        logging.info("哈希函数是均匀的")
    else:
        logging.info("哈希函数不是均匀的")

    # 绘制直方图
    bin_edges = np.linspace(min_value, max_value, num_bins+1)
    plt.figure(figsize=(5, 3))  # Adjust the figure size
    plt.bar(bin_edges[:-1], frequency, width=bin_size, color='skyblue', edgecolor='black')
    plt.xlabel('数值范围')
    plt.ylabel('频率')
    plt.title('均匀性测试')
    plt.grid(True)

    # Move the legend to a better position
    plt.legend(['Frequency'], loc='upper right')
    plt.subplots_adjust(left=0.15, bottom=0.18)
    # 保存图片为png
    plt.savefig('uniformity_test.png')

    plt.show()

if __name__ == "__main__":
    uniformity_test('./results_new2.txt')
