def generate_circom_file(position_len):
    # 基本的文件头部信息
    circom_content = [
        "pragma circom 2.1.1;",
        'include "../each_position_detect.circom";',
        ""
    ]

    # 模板定义的头部
    template_header = f"template total_detect(seq_len, each_position_infor, nLevels, position_len, {', '.join([f'token_num_{i+1}' for i in range(position_len)])})"
    circom_content.append(template_header)
    circom_content.append("{")

    # 信号输入部分
    signal_inputs = [
        "    signal input key;",
        "    // signal input position[seq_len];",
        "    //public input",
        "    signal input public_root;"
    ]
    for i in range(position_len):
        signal_inputs.extend([
            f"    signal input ngrams_{i+1}[token_num_{i+1}];",
            f"    signal input current_token_index_{i+1}[token_num_{i+1}];",
            f"    signal input pathIndices_{i+1}[token_num_{i+1}][nLevels];",
            f"    signal input siblings_{i+1}[token_num_{i+1}][nLevels];"
        ])
    circom_content.extend(signal_inputs)

    # 信号输出部分
    circom_content.append(f"    signal output out[position_len * each_position_infor];")

    # 组件声明部分
    for i in range(position_len):
        circom_content.append(f"    component each_position_detect_{i+1} = each_position_detect(seq_len, each_position_infor, nLevels);")

    # 组件连接部分
    for i in range(position_len):
        circom_content.extend([
            f"    each_position_detect_{i+1}.key <== key;",
            f"    each_position_detect_{i+1}.position <== current_token_index_{i+1};",
            f"    each_position_detect_{i+1}.ngrams <== ngrams_{i+1};",
            f"    each_position_detect_{i+1}.current_token_index <== current_token_index_{i+1};",
            f"    each_position_detect_{i+1}.public_root <== public_root;",
            f"    each_position_detect_{i+1}.pathIndices <== pathIndices_{i+1};",
            f"    each_position_detect_{i+1}.siblings <== siblings_{i+1};"
        ])

    # 输出连接部分
    circom_content.append("    for (var i = 0; i < each_position_infor; i ++)")
    circom_content.append("    {")
    for i in range(position_len):
        circom_content.append(f"        out[i + {i} * each_position_infor] <== each_position_detect_{i+1}.out[i];")
    circom_content.append("    }")

    circom_content.append("}")

    # 主组件定义
    main_component = f"component main {{public [public_root, {', '.join([f'ngrams_{i+1}, current_token_index_{i+1}, pathIndices_{i+1}, siblings_{i+1}' for i in range(position_len)])}]}} = total_detect(2, 2, 2, {position_len}, {', '.join(['2'] * position_len)});"
    circom_content.append(main_component)

    # 将内容写入文件
    with open("generated.circom", "w") as f:
        f.write("\n".join(circom_content))

# 示例：生成 position_len 为 3 的 circom 文件
position_len = 3
generate_circom_file(position_len)
