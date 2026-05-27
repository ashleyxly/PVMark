import argparse
from pathlib import Path

import pandas as pd


def parse_txt_file(file_path):
    data = {
        "Type": [],
        "Compile Time": [],
        "Setup Time": [],
        "Prove Time": [],
        "Verify Time": [],
    }
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split(": ")
            if "Compile time" in line:
                data["Type"].append(parts[0].split(" for ")[1])
                data["Compile Time"].append(float(parts[1]))
            elif "Setup time" in line:
                data["Setup Time"].append(float(parts[1]))
            elif "Prove time" in line:
                data["Prove Time"].append(float(parts[1]))
            elif "Verify time" in line:
                data["Verify Time"].append(float(parts[1]))
    return data


def save_to_excel(data, output_path):
    df = pd.DataFrame(data)
    df.set_index("Type", inplace=True)
    df.T.to_excel(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("zkp_timing_summary.xlsx"))
    args = parser.parse_args()
    save_to_excel(parse_txt_file(args.input), args.output)


if __name__ == "__main__":
    main()
