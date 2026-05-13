import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

FILE_PATH = "artifacts"
MODEL_FILE = Path("artifacts/model.pkl")
PIPELINE_FILE = Path("artifacts/pipeline.pkl")

def main():
    parser = argparse.ArgumentParser(description="Enter a valid file name ")
    parser.add_argument("filename", 
                        help="Enter a valid csv file with headings and value\"")
    args = parser.parse_args()
    file_name = Path(args.filename).name

    print(file_name)

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    test_data = pd.read_csv(file_name)
    tranformed_input = pipeline.transform(test_data)
    predictions = model.predict(tranformed_input)
    test_data['median_house_value'] = predictions

    output_file_name = os.path.join(FILE_PATH, "test_output.csv")
    test_data.to_csv(output_file_name, index=False)
    print(f"Inference is complete, Results saved to {output_file_name}")

if __name__ == "__main__":
    main()