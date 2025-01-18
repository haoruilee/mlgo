#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import re
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def predict_with_mips_inference(features):
    """
    Calls the local ./mips_inference executable file and passes 4 floating-point parameters.
    Returns the predicted class (integer) from the program's output.
    """
    str_args = [str(f) for f in features]

    try:
        result = subprocess.run(
            ["./mips_inference"] + str_args,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"mips_inference execution failed, error message: {result.stderr}")

        # Assume the output contains "Predicted class: X"
        output = result.stdout.strip()
        match = re.search(r"Predicted class:\s*(\d+)", output)
        if match:
            predicted_class = int(match.group(1))
            return predicted_class
        else:
            raise ValueError(f"Failed to parse the prediction result, please check the mips_inference output format: {output}")

    except FileNotFoundError:
        raise FileNotFoundError("The './mips_inference' executable file was not found. Please ensure it is compiled and in the current directory.")

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = []
    for i, features in enumerate(X_test):
        predicted = predict_with_mips_inference(features)
        y_pred.append(predicted)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the Iris test set: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()