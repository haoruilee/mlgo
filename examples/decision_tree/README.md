# Decision Tree Example (Iris Dataset)

This folder demonstrates how to train and use a simple decision tree model on the Iris dataset, then converte a trained model to GGML format and run inference in a Go-based MIPS environment.

------

# Folder Description

Below is a quick overview of each folder:

1. **`converet/`**
   - **`convert.py`**: An example script that demonstrates converting models to GGML format.  
   - **`iris_ggml_model.bin`**: An example model already converted to GGML format.

2. **`go-mips-inference/`**
   - **`build.sh`**: A Bash script to build the Go inference program.
   - **`mips_inference`**: Compiled Go inference program.
   - **`mips_inference.go`**: Main Go source code implementing the MIPS inference logic.
   - **`val.py`**: Python script for verifying  outputs.

3. **`train/`**
   - **`iris_decision_tree_model.pkl`**: A serialized scikit-learn decision tree model trained on the Iris dataset.
   - **`train.py`**: Python script that trains a decision tree on the Iris dataset, saving the model to `iris_decision_tree_model.pkl`.

------

# How to Run the Decision Tree Example

## 1. Train a Decision Tree

1. Navigate to the `train/` folder
2. Run `python train.py`

## 2. Convert to a GGML 

1. Navigate to the `convert/` folder
2. Run `python convert.py`

## 3. Build MIPS Inference

1. Navigate to the `go-mips-inference` folder
2. Run `./build.sh`

## 4. Validate the result

We provide `val.py` (in the go-mips-inference folder) that can be used to check or compare inference outputs.

```
$ python val.py 
Accuracy on the Iris test set: 100.00%
```
