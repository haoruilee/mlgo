package main

import (
    "fmt"
    "os"
	"strconv"
    "go-mips-inference/common"
)

type irisModel struct {
    nFeatures int32
    nNodes    int32
    features  []int32
    thresholds []float32
    lefts     []int32
    rights    []int32
    values    [][]float32
}

func loadIrisModel(model *irisModel, filePath string) error {
    // 1) Read the file instead of memory address
    modelBytes := common.ReadBytesFromFile(filePath)

    idx := 0

    // 2) Check magic number
    magic := common.ReadInt32FromBytes(modelBytes, &idx)
    if magic != 0x67676d6c {
        return fmt.Errorf("invalid magic number: 0x%x", magic)
    }

    // 3) Read n_features, n_nodes
    model.nFeatures = common.ReadInt32FromBytes(modelBytes, &idx)
    model.nNodes    = common.ReadInt32FromBytes(modelBytes, &idx)

    // 4) Allocate slices
    model.features   = make([]int32, model.nNodes)
    model.thresholds = make([]float32, model.nNodes)
    model.lefts      = make([]int32, model.nNodes)
    model.rights     = make([]int32, model.nNodes)
    model.values     = make([][]float32, model.nNodes)

    // 5) Read node data
    for i := int32(0); i < model.nNodes; i++ {
        model.features[i]   = common.ReadInt32FromBytes(modelBytes, &idx)
        model.thresholds[i] = common.ReadFloat32FromBytes(modelBytes, &idx)
        model.lefts[i]      = common.ReadInt32FromBytes(modelBytes, &idx)
        model.rights[i]     = common.ReadInt32FromBytes(modelBytes, &idx)

        // 3 classes for Iris
        classProb := make([]float32, 3)
        for j := 0; j < 3; j++ {
            classProb[j] = common.ReadFloat32FromBytes(modelBytes, &idx)
        }
        model.values[i] = classProb
    }
    return nil
}

// Evaluate the decision tree
func evalIrisModel(model *irisModel, features []float32) int {
    var node int32 = 0

    for {
        // Leaf check
        if model.lefts[node] == -1 && model.rights[node] == -1 {
            // return class with highest probability
            maxIndex := 0
            for i := 1; i < len(model.values[node]); i++ {
                if model.values[node][i] > model.values[node][maxIndex] {
                    maxIndex = i
                }
            }
            return maxIndex
        }

        // Branch
        if features[model.features[node]] <= model.thresholds[node] {
            node = model.lefts[node]
        } else {
            node = model.rights[node]
        }
    }
}

func main() {
    // 1) Create model struct
    var model irisModel

    // 2) Load your model file from local path
    modelPath := "models/iris/ggml-model-small-f32-big-endian.bin"
    err := loadIrisModel(&model, modelPath)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
        os.Exit(1)
    }

    // 3) Example input


	if len(os.Args) < 5 {
        fmt.Fprintf(os.Stderr, "Usage: %s f1 f2 f3 f4\n", os.Args[0])
        os.Exit(1)
    }

    // 2) Convert them to float32
    input := make([]float32, 4)
    for i := 0; i < 4; i++ {
        val, err := strconv.ParseFloat(os.Args[i+1], 32)
        if err != nil {
            fmt.Fprintf(os.Stderr, "Invalid feature: %s\n", os.Args[i+1])
            os.Exit(1)
        }
        input[i] = float32(val)
    }

    // 4) Evaluate
    predictedClass := evalIrisModel(&model, input)
    fmt.Printf("Predicted class: %d\n", predictedClass)
}
