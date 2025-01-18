package common

import (
	"io/ioutil"
	"log"
    "math"
)

// vm only ===================================================================================

// memory layout in MIPS
const (
	INPUT_ADDR = 0x31000000
	OUTPUT_ADDR = 0x32000000
	MODEL_ADDR = 0x33000000
	MAGIC_ADDR = 0x30000800
)

// ReadBytesFromFile reads the entire file into a byte slice
func ReadBytesFromFile(filePath string) []byte {
    data, err := ioutil.ReadFile(filePath)
    if err != nil {
        log.Fatalf("Error reading file %s: %v\n", filePath, err)
    }
    return data
}

// ReadInt32FromBytes extracts an int32 from `data` at the current index `idx`
func ReadInt32FromBytes(data []byte, idx *int) int32 {
    val := uint32(data[*idx]) |
            uint32(data[*idx+1])<<8 |
            uint32(data[*idx+2])<<16 |
            uint32(data[*idx+3])<<24
    *idx += 4
    return int32(val)
}

// ReadFloat32FromBytes extracts a float32 from `data` at the current index `idx`
func ReadFloat32FromBytes(data []byte, idx *int) float32 {
    bits := uint32(data[*idx]) |
            uint32(data[*idx+1])<<8 |
            uint32(data[*idx+2])<<16 |
            uint32(data[*idx+3])<<24
    *idx += 4
    return math.Float32frombits(bits)
}
