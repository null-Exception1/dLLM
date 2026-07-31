package quant

import (
	"math"
	"unsafe"
)

// BlockQuantize64 processes raw Float16 bytes into block-quantized INT8 vectors.
// It groups elements into 64-wide hardware chunks to preserve local outlier peaks.
func BlockQuantize64(rawFloat16Bytes []byte) ([]int8, []float32) {
	// 1. Zero-Copy Pointer Cast: Interpret the raw byte slice directly as uint16 values
	elementCount := len(rawFloat16Bytes) / 2
	if elementCount == 0 {
		return nil, nil
	}

	// Unsafe mapping to avoid allocating a temporary intermediate array
	f16Slice := unsafe.Slice((*uint16)(unsafe.Pointer(&rawFloat16Bytes[0])), elementCount)

	blockSize := 64
	blockCount := int(math.Ceil(float64(elementCount) / float64(blockSize)))

	quantizedData := make([]int8, elementCount)
	scaleVectors := make([]float32, blockCount)

	// 2. Loop through each 64-element block independently
	for b := 0; b < blockCount; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > elementCount {
			end = elementCount
		}

		// Step A: Find the local absolute maximum value within this cache block
		var maxAbs float32 = 0.0
		for i := start; i < end; i++ {
			val := decodeFloat16(f16Slice[i])
			absVal := float32(math.Abs(float64(val)))
			if absVal > maxAbs {
				maxAbs = absVal
			}
		}

		// Guard against a block of absolute zeros
		if maxAbs == 0 {
			scaleVectors[b] = 0.0
			continue
		}

		// Step B: Calculate the local hyper-precise Scale Factor
		scale := maxAbs / 127.0
		scaleVectors[b] = scale

		// Step C: Scale and clamp the float values into signed 8-bit integers
		for i := start; i < end; i++ {
			val := decodeFloat16(f16Slice[i])
			quantizedInt := int(math.Round(float64(val / scale)))

			// Hard hardware clamping safety check
			if quantizedInt > 127 {
				quantizedInt = 127
			} else if quantizedInt < -128 {
				quantizedInt = -128
			}
			quantizedData[i] = int8(quantizedInt)
		}
	}

	return quantizedData, scaleVectors
}

// decodeFloat16 is a high-speed bitwise decoder that expands IEEE 754 Half-Precision bits to Float32
func decodeFloat16(f16 uint16) float32 {
	sign := (f16 >> 15) & 0x1
	exponent := (f16 >> 10) & 0x1F
	fraction := f16 & 0x3FF

	var f32Bits uint32
	if exponent == 0 {
		if fraction == 0 {
			f32Bits = uint32(sign) << 31
		} else {
			// Subnormal number adjustment adjustments
			for (fraction & 0x400) == 0 {
				fraction <<= 1
				exponent--
			}
			exponent++
			fraction &= 0x3FF
			f32Bits = (uint32(sign) << 31) | (uint32(exponent+(-15+127)) << 23) | (uint32(fraction) << 13)
		}
	} else if exponent == 0x1F {
		// Infinity / NaN handling mapping
		f32Bits = (uint32(sign) << 31) | (0xFF << 23) | (uint32(fraction) << 13)
	} else {
		// Normal numbers mapping vector conversion
		f32Bits = (uint32(sign) << 31) | (uint32(exponent+(-15+127)) << 23) | (uint32(fraction) << 13)
	}

	return *(*float32)(unsafe.Pointer(&f32Bits))
}

// DequantizeBlock64 reconstructs compressed INT8 matrices back into precise Float32 arrays.
// It leverages the local scale factor vectors to recover the native tensor balances.
func DequantizeBlock64(quantizedData []int8, scaleVectors []float32) []float32 {
	elementCount := len(quantizedData)
	if elementCount == 0 {
		return nil
	}

	reconstructed := make([]float32, elementCount)
	blockSize := 64

	// Rebuild element positions by multiplying out the local block scales
	for i := 0; i < elementCount; i++ {
		blockIndex := i / blockSize
		if blockIndex >= len(scaleVectors) {
			break
		}

		scale := scaleVectors[blockIndex]
		reconstructed[i] = float32(quantizedData[i]) * scale
	}

	return reconstructed
}

// PackBitmask squeezes an array of booleans into a dense byte buffer.
// It packs exactly 8 individual flags into a single 8-bit byte to save network bandwidth.
func PackBitmask(bools []bool) []byte {
	length := len(bools)
	byteLength := (length + 7) / 8
	packed := make([]byte, byteLength)

	for i, b := range bools {
		if b {
			byteIdx := i / 8
			bitIdx := uint(i % 8)
			packed[byteIdx] |= (1 << bitIdx)
		}
	}
	return packed
}

// UnpackBitmask expands a dense compressed byte buffer back into a raw boolean array.
func UnpackBitmask(packed []byte, totalBits int) []bool {
	bools := make([]bool, totalBits)
	for i := 0; i < totalBits; i++ {
		byteIdx := i / 8
		bitIdx := uint(i % 8)
		if byteIdx < len(packed) {
			bools[i] = (packed[byteIdx] & (1 << bitIdx)) != 0
		}
	}
	return bools
}
