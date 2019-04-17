package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	nn "./gonn"
	gonn "./network"
	mnist "github.com/petar/GoMNIST"
)

func main() {
	trainDigits()
}

func trainDigits() {
	train, test, err := mnist.Load("./data")
	if err != nil {
		fmt.Println("there was an error loading the data", err)
		return
	}
	fmt.Println("Test Size", len(test.Images))

	inputSize := 28 * 28
	x := make([]*mat.Dense, len(train.Images))
	testX := make([]*mat.Dense, len(test.Images))

	for i := 0; i < len(train.Images); i++ {
		inputs := make([]float64, len(train.Images[i]))
		for j := 0; j < len(train.Images[i]); j++ {
			inputs[j] = float64(train.Images[i][j])
		}
		x[i] = mat.NewDense(1, len(inputs), inputs)
	}
	pm("x.0", x[0])
	for i := 0; i < len(test.Images); i++ {
		inputs := make([]float64, len(test.Images[i]))
		for j := 0; j < len(test.Images[i]); j++ {
			inputs[j] = float64(test.Images[i][j])
		}
		testX[i] = mat.NewDense(1, len(inputs), inputs)
	}

	y := make([]*mat.Dense, len(train.Labels))
	testY := make([]*mat.Dense, len(test.Labels))

	for i := 0; i < len(train.Labels); i++ {
		label := train.Labels[i]
		yy := make([]float64, 10)
		for j := 0; j < 10; j++ {
			if j == int(label) {
				yy[j] = 1
			} else {
				yy[j] = 0
			}
		}
		y[i] = mat.NewDense(1, 10, yy)
	}

	for i := 0; i < len(test.Labels); i++ {
		label := test.Labels[i]
		yy := make([]float64, 10)
		for j := 0; j < 10; j++ {
			if j == int(label) {
				yy[j] = 1
			} else {
				yy[j] = 0
			}
		}
		testY[i] = mat.NewDense(1, 10, yy)
	}

	hiddenLayers := make([]nn.LayerConfig, 2)
	hiddenLayers[0] = nn.LayerConfig{
		InputCount: inputSize,
		Size:       30,
	}
	hiddenLayers[1] = nn.LayerConfig{
		InputCount: 30,
		Size:       15,
	}

	network := nn.NewNetwork(inputSize, 10, hiddenLayers)
	network.Train(20, x, y)

	acc := 0
	for t := 0; t < len(test.Images); t++ {
		testInput := test.Images[t]
		inputs := make([]float64, len(testInput))
		for j := 0; j < len(testInput); j++ {
			inputs[j] = float64(testInput[j])
		}

		_, a := network.FeedForward(mat.NewDense(1, len(testInput), inputs))
		found := mnistDigitMax(a)
		//fmt.Println("Expected", test.Labels[t], "Found", found)
		if int(test.Labels[t]) == found {
			acc++
		}
	}

	fmt.Println("\n\n\n\n\n\nAcc:", acc, "total:", len(test.Images))

}

func mnistDigitMax(m *mat.Dense) int {
	max := m.At(0, 0)
	maxIndx := 0
	for i := 0; i < 10; i++ {
		if m.At(0, i) > max {
			maxIndx = i
			max = m.At(0, i)
		}
	}
	return maxIndx
}

func trainXORNew() {

	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	outputs := []float64{
		0,
		1,
		1,
		0,
	}

	hiddenLayers := make([]nn.LayerConfig, 2)
	hiddenLayers[0] = nn.LayerConfig{
		InputCount: 2,
		Size:       2,
	}
	hiddenLayers[1] = nn.LayerConfig{
		InputCount: 2,
		Size:       2,
	}

	network := nn.NewNetwork(2, 1, hiddenLayers)

	len := 100000
	x := make([]*mat.Dense, len*4)
	y := make([]*mat.Dense, len*4)
	m := 0
	for j := 0; j < len; j++ {
		for k := 0; k < 4; k++ {
			x[m] = mat.NewDense(1, 2, inputs[k])
			y[m] = mat.NewDense(1, 1, []float64{outputs[k]})
			m++
		}

	}
	network.Train(10, x, y)

	for h := 0; h < 4; h++ {
		_, activations := network.FeedForward(mat.NewDense(1, 2, inputs[h]))
		fmt.Println("input", inputs[h], "expected", outputs[h])
		pm("a", activations)
	}

}
func main3() {
	fmt.Println()
	fmt.Println("====================================")
	fmt.Println("testXOR")
	fmt.Println("====================================")
	testXOR()

	fmt.Println()
	fmt.Println("====================================")
	fmt.Println("testGendorBasedOnWeightAndHeight")
	fmt.Println("====================================")
	testGendorBasedOnWeightAndHeight()
}
func testGendorBasedOnWeightAndHeight() {
	inputs := [][]float64{
		{-2, -1},
		{25, 6},
		{17, 4},
		{-15, -6},
	}
	outputs := []float64{
		1,
		0,
		0,
		1,
	}
	fmt.Println("inputs", inputs)
	fmt.Println("outputs", outputs)

	network := gonn.NewNetwork(2, 1, 1)

	network.Train(100000, 0.1, inputs, outputs)
	output, _ := network.FeedForward([]float64{115, 16})
	fmt.Println("115, 16 output", output, "Expected", 0)

	output, _ = network.FeedForward([]float64{-25, -26})
	fmt.Println("-25, -26 output", output, "Expected", 1)

	for x := 0; x < len(inputs); x++ {
		output, _ := network.FeedForward(inputs[x])
		fmt.Println("Test, Input", inputs[x], "Output", output)
	}

}
func testXOR() {
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	outputs := []float64{
		0,
		1,
		1,
		0,
	}
	fmt.Println("inputs", inputs)
	fmt.Println("outputs", outputs)

	network := gonn.NewNetwork(2, 1, 2)

	network.Train(1000000, 0.2, inputs, outputs)

	for x := 0; x < len(inputs); x++ {
		output, output2 := network.FeedForward(inputs[x])
		fmt.Println("Test, Input", inputs[x], "Output", output, "Expected", outputs[x], "output2", output2)
	}
}
func pm(name string, m *mat.Dense) {
	fmt.Println("Printing -> ", name)
	fmt.Println(mat.Formatted(m))
	fmt.Println()
}
