package gonn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

//Network structure
type Network struct {
	inputSize    int
	outputSize   int
	hiddenLayers int
	layers       []*Layer
}

//Layer is an input, hidden or output layer in the Network
type Layer struct {
	weights     *mat.Dense
	biases      *mat.Dense
	activations *mat.Dense
	outputs     *mat.Dense
	size        int
}

//Size returns the no. of neuron in the layer
func (layer *Layer) Size() int {
	//return len(layer.neurons)
	return layer.size
}

//LayerConfig is the config for a layer
type LayerConfig struct {
	InputCount int
	Size       int
}

//NewNetwork creates a new network
func NewNetwork(inputSize, outputSize int, hiddenLayers []LayerConfig) *Network {
	layers := make([]*Layer, len(hiddenLayers)+2)
	layers[0] = inputLayer(inputSize) //input layer

	for i := 0; i < len(hiddenLayers); i++ {
		layers[i+1] = newLayer(hiddenLayers[i].Size, hiddenLayers[i].InputCount)
	}

	//Output Layer
	layers[len(hiddenLayers)+1] = newLayer(outputSize, hiddenLayers[len(hiddenLayers)-1].Size) //output layer

	network := Network{
		inputSize:    inputSize,
		outputSize:   outputSize,
		hiddenLayers: len(hiddenLayers),
		layers:       layers,
	}

	return &network
}

func newLayer(size, inputCount int) *Layer {

	ws := make([]float64, size*inputCount)
	x := 0
	biases := make([]float64, size)
	for i := 0; i < size; i++ {
		weights := make([]float64, inputCount)
		for j := 0; j < inputCount; j++ {
			weights[j] = rand.NormFloat64()
			ws[x] = weights[j]
			x++
		}
		biases[i] = rand.NormFloat64()
	}
	layerWeights := mat.NewDense(inputCount, size, ws)
	layer := Layer{
		size:    size,
		weights: layerWeights,
		biases:  mat.NewDense(size, 1, biases),
	}
	return &layer
}

func inputLayer(size int) *Layer {

	biases := make([]float64, size)
	ws := make([]float64, size)
	for i := 0; i < size; i++ {
		biases[i] = 0
		ws[i] = rand.NormFloat64()
	}
	layer := Layer{
		size:    size,
		weights: mat.NewDense(1, size, ws),
		biases:  mat.NewDense(size, 1, biases),
	}
	return &layer
}
