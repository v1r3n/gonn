package gonn

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Neuron structure
type Neuron struct {
	weights []float64
	bias    float64
	output  float64
	opsum   float64
}

//Network is a neural network implementation
type Network struct {
	inputSize  int
	outputSize int
	layers     []Layer
	op2        Neuron
	op         []Neuron
}

//Layer of a neural network - hidden our output
type Layer struct {
	outputSize int
	neurons    []Neuron
}

func newLayer(inputSize int) Layer {
	bias := float64(0)
	neurons := make([]Neuron, inputSize)
	weights := make([]float64, inputSize)

	for i := 0; i < inputSize; i++ {
		weights[i] = rand.NormFloat64()
	}

	for i := 0; i < inputSize; i++ {
		neurons[i] = Neuron{
			bias:    bias,
			weights: weights,
		}
	}

	return Layer{
		neurons:    neurons,
		outputSize: inputSize,
	}

}

//NewNetwork creates a new network
func NewNetwork(inputSize, outputSize, hiddenLayersCount int) Network {
	bias := rand.NormFloat64()
	weights := make([]float64, inputSize)
	hiddenLayers := make([]Layer, hiddenLayersCount)

	for i := 0; i < hiddenLayersCount; i++ {
		hiddenLayers[i] = newLayer(inputSize)
	}

	for i := 0; i < inputSize; i++ {
		weights[i] = rand.NormFloat64()
	}

	op := Neuron{
		bias:    bias,
		weights: weights,
	}

	outputs := make([]Neuron, outputSize)
	for i := 0; i < outputSize; i++ {
		outputs[i] = Neuron{
			bias:    rand.NormFloat64(),
			weights: weights,
		}
	}

	network := Network{
		inputSize:  inputSize,
		outputSize: outputSize,
		op2:        op,
		op:         outputs,
		layers:     hiddenLayers,
	}

	return network
}

//FeedForward comptation of the network
func (network *Network) FeedForward(inputs []float64) (float64, []float64) {
	if len(inputs) != network.inputSize {
		fmt.Println("Inputs do not match the network size", len(inputs), "!=", network.inputSize)
		//return
	}

	hiddenOps := make([][]float64, len(network.layers))
	inputToNextLayer := inputs
	for i := 0; i < len(network.layers); i++ {
		hiddenOps[i] = network.layers[i].feedForward(inputToNextLayer)
		inputToNextLayer = hiddenOps[i]
	}

	opVector := mat.NewDense(1, len(inputToNextLayer), inputToNextLayer).RowView(0)
	opsum := network.op2.Sum(opVector)
	outO1 := Sigmoid(opsum)

	network.op2.opsum = opsum
	network.op2.output = outO1

	all := make([]float64, network.outputSize)
	for i := 0; i < network.outputSize; i++ {
		opVector := mat.NewDense(1, len(inputToNextLayer), inputToNextLayer).RowView(0)
		opsum := network.op[i].Sum(opVector)
		outO1 := Sigmoid(opsum)

		network.op[i].opsum = opsum
		network.op[i].output = outO1
	}

	return outO1, all
}

func (layer *Layer) feedForward(inputs []float64) []float64 {
	sigmoids := make([]float64, layer.outputSize)
	for i := 0; i < len(layer.neurons); i++ {
		sum := layer.neurons[i].Sum(mat.NewDense(1, len(inputs), inputs).RowView(0))
		sigmoids[i] = Sigmoid(sum)
		layer.neurons[i].output = sigmoids[i]
		layer.neurons[i].opsum = sum
	}
	return sigmoids
}

func (layer *Layer) train(inputs []float64) {
	outputs := make([]float64, layer.outputSize)
	for i := 0; i < len(layer.neurons); i++ {
		outputs[i] = layer.neurons[i].Feedforward(mat.NewDense(1, len(inputs), inputs).RowView(0))
	}
}

//Train the neural network
func (network *Network) Train(epochs int, learnRate float64, inputs [][]float64, outputs []float64) {
	for v := 0; v < epochs; v++ {
		for k := 0; k < len(inputs); k++ {
			input := inputs[k]
			output := outputs[k]

			_, allPredicted := network.FeedForward(input)

			//Handle the output layer
			z := len(network.layers) - 1
			dLdYs := make([]float64, network.outputSize)
			dydHs := make([][]float64, network.outputSize)
			dLdY := float64(0)
			for k := 0; k < network.outputSize; k++ {
				dLdYs[k] = -2 * (output - allPredicted[k])
				dLdY += dLdYs[k]
				dydH := make([]float64, len(network.op[k].weights))
				dYdB := DerivSigmoid(network.op[k].opsum)

				for y := 0; y < len(network.op[k].weights); y++ {
					dYdW := network.layers[z].neurons[y].output * DerivSigmoid(network.op[k].opsum)
					dydH[y] = network.op[k].weights[y] * DerivSigmoid(network.op[k].opsum)
					network.op[k].weights[y] -= learnRate * dLdYs[k] * dYdW
				}
				dydHs[k] = dydH
				network.op[k].bias -= learnRate * dLdYs[k] * dYdB

			}
			//dLdY3 := -2 * (output - predicted2)
			//dydH := make([]float64, len(network.op2.weights))
			//dYdB := DerivSigmoid(network.op2.opsum)

			//for y := 0; y < len(network.op2.weights); y++ {
			//dYdW := network.layers[z].neurons[y].output * DerivSigmoid(network.op2.opsum)
			//dydH[y] = network.op2.weights[y] * DerivSigmoid(network.op2.opsum)
			//network.op2.weights[y] -= learnRate * dLdY * dYdW
			//}
			//network.op2.bias -= learnRate * dLdY * dYdB

			inputToNextLayer := input

			for ; z >= 0; z-- {
				layer := network.layers[z]

				for m := 0; m < len(layer.neurons); m++ {

					dHdW := inputToNextLayer[m] * DerivSigmoid(layer.neurons[m].opsum)
					dHdB := DerivSigmoid(layer.neurons[m].opsum)
					layer.neurons[m].weights[m] -= learnRate * dLdY * dydHs[0][m] * dHdW
					layer.neurons[m].bias -= learnRate * dLdY * dydHs[0][m] * dHdB
				}

			}

			if v%100 == 0 {
				//fmt.Println("Epoch", v, "Predicted", predicted, "Expected", output)
			}
		}

	}
}

//MseLoss calculates mean squared loss
func mseLoss(ytrue, y mat.Vector) float64 {
	r, _ := ytrue.Dims()
	var result = make([]float64, r)
	var mean = float64(0)
	for i := 0; i < r; i++ {
		result[i] = math.Pow((ytrue.At(i, 0) - y.At(i, 0)), 2)
		mean += result[i]
	}

	return mean / float64(r)
}

// Feedforward function implementation
func (neuron *Neuron) Feedforward(inputs mat.Vector) float64 {
	weightVector := mat.NewDense(1, len(neuron.weights), neuron.weights).RowView(0)
	var total = mat.Dot(weightVector, inputs) + neuron.bias
	return Sigmoid(total)
}

// Sum function implementation
func (neuron *Neuron) Sum(inputs mat.Vector) float64 {
	weightVector := mat.NewDense(1, len(neuron.weights), neuron.weights).RowView(0)
	var total = mat.Dot(weightVector, inputs) + neuron.bias
	return total
}

//Sigmoid function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// DerivSigmoid computes derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
func DerivSigmoid(x float64) float64 {
	fx := Sigmoid(x)
	return fx * (1 - fx)
}
