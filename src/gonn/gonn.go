package gonn

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

//FeedForward the network
func (network *Network) FeedForward(input *mat.Dense) (*mat.Dense, *mat.Dense) {
	activations := input
	outputs := input
	network.layers[0].activations = input
	network.layers[0].outputs = nil
	for i := 1; i < len(network.layers); i++ {
		layer := network.layers[i]
		layer.feedForward(activations)
		as := layer.activations
		zs := layer.outputs

		activations = as
		outputs = zs
	}
	return outputs, activations
}

//Print the network details
func (network *Network) Print() {
	for i := 0; i < len(network.layers); i++ {
		fmt.Println("Layer", i)
		weights := network.layers[i].weights
		fmt.Println(mat.Formatted(weights))
		fmt.Println("<=========>")
	}
}

func (layer *Layer) feedForward(input *mat.Dense) {

	w := layer.weights
	b := layer.biases

	m := new(mat.Dense)
	m.Mul(input, w)

	z := new(mat.Dense)
	addB := func(_, col int, v float64) float64 { return v + b.At(col, 0) }
	z.Apply(addB, m)

	a := new(mat.Dense)
	a.Apply(applySigmoid, z)

	layer.activations = a
	layer.outputs = z

	/*
		activations := make([]float64, layer.Size())
		outputs := make([]float64, layer.Size())

		for i := 0; i < layer.Size(); i++ {
			weights := layer.weights.ColView(i)
			outputs[i], activations[i] = eval(weights, layer.biases.At(0, i), input)
		}
		layer.activations = mat.NewDense(1, len(activations), activations)
		layer.outputs = mat.NewDense(1, len(outputs), outputs)
	*/
}

func eval(weights mat.Vector, bias float64, input *mat.Dense) (z, a float64) {
	dot := mat.Dot(weights, input.RowView(0))
	z = dot + bias
	a = sigmoid(z)

	return z, a
}

//Sigmoid function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// DerivSigmoid computes derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
func derivSigmoid(x float64) float64 {
	fx := sigmoid(x)
	return fx * (1 - fx)
}
func applySigmoidprime(_, _ int, v float64) float64 {
	return derivSigmoid(v)
}

func applySigmoid(_, _ int, v float64) float64 {
	return sigmoid(v)
}
