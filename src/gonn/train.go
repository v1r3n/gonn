package gonn

import (
	"fmt"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

//Train the network
func (network *Network) Train(epoch int, x, y []*mat.Dense) {
	for i := 0; i < epoch; i++ {
		fmt.Println("epoch", i)
		network.updateMiniBatch(x, y, 0.1)
	}

}
func (network *Network) updateMiniBatch(miniX, miniY []*mat.Dense, eta float64) {

	for i := 0; i < len(miniX); i++ {
		x := miniX[i]
		y := miniY[i]
		N, _ := x.Dims()
		dnbs, dnws := network.backprop(x, y)

		//Skip the input layer, so starting from first hidden layer (if any)
		for k := 1; k < len(network.layers); k++ {

			layer := network.layers[k]
			weights := layer.weights
			nw := dnws[k-1]

			b := layer.biases
			nb := addColumns(dnbs[k-1]).T()

			alpha := eta / float64(N)
			scalednw := new(mat.Dense)
			scalednw.Scale(alpha, nw)

			scalednb := new(mat.Dense)
			scalednb.Scale(alpha, nb)

			wprime := new(mat.Dense)
			wprime.Sub(weights, scalednw)

			bprime := new(mat.Dense)
			bprime.Sub(b, scalednb)

			//Update the weights and biases now here...
			layer.weights = wprime
			layer.biases = bprime
		}

	}
}

func (network *Network) backprop(x, y *mat.Dense) (nbs, nws []*mat.Dense) {
	nablaBs := make([]*mat.Dense, len(network.layers)-1)
	nablaWs := make([]*mat.Dense, len(network.layers)-1)
	m := len(nablaBs) - 1

	outputs, activations := network.FeedForward(x)
	delta := costFn(outputs, activations, y)

	//For the output layer
	nablaW := new(mat.Dense)
	prevToLast := len(network.layers) - 2
	a := network.layers[prevToLast].activations
	nablaW.Mul(a.T(), delta)

	nablaBs[m] = delta
	nablaWs[m] = nablaW
	m--

	for k := len(network.layers) - 2; k > 0; k-- {
		z := network.layers[k].outputs

		sp := new(mat.Dense)
		sp.Apply(applySigmoidprime, z)

		wdelta := new(mat.Dense)
		weights := network.layers[k+1].weights

		wdelta.Mul(delta, weights.T())

		nextdelta := new(mat.Dense)
		nextdelta.MulElem(wdelta, sp)
		delta = nextdelta
		nablaBs[m] = delta

		a := network.layers[k-1].activations
		nw := new(mat.Dense)
		nw.Mul(a.T(), delta)

		nablaWs[m] = nw
		m--

	}

	return nablaBs, nablaWs
}

//TODO: cost function should be pluggable
func costFn(z, out, y *mat.Dense) *mat.Dense {
	err := new(mat.Dense)
	err.Sub(out, y)

	// delta of last layer
	// delta = (out - y).sigmoidprime(last_z)
	sp := new(mat.Dense)
	sp.Apply(applySigmoidprime, z)

	delta := new(mat.Dense)
	delta.MulElem(err, sp)

	return delta
}

func addColumns(m *mat.Dense) *mat.Dense {

	_, c := m.Dims()

	data := make([]float64, c)
	for i := 0; i < c; i++ {
		col := mat.Col(nil, i, m)
		data[i] = floats.Sum(col)
	}
	return mat.NewDense(1, c, data)
}
func pm(name string, m *mat.Dense) {
	fmt.Println("Printing -> ", name)
	fmt.Println(mat.Formatted(m))
	fmt.Println()
}
