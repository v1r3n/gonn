package gonn

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

//MLP is the MLP
type MLP struct {
	numLayers int
	sizes     []int
	biases    []*mat.Dense
	weights   []*mat.Dense
}

//NewMLP creates NewMLP ;)
func NewMLP(sizes ...int) *MLP {

	// generate some random weights and biases
	bs := []*mat.Dense{}
	ws := []*mat.Dense{}

	// len of slices we will make
	// don't need any biases for input layer
	// don't need any weights for output layer
	l := len(sizes) - 1

	for j := 0; j < l; j++ {
		y := sizes[1:][j] // y starts from layer after input layer to output layer
		x := sizes[:l][j] // x starts from input layer to layer before output layer

		// make a random init biases matrix of y*1
		b := make([]float64, y)
		for i := range b {
			b[i] = rand.NormFloat64()
		}
		bs = append(bs, mat.NewDense(y, 1, b))

		// make a random init weights matrix of y*x
		w := make([]float64, y*x)
		for i := range w {
			w[i] = rand.NormFloat64()
		}
		ws = append(ws, mat.NewDense(x, y, w)) // P:changed the order of row and column

	}
	fmt.Println("\n\n\n.")
	fmt.Println("======-> MLP <-==========")
	fmt.Println("sizes", sizes)
	fmt.Println("biases", len(bs))
	fmt.Println(mat.Formatted(bs[0]))
	fmt.Println()
	fmt.Println(mat.Formatted(bs[1]))
	fmt.Println()
	fmt.Println("weights", len(ws))
	fmt.Println(mat.Formatted(ws[0]))
	fmt.Println()
	fmt.Println(mat.Formatted(ws[1]))
	fmt.Println()
	fmt.Println("=========================")
	fmt.Println("\n\n\n.")
	return &MLP{
		numLayers: len(sizes),
		sizes:     sizes,
		biases:    bs,
		weights:   ws,
	}
}

//Forward forwards the request ;-
func (n *MLP) Forward(x mat.Matrix) (as, zs []mat.Matrix) {

	as = append(as, x)
	_x := x

	for i := 0; i < len(n.weights); i++ {

		w := n.weights[i]
		b := n.biases[i]

		// z = w.x + b

		m := new(mat.Dense)

		m.Mul(_x, w)

		z := new(mat.Dense)
		addB := func(_, col int, v float64) float64 { return v + b.At(col, 0) }
		z.Apply(addB, m)

		zs = append(zs, z)

		// a = sigmoid(z)
		a := new(mat.Dense)
		as = append(as, a)

		_x = a
	}

	return

}
