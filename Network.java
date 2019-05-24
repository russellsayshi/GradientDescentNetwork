public class Network {
	private Neuron[][] neurons;
	private int numInputs;
	private int numTotalNeurons;
	private int numWeights;

	public Network(int numInputs, int[] numNeuronsPerLevel) {
		this.numInputs = numInputs;
		neurons = new Neuron[numNeuronsPerLevel.length][];
		for(int i = 0; i < neurons.length; i++) {
			Neuron[] nlist = new Neuron[numNeuronsPerLevel[i]];
			int numWeightsPerNeuron = i == 0 ? numInputs : numNeuronsPerLevel[i-1];
			for(int o = 0; o < nlist.length; o++) {
				nlist[o] = new Neuron(numWeightsPerNeuron);
				nlist[o].randomize();
			}
			numWeights += numWeightsPerNeuron * nlist.length;
			numTotalNeurons += nlist.length;
			neurons[i] = nlist;
		}
	}

	public double[] compute(double[] input) {
		if(input.length != numInputs) throw new IllegalArgumentException("size mismatch.");

		double[] prevOutputs = input;
		for(int layer = 0; layer < neurons.length; layer++) {
			double[] outputStore = new double[neurons[layer].length];
			for(int i = 0; i < outputStore.length; i++) {
				outputStore[i] = Nonlinearity.normalize(neurons[layer][i].getOutput(prevOutputs));
			}
			prevOutputs = outputStore;
		}
		return prevOutputs;
	}

	//TODO delete me and replace with something better
	public double computeZ(double[] input, int neuron_layer, int neuron) {
		if(input.length != numInputs) throw new IllegalArgumentException("size mismatch.");

		double[] prevOutputs = input;
		for(int layer = 0; layer < neurons.length; layer++) {
			if(neuron_layer == layer) {
				return neurons[layer][neuron].getOutput(prevOutputs);
			}
			double[] outputStore = new double[neurons[layer].length];
			for(int i = 0; i < outputStore.length; i++) {
				outputStore[i] = Nonlinearity.normalize(neurons[layer][i].getOutput(prevOutputs));
			}
			prevOutputs = outputStore;
		}
		throw new IllegalArgumentException("ee");
	}

	public double dCostdA(double[] input, int layer, int neuron) {
		double sum = 0;
		for(int i = 0; i < neurons[layer+1].length; i++) {
			double dzda = neurons[layer+1][i].getWeight(neuron);
			double dadz = Nonlinearity.derivativeAt(computeZ(input, layer, neuron));
			double dCda = 2*(Nonlinearity.normalize(computeZ(input, layer, neuron)));
			sum += dzda*dadz*dCda;
		}
		return sum;
	}

	public double dCostdw(double[] input, double[] output, int layer, int neuron, int prev_neuron) {
		double dzdw = (layer == 0 ? input[prev_neuron] : Nonlinearity.normalize(computeZ(input, layer-1, prev_neuron)));
		double dadz = Nonlinearity.derivativeAt(computeZ(input, layer, neuron));
		double dCda = (layer+1 == neurons.length ? 2*(Nonlinearity.normalize(computeZ(input, layer, neuron)) - output[neuron]) : dCostdA(input, layer, neuron));
		return dzdw*dadz*dCda;
	}

	public double dCostdb(double[] input, double[] output, int layer, int neuron) {
		double dzdb = 1;
		double dadz = Nonlinearity.derivativeAt(computeZ(input, layer, neuron));
		double dCda = (layer+1 == neurons.length ? 2*(Nonlinearity.normalize(computeZ(input, layer, neuron)) - output[neuron]) : dCostdA(input, layer, neuron));
		return dzdb*dadz*dCda;
	}

	public double[] gradientOfCost(double[] input, double[] output) {
		double[] gradient = new double[numWeights + numTotalNeurons];
		int gradient_index = 0;
		for(int i = 0; i < neurons.length; i++) {
			for(int o = 0; o < neurons[i].length; o++) {
				int numPrevNeurons = (i == 0 ? numInputs : neurons[i-1].length);
				for(int k = 0; k < numPrevNeurons; k++) {
					gradient[gradient_index++] = dCostdw(input, output, i, o, k);
				}
			}
		}
		for(int i = 0; i < neurons.length; i++) {
			for(int o = 0; o < neurons[i].length; o++) {
				gradient[gradient_index++] = dCostdb(input, output, i, o);
			}
		}
		if(gradient_index != gradient.length) throw new RuntimeException("gradient mismatch");
		return gradient;
	}

	public void nudgeWithGradient(double[] gradient) {
		int gradient_index = 0;
		for(int i = 0; i < neurons.length; i++) {
			for(int o = 0; o < neurons[i].length; o++) {
				int numPrevNeurons = (i == 0 ? numInputs : neurons[i-1].length);
				for(int k = 0; k < numPrevNeurons; k++) {
					neurons[i][o].nudgeWeight(k, -gradient[gradient_index++]);
				}
			}
		}
		for(int i = 0; i < neurons.length; i++) {
			for(int o = 0; o < neurons[i].length; o++) {
				neurons[i][o].nudgeBias(-gradient[gradient_index++]);
			}
		}
		if(gradient_index != gradient.length) throw new RuntimeException("gradient mismatch");
	}
}
