import java.util.Random;

public class Neuron {
	private final int numWeights;
	private double[] weights;
	private double bias;

	public Neuron(int numWeights) {
		this.numWeights = numWeights;
		weights = new double[numWeights];
	}

	public void randomize() {
		Random random = new Random();
		for(int i = 0; i < numWeights; i++) {
			weights[i] = random.nextGaussian();
		}
		bias = random.nextGaussian();
	}

	public double getOutput(double[] rawInput) {
		if(rawInput.length != numWeights) {
			throw new IllegalArgumentException("raw input does not match # weights");
		}
		double out = bias;
		for(int i = 0; i < numWeights; i++) {
			out += weights[i] * rawInput[i];
		}
		return out;
	}

	public double getWeight(int i) {
		return weights[i];
	}

	public double getBias() {
		return bias;
	}

	public void setWeight(int i, double weight) {
		weights[i] = weight;
	}

	public void nudgeWeight(int i, double nudge) {
		weights[i] += nudge;
	}

	public void setBias(double b) {
		bias = b;
	}

	public void nudgeBias(double nudge) {
		bias += nudge;
	}
}
