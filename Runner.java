import java.util.Arrays;
import java.util.Random;
public class Runner {
	public static void main(String[] args) {
		Network net = new Network(5, new int[] {3});
		double[] dummyInput = new double[] {1, 0.2, 0.3, 0.4, 0.5};
		double[] dummyOutput = new double[] {0, 0, 0};
		Random random = new Random();
		for(int i = 0; i < 100000; i++) {
			double[] gradient = net.gradientOfCost(dummyInput, dummyOutput);
			double acost = 0;
			for(int o = 0; o < gradient.length; o++) gradient[o] = 0;
			for(int k = 0; k < 100; k++) {
				double r = random.nextDouble();
				double[] input = new double[] {1, 0.2, r, 0.4, 0.5};
				double[] output = new double[] {0.6, r, 0.3};
				acost += net.computeCost(input, output);
				double[] m_gradient = net.gradientOfCost(input, output);
				for(int o = 0; o < gradient.length; o++) {
					gradient[o] += m_gradient[o];
				}
			}
			for(int o = 0; o < gradient.length; o++) {
				gradient[o] /= 100 * 1000;
			}
			net.nudgeWithGradient(gradient);

			// if it's real good, we're done
			double cost = acost/100;
			System.out.println(cost);
			if(cost < 0.001) {
				break;
			}

			// if it's real bad, fix it away
			if(cost > 0.1) {
				net.randomize();
				i--;
			}
		}
		for(double k = 0; k <= 1; k += 0.05) {
			System.out.println("" + k + "k: " + Arrays.toString(net.compute(new double[] {1, 0.2, k, 0.4, 0.4})));
		}
	}
}
