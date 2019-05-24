import java.util.Arrays;
public class Runner {
	public static void main(String[] args) {
		Network net = new Network(5, new int[] {4, 6, 3});
		double[] input = new double[] {1, 0.2, 0.3, 0.4, 0.5};
		double[] output = new double[] {0.6, 0.4, 0.3};
		System.out.println(Arrays.toString(net.compute(input)));
		for(int i = 0; i < 10000; i++) {
			double[] gradient = net.gradientOfCost(input, output);
			for(int o = 0; o < gradient.length; o++) {
				gradient[o] /= 100;
			}
			net.nudgeWithGradient(gradient);
			System.out.println(Arrays.toString(net.compute(input)));
		}
	}
}
