public class Runner {
	public static void main(String[] args) {
		Network net = new Network(5, new int[] {5, 3, 1});
		System.out.println(net.gradientOfCost(new double[] {1, 2, 3, 4, 5}));
	}
}
