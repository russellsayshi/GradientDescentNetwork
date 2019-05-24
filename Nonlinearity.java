public final class Nonlinearity {
	private Nonlinearity() {}
	public static double normalize(double x) {
		double pos = Math.exp(x);
		double neg = Math.exp(-x);
		double tanh = (pos - neg)/(pos + neg);
		return tanh;
	}
	public static double derivativeAt(double x) {
		double pos = Math.exp(2*x);
		double neg = Math.exp(-2*x);
		double dtanh = 4/(pos + neg + 2);
		return dtanh;
	}
}
