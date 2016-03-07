public class ActivationFunctions {

    public static double alpha = 1;

    public static void setAlpha(double a) {
        alpha = a;
    }

    public static double sigmoid(double S) {
        return 1 / (1 + Math.exp(-alpha * S));
    }

    public static double derivativeSigmoid(double S) {
        return alpha * sigmoid(S) * (1 - sigmoid(S));
    }

    public static double tanh(double S) {
        return Math.tanh(alpha * S);
    }

    public static double derivativeTanh(double S) {
        return alpha * (1 - tanh(S) * tanh(S));
    }

    public static double signum(double S) {
        return Math.signum(S);
    }

    public static double derivativeSignum(double S) {
        return 0;
    }

}
