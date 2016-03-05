public abstract class Perceptron implements ActivationFunction {

    private double []weights;
    private double []inputs;
    private double desiredOutput;
    private double epsilon = 0.1;

    public Perceptron(double []i, double d) {
        this.inputs = i;
        this.desiredOutput = d;
        this.weights = new double[i.length];
        for (int j = 0; j < inputs.length; ++j) {
            int sign = Math.random() < 0.5 ? -1 : 1;
            weights[j] = sign * Math.random();
        }
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setInputs(double []i) {
        this.inputs = i;
    }

    public void setDesiredOutputs(double d) {
        this.desiredOutput = d;
    }
    
    public void setEpsilon(double e) {
        this.epsilon = e;
    }

    public double[] getWeights() {
        return this.weights;
    }

    double sum() {
        double sum = 0;
        for (int i = 0; i < weights.length; ++i) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    double output() {
        return activationFunction(sum());
    }

    void correctWeights() {
        for (int i = 0; i < weights.length; ++i) {
            /*W(i + 1)[k] = W(i)[k] + (desired_output - network_output) * input[k] * ABS(W(i)[k]) * epsilon
            * epsilon1 = ABS(W(i)[k] * epsilon
            */
            weights[i] = weights[i] + (desiredOutput - output()) * inputs[i] * epsilon;
        }
    }
}
