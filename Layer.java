public abstract class Layer implements ActivationFunction {
    protected Perceptron[] perceptrons;
    protected double[] inputs;
    protected int perceptronsCount;

    public Layer(double []i, double []d, int pCount) {
        this.inputs = i;
        this.perceptronsCount = pCount;
        perceptrons = new Perceptron[perceptronsCount];
        for (int j = 0; j < perceptronsCount; ++j) {
            perceptrons[j] = new Perceptron(inputs, d[j]) {
                @Override
                public double activationFunction(double S) {
                    return Layer.this.activationFunction(S);
                }

                @Override
                public double functionDerivative(double S) {
                    return Layer.this.functionDerivative(S);
                }
            };
        }
    }

    public double[] layerOutput() {
        double []res = new double[perceptronsCount];
        for (int i = 0; i < perceptronsCount; ++i) {
            res[i] = perceptrons[i].output();
        }
        return res;
    }

    public void setWeights(double [][]w) {
        for (int j = 0; j < perceptrons.length; ++j) {
            perceptrons[j].setWeights(w[j]);
        }
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
        for (int j = 0; j < perceptronsCount; ++j) {
            perceptrons[j].setInputs(inputs);
        }
    }

    public Perceptron[] getPerceptrons() {
        return perceptrons;
    }
}
