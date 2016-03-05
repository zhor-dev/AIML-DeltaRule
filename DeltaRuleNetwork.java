public class DeltaRuleNetwork extends Layer {

    private double learningRate = 0.1;
    protected double [][]desiredOutput;
    private int dOutputIndex;

    public DeltaRuleNetwork(double[] i, double [][]dOutput, int pCount, int dOutputIndex) {
        super(i, dOutput[dOutputIndex], pCount);
        this.desiredOutput = dOutput;
        this.dOutputIndex = dOutputIndex;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public double activationFunction(double S) {
        return ActivationFunctions.sigmoid(S);
    }

    @Override
    public double functionDerivative(double S) {
        return ActivationFunctions.derivativeSigmoid(S);
    }

    public void setDesiredOutput(double[][] desiredOutput) {
        this.desiredOutput = desiredOutput;
    }

    public void setdesiredOutputIndex(int dOutputIndex) {
        this.dOutputIndex = dOutputIndex;
    }

    public void trainNetwork() {
        double e = 0;
        for (int i = 0; i < perceptronsCount; ++i) {
            e += Math.pow(perceptrons[i].output() - desiredOutput[dOutputIndex][i], 2);
        }
        if (!isCorrect(e)) {
            updateWeights();
            System.out.println("NO");
        } else {
            System.out.println("YES");
        }
    }

    private void updateWeights() {
        for (int i = 0; i < perceptrons.length; ++i) {
            for (int j = 0; j < perceptrons[i].getWeights().length; ++j) {
                perceptrons[i].getWeights()[j] = perceptrons[i].getWeights()[j]
                        + learningRate
                        * (desiredOutput[dOutputIndex][i] - perceptrons[i].output())
                        * functionDerivative(perceptrons[i].sum())
                        * inputs[j];
            }
        }
    }

    private boolean isCorrect(double e) {
        double err = 0;
        for (double[] outputs : desiredOutput) {
            if (notCurrentOutput(outputs)) {
                for (int j = 0; j < perceptronsCount; ++j) {
                    err += Math.pow(perceptrons[j].output() - outputs[j], 2);
                }
                if (err < e) {
                    return false;
                }
            }
            err = 0;
        }
        return true;
    }

    private boolean notCurrentOutput(double []ouput) {
        for (int i = 0; i < ouput.length; ++i) {
            if (ouput[i] != desiredOutput[dOutputIndex][i]) return true;
        }
        return false;
    }

}
