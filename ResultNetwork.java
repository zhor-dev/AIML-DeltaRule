import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ResultNetwork {

    public double[][] dOut = {
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
            { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
            { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 },
            { 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 },
            { 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 },
            { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 },
            { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 }
    };

    private double [][]weights;
    private Layer layer;
    private double []input;

    public ResultNetwork(double []in) {
        this.input = in;
        layer = new Layer(in, dOut[0], dOut.length) {
            @Override
            public double activationFunction(double S) {
                return ActivationFunctions.sigmoid(S);
            }

            @Override
            public double functionDerivative(double S) {
                return ActivationFunctions.derivativeSigmoid(S);
            }
        };
        weights = new double[dOut.length][layer.getPerceptrons()[0].getWeights().length];
    }

    public int recognize(double []i) {
        setWeights();
        layer.setInputs(i);
        double err = getErr(layer.layerOutput(), dOut[0]);
        int index = 0;
        for (int j = 1; j < layer.layerOutput().length; ++j) {
            if (getErr(dOut[j], layer.layerOutput()) < err) {
                index = j;
                err = getErr(dOut[j], layer.layerOutput());
            }
        }
        return index;
    }

    private void setWeights() {
        BufferedReader buffReader = null;
        try {
            FileReader fileReader = new FileReader("src/weights.txt");
            buffReader = new BufferedReader(fileReader);
            for (int i = 0; i < weights.length; ++i) {
                for (int j = 0; j < weights[0].length; ++j) {
                    weights[i][j] = Double.parseDouble(buffReader.readLine());
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        } finally {
            try {
                assert buffReader != null;
                buffReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        layer.setWeights(weights);
    }

    private double getErr(double []o1, double []o2) {
        double err = 0;
        for (int j = 0; j < o1.length; ++j) {
            err += Math.pow(o1[j] - o2[j], 2);
        }
        return err;
    }
}
