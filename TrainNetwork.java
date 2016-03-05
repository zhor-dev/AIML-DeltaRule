import java.io.*;

public class TrainNetwork extends DeltaRuleNetwork {

    private static double[][] in = {
            { 0, 0, 0, 0 },
            { 0, 0, 0, 1 },
            { 0, 0, 1, 0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0, 0 },
            { 0, 1, 0, 1 },
            { 0, 1, 1, 0 },
            { 0, 1, 1, 1 },
            { 1, 0, 0, 0 },
            { 1, 0, 0, 1 }
    };

    private static double[][] dOut = {
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

    public TrainNetwork() {
        super(in[0], dOut, dOut.length, 0);
    }

    public void train(int trainCicle) {
        for (int j = 0; j < trainCicle; ++j) {
            getNetworkDataAndTrain();
        }
        saveWeights();
    }

    private void getNetworkDataAndTrain() {
        int k = 0;
        while (k < desiredOutput.length) {
            setInputs(in[k]);
            setdesiredOutputIndex(k);
            trainNetwork();
            ++k;
        }
    }


    public void saveWeights() {
        String res = "";
        BufferedWriter output = null;
        try {
            File file = new File("src/weights.txt");
            output = new BufferedWriter(new FileWriter(file));
            for (Perceptron perceptron : perceptrons) {
                for (int j = 0; j < perceptron.getWeights().length; ++j) {
                    res += perceptron.getWeights()[j] + "\n";
                }
            }
            output.write(res);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (output != null) try {
                output.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
