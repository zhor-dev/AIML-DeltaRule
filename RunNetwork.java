import java.util.Arrays;

public class RunNetwork {

    public static void main(String... args) {
        TrainNetwork t = new TrainNetwork();
        t.train(200);
        double []i = { 0, 1, 1, 0 };
        ResultNetwork r = new ResultNetwork(i);
        System.out.println(Arrays.toString(r.dOut[r.recognize(i)]));
    }
}
