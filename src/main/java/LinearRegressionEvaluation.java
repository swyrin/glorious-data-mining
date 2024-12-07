import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.WekaPackageManager;

import java.util.Random;

public class LinearRegressionEvaluation {
    public static void main(String[] args) throws Exception {
        WekaPackageManager.loadPackages(false);

        // ================== DATA LOADING ==================
        // load the data
        Instances data = Utility.readData();
        Utility.printAttributes(data);

        // ================== DATA CLEANING ==================
        // delete IND
        data.deleteAttributeAt(2);
        Utility.printAttributes(data);

        // set RAIN as the class target
        data.setClassIndex(2);

        // ================== DATA EVAL ==================
        try {
            Evaluation evaluator = new Evaluation(data);
            LinearRegression lr = new LinearRegression();

            lr.setDebug(false);
            lr.buildClassifier(data);

            evaluator.crossValidateModel(lr, data, 10, new Random(31102003));

            System.out.println(evaluator.toSummaryString());
            System.out.println(lr);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
