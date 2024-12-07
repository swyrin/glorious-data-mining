import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.WekaPackageManager;

import java.util.Random;

public class RandomForestEvaluation {
    public static void main(String[] args) throws Exception {
        WekaPackageManager.loadPackages(false);

        // ================== DATA LOADING ==================
        // load the data
        Instances data = Utility.readData();
        Utility.printAttributes(data);

        // ================== DATA CLEANING ==================
        // delete DATE, IND, IND.1, IND.2
        data.deleteAttributeAt(0);
        data.deleteAttributeAt(1);
        data.deleteAttributeAt(2);
        data.deleteAttributeAt(3);
        Utility.printAttributes(data);

        // set last attribute as the class target
        data.setClassIndex(data.numAttributes() - 1);

        // ================== DATA EVAL ==================
        try {
            Evaluation evaluator = new Evaluation(data);
            RandomForest rf = new RandomForest();

            rf.setDebug(false);
            rf.buildClassifier(data);

            evaluator.crossValidateModel(rf, data, 10, new Random(31102003));

            System.out.println(evaluator.toSummaryString());
            System.out.println(rf);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
