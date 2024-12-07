import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.WekaPackageManager;

public class KNearestNeighborClusteringEvaluation {
    public static void main(String[] args) throws Exception {
        WekaPackageManager.loadPackages(false);

        // ================== DATA LOADING ==================
        // load the data
        Instances data = Utility.readData();
        Utility.printAttributes(data);

        // ================== DATA CLEANING ==================
        // delete IND and Date
        data.deleteAttributeAt(0);
        data.deleteAttributeAt(1);
        Utility.printAttributes(data);

        // ================== DATA EVAL ==================
        try {
            SimpleKMeans knn = new SimpleKMeans();

            knn.setDebug(false);
            knn.buildClusterer(data);
            knn.setPreserveInstancesOrder(true);

            System.out.println(knn);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
