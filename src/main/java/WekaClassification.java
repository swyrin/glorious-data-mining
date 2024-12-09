import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.Add;
import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class WekaClassification {
    private RandomForest classifier;
    private Instances data;

    public WekaClassification() {
        // 3.1 Model Selection - Random Forest
        System.out.println("3.1 Model Selection:");
        System.out.println("Using Random Forest classifier because:");
        System.out.println("- Handles high-dimensional data well");
        System.out.println("- Provides feature importance rankings");
        System.out.println("- Resistant to overfitting");
        System.out.println("- Handles both numerical and categorical features\n");

        classifier = new RandomForest();
    }

    public void convertCSVtoARFF(String csvPath, String arffPath) throws Exception {
        System.out.println("3.2 Implementation Process:");
        System.out.println("Step 1: Loading and converting data format");

        File csvFile = new File(csvPath);
        if (!csvFile.exists()) {
            throw new IllegalArgumentException("CSV file not found: " + csvPath);
        }

        Path arffDirectory = Paths.get(arffPath).getParent();
        if (arffDirectory != null && !Files.exists(arffDirectory)) {
            Files.createDirectories(arffDirectory);
        }

        System.out.println("Loading CSV file from: " + csvPath);
        CSVLoader loader = new CSVLoader();
        loader.setSource(csvFile);
        data = loader.getDataSet();

        System.out.println("Saving ARFF file to: " + arffPath);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(arffPath))) {
            writer.write("@relation WindPrediction\n\n");

            for (int i = 0; i < data.numAttributes(); i++) {
                writer.write("@attribute '" + data.attribute(i).name() + "' " +
                        (i == data.numAttributes() - 1 ? "{low,medium,high}" : "numeric") + "\n");
            }

            writer.write("\n@data\n");
            for (int i = 0; i < data.numInstances(); i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < data.numAttributes(); j++) {
                    if (j > 0) sb.append(",");
                    if (j == data.numAttributes() - 1) {
                        // Discretize the class value manually
                        double val = data.instance(i).value(j);
                        if (val < 5.0) sb.append("low");
                        else if (val < 10.0) sb.append("medium");
                        else sb.append("high");
                    } else {
                        sb.append(data.instance(i).value(j));
                    }
                }
                writer.write(sb.toString() + "\n");
            }
        }

        // Reload the data from ARFF to ensure proper formatting
        data = new weka.core.Instances(new java.io.BufferedReader(new java.io.FileReader(arffPath)));

        System.out.println("Data conversion completed successfully");
        System.out.println("Total instances: " + data.numInstances());
        System.out.println("Total attributes: " + data.numAttributes());
    }

    public void prepareData(int classIndex) throws Exception {
        System.out.println("\nStep 2: Preparing data for classification");

        if (classIndex < 0 || classIndex >= data.numAttributes()) {
            throw new IllegalArgumentException("Invalid class index");
        }

        // Set class attribute
        data.setClassIndex(classIndex);

        // Configure Random Forest
        classifier.setMaxDepth(100);
        classifier.setNumFeatures((int) Math.log(data.numAttributes()) + 1);
        classifier.setSeed(42);

        System.out.println("Data preparation completed:");
        System.out.println("- Number of classes: " + data.numClasses());
        System.out.println("- Number of features: " + (data.numAttributes() - 1));
    }

    public Evaluation trainAndEvaluate(boolean isImprovedModel) throws Exception {
        if (isImprovedModel)
            System.out.println("\nStep 5: Model Evaluation");
        else
            System.out.println("\nStep 3: Training and evaluation");

        long startTime = System.currentTimeMillis();
        Evaluation eval = new Evaluation(data);

        if (isImprovedModel) {
            // Use 10-fold cross-validation for improved model
            System.out.println("Performing 10-fold cross-validation for improved model...");
            eval.crossValidateModel(classifier, data, 10, new Random(42));
        } else {
            // Use 5-fold cross-validation for initial model
            System.out.println("Performing 5-fold cross-validation for initial model...");
            eval.crossValidateModel(classifier, data, 5, new Random(42));

            // Additional evaluation metrics for initial model
            System.out.println("\nPerforming additional evaluation steps:");

            // Holdout validation (70-30 split)
            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;

            // Randomize the data
            Random rand = new Random(42);
            Instances randData = new Instances(data);
            randData.randomize(rand);

            // Create train and test sets
            Instances trainSet = new Instances(randData, 0, trainSize);
            Instances testSet = new Instances(randData, trainSize, testSize);

            // Build classifier on training set
            RandomForest holdoutClassifier = new RandomForest();
            holdoutClassifier.setOptions(classifier.getOptions());
            holdoutClassifier.buildClassifier(trainSet);

            // Evaluate on test set
            Evaluation holdoutEval = new Evaluation(trainSet);
            holdoutEval.evaluateModel(holdoutClassifier, testSet);

            // Calculate weighted metrics manually
            double weightedPrecision = 0.0;
            double weightedRecall = 0.0;
            double weightedF1 = 0.0;
            double totalInstances = testSet.numInstances();

            for (int i = 0; i < testSet.numClasses(); i++) {
                double classCount = holdoutEval.numTruePositives(i) +
                        holdoutEval.numFalseNegatives(i);

                // Calculate precision, recall, and F1 for each class
                double precision = holdoutEval.precision(i);
                double recall = holdoutEval.recall(i);
                double f1 = holdoutEval.fMeasure(i);

                // Weight by class distribution
                double weight = classCount / totalInstances;
                weightedPrecision += precision * weight;
                weightedRecall += recall * weight;
                weightedF1 += f1 * weight;
            }

            System.out.println("Holdout Validation Results (70-30 split):");
            System.out.printf("Accuracy: %.2f%%\n", holdoutEval.pctCorrect());
            System.out.printf("Weighted Precision: %.3f\n", weightedPrecision);
            System.out.printf("Weighted Recall: %.3f\n", weightedRecall);
            System.out.printf("Weighted F1-Score: %.3f\n", weightedF1);
        }

        // Build final classifier on full dataset
        classifier.buildClassifier(data);

        long runtime = System.currentTimeMillis() - startTime;
        System.out.println("Training and evaluation completed in " + runtime + "ms");

        return eval;
    }

    public void printResults(Evaluation eval, boolean isImprovedModel) throws Exception {
        if (!isImprovedModel)
            System.out.println("\n3.3 Results:");
        else
            System.out.println("\n5.1 Performance Metrics:");

        System.out.println("============");

        // Model Accuracy
        System.out.printf("Model Accuracy: %.2f%%\n", eval.pctCorrect());

        // Calculate weighted precision, recall, and F1-score manually
        double weightedPrecision = 0.0;
        double weightedRecall = 0.0;
        double weightedF1 = 0.0;
        double totalInstances = data.numInstances();

        for (int i = 0; i < data.numClasses(); i++) {
            // Get class count (number of instances for class i)
            double classCount = eval.numTruePositives(i) + eval.numFalsePositives(i) + eval.numTrueNegatives(i) + eval.numFalseNegatives(i);

            // Precision, Recall, F1 for the class
            double precision = eval.precision(i);
            double recall = eval.recall(i);
            double f1 = eval.fMeasure(i);

            // Weighted metrics
            weightedPrecision += precision * classCount / totalInstances;
            weightedRecall += recall * classCount / totalInstances;
            weightedF1 += f1 * classCount / totalInstances;
        }

        // Print weighted precision, recall, and F1
        System.out.printf("Weighted Precision: %.3f\n", weightedPrecision);
        System.out.printf("Weighted Recall: %.3f\n", weightedRecall);
        System.out.printf("Weighted F1-Score: %.3f\n", weightedF1);

        System.out.println("\nDetailed Performance by Class:");
        System.out.println(eval.toClassDetailsString());

        System.out.println("\nConfusion Matrix:");
        System.out.println(eval.toMatrixString());
    }

    public void improveWithClustering(int numClusters) throws Exception {
        System.out.println("\nStep 4: Improvement of Results");
        System.out.println("4.1 Methodology:");
        System.out.println("Using K-Means clustering to enhance the model's performance");
        System.out.println("Number of clusters: " + numClusters);

        // Create a copy of data for clustering (excluding class attribute)
        Instances clusterData = new Instances(data);
        clusterData.setClassIndex(clusterData.numAttributes() - 1);
        Remove removeClass = new Remove();
        removeClass.setAttributeIndices("last");
        removeClass.setInputFormat(clusterData);
        clusterData = Filter.useFilter(clusterData, removeClass);

        // Configure and build K-Means clusterer
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClusters);
        kMeans.setSeed(42);  // For reproducibility
        kMeans.buildClusterer(clusterData);

        // Add cluster attribute to original data
        Add addCluster = new Add();
        addCluster.setAttributeName("cluster");
        addCluster.setAttributeIndex("last");
        addCluster.setNominalLabels("cluster0,cluster1,cluster2");  // Set nominal labels directly
        addCluster.setInputFormat(data);
        Instances newData = Filter.useFilter(data, addCluster);

        // Assign cluster numbers
        for (int i = 0; i < newData.numInstances(); i++) {
            Instance inst = clusterData.instance(i);
            int cluster = kMeans.clusterInstance(inst);
            newData.instance(i).setValue(newData.numAttributes() - 1, cluster);
        }

        // Update the data reference
        data = newData;

        System.out.println("Clustering completed successfully");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes (including cluster): " + data.numAttributes());

        // Set class index to the attribute before cluster
        data.setClassIndex(data.numAttributes() - 2);
    }

    public void compareResults(Evaluation initialEval, Evaluation improvedEval) {
        System.out.println("\n4.2 Comparison of Results:");
        System.out.println("Metric\t\tInitial Model\tImproved Model\tDifference");
        System.out.println("------\t\t-------------\t--------------\t----------");

        // Compare accuracy
        double initAcc = initialEval.pctCorrect();
        double impAcc = improvedEval.pctCorrect();
        System.out.printf("Accuracy\t\t\t%.2f%%\t\t\t%.2f%%\t\t\t%+.2f%%\n", initAcc, impAcc, (impAcc - initAcc));

        // Manually calculate and compare weighted precision, recall, and F1
        double initPrecision = 0.0;
        double impPrecision = 0.0;
        double initRecall = 0.0;
        double impRecall = 0.0;
        double initF1 = 0.0;
        double impF1 = 0.0;

        // Get the number of classes from the dataset
        int numClasses = data.numClasses();
        double totalInstances = data.numInstances();

        for (int i = 0; i < numClasses; i++) {
            // Initial model metrics
            double initTP = initialEval.numTruePositives(i);
            double initFP = initialEval.numFalsePositives(i);
            double initFN = initialEval.numFalseNegatives(i);
            double initClassSize = initTP + initFN; // Number of actual instances of this class

            // Improved model metrics
            double impTP = improvedEval.numTruePositives(i);
            double impFP = improvedEval.numFalsePositives(i);
            double impFN = improvedEval.numFalseNegatives(i);
            double impClassSize = impTP + impFN;

            // Calculate class weight (proportion of instances in this class)
            double classWeight = initClassSize / totalInstances;

            // Calculate metrics for initial model
            double precisionInit = initTP + initFP > 0 ? initTP / (initTP + initFP) : 0;
            double recallInit = initTP + initFN > 0 ? initTP / (initTP + initFN) : 0;
            double f1Init = (precisionInit + recallInit > 0) ?
                    2 * (precisionInit * recallInit) / (precisionInit + recallInit) : 0;

            // Calculate metrics for improved model
            double precisionImp = impTP + impFP > 0 ? impTP / (impTP + impFP) : 0;
            double recallImp = impTP + impFN > 0 ? impTP / (impTP + impFN) : 0;
            double f1Imp = (precisionImp + recallImp > 0) ?
                    2 * (precisionImp * recallImp) / (precisionImp + recallImp) : 0;

            // Update weighted metrics
            initPrecision += precisionInit * classWeight;
            impPrecision += precisionImp * classWeight;
            initRecall += recallInit * classWeight;
            impRecall += recallImp * classWeight;
            initF1 += f1Init * classWeight;
            impF1 += f1Imp * classWeight;
        }

        // Print weighted metrics
        System.out.printf("Precision\t\t\t%.2f%%\t\t\t%.2f%%\t\t%+.2f%%\n",
                initPrecision * 100, impPrecision * 100, (impPrecision - initPrecision) * 100);
        System.out.printf("Recall\t\t\t\t%.2f%%\t\t\t%.2f%%\t\t%+.2f%%\n",
                initRecall * 100, impRecall * 100, (impRecall - initRecall) * 100);
        System.out.printf("F1-Score\t\t\t%.2f%%\t\t\t%.2f%%\t\t%+.2f%%\n",
                initF1 * 100, impF1 * 100, (impF1 - initF1) * 100);

        // Print additional statistics
        System.out.println("\nDetailed Statistics:");
        System.out.printf("Total Instances: %.0f\n", totalInstances);
        System.out.printf("Number of Classes: %d\n", numClasses);

        // Print class distribution
        System.out.println("\nClass Distribution:");
        for (int i = 0; i < numClasses; i++) {
            double classSize = initialEval.numTruePositives(i) + initialEval.numFalseNegatives(i);
            System.out.printf("Class %d: %.0f instances (%.1f%%)\n",
                    i, classSize, (classSize/totalInstances) * 100);
        }
    }

    public static void main(String[] args) {
        try {
            WekaClassification classifier = new WekaClassification();

            String csvFilePath = "cleaned_wind_dataset.csv";
            String arffFilePath = "cleaned_wind_dataset.arff";

            // Initial classification with 5-fold cross-validation
            System.out.println("Running initial classification...");
            classifier.convertCSVtoARFF(csvFilePath, arffFilePath);
            classifier.prepareData(classifier.data.numAttributes() - 1);
            Evaluation initialEval = classifier.trainAndEvaluate(false);
            classifier.printResults(initialEval, false);

            // Improve with clustering and 10-fold cross-validation
            System.out.println("\nImproving model with clustering...");
            classifier.improveWithClustering(3);
            Evaluation improvedEval = classifier.trainAndEvaluate(true);
            classifier.printResults(improvedEval, true);

            classifier.compareResults(initialEval, improvedEval);
        } catch (Exception e) {
            System.err.println("Error during classification:");
            System.err.println("Message: " + e.getMessage());
            e.printStackTrace();
        }
    }
}