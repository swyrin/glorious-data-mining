import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Utility {
    public static Instances readData() throws Exception {
        DataSource dataSource = new DataSource("cleaned_wind_data.arff");
        return dataSource.getDataSet();
    }

    public static void printAttributes(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println("Attribute " + i + ": " + data.attribute(i));
        }
        System.out.println();
    }
}
