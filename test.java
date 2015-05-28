import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
 
public class test {
 
    public static void main(String[] args) throws Exception{
    	
    	 DataSource source = new DataSource("teste.arff");
    	 Instances data = source.getDataSet();
    	 if (data.classIndex() == -1)
    		   data.setClassIndex(data.numAttributes() - 1);
    	 
         MultilayerPerceptron cModel = new MultilayerPerceptron(); 
         cModel.buildClassifier(data);
         
         
         // Test the model
         Evaluation eTest = new Evaluation(data);
         eTest.evaluateModel(cModel,data);
          
         // Print the result à la Weka explorer:
         String strSummary = eTest.toSummaryString();
         System.out.println(strSummary);
          
         // Get the confusion matrix
         double[][] cmMatrix = eTest.confusionMatrix();
         for(int row_i=0; row_i<cmMatrix.length; row_i++){
             for(int col_i=0; col_i<cmMatrix.length; col_i++){
                 System.out.print(cmMatrix[row_i][col_i]);
                 System.out.print("|");
             }
             System.out.println();
         }
    }
}