import java.io.PrintWriter;
import java.io.Writer;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
 
public class test {
 
    public static void main(String[] args) throws Exception{
        PrintWriter writer = new PrintWriter("log.txt", "UTF-8");

    	 //train
    	 DataSource source1 = new DataSource("train.arff");
    	 Instances train = source1.getDataSet();
    	 if (train.classIndex() == -1)
    		   train.setClassIndex(train.numAttributes() - 1);
    	 //test
    	 DataSource source2 = new DataSource("teste.arff");
    	 Instances test = source2.getDataSet();
    	 if (test.classIndex() == -1)
    		   test.setClassIndex(test.numAttributes() - 1);
    	 
    	 //modelo que executa o backpropagation
         MultilayerPerceptron cModel = new MultilayerPerceptron(); 
         
         // atributos + classes * 2/3
         cModel.setHiddenLayers("67");
         
         cModel.setGUI(true);
         cModel.buildClassifier(train);
         
         // testar o modelo
         Evaluation eTest = new Evaluation(test);
         eTest.evaluateModel(cModel,test);
         
         // print dos resultados
         String strSummary = eTest.toSummaryString();
         writer.print("" + strSummary.substring(0, 135) + "\n");

         
         
         // matriz confusão
         double[][] cmMatrix = eTest.confusionMatrix();
         for(int row_i=0; row_i<cmMatrix.length; row_i++){
             for(int col_i=0; col_i<cmMatrix.length; col_i++){
            	 writer.print(cmMatrix[row_i][col_i]);
            	 writer.print("|");
             }
             writer.println();
         }
         writer.close();
    }
}