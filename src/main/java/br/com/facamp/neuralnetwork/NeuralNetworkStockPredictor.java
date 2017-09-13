package br.com.facamp.neuralnetwork;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

public class NeuralNetworkStockPredictor {

    private int slidingWindowSize; //slinding window = # of entries used in prediction
    private double max = 0; //max set to a min
    private double min = Double.MAX_VALUE; //min set to a max
    private String rawDataFilePath; //path for raw data file
    private double learningRate;
    private double finalError;

    private String learningDataFilePath = "input/learningData.csv";
    private String neuralNetworkModelFilePath = "stockPredictor.nnet";

    private static XYSeries graphDataset;

    public NeuralNetworkStockPredictor(int slidingWindowSize, String rawDataFilePath, double learningRate) {
        this.rawDataFilePath = rawDataFilePath;
        this.slidingWindowSize = slidingWindowSize;
        this.graphDataset = new XYSeries("Error", false);
        this.learningRate = learningRate;
    }

    void prepareData() throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(rawDataFilePath));
        // find the min and max values for normalization
        try {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                double currentValue = Double.valueOf(tokens[1]);
                if (currentValue > max) {
                    max = currentValue;
                }
                if (currentValue < min) {
                    min = currentValue;
                }
            }
        } finally {
            reader.close();
        }

        reader = new BufferedReader(new FileReader(rawDataFilePath));
        BufferedWriter writer = new BufferedWriter(new FileWriter(learningDataFilePath));

        // keep a queue with slidingWindowSize + 1 values
        LinkedList<Double> valuesQueue = new LinkedList<Double>();
        try {
            String line;
            while ((line = reader.readLine()) != null) {
                double currentValue = Double.valueOf(line.split(",")[1]);
                // normalize value and add it to queue
                double normalizedValue = normalizeValue(currentValue);
                valuesQueue.add(normalizedValue);

                if (valuesQueue.size() == slidingWindowSize + 1) {
                    String valueLine = valuesQueue.toString().replaceAll("\\[|\\]", "");
                    writer.write(valueLine);
                    writer.newLine();
                    // remove the first element in queue to make place for a new one
                    valuesQueue.removeFirst();
                }
            }
        } finally {
            reader.close();
            writer.close();
        }
    }

    double normalizeValue(double input) {
        return (input - min) / (max - min) * 0.8 + 0.1;
    }

    double deNormalizeValue(double input) {
        return min + (input - 0.1) * (max - min) / 0.8;
    }

    void trainNetwork() throws IOException {
        NeuralNetwork<BackPropagation> neuralNetwork = new MultiLayerPerceptron(slidingWindowSize,
                2 * slidingWindowSize + 1, 1);

        int maxIterations = 1000;
        double maxError = 0.00001;
        SupervisedLearning learningRule = neuralNetwork.getLearningRule();
        learningRule.setMaxError(maxError);
        learningRule.setLearningRate(learningRate);
        learningRule.setMaxIterations(maxIterations);
        learningRule.addListener(new LearningEventListener() {
            public void handleLearningEvent(LearningEvent learningEvent) {
                SupervisedLearning rule = (SupervisedLearning) learningEvent.getSource();               
                System.out.println("Network error for iteration "
                        + rule.getCurrentIteration() + ": "
                        + rule.getTotalNetworkError());
            }
        });
        DataSet trainingSet = loadTrainingData(learningDataFilePath);
        neuralNetwork.learn(trainingSet);
        neuralNetwork.save(neuralNetworkModelFilePath);
    }

    DataSet loadTrainingData(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        DataSet trainingSet = new DataSet(slidingWindowSize, 1);

        try {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");

                double trainValues[] = new double[slidingWindowSize];
                for (int i = 0; i < slidingWindowSize; i++) {
                    trainValues[i] = Double.valueOf(tokens[i]);
                }
                double expectedValue[] = new double[]{
                    Double.valueOf(tokens[slidingWindowSize])};
                trainingSet.addRow(new DataSetRow(trainValues, expectedValue));
            }
        } finally {
            reader.close();
        }
        return trainingSet;
    }

    void testNetwork() throws IOException {
        
        BufferedReader reader = new BufferedReader(new FileReader("input/rawTestingData.csv"));    
        LinkedList<Double> valuesQueue = new LinkedList<>();
        int anchor = 63;
        try {
            String line;
            for (int i = 0; i < anchor-slidingWindowSize-1; i++) {
                line = reader.readLine();                
            }
            for (int i = 0; i <= slidingWindowSize; i++) {
                line = reader.readLine();    
                double currentValue = Double.valueOf(line.split(",")[1]);
                valuesQueue.add(currentValue);               
            }
        } finally {
            reader.close();
        }
        NeuralNetwork neuralNetwork = NeuralNetwork.createFromFile(
                neuralNetworkModelFilePath);
        double[] windowList = new double[slidingWindowSize];      

        int size = valuesQueue.size();
        for (int i = 0; i < size-1; i++)
            windowList[i] = normalizeValue(valuesQueue.pop());

        neuralNetwork.setInput(windowList);
        neuralNetwork.calculate();
        double[] networkOutput = neuralNetwork.getOutput();
        System.out.println("Expected value: "+valuesQueue.element());
        System.out.println("Predicted value: " + deNormalizeValue(networkOutput[0]));
        finalError = (valuesQueue.element() - deNormalizeValue(networkOutput[0]))
                *(valuesQueue.element() - deNormalizeValue(networkOutput[0]));

    }

    public static void main(String[] args) throws IOException {

        List<NeuralNetworkStockPredictor> networks = new ArrayList<>();
        for(int i = 1; i <= 10; i++) {
            NeuralNetworkStockPredictor predictor = new NeuralNetworkStockPredictor(i, 
                    "input/rawTrainingData.csv", 0.5);
            networks.add(predictor);
            predictor.prepareData();
            System.out.println("Training starting");
            predictor.trainNetwork();

            System.out.println("Testing network");
            predictor.testNetwork();
        }
        
        for (NeuralNetworkStockPredictor network : networks)
            graphDataset.add(network.slidingWindowSize, network.finalError);
        
        displaySlidingWindowGraph();
        
        networks.clear();
        
        for(int i = 1; i <= 10; i++) {
            NeuralNetworkStockPredictor predictor = new NeuralNetworkStockPredictor(5,
                    "input/rawTrainingData.csv", i/10.0);
            networks.add(predictor);
            predictor.prepareData();
            System.out.println("Training starting");
            predictor.trainNetwork();

            System.out.println("Testing network");
            predictor.testNetwork();
        }

        for (NeuralNetworkStockPredictor network : networks)
            graphDataset.add(network.learningRate, network.finalError);
        
        displayLRateGraph();
    }

    public static void displayLRateGraph() {
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(graphDataset);
        LineChart chart = new LineChart(
                "Learning Rate Variation",
                "Learning Rate Variation", dataset);

        chart.pack();
        RefineryUtilities.centerFrameOnScreen(chart);
        chart.setVisible(true);
    }
    
    public static void displaySlidingWindowGraph() {
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(graphDataset);
        LineChart chart = new LineChart(
                "Sliding Window Size Variation",
                "Sliding Window Size Variation", dataset);

        chart.pack();
        RefineryUtilities.centerFrameOnScreen(chart);
        chart.setVisible(true);
    }
}
