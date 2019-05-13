package neural.network.myNeuralNetwork;

//Vincent Tomie

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
* Hello world!
*
*/
public class App 
{
	private static final String TRAIN_IMAGES_FILE_DIGIT = "/Users/vincenttomie/git/440-project-1/data/digitdata/trainingimages.txt";
  private static final String TRAIN_LABELS_FILE_DIGIT = "/Users/vincenttomie/git/440-project-1/data/digitdata/traininglabels.txt";
  private static final String TEST_IMAGES_FILE_DIGIT = "/Users/vincenttomie/git/440-project-1/data/digitdata/testimages.txt";
  private static final String TEST_LABELS_FILE_DIGIT = "/Users/vincenttomie/git/440-project-1/data/digitdata/testlabels.txt";

  private static final String TRAIN_IMAGES_FILE_FACE = "/Users/vincenttomie/git/440-project-1/data/facedata/facedatatrain.txt";
  private static final String TRAIN_LABELS_FILE_FACE = "/Users/vincenttomie/git/440-project-1/data/facedata/facedatatrainlabels.txt";
  private static final String TEST_IMAGES_FILE_FACE = "/Users/vincenttomie/git/440-project-1/data/facedata/facedatatest.txt";
  private static final String TEST_LABELS_FILE_FACE = "/Users/vincenttomie/git/440-project-1/data/facedata/facedatatestlabels.txt";
  
	//private static final Logger LOGGER = LoggerFactory.getLogger(App.class);
  //apparently this logger throws an error, operating system based?
  public static void main( String[] args ) throws IOException
  {
  
     // BalancedPathFilter pathFilter = new BalanedPathFilter(rng, labelMaker, 10, 10, maxPathsPerLabel);
      int numExamples = 5000;
      //number of examples in digits
      int numLabels = 50;
      //random number I made up that doesn't do anything
  			
  	int numChannels = 1;
  	int numOutput = 10;
  	int batchSize = 50;
  	int numofEpochs = 1;
  	//number of forward and backward iterations so one forward propogation and one backward propogation
  	int numofIterations = 1;
  	//this is strictly based on the batch size apparently
  	int seed = 123;
  	//just some random number generator I probably won't use
  	Random randNumGen = new Random(seed);
  	
  	ArrayList<Double[][]> digitData = loadDataIntegers(TRAIN_IMAGES_FILE_DIGIT, 5000, 28, 28);
  	//System.out.println(digitData.size());
  	
  	ArrayList<Integer> digitLabels = loadLabelsFile(TRAIN_LABELS_FILE_DIGIT, 5000);
  	//System.out.println(digitLabels.size());
  	
  	ArrayList<Double[][]> digitTestData = loadDataIntegers(TEST_IMAGES_FILE_DIGIT, 1000, 28, 28);
  	//System.out.println(digitTestData.size());
  	
  	ArrayList<Integer> digitTestLabels = loadLabelsFile(TEST_LABELS_FILE_DIGIT, 1000);
  	//System.out.println(digitTestLabels.size());
  	
  	ArrayList<Double[][]> faceTrainData = loadDataIntegers(TRAIN_IMAGES_FILE_FACE, 451, 60, 70);
  	//System.out.println(faceTrainData.size());
  	
  	ArrayList<Integer> faceTrainLabels = loadLabelsFile(TRAIN_LABELS_FILE_FACE, 451);
  	//System.out.println(faceTrainLabels.size());
  	
  	ArrayList<Double[][]> faceTestData = loadDataIntegers(TEST_IMAGES_FILE_FACE, 150, 60, 70);
  	//System.out.println(faceTestData.size());
  	
  	ArrayList<Integer> faceTestLabels = loadLabelsFile(TEST_LABELS_FILE_FACE, 150);
  	//System.out.println(faceTestLabels.size());
  	
  	Integer[] digitLabelsTrain = new Integer[numExamples];
  	Integer[] digitLabelsTest = new Integer[numExamples];
  	
  	for (int i = 0; i < numExamples; i++)
  	{
  		digitLabelsTrain[i] = digitLabels.get(i);
  	}
  	
  	
  	DataSet d = new DataSet();
  	DataSet dt = new DataSet();
  	
  	DataSet f = new DataSet();
  	DataSet ft = new DataSet();
      //d.setFeatures(data);
      //d.setLabels(labels);
  	NDArray data = new NDArray();
  	//INDArray arr = Nd4j.createFromArray(vectorifiedData);
  	
  	double[][] vectorFaceTrainData = new double[451][60 * 70];
  	double[][] vectorFaceTestData = new double[150][60 * 70];
  	double[][] vectorTestData = new double[1000][28 * 28];
  	double[][] vectorifiedData = new double[numExamples][28 * 28];
  	
  	for (int i = 0; i < 451; i++)
  	{
  		int z = 0;
  		for (int j = 0; j < 70; j++)
  		{
  			for (int k = 0; k < 60; k++)
  			{
  				if (i < 150)
  					vectorFaceTestData[i][z] = faceTestData.get(i)[j][k];
  				
  				vectorFaceTrainData[i][z] = faceTrainData.get(i)[j][k];
  				z++;
  			}
  		}
  	}
  	
  	for (int i = 0; i < 5000; i++)
  	{
  		int m = 0;
  		for (int j = 0; j < 28; j++)
  		{
  			for (int k = 0; k < 28; k++)
  			{    				
  				if (i < 1000)
  					vectorTestData[i][m] = digitTestData.get(i)[j][k];
  				
  				vectorifiedData[i][m] = digitData.get(i)[j][k];
  				if (i < 1)
  					System.out.print(vectorifiedData[i][m]);
  				m++;
  			}
  			if (i < 1)
  				System.out.println();
  		}
  		
  		//INDArray datum = Nd4j.create(vectorData, new int[]{28, 28}, 'c');
  		
  	}
  	
  	
  	double[][] faceLabelTrain = new double[451][2];
  	double[][] faceLabelTest = new double[150][2];
  	double[][] labelTest = new double[1000][10];
  	double[][] labelTraining = new double[5000][10];
  	
  	for (int i = 0; i < 451; i++)
  	{
  		int x = faceTrainLabels.get(i);
  		for (int j = 0; j < 2; j++)
  		{
  			if (x == j)
  				faceLabelTrain[i][j] = 1.0;
  			else
  				faceLabelTrain[i][j] = 0.0;
  		}
  	}
  	
  	for (int i = 0; i < 150; i++)
  	{
  		int x = faceTestLabels.get(i);
  		for (int j = 0; j < 2; j++)
  		{
  			if (x == j)
  				faceLabelTest[i][j] = 1.0;
  			else
  				faceLabelTest[i][j] = 0.0;
  		}
  	}
  	
  	
  	for (int i = 0; i < 1000; i++)
  	{
  		double[] tempTrain = new double[10];
  		int x = digitTestLabels.get(i);
  		for (int j = 0; j < 10; j++)
  		{
  			if (x == j)
  				labelTest[i][j] = 1.0;
  			else
  				labelTest[i][j] = 0.0;
  			//System.out.print(labelTest[i][j]);
  		}
  		//System.out.println();

  	}

  	for (int i = 0; i < 5000; i++)
  	{
  		double[] tempTrain = new double[10];
  		int x = digitLabels.get(i);
  		for (int j = 0; j < 10; j++)
  		{
  			if (x == j)
  				labelTraining[i][j] = 1.0;
  			else
  				labelTraining[i][j] = 0.0;
  			//System.out.print(labelTraining[i][j]);
  		}
  		//System.out.println();
  		
  	}
  	//System.out.println(labelTraining.toString());
  	INDArray datum = Nd4j.createFromArray(vectorifiedData);
  	INDArray labels = Nd4j.createFromArray(labelTraining);
  	INDArray datumTest = Nd4j.createFromArray(vectorTestData);
  	INDArray labelsTest = Nd4j.createFromArray(labelTest);
  	
  	INDArray facedatum = Nd4j.createFromArray(vectorFaceTrainData);
  	INDArray facelabels = Nd4j.createFromArray(faceLabelTrain);
  	INDArray facedatumTest = Nd4j.createFromArray(vectorFaceTestData);
  	INDArray facelabelsTest = Nd4j.createFromArray(faceLabelTest);
  	
  	
  	//Integer[][] labelArray = new Integer[numExamples][numOutput];
  	//INDArray labels = Nd4j.createFromArray(labelArray); 
  	
  	
  	//System.out.println(datum.toString());
  	//d.setFeatures(datum);
  	//System.out.println(d.getFeatures().toString());
  	d.setFeatures(datum);
  	System.out.println(d.getFeatures().toString());
  	d.setLabels(labels);
  	System.out.println(d.getLabels().toString());
  	
  	f.setFeatures(facedatum);
  	f.setLabels(facelabels);
  	ft.setFeatures(facedatumTest);
  	ft.setLabels(facelabelsTest);
  	
  	
  	dt.setFeatures(datumTest);
  	dt.setLabels(labelsTest);
  	
  	System.out.println(d.getFeatures().length());
  	System.out.println(d.getLabels().length());
  	
  	System.out.println(dt.getFeatures().length());
  	System.out.println(dt.getLabels().length());
  	
  	//d.batchBy(64);
  	
  	
  	//these are our iterators for running the neural network on
  	DataSetIterator digitTrainIterator = new ExistingDataSetIterator(d);
  	DataSetIterator digitTestIterator = new ExistingDataSetIterator(dt);
  	
  	DataSetIterator faceTrainIterator = new ExistingDataSetIterator(f);
  	DataSetIterator faceTestIterator = new ExistingDataSetIterator(ft);
  	
  	//System.out.println(digitTestIterator.next().getFeatures().toString());
  	//System.out.println(digitTestIterator.next().getLabels().toString());
  	//System.out.println(digitTrainIterator.batch());
  	

  	
  	
      //System.out.println( "Hello World!" );
      
      
      //this is the MNIST dataset, not ours
      DataSetIterator trainingSet = new MnistDataSetIterator(batchSize, true, seed);
      
      DataSet dataMNIST = trainingSet.next();
      System.out.println(dataMNIST.getFeatures().toString());
      System.out.println(dataMNIST.getLabels().toString());
      
      DataSetIterator testDataSet = new MnistDataSetIterator(batchSize, false, seed);
      


      
      Map<Integer, Double> learningRateSchedule = new HashMap<Integer, Double>();
      learningRateSchedule.put(0, 0.05);
      //solid learning rate instead of dynamic
      
      
      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
              .seed(seed)
              .l2(0.0005) 
              .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
              .weightInit(WeightInit.XAVIER)
              .list()
              .layer(new ConvolutionLayer.Builder(3, 3)
                  .nIn(numChannels)
                  .stride(2, 2)
                  .nOut(20)
                  .activation(Activation.IDENTITY)
                  .build())
              .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                  .kernelSize(2, 2)
                  .stride(2, 2)
                  .build())
              .layer(new ConvolutionLayer.Builder(3, 3)
                  .stride(2, 2) // nIn need not specified in later layers
                  .nOut(5)
                  .activation(Activation.IDENTITY)
                  .build())
              .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                  .kernelSize(2, 2)
                  .stride(2, 2)
                  .build())
              .layer(new DenseLayer.Builder().activation(Activation.RELU)
                  .nOut(20)
                  .build())
              .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                  .nOut(numOutput)
                  .activation(Activation.SOFTMAX)
                  .build())
              .setInputType(InputType.convolutionalFlat(28, 28, 1))
              .build();
      
      MultiLayerNetwork neuralNet = new MultiLayerNetwork(conf);
      neuralNet.init();
      
      neuralNet.setListeners(new ScoreIterationListener(1));
      
      MultiLayerConfiguration faceconf = new NeuralNetConfiguration.Builder()
              .seed(seed)
              .l2(0.0005) // ridge regression value
              .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
              .weightInit(WeightInit.XAVIER)
              .list()
              .layer(new ConvolutionLayer.Builder(3, 3)
                  .nIn(numChannels)
                  .stride(2, 2)
                  .nOut(20)
                  .activation(Activation.IDENTITY)
                  .build())
              .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                  .kernelSize(2, 2)
                  .stride(2, 2)
                  .build())
              .layer(new ConvolutionLayer.Builder(3, 3)
                  .stride(2, 2) // nIn need not specified in later layers
                  .nOut(5)
                  .activation(Activation.IDENTITY)
                  .build())
              .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                  .kernelSize(2, 2)
                  .stride(2, 2)
                  .build())
              .layer(new DenseLayer.Builder().activation(Activation.RELU)
                  .nOut(20)
                  .build())
              .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                  .nOut(2)
                  .activation(Activation.SOFTMAX)
                  .build())
              .setInputType(InputType.convolutionalFlat(70, 60, 1))
              .build();
      
      MultiLayerNetwork neuralNet2 = new MultiLayerNetwork(faceconf);
      neuralNet2.init();
      
      neuralNet2.setListeners(new ScoreIterationListener(1));
      
      
      
      
      
      double standardDeviation = 0.0;
      double[] sd = new double[10];
      double accuracy = 0.0;
      long time = 0;
      
      List<DataSet> sets = d.batchBy(64);
      for (int i = 0; i < 5; i++)
      {
      	for (int j = 0; j < sets.size(); j++)
      	{
      		neuralNet.fit(sets.get(j));    
      		//neuralNet.evaluate(testDataSet);
      		//testDataSet.reset();
      	}
      	long beginTime = System.nanoTime();
      	//neuralNet.fit(digitTrainIterator);
      	Evaluation evaluation = neuralNet.evaluate(digitTestIterator);
      	long endTime = System.nanoTime();
      	System.out.println(evaluation.stats());
      	System.out.println(evaluation.accuracy());
      	accuracy += evaluation.accuracy();
      	time += endTime - beginTime;
      	sd[i] = evaluation.accuracy();
      	testDataSet.reset();
      }

      
      /*System.out.println("Average Time is " + time / (sets.size() * 5);
      System.out.println("Average Accuracy is " + accuracy / (sets.size() * 5);
      
      for (int i = 0; i < sd.length; i++)
      {
      	standardDeviation += Math.pow(sd[i] - accuracy, 2);
      }
      System.out.println("Standard Deviation is " + Math.sqrt(standardDeviation / 5));*/
     
      List<DataSet> facesets = f.batchBy(64);
      for (int i = 0; i < 5; i++)
      {
      	for (int j = 0; j < facesets.size(); j++)
      	{
      		neuralNet2.fit(facesets.get(j));    
      		//neuralNet.evaluate(testDataSet);
      		//testDataSet.reset();
      	}
      	long beginTime = System.nanoTime();
      	//neuralNet.fit(digitTrainIterator);
      	Evaluation evaluation = neuralNet2.evaluate(faceTestIterator);
      	long endTime = System.nanoTime();
      	System.out.println(evaluation.stats());
      	System.out.println(evaluation.accuracy());
      	accuracy += evaluation.accuracy();
      	time += endTime - beginTime;
      	sd[i] = evaluation.accuracy();
      	testDataSet.reset();
      }
  }
  
	public static ArrayList<Integer> loadLabelsFile(String filename, int numImages) throws IOException
	{
		//Read numImages labels from file
		File b = new File(filename);
		ArrayList<Integer> labels = new ArrayList<Integer>();
		BufferedReader br = new BufferedReader(new FileReader(b));
		String currentLine;
		
		for (int i = 0; i < numImages; i++)
		{
			if (!br.ready())
			{
				System.out.println("Not ready at " + i);
				break;
			}
			currentLine = br.readLine();
			if (currentLine.equals(" "))
				break;
			int currentInt = Integer.parseInt(currentLine);
			labels.add(currentInt);
		}
		br.close();
		return labels;
	}
  
  public static ArrayList<Double[][]> loadDataIntegers(String filename, int numImages, int width, int height) throws IOException 
	{
		//if we need to both at same time we can just combine functions
		//we probably need to combine them tbh
		File a = new File(filename);
		
		ArrayList<Double[][]> items = new ArrayList<Double[][]>();
		//ArrayList<String> data = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(a));
		String currentLine;
		char c;
		
		for (int i = 0; i < numImages; i++)
		{
			Double[][] dataConversion = new Double[height][width];
			//ArrayList<String> data = new ArrayList<String>(height);
			if (!br.ready())
			{
				System.out.println("Not ready at " + i);
				break;
			}
			for (int j = 0; j < height; j++)
			{
				currentLine = br.readLine();
				for (int k = 0; k < currentLine.length(); k++)
				{
					c = currentLine.charAt(k);
					if (c == ' ')
						dataConversion[j][k] = 0.0;
					else if (c == '#' || c == '+')
						dataConversion[j][k] = 1.0;				
					
				}
			}
			if (dataConversion[0].length < (width-1))
			{
				System.out.println("We got to end of file at image " + (i+1));
				break;
			}
			items.add(dataConversion);
		}
		br.close();
		return items;
	}
}

