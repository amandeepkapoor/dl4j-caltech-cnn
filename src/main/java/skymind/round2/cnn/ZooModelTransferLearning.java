package skymind.round2.cnn;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
//import org.deeplearning4j.examples.utilities.DataUtilities;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.model.VGG16;



import javax.imageio.ImageTranscoder;
//import javax.xml.crypto.Data;
import java.io.*;
import java.util.Random;


public class ZooModelTransferLearning {


    public static final String File_Sub_Path = "src\\main\\resources\\101_ObjectCategories";


    private static Logger log = LoggerFactory.getLogger(ZooModelTransferLearning.class);

    public static void main(String[] args) throws Exception {
        int height = 224;
        int width = 224;
        int channels = 3;
        int seed = 123;
        Random randNumGen = new Random(seed);
        int batchSize = 50;
        int outputNum = 101;
        //int epoch = 15;
        File trainData = new File(System.getProperty("user.dir"),File_Sub_Path);



        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, labelMaker, 0, outputNum, batchSize);
        InputSplit[] filesInDirSplit = train.sample(pathFilter, 80, 20);
        InputSplit trainData1 = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);


        recordReader.initialize(trainData1);

        DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
        //DataNormalization scalar = new VGG16ImagePreProcessor();
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        dataIter.setPreProcessor(preProcessor);
        //scalar.fit(dataIter);
        //dataIter.setPreProcessor(scalar);

        ZooModel zooModel = new VGG16();

        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrainedNet.summary());
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(5e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(outputNum)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0,0.2*(2.0/(4096+outputNum))))
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();


        log.info("*****Evaluate Test MODEL after some Iteration on training data********");
        recordReader.reset();
        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        testIter.setPreProcessor(preProcessor);

        log.info("Before----------------");
        Evaluation eval = new Evaluation(outputNum);
        eval = vgg16Transfer.evaluate(testIter);
        log.info(eval.stats() + "\n");

        // Evaluate the network
        int iter = 0;
        while(dataIter.hasNext()){
            log.info("iteration = "+iter);
            vgg16Transfer.fit(dataIter.next());

            if (iter % 10 == 0){
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;

        }



    }
}