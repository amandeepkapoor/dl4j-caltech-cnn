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
import org.deeplearning4j.util.ModelSerializer;
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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;

/*
This program is an Image Classifier, capable of classifying 101 categories of the images. This is implemented using the
concept of "Transfer Learning - Zoo Model in Deep Learning 4 J. I have shared the links which you would find useful in github
 (Refer below). The Dataset is hosted at http://www.vision.caltech.edu/Image_Datasets/Caltech101/.
Please refer the read me text on the link https://github.com/amandeepkapoor/dl4j-caltech-cnn to know more about the
implementation and the useful link. Also I have added few useful comments, would be helpful in understanding the flow of the code.
*/


public class ZooModelTransferLearning {

    //Defining the Maven Directory path as the sub path so that it can be accessible everywhere. It contains the downloaded data
    public static final String File_Sub_Path = "src\\main\\resources\\101_ObjectCategories";


    private static Logger log = LoggerFactory.getLogger(ZooModelTransferLearning.class);

    /* Even though the input Height*Width of our data is different but for this model to work we need to modify the
    the images in 224*224 dimension as the pretrained model VGG16 assumes this input size
     */
    public static void main(String[] args) throws Exception {
        int height = 224;
        int width = 224;
        int channels = 3;//As the image is in RGB so Depth defines the intensity across them
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

        //After reducing the size of the image, know we are doing the mean subtraction.
        DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
        //DataNormalization scalar = new VGG16ImagePreProcessor();
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        dataIter.setPreProcessor(preProcessor);

        /*
         Initiating below the ZooModel which is will be utilizing the already learned model, in this case VGG16.
         That's what called as Tranfer Learning. Already trained parameters can be utilized and these are trained
         on large data sets.
         The method that we are using here is of "Modifying the Last Layer to fit our Data Classification Requirement".
         Rest all would be frozen.
         */

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
        ImageRecordReader recordReader1 = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader1.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader1,batchSize,1,outputNum);
        testIter.setPreProcessor(preProcessor);

        log.info("Before----------------");
        Evaluation eval = new Evaluation(outputNum);
        eval = vgg16Transfer.evaluate(testIter);
        log.info(eval.stats() + "\n");

        // Evaluate the network. This model is very very fast in convergence.
        int iter = 0;
        while(dataIter.hasNext()){
            log.info("iteration = "+iter);
            vgg16Transfer.fit(dataIter.next());

            if (iter % 2 == 0){
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;

        }
        File locationToSave = new File(System.getProperty("user.dir"),"zoomodel.zip");
        ModelSerializer.writeModel(vgg16Transfer,locationToSave,false);


    }
}