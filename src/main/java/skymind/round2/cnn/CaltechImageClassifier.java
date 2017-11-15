package skymind.round2.cnn;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
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


public class CaltechImageClassifier {


    public static final String File_Sub_Path = "src\\main\\resources\\101_ObjectCategories";


    private static Logger log = LoggerFactory.getLogger(CaltechImageClassifier.class);

    public static void main(String[] args) throws Exception {
        int height = 200;
        int width = 300;
        int channels = 3;
        int seed = 123;
        Random randNumGen = new Random(seed);
        int batchSize = 50;
        int outputNum = 101;
        int epoch = 10;
        File trainData = new File(System.getProperty("user.dir"),File_Sub_Path);
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, labelMaker,0, outputNum, batchSize);
        InputSplit[] filesInDirSplit = train.sample(pathFilter, 80, 20);
        InputSplit trainData1 = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        ImageTransform transform1 = new FlipImageTransform(randNumGen);
        ImageTransform transform2 = new FlipImageTransform(new Random(seed));
        ImageTransform warptranform = new WarpImageTransform(randNumGen,42);
        List<ImageTransform> tranforms = Arrays.asList(new ImageTransform[] {transform1, warptranform, transform2});


        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(trainData1);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        DataSetIterator dataIter;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .seed(seed)
                .iterations(1)
                .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(.006)
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(.0001)
                .convolutionMode(ConvolutionMode.Same).list()
                // block 1
                .layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}).name("image_array").stride(new int[]{1, 1})
                        .nIn(3)
                        .nOut(16).build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(3, new ConvolutionLayer.Builder(new int[] {5, 5}).stride(new int[]{1, 1}).nIn(16).nOut(16)
                        .build())
                .layer(4, new BatchNormalization.Builder().build())
                .layer(5, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(6, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                        new int[] {2, 2}).build())
                .layer(7, new ConvolutionLayer.Builder(new int[] {5, 5}).stride(new int[]{2, 2}).nIn(16).nOut(16)
                        .build())
                .layer(8, new BatchNormalization.Builder().build())
                .layer(9, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(10, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG,
                        new int[] {2, 2}).build())
                .layer(11, new DropoutLayer.Builder(0.5).build())
                .layer(12, new DenseLayer.Builder().name("ffn2").nOut(256).build())
                .layer(13, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output").nOut(outputNum).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(height, width, channels))
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");

        for (ImageTransform transform: tranforms) {

            System.out.println("Training on"+transform.getClass().toString());
            recordReader.initialize(trainData1,transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            for (int j = 0; j < epoch; j++) {
                model.fit(dataIter);
            }
        }
        recordReader.reset();
        recordReader.initialize(trainData1);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        for (int j = 0; j < epoch; j++) {

            model.fit(dataIter);
        }
        log.info("*****Test MODEL********");

        recordReader.reset();

        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(outputNum);

        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);

        }

        log.info(eval.stats());

    }

}