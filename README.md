# dl4j-caltech-cnn
Image Classifier for Caltech 101 Dataset

Convolution Neural Network on CalTech-101 Dataset

**Brief Overview:**
There are 2 programs created to classify the 101 Image Categories from the CalTech-101 Dataset. It is hosted at http://www.vision.caltech.edu/Image_Datasets/Caltech101/ .
These are: 
1. CaltechImageClassifier.java – Created using the Hybrid neural network. The structure of the network is 
[Conv -> Normalization -> Relu]*2 -> Pool -> Conv -> Normalization -> Relu -> Pool -> drop out (the features which are not significant) -> Fully connected layer (it will flatten the data and will combine with all the neurons) -> Softmax (Classifier on 101 classes - Kind of Logistic Regression at the end)

   * In order to increase the input data size, I have also applied the Image Transformation functions.
   * This model achieves the overall accuracy and precision between the range of 25 to 30 percent.  Depending upon the various parameter setting done by me.

2. ZooModalTransferLearning – This model is created using the already learned parameter from VGG16. This is called transfer learning and in DL4J – Zoo Modelling.

   * The version which is used by me in this program is “Modifying only the last layer keeping other frozen”.
   * This model very quickly converges with minimal training. After few iterations  of individual datasets. The accuracy was 		increased from .008 towards .7.
   * The final outcome was (no. of epoch was 1) :-	
   ```
   	==========================Scores========================================
 		# of classes:    101
 		Accuracy:        0.6869
 		Precision:       0.8305	(9 classes excluded from average)
 		Recall:          0.6912
 		F1 Score:        0.7344	(9 classes excluded from average)
		Precision, recall & F1: macro-averaged (equally weighted avg. of 101 classes)
		========================================================================
   ```
   * This learning is very efficient. But need to take 1 caution i.e. before implementing this model that configuring the 			Heap Size, as the number of parameter is very high in this model. Will explain below.

**How to Install:**

1. The complete Maven project is hosted at GitHub - https://github.com/amandeepkapoor/dl4j-caltech-cnn. 

2. For running this in your system, please follow the below steps :-

   * Install Maven - https://maven.apache.org/download.cgi . (Do check the Maven Version – “mvn –v” in commandline.
   * Download Git in to your system.
3. Once both the things are completed. Execute below commands:-
   * git clone https://github.com/amandeepkapoor/dl4j-caltech-cnn
   * A new directory would be created. “cd” in to the directory and execute the command – “mvn install” (Please don’t execute “mvn –clean install”, it is currently giving an error. Working on that as of now)
   * Once the above commands are executed, a Jar file would be created in the target folder. Either execute that file on command line or to have better view, please install “Intelli J – Community Edition”. It is free .
     * The major advantage is you can easily increase the heap size, specifically for the second program. 
     * Also, for creating the maven project it is very effective.

**Heap Size Allocation:**

1. If you have Intelli J installed then it is fairly quick. Go to the Run Configuration -> Set the VM Option for the class you are executing. Commands are –Xms<size> and –Xmx<size>

**Additional Useful Link:**

1. Numpy memory allocation - http://nd4j.org/userguide#inmemory 

2. Zoo Model Setting - https://deeplearning4j.org/build_vgg_webapp 

3. Convolutional Neural Nets Learning – 
   - https://deeplearning4j.org/convolutionalnets 
   - https://www.mathworks.com/help/nnet/ug/layers-of-a-convolutional-neural-network.html 
		
4. DL4J Examples - https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/transferlearning/vgg16






