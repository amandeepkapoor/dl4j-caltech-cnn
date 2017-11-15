# dl4j-caltech-cnn
Image Classifier for Caltech 101 Dataset

Convolution Neural Network on CalTech-101 Dataset

Brief Overview:
There are 2 programs created to classify the 101 Image Categories from the CalTech-101 Dataset. It is hosted at http://www.vision.caltech.edu/Image_Datasets/Caltech101/ .
These are: 
1.	CaltechImageClassifier.java – Created using the Hybrid neural network. The structure of the network is 
[Conv -> Normalization -> Relu]*2 -> Pool -> Conv -> Normalization -> Relu -> Pool -> drop out (the features which are not significant) -> Fully connected layer (it will flatten the data and will combine with all the neurons) -> Softmax (Classifier on 101 classes - Kind of Logistic Regression at the end)

			a.	In order to increase the input data size, I have also applied the Image Transformation functions.

			b.	This model achieves the overall accuracy and precision between the range of 25 to 30 percent.  Depending upon the various parameter setting done by me.

2.	ZooModalTransferLearning – This model is created using the already learned parameter from VGG16. This is called transfer learning and in DL4J – Zoo Modelling.

			a.	The version which is used by me in this program is “Modifying only the last layer keeping other frozen”.

			b.	This model very quickly converges with minimal training. After few (30) iterations  of individual datasets. The accuracy was increased from .008 to .7 which is 70 percent. 

			c.	This learning is very efficient. But need to take 1 caution before implementing this model that configuring the Heap Size, as the number of parameter is very high in this model. Will explain below.

How to Install:

1.	The complete Maven project is hosted at GitHub - https://github.com/amandeepkapoor/dl4j-caltech-cnn. 

2.	For running this in your system, please follow the below steps :-

			a.	Install Maven - https://maven.apache.org/download.cgi . (Do check the Maven Version – “mvn –v” in commandline.

			b.	Download Git in to your system.

			c.	Once both the things are completed. Type below commands:-

					i.	git clone https://github.com/amandeepkapoor/dl4j-caltech-cnn

					ii.	A new directory would be created. “cd” in to the directory and execute the command – “mvn install” (Please don’t execute “mvn –clean install”, it is currently giving an error. Working on that as of now)

d.	Once the above commands are executed, a Jar file would be created in the target folder. Either execute that file on command line or to have better view, please install “Intelli J – Community Edition”. It is free .

					i.	 The major advantage is you can easily increase the heap size specifically for the second program. 

					ii.	Also for creating the maven project is very effective.

Heap Size Allocation:

1.	If you have Intelli J installed then it is fairly quick. Go to the Run Configuration -> Set the VM Option for the class your executing. Commands are –Xms<size> and –Xmx<size>

Additional Useful Link:

1.	 Numpy memory allocation - http://nd4j.org/userguide#inmemory 

2.	Zoo Model Setting - https://deeplearning4j.org/build_vgg_webapp 

3.	Convolutional Neural Nets Learning – 

		a.	https://deeplearning4j.org/convolutionalnets 

		b.	https://www.mathworks.com/help/nnet/ug/layers-of-a-convolutional-neural-network.html 





