**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bargraphbefore.png "Original data"
[image2]: ./examples/bargraphafter.png "After expansion"
[image3]: ./examples/rotation.png "rotation"
[image4]: ./examples/scaling.png "scaling"
[image5]: ./examples/affine.png "affine"
[image6]: ./examples/preprocess.png "preprocess"
[image7]: ./internet_test/1.jpg "Traffic Sign 1"
[image8]: ./internet_test/2.jpg "Traffic Sign 2"
[image9]: ./internet_test/3.jpg "Traffic Sign 3"
[image10]: ./internet_test/4.jpg "Traffic Sign 4"
[image11]: ./internet_test/5.jpg "Traffic Sign 5"
[image12]: ./internet_test/stop.jpg "Traffic Sign 6"
[image13]: ./examples/internet_confidence.png "confidence"
[image14]: ./examples/softmax.png "softmax"

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* number of training examples = 34,799
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43
* Average examples per class = 809

After expanding the database:

* Number of training examples = 252,888
* Number of validation examples = 4410
* Image data shape = (32, 32, 1)
* Number of classes = 43
* Average examples per class = 5881



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.I have included a few set images and a graph of the distibution of examples across the classes. We can see that the data is terribly uneven, which will lead to low quality training later on, so I've spent plenty of effort expanding this data and evening it out.

![Before][image1]
![After][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

My first step was to expand the dataset, I decided to do this becuase I had heard that is was a very importand step in making goo predictions, so, starting with the highest quality additonal data and going down to the lowest, I doubled or tripled the dataset many times.

First step was flipping images that can be flipped in the directions that preserve their sign names. This includes images that can be flipped to become a different sign, such as turn left becomeing tern right.

Next, I begin expanding data in the classes that need to "catch up" with the higher classes.
I used rotation from from -10 to -2 and 2 to 10 degrees:
![rotate][image3]

After that, I used scaling up and down, with seperate and random x and y to additonally cause some warping, up to ,

![scale][image4]

Then I intruduced both translating the images to different areas as well as shifts in global brightness, randomly up and down.

![move][image5]

Finally, I applied grayscaling and then mean zero normalization, I wanted to use histogram equalization, but it was causeing worser results and bugs, so it wasn't included in the final product.

![prep][image6]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, it is an unchanged lenet:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 B&W image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| outputs 1x400   								|
| Fully connected		| 400->120        								|
| Fully connected		| 120->84      									|
| Fully connected		| 84->43     									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

    I used a learning rate of 0.001, as default. I set the Epoch to an arbitrarily high number of 75, relying on early stopping to halt training when it was ready. 
    It saved the model when it beat the last validation accuracy by being 2% better at least, and was willing to wait 10 epochs before stopping early. It stopped training at epoch 28, reverting to epoch 18 where the last saved model was.
    I used the adam optimizer, reducing mean cross entropy, and simply minimizing the loss. All were as default in the lenet example.
    

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 95.2% 
* test set accuracy of 93.4%

If an iterative approach was chosen:
    
    My first failure was to change the achitecture into a more modern form. I wanted to expand the number of feature maps by ten times what they were, add an additonal convolution layer, change to same rather than valid padding, add inception modules with dimentionality reduction, and then have a single fully connected layer before a softmax. After spending a week on this change I was forced to give up on it becuase I couldn't get the code to run. Reverted to the old lenet model.
    Next I wanted to add a CLAHE step to my preproccessing, but after a series of crashes and another week, I was again forced to move on without any progress.
    I spent another week attempting to add gpu support to my training, but only served to brick my system and I was forced to reinstall, with backups so no progress was lost, but non gained either. I ended up using the amazon  instances to train my model


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, plus one stop sign I found while on a walk:

![1][image7] ![2][image8] ![3][image9] ![4][image10] ![5][image11]


I assumed that these images would be easy to classify, except for the stop sign here:

![6][image12]

The reason being is the STOP is not using the same font that it has been trained on this whole time. With thinner text I'm fairly confident it will fail this classification.

#### 2. Predictions on internet images

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50km/h          		| X 80km/h       								| 
| Right-of-way at the...| Right-of-way at the next intersection 		|
| 30km/h				| 30km/h										|
| No passing	      	| No passing						 			|
| Ahead only			| Ahead only        							|
| Stop					| X No entry       								|



Excluding the odd stop sign, the model had an 80% accuracy, with it, 66.6%.

#### 3. Softmax and confidence in predictions.

    The model is confident on every prediction it makes, even when wrong. This seems like more evidence that the model is too small for the size of the dataset given to it and that the data soesn't have enough variety to it. Too many similiar images. I will have to change this for next time.
    
![confidence][image13]

    Here, in this softmax visual, the confidence is plain to see, all images had near 100% confidence.
![softmax][image14]

#### 4. Conclusion.

    It is clear now that the model needs to have a deeper representation of the images. six, then sixteen maps were not enough. Additionally I needed to do more to each image to avoid training the model too narrowly. While 93.4% may be enough to pass the assignment, this is terrible in comparison to what has been acheived. I will look into what specifically these better models have done to learn from them. I have learned plenty about monipulating data and working with it in python, while making many mistakes with the deep learning portion.