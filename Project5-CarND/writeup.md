**Vehicle Detection Project**

**Brian Stiner** 

**MathMagicMachine on Github and Linkedin**

The steps for detecting a car on the road:

* Read in all images, car and not car
* Extract features, HOG for all three channels, reduced resololution image, and color histograms of all three channels.
* Train a linear svc to seperate cars features from not car features.
* Now, move on to using the trained model...



* Create slices of an image using boxes at different scales
* Extract all features, normalize the batch, predict the batch.
* Save predicted boxes, use them to create a heatmap, with a fram buffer
* Threshold the heatmap, output the labeled predictions of cars.

[//]: # (Image References)
[image1]: ./data/examples/carstack.png
[image2]: ./data/examples/notcarstack.png
[image3]: ./data/examples/carhog.png
[image4]: ./data/examples/notcarhog.png
[image5]: ./data/examples/boxes-example.png
[image6]: ./data/examples/carbox.png
[image7]: ./data/examples/finalstack.png
[video1]: ./data/SVC_video.mp4
[video2]: ./data/Yolo_video.mp4
[video3]: ./data/Personal_video.mp4


---

### Histogram of Oriented Gradients (HOG)

  The code for this step is contained within the 'featureprocesser.py' file between lines 7-25. It is a simple wrapper for the hog function from skimage.feature, and is used inside of 'single_img_features' from lines 127-136.

  I started by reading in all the `vehicle` and `non-vehicle` images from the datasets provided. Here are the example image I created for this presentation, unaltered:

![car_stack][image1]
![notcar_stack][image2]

  I then explored different options for this function using past examples as starting points. After running experiments on the options given, I settled on the following:
`YCrCb` color space and parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` which are the defaults as I've set them, as they gave the best results:

![car_hog][image3]
![notcar_hog][image4]

  These settings, when trained on a three-channel hog feature vector, are the most accuracte on the training data

  I trained the model as shown in cells 11-13 in the jupyter notebook. I modified the loss to hinge, as it increased speed of training and increased accuracy by more than one percent.

### Sliding Window Search

  I wanted to test where bounding boxes were likly to occur in the test videos. I used gimp to measure pixels in the sample images and using that information decided to use a line of 64x64 at 400y to 464y, another with 96x96, and 128x128. All have the high points at 400, and go down equal to their size. I tried larger and smaller boxes at different heights but found no success there.

![alt text][image5]

I attempted to find the optimal setup by first studying the work of those before me. After compareing the results of each setting I settled on Ycrcb, using 8 channels for the hog, and leaving out the histogram of gradients. However dispite this, I was unable to get the results I wanted from this model.


It was at this point in the process that I decided to stretch the project in order to get the results I wanted. I switched to using a single-shot convolutional network, specifically Yolo-v2 after initially trying tiny-yolo. Although I was capable of training the network using the dataset provided or creating my own, due to time constraints I used a weights file. The results were worth the change.

![alt text][image6]

---

### Video Implementation

I have three videos, two of the project video and one of my own shooting.

LinearSVC and preprocessing video [Project video](./data/SVC_video.mp4)

Yolo v2 [Project video](./data/Yolo_video.mp4)

Video of mine, using Yolo [My video](./data/Personal_video.mp4)

By using a simple convnet to predict boxes and confidence in the image directly, the Yolo network very quickly, with higher accuracy than the svc could accomplish.

### Pipeline Details

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
This code is found in cell 30 of the main notebook.

### Here are nine frames making their way through my pipeline. They start unaltered and use heatmaps and labels to seperate out the cars before putting a box on the original image.

![The pipeline in nine frames][image7]


---

### Discussion


Something is certainly wrong with the training data, the classifier, or the preprocessing as something caused the results to be chaotic with the svc.
The yolo results are brilliant, and I look forward to doing more experiments with the yolo structure. Being able to get realtime data from cameras is a great boon, and keeping tiny-yolo on my phone for my use is something I plan to do later.

