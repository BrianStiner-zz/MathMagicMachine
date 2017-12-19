---

**Advanced Lane Finding Project**

**Brian Stiner**

**MathMagicMachine**

The goals / steps of this project are the following:

* Fix the camera lens distortionoutput_images
* Use color transforms and gradients to create a binary image.
* Apply a perspective transform
* Detect lane pixels and fit to find the lines
* Determine the curvature of the lane and vehicle position with respect to center.
* Average the lines over time and reject bad lines.data/
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./data/output_images/undistorted.png "Undistorted"
[image2]: ./data/output_images/project_undistorted.png "Road Undistorted"
[image3]: ./data/output_images/binary.png "Binary Example"
[image4]: ./data/output_images/birdseye.png "Road Transformed"
[image5]: ./data/output_images/fitted_lines.png "Fitting Example"
[image6]: ./data/output_images/lanelines.png "Warp Example"
[image7]: ./data/output_images/filled_lane.png "Fit Visual"
[image8]: ./data/output_images/example_output.png "Output"
[video1]: ./data/project_video.mp4 "Video"

---


### Camera Calibration

#### 1. Calculating the distortion

The code for this step is contained in the first code cell of the IPython notebook located in "./example.ipynb" 

Used as tool in lines 27 through 37 in `utils.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline

#### Distortion correction

Here is an example of a distortion corrected image from the project:

![alt text][image2]

#### Perspective transformation

The function 'first2bird()' located in 'utils.py' at line 39:

------
def first2bird(image):

    points1 = np.float32([[265, 600],[410, 500],[825, 500],[975, 600]])
    points2 = np.float32([[150,600],[150,520],[450,520],[450,600]])
    M = cv2.getPerspectiveTransform(points1,points2)
    bird_eye = cv2.warpPerspective(image,M,(600,600))

    return bird_eye
------

These point convert the image into a 600x600, flat, pixel space:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 265, 600      | 150, 600      | 
| 410, 500      | 150, 520      |
| 825, 500      | 450, 520      |
| 975, 600      | 450, 600      |

Here is a visual of the transformation:

![alt text][image4]

#### Computer vision transforms

I used two functions which use hsv and lab to specifically target the yellow and white on the road used for the lanes. I also use sobelx as is as well. This created a high quality map of the lane lines, and I excluded using other calculations as they generally made the results worse.

![alt text][image3]


#### Fitting to the lines

In the file 'utils.py' lines 147 through 201, function blindlanefinder(warped). Using np.slices and histogram maxes, I follow the lane with ten boxes. Using np.nonzero to extract points and add them to list. 

Finally, I use np.polyfit() to create the three coefficients needed for a second order polynomial.

![alt text][image5]

#### Finding lane curvature

In my pipeline, lane curvature is calulated in the 'Line()' class, in function 'calc_radius_of_curvature(self)'

line 238 of 'utils.py' does the work of calculating the center of the road, just after the function for calculating average curvature

I average both curvatures together after in order to display

#### Lane, identified clearly

After the last step, we draw in the shape filled between the two lines and then use 'bird2first()' to draw the lane on the road, the original undistorted image.

The drawlane function is on 204 of 'utils.py'.

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 My pipeline still has issues with discarding bad lines. 
 
 Going back the preprocessing step to further play with threshold values may help, but might be video specific and would therefore overfit the pipeline. Having a better method in the 'seems_reasonable()' function in the 'Line()' class would be a more robust solution.
