To start, I detailed all the steps I wanted to make.
I originally only used 1,5,6,7,8,9,11. After getting a working video, I added color focusing to the algorithm.
Step 10 parameters were added as the last feature to remove false positives from the lines created.
Final steps are as follows:

#Steps:            Parameters:
#1:Get image       (nothing)        
#2:Get Masked      (Thresholds)            
#3:Darken          (level of change) 
#4:Merge all       (nothing)        
#5:Convert to grey (nothing)  
#6:Blur            (kernel_size)
#7:Canny edge      (LowT,HighT)
#8:Area            (shape, size)
#9:Hough lines     (Threshold, min line length, max line gap, rho, theta)
#10:Draw           (Lines to ignore)

I needed to know what regions and colors would work best, so I took all the sample images given to us so far as well as a few taken from my own car, opened them in gimp photo editor, and made both color and region Measurements.

After getting a working video with all features, I spent a few hours finetuning variables to remove false positives and false negatives. This finetuning has certainly overfit this algorithm to this video. Specifically, the video must be the same resolution as all the numbers are fine tuned to 720x1280.

# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./TestImages/solidYellowCurve.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

