"""
p4pipeline.py: version 0.1: Nov. 7
"""
# Import statements
import numpy as np
import cv2
import utils as u
import


global left_lane_line, right_lane_line
left_lane_line, right_lane_line = [],[]

def lanedetector(original_image):

    image = np.copy(original_image)

    global width,height
    height,width = image.shape[0], image.shape[1]

    undistorted_image = utils.undistort(image)

    binary_image = utils.binarify(undistorted_image)

    birdseye_image = utils.perspectiveshift(binary_image)

    curvature = utils.calulatecurvature(birdseye_image)
