import cv2, os
import numpy as np
import matplotlib.image as mpimg
import os

IN_HEIGHT, IN_WIDTH, IMAGE_CHANNELS = 160, 320, 3
OUT_HEIGHT, OUT_WIDTH = 100, 100
INPUT_SHAPE = (OUT_HEIGHT, OUT_WIDTH, IMAGE_CHANNELS)

def load_image(image_dir):
    """
    Load RGB images from a file
    """
    return np.array(mpimg.imread(image_dir))

def load_undistort_file():
    if os.path.isfile('saved_undistort.npz'):
        npfile = np.load('saved_undistort.npz')
        return npfile['mtx'], npfile['dist'], npfile['newcameramtx'], npfile['region_of_interest']
"""
These two functions handle the calulation (if needed) of the undistortion.
Then saves those variables for later.
If the saved file exists, then it applies the undistortion to the image sent.
"""
def undistort(image):
    if not os.path.isfile('saved_undistort.npz'): #if we don't have saved values we need to calulate them
        imgs = [0 for x in range(20)]
        # Make a list of calibration images
        for i in range(0, 20):
            imgs[i] = cv2.imread("./camera_cal/calibration{}.jpg".format(i+1), 1)

        objpoints = [] # 3d points
        imgpoints = [] # 2d points

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # always the same, so it's outside the loop
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Step through the list and search for chessboard corners
        for img in imgs:
            # First, convert to gray so the next function can accept it.
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Give: gray image, number of corners to find, and flags (none)
            # Get: flag for success, 2d coordinates for the corners
            returned, imgp = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if returned == True:
                objpoints.append(objp)
                imgpoints.append(imgp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # The final co-efficients that can be saved and reused for the video pipeline
        newcameramtx, region_of_interest = cv2.getOptimalNewCameraMatrix(mtx,dist,(1280, 720),1,(1280, 720))
        np.savez('saved_undistort.npz', mtx=mtx, dist=dist, newcameramtx=newcameramtx, region_of_interest=region_of_interest)

    if os.path.isfile('saved_undistort.npz'):
        mtx, dist, newmtx, roi = load_undistort_file()

        # image sent to this function
        height, width = image.shape[:2]
        # undistort
        undistorted_image = cv2.undistort(image, mtx, dist, None, newmtx)

        # crop the image
        left, top, width, height = roi
        cropped_image = undistorted_image[top:top+height, left:left+width]

    return cropped_image









def perspective_shift(image):
    """
    Crop the image by 60 from top, 25 from bottom.
    Then, warp perspective to a square
    """
    points1 = np.float32([[60,60],[240,60],[0,135],[320,135]])
    points2 = np.float32([[0,0],[100,0],[0,100],[100,100]])
    M = cv2.getPerspectiveTransform(points1,points2)
    bird_eye = cv2.warpPerspective(image,M,(100,100))
    return bird_eye

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
