import cv2, os
import numpy as np
import matplotlib.image as mpimg
import os

def load_image(image_dir):
    """
    Load RGB images from a file
    """
    return np.array(mpimg.imread(image_dir))

def load_undistort_file():
    if os.path.isfile('saved_undistort.npz'):
        npfile = np.load('saved_undistort.npz')
        return npfile['mtx'], npfile['dist'], npfile['newcameramtx'], npfile['region_of_interest']

def undistort(image):
    """
    These two functions handle the calulation (if needed) of the undistortion.
    Then saves those variables for later.
    If the saved file exists, then it applies the undistortion to the image sent.
    """
    # If we don't have saved values we need to calulate them
    if not os.path.isfile('saved_undistort.npz'):
        imgs = [0 for x in range(20)]
        # Make a list of calibration images
        for i in range(0, 20):
            imgs[i] = cv2.imread("./camera_cal/calibration{}.jpg".format(i+1), 1)

        objpoints = [] # 3d points
        imgpoints = [] # 2d points

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
        #left, top, width, height = roi
        #cropped_image = undistorted_image[top:top+height, left:left+width]

    return undistorted_image

def fastlanefinder(leftfit, rightfit):


    return lane_lines

def lanefinder(image, num_windows = 10, window_offset = 100):

    image_height, image_width  = image.shape
    midpoint = np.int(image_width/2)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[int(4*image_height/7):,:], axis=0)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = image_height/num_windows

    leftx, rightx = leftx_base, rightx_base
    leftleft, leftright, rightleft, rightright = leftx-window_offset, leftx+window_offset, rightx-window_offset, rightx+window_offset
    lane_lines = np.zeros_like(image)
    #boxes = []

    for y_high in range(int(num_windows*window_height), 0, int(-window_height)):
            y_low = int(y_high-window_height)

            lhistogram = np.sum(image[y_low:y_high,leftleft:leftright], axis=0)
            rhistogram = np.sum(image[y_low:y_high,rightleft:rightright], axis=0)
            if lhistogram.size !=0:
                if np.max(lhistogram):
                    leftx = np.argmax(lhistogram)+leftleft
            if rhistogram.size !=0:
                if np.max(rhistogram):
                    rightx = np.argmax(rhistogram)+rightleft

            leftleft, leftright, rightleft, rightright = leftx-window_offset, leftx+window_offset, rightx-window_offset, rightx+window_offset

            #boxes += coord2box([leftleft,y_low],[leftright,y_high]), coord2box([rightleft,y_low],[rightright,y_high])

            lane_lines[y_low:y_high, leftleft: leftright] =  image[y_low:y_high, leftleft:leftright]
            lane_lines[y_low:y_high, rightleft:rightright] = image[y_low:y_high, rightleft:rightright]

    return lane_lines    #, boxes


def fitlanes(laned_image):

    y, x = np.nonzero(laned_image)

    number = 10
    image_height, image_width = laned_image.shape

    left_fit = np.polyfit(lefty, leftx, 2)

    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def drawlane(left_fit, right_fit, laned_image):

    if ((np.all(left_fit != 0)) & (np.all(right_fit != 0))):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(laned_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        #
        ploty = np.linspace(0, 719, num=720)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        y_eval = np.max(ploty)

        #
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        #Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = u.bird2first(color_warp)
        result = cv2.addWeighted(laned_image, 1, newwarp, 0.3, 0)

        return result

    else:
        return laned_image


def calculate_radius3d(leftx, lefty, rightx, righty):

    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7 / 650     # meters per pixel in x dimension (lane width in US = 3.7 m)
    ym_per_pix = 3.0 / 80       # meters per pixel in y dimension (dashed marker length in US = 3.0 m)
    cam_pos_x = 1280 / 2.       # camera x-position in pixel (center of image)

    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    # multiply by meters, divide by pixels.
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def coord2box(coord1, coord2):

    if coord2[:]<coord1[:]:
        temp = coord1
        coord1 = coord2
        coord2 = temp

    height = coord2[0]-coord1[0]
    width = coord2[1]-coord1[1]

    return (coord1[0], coord1[1]), height, width

def yellowandwhite(hsv_image):
    masks = []
    # First, we define what values, in hsv format, define yellow and white.
    low_white,  high_white =  np.array([0, 0, 200]),  np.array([200, 30, 255])
    low_yellow, high_yellow = np.array([15, 70, 205]),np.array([25, 255, 255])

    # Then, we use those thresholds to extract the colors we want from the image.
    white = cv2.inRange(hsv_image, low_white, high_white)
    masks.append(white)
    yellow = cv2.inRange(hsv_image, low_yellow, high_yellow)
    masks.append(yellow)

    #Finally, we convert the masks to binary with a low threshold and return.
    yawbinary = np.zeros_like(hsv_image[:,:,0])
    yawbinary[(cv2.add(*masks) >= 20)] = 1

    return yawbinary

def sobelthis(gray_image):
    # Sobel is a filter for left edges that is convolved over the image.
    # Then we take the absolute value to retreive the right edges.
    # Finally, we scale this to between 0-255
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshholding. Create a blank image or zeroes.
    # Changing to 1 if the pixel value was within the threshold values.
    thresh_min, thresh_max = (20, 100)
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary

def OR_binaryimage(image1, image2):
    combined_binary = np.zeros_like(image1)
    combined_binary[(image1 == 1) | (image2 == 1)] = 1
    return combined_binary

def AND_binaryimage():
    combined_binary = np.zeros_like(image1)
    combined_binary[(image1 == 1) & (image2 == 1)] = 1
    return combined_binary

def first2bird(undist_image):
    image_width, image_height = undist_image.shape[1], undist_image.shape[0]
    #src is a list of starting locations, dst is where those points finish at.
    #In this case, we're straigtening two subsections of the lanes near us,
    #which straightenes the rest of the image as well.
    src = np.float32 ([
            [335, 630],
            [558, 480],
            [754, 480],
            [988, 630]
        ])
    dst = np.float32 ([
            [275, 705],
            [275, 450],
            [960, 450],
            [960, 705]
        ])

    #First, calulate the matrix based on the above points.
    #Then, use the matrix to do the transformaton and save the result, then return it.
    M = cv2.getPerspectiveTransform(src, dst)
    image_warped = cv2.warpPerspective(undist_image, M, (image_width, image_height))
    return image_warped

def bird2first(distorted_image):
    image_width, image_height = distorted_image.shape[1], distorted_image.shape[0]
    src = np.float32 ([
            [335, 630],
            [558, 480],
            [754, 480],
            [988, 630]
        ])
    dst = np.float32 ([
            [275, 705],
            [275, 450],
            [960, 450],
            [960, 705]
        ])
    Minv = cv2.getPerspectiveTransform(dst, src)
    image_unwarped = cv2.warpPerspective(distorted_image, Minv, (image_width, image_height))
    return image_unwarped

def binarify(hsv):

    yaw_binary = yellowandwhite(hsv)
    #Save a copy of a converted to grayscale image.
    gray = hsv[:,:,2]
    sobel_binary = sobelthis(gray)

    # Combine the two binary thresholds
    combined_binary = OR_binaryimage(sobel_binary, yaw_binary)

    return combined_binary
