import cv2, os
import numpy as np
import matplotlib.image as mpimg


IN_HEIGHT, IN_WIDTH, IMAGE_CHANNELS = 160, 320, 3 #Change this
OUT_HEIGHT, OUT_WIDTH = 100, 100
INPUT_SHAPE = (OUT_HEIGHT, OUT_WIDTH, IMAGE_CHANNELS)

# RGB image, image_dir is a string
def load_image(image_dir):
    return np.array(mpimg.imread(image_dir))

def resize(image):
    return cv2.resize(image, (OUT_WIDTH, OUT_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image): #Change this for later
    image = first2bird(image)
    image = rgb2yuv(image)
    image = resize(image)
    return image


def undistort(image, directory = None):
    if directory == None:
        directory = "./data/saved_undistort.npz"

    data = np.load(file=directory)
    dst = cv2.undistort(image, data['mtx'], data['dist'], None, data['newcameramtx'])
    x,y,w,h = data['region_of_interest']
    image = dst[y:y+h, x:x+w]

    return image


def first2bird(image):

    points1 = np.float32([[265, 600],[410, 500],[825, 500],[975, 600]])
    points2 = np.float32([[150,600],[150,520],[450,520],[450,600]])

    M = cv2.getPerspectiveTransform(points1,points2)
    bird_eye = cv2.warpPerspective(image,M,(600,600))

    return bird_eye

def bird2first(image):

    points1 = np.float32([[265, 600],[410, 500],[825, 500],[975, 600]])
    points2 = np.float32([[150,600],[150,520],[450,520],[450,600]])

    M = cv2.getPerspectiveTransform(points2,points1)
    firstperson = cv2.warpPerspective(image,M,(1200,617))

    return firstperson

"""                               """                                 """
                  COMPUTER VISION FUCTIONS FOR BINARIFY
"""                               """                                 """

def binarify(image):

    sobelx = sobel_image(image, orient='x')
    sobelx_binary = thresh_binary(sobelx, thresh=(30, 255))

    white  = white_mask(image)
    yellow = yellow_mask(image)

    combined_binary = np.zeros_like( cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) )
    combined_binary[( (white>0)|(yellow>0)|(sobelx_binary>0) )] = 255

    return combined_binary

def thresh_binary(image, thresh):

    # Create an all zeros copy and apply the threshold.
    # Anything within the threshold is a one, not, then a zero
    binary_output = np.zeros_like(image)
    binary_output[(image >= thresh[0]) & (image <= thresh[1])] = 1

    return binary_output


# Uses an absolute sobel to detect lines
def sobel_image(image, orient='x'):

    # Create a grayscale copy of the image.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Choose which options to use for cv2.Sobel
    # Absolute value the output and merge variable names
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobel = np.absolute(sobelx)
    if orient == 'y':
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.absolute(sobely)

    # Scale abs_sobel from 0 to 255 integer / uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return scaled_sobel


# Uses lab and hsv to ignore shadows and detect white
def white_mask(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    hsv2 = (185, 255)
    hsv1 = (0, 25)
    lab0 = (185, 255)

    white_hsv2 = thresh_binary(hsv[:,:,2], hsv2)
    white_hsv1 = thresh_binary(hsv[:,:,1], hsv1)
    white_lab0 = thresh_binary(lab[:,:,0], lab0)

    white = np.zeros_like(image)
    white[(white_lab0>0)&(white_hsv1>0)&(white_hsv2>0)] = 255

    return cv2.cvtColor(white, cv2.COLOR_RGB2GRAY)


# Uses lab and hsv to ignore shadows and detect yellow
def yellow_mask(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    hsv0 = (17, 30)
    lab2 = (148, 255)

    yellow_hsv0 = thresh_binary(hsv[:,:,0], hsv0)
    yellow_lab2 = thresh_binary(lab[:,:,2], lab2)

    yellow = np.zeros_like(hsv)
    yellow[(yellow_lab2>0)&(yellow_hsv0>0)] = 255 #

    return cv2.cvtColor(yellow, cv2.COLOR_RGB2GRAY)

"""                               """                                 """
            LANEFINDER AFTER BINARY 600X600 IMAGE IN CALCULATED
"""                               """                                 """

def blindlanefinder(warped):

    num_windows = 10
    window_offset = 100
    image_height, image_width  = warped.shape[:2]
    midpoint = np.int(image_width/2)
    window_height = image_height/num_windows

    # 5/7ths of the lower part of the image is measured to find where the most pixels are
    # Then we use this as a starting point to find the lane lines.
    histogram = np.sum(warped[int(5*image_height/7):,:], axis=0)

    # Now we split the image in half, which caused problems becuase sometimes the lanes are on one side of the image
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    leftx, rightx = leftx_base, rightx_base
    leftleft, leftright, rightleft, rightright = leftx-window_offset, leftx+window_offset, rightx-window_offset, rightx+window_offset

    lane_lines = np.zeros_like(warped)

    leftpointy, leftpointx, rightpointy, rightpointx = [],[],[],[]

    for y_high in range(int(num_windows*window_height), 0, int(-window_height)):

        y_low = int(y_high-window_height)

        lhistogram = np.sum(warped[y_low:y_high,leftleft:leftright], axis=0)
        rhistogram = np.sum(warped[y_low:y_high,rightleft:rightright], axis=0)

        if lhistogram.size !=0:
            if np.max(lhistogram):
                leftx = np.argmax(lhistogram)+leftleft
        if rhistogram.size !=0:
            if np.max(rhistogram):
                rightx = np.argmax(rhistogram)+rightleft

        leftleft, leftright, rightleft, rightright = leftx-window_offset, leftx+window_offset, rightx-window_offset, rightx+window_offset

        tempy, tempx   =  np.nonzero( warped[y_low:y_high, leftleft:leftright] )
        leftpointy.extend(tempy + y_low)
        leftpointx.extend(tempx + leftleft)

        tempy, tempx =  np.nonzero( warped[y_low:y_high, rightleft:rightright] )
        rightpointy.extend(tempy + y_low)
        rightpointx.extend(tempx + rightleft)

        lane_lines[y_low:y_high, leftleft:leftright]   = warped[y_low:y_high, leftleft:leftright]
        lane_lines[y_low:y_high, rightleft:rightright] = warped[y_low:y_high, rightleft:rightright]


    left_fit  = np.polyfit(leftpointy, leftpointx, 2)
    right_fit = np.polyfit(rightpointy, rightpointx, 2)

    return lane_lines, left_fit, right_fit


def drawlane(left_fit, right_fit, image):

    height, width = 600,600

    if ((np.all(left_fit != 0)) & (np.all(right_fit != 0))):

        # Create an image to draw the lines on
        warp_zero = np.zeros((height, width)).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        #
        ploty = np.linspace(0, 599, num=600)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        y_eval = np.max(ploty)

        # Organization of the points.
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        unwarped = bird2first(color_warp)

        final = cv2.addWeighted(image, 1, unwarped, 0.3, 0)

        return final

    else:
        return image

def measure_center(left, right):

    # The lines are plotted in flat, 600X600 space
    # So 300 is the center, 599 is the closest to us
    ploty = np.linspace(0, 599, num=600)
    leftx   = left[0] *ploty**2 + left[1] *ploty + left[2]
    rightx  = right[0]*ploty**2 + right[1]*ploty + right[2]
    xm_per_pix = 3.7/350

    left, right = 300-leftx[599], rightx[599]-300

    side = "left"
    if right < left:
        side = "right"

    pixels = int(abs(left-right))
    return [pixels * xm_per_pix, side]

def average_curve(leftcurve, rightcurve):
    return round((leftcurve+rightcurve)/2, 3)
