import numpy as np
from myImageFilter import myImageFilter
from scipy import signal    # For signal.gaussian function

def nonMaximumSuppression(img, angle):
    angle = roundAngle(angle)
    width = img.shape[1]
    height = img.shape[0]
    # Shift the image in different directions based on the angle
    # 0 degree
    shifted_left = np.roll(img, 1, axis=1)
    shifted_right = np.roll(img, -1, axis=1)
    # 90 degree
    shifted_up = np.roll(img, 1, axis=0)
    shifted_down = np.roll(img, -1, axis=0)
    # 45 degree
    shifted_up_right = np.roll(shifted_up, -1, axis=1)
    shifted_down_left = np.roll(shifted_down, 1, axis=1)
    # 135 degree
    shifted_down_right = np.roll(shifted_down, -1, axis=1)
    shifted_up_left = np.roll(shifted_up, 1, axis=1)
    
    # Check the neighboring pixels based on the angle
    angle_0_mask = (angle == 0) & ((img < shifted_left) | (img < shifted_right))
    angle_45_mask = (angle == np.pi/4) & ((img < shifted_up_left) | (img < shifted_down_right))
    angle_90_mask = (angle == np.pi/2) & ((img < shifted_up) | (img < shifted_down))
    angle_135_mask = (angle == 3*np.pi/4) & ((img < shifted_up_right) | (img < shifted_down_left))
    # Zero out the pixels that don't meet the criteria
    img[angle_0_mask | angle_45_mask | angle_90_mask | angle_135_mask] = 0
    return img
def roundAngle(angle):
    pi = np.pi
    angle = np.where(angle < pi/8, 0, angle)
    angle = np.where((angle >= pi/8) & (angle < 3*pi/8), pi/4, angle)
    angle = np.where((angle >= 3*pi/8) & (angle < 5*pi/8), pi/2, angle)
    angle = np.where((angle >= 5*pi/8) & (angle < 7*pi/8), 3*pi/4, angle)
    angle = np.where(angle >= 7*pi/8, 0, angle)
    return angle
def myEdgeFilter(img0, sigma):
    '''
    Basically a Canny Filter.
    Blurres image.
    Finds gradient magnitude and direction.
    Applies non-maximum suppression.
    Returns edge magnitude image.
    '''
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    gauss_signal = signal.gaussian(2*np.ceil(sigma*3)+1,sigma)
    gauss_filter = np.outer(gauss_signal,gauss_signal) # outer product to create matrix
    gauss_filter = gauss_filter / np.sum(gauss_filter) # normalize to make sum 1 
    blurred_image = myImageFilter(img0,gauss_filter)
    #blurred_image = ndimage.gaussian_filter(img0,sigma) # used to compare with myImageFilter
    imgx = myImageFilter(blurred_image,sobel_x)
    imgy = myImageFilter(blurred_image,sobel_y)
    gradMagnitude = np.sqrt(np.square(imgx) + np.square(imgy)) 
    before_nms = gradMagnitude.copy()
    Angle = np.arctan2(imgy,imgx)
    # reason to use arctan2 -> https://geo.libretexts.org/Courses/University_of_California_Davis/GEL_056%3A_Introduction_to_Geophysics/Geophysics_is_everywhere_in_geology.../zz%3A_Back_Matter/Arctan_vs_Arctan2
    ## Non maximum suppression helped by chatgpt
    final_img = nonMaximumSuppression(gradMagnitude, Angle)
    # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    return final_img 
if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import cv2
    from scipy import ndimage  # For ndimage.gaussian_filter function
    # read in images
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/img02.jpg')
    # parameters
    sigma     = 1
    # end of parameters
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # binarize the image
    img = np.float32(img) / 255

    # actual Hough line code function calls
    img_edge= myEdgeFilter(img, sigma)
    cv2.imshow('img_edge',img_edge)
    cv2.waitKey(0)
    