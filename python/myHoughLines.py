import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    """
    H: Hough transform accumulator array
    nLines: number of lines to return
    returns: rhos, thetas
    """
    # Using dilate to remove duplicate lines
    cv2.dilate(H, np.ones((3, 3)), iterations=1)
    rhos = []
    thetas = []
    for i in range(nLines):
        # Find max element of H and use unravel_index to find corresponding rho and theta
        max_index = np.unravel_index(np.argmax(H), H.shape)
        rhos.append(max_index[0])
        thetas.append(max_index[1])
        H[max_index[0], max_index[1]] = 0
    return rhos, thetas



