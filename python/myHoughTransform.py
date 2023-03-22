import numpy as np 
from myEdgeFilter import myEdgeFilter

def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    """
    Show the Hough transform result. Taken from the below example:
    https://github.com/alyssaq/hough_transform/blob/master/hough_transform.py
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
def myHoughTransform(img, rhoRes, thetaRes):
    """
    Hough transform for line detection
    :param img: binary edge image
    :param rhoRes: resolution of rho
    :param thetaRes: resolution of theta
    :return: accumulator, thetaValues, rhoValues
    """

    # Theta and rho values are different than in the pdf. 
    # The pdf uses the range [0,2*pi] for theta, but we use -pi/2, pi/2
    # Reason for that is -pi/2, pi/2 range covers the half circle. Since we are using sin and cos values, 
    # this range covers the whole values of sin and cos.
    # The pdf uses the range [0, 2*diag_len] for rho, but we use [-diag_len, diag_len]
    # Reason for that is -diag_len, diag_len range covers the whole coordinate system. If you put the image in center,
    # rotate it, max length will be from -diag_len to diag_len.
    # Additional to these information, these values are standard for the hough transform. For example :
    # https://sbme-tutorials.github.io/2021/cv/notes/4_week4.html#basic-implementation
    
    diag_len = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
    rhoValues = np.linspace(-diag_len/rhoRes, diag_len/rhoRes, 2*diag_len)
    thetaValues = np.arange(-np.pi/2,np.pi/2, thetaRes)
    img_hough = np.zeros((len(rhoValues), len(thetaValues)))
    cos_theta = np.cos(thetaValues)
    sin_theta = np.sin(thetaValues)

    y, x = np.nonzero(img)   
    # diag_len added as a normalization parameter. It can be visually seen if you remove it.
    rho = (np.round(np.outer(x, cos_theta) + np.outer(y, sin_theta)) + diag_len).astype(np.int16)
    for i in range(len(thetaValues)):
        # counts the number of votes for each rho value
        rhos,counts = np.unique(rho[:,i], return_counts=True)
        img_hough[rhos,i] = counts
    
    return img_hough, rhoValues, thetaValues

if __name__ == '__main__':
    import cv2
    import os
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/img03.jpg')
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # using edge filter to get the edge image
    img_edge= myEdgeFilter(img, 1)
    accumulator, rhos, thetas = myHoughTransform(img, 1, np.pi/90)
    show_hough_line(img_edge, accumulator, thetas, rhos, save_path='output.png')
    #cv2.imshow('accumulator', accumulator)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
