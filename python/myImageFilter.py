import numpy as np

def myImageFilter(img0, h):
   '''
   Huge thanks to https://lucasdavid.github.io/blog/computer-vision/vectorization/ for the idea
   
   Takes input image and kernel and returns convolved image
   Kernel is flipped to match signal library's convolve2d function.
   Assumes kernel is odd in both dimensions.
   Pads image with edge values to match output size.
   '''
   KH, KW = h.shape
   # Kernel is not square so we need to pad differently
   pad_height = (KH-1)//2 
   pad_width  = (KW-1)//2
   # Pad image equally on start and end
   img0 = np.pad(img0, ((pad_height,pad_height), (pad_width, pad_width)))
   H, W = img0.shape

   # Create a matrix from the input image to directly multiply with the kernel
   # This will speed up the convolution process significantly
   # Detailed explanation can be found in the link above

   r0 = np.arange(H-KH+1)
   r0 = np.repeat(r0, W-KW+1)
   r0 = r0.reshape(-1, 1)

   r1 = np.arange(KH).reshape(1, KH)
   r = np.repeat(r0 + r1, KW, axis=1)

   c0 = np.arange(KW)
   c0 = np.tile(c0, KH).reshape(1, -1)

   c1 = np.arange(W-KW+1).reshape(-1, 1)
   c = np.tile(c0 + c1, [H-KH+1, 1])

   # https://numpy.org/doc/stable/user/quickstart.html#indexing-with-arrays-of-indices
   y = img0[r, c] @ h.reshape(-1) # @ is matrix multiplication dot product
   y = y.reshape(H-KH+1, W-KW+1)
   return y

if __name__ == '__main__':
   import scipy.signal as signal
   import time
   import cv2
   import os
   '''
   Helper function to test your code and measure performance
   my implementation works 4x slower than signal convolve2d function
   main reason is because underlying implementation of signal functions
   written in C and optimized for speed
   '''
   dirname = os.path.dirname(__file__)
   filename = os.path.join(dirname, '../data/img01.jpg')
   img0 = cv2.imread(filename , 0)
   h = np.arange(1,10).reshape(3,3)
   my_start = time.time()
   y_mine = myImageFilter(img0, h)
   my_end = time.time()
   y_scipy = signal.correlate2d(img0,h, mode='same')
   end = time.time()
   np.testing.assert_almost_equal(y_mine,y_scipy)
   print('All tests passed (no exceptions raised).')
   print('myImageFilter time:',my_end-my_start)
   print('signal.convolve2d time:',end-my_end)
   print("difference", (my_end-my_start)/(end-my_end))
