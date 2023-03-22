import matplotlib.pyplot as plt
import numpy as np
def plot():
    theta = np.arange(0,2*np.pi,0.01)
    ro1 = 10*np.cos(theta) + 10*np.sin(theta) 
    ro2 = 20*np.cos(theta) + 20*np.sin(theta)
    ro3 = 30*np.cos(theta) + 30*np.sin(theta)
    fig = plt.figure(figsize = (10, 5))	#identifies the figure 
    plt.title("Hough Space", fontsize='16')	#title
    plt.plot(theta,ro1)
    plt.plot(theta,ro2)
    plt.plot(theta,ro3)    
    plt.show()
plot()