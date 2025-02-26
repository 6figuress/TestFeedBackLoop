import numpy as np  

class ColorBound:
    def __init__(self): # Initialize the min and max values for each HSV channel
        self.min_red = np.array([170, 120, 70])
        self.max_red =np.array([180, 255, 255])
        self.min_yellow =  np.array([50, 50, 50])
        self.max_yellow = np.array([70, 100, 100])
        self.min_black = np.array([0, 0, 0]) 
        self.max_black = np.array([360, 50, 50])

        self.min_white = np.array([0, 0, 200])    # Low saturation, high valuese
        self.max_white = np.array([180, 50, 255]) # Allow slight saturation variation
