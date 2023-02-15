import actionDetection
import util
import os
import numpy as np


DATA_PATH = os.path.join('C:/Users/Jack/Documents/WLASL_Refined_Dataset')
# Actions that we try to detect
if __name__ == '__main__':
    actionDetection.detect_tflite('res/model.tflite')