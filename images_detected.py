

'''
/**************************************************************
 *                                                            *
 *                     Done By: Alqoary                       *
 *            Github: https://github.com/Alqoary              *
 *                    Date: 05/11/2021                        *
 *                                                            * 
 *************************************************************/
'''


# libraries
import torch
import time
import cv2
import numpy as np
import os

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def predict_images():
    files = []
    path = "./images/before_prediction"
    save_path = "./images/after_prediction"
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
    
    for i in range(len(files)):
        img = os.path.join(path, files[i])
        img = model(img)
        #img.show()
        img = np.squeeze(img.render())
        #cv2.imshow('n', img)
        img_path = os.path.join(save_path, str(i) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(img_path, img)


# main program
if __name__ == '__main__':
    predict_images()