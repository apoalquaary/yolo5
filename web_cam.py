
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

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# objects' name
names =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names


def detect_web_cam():
    # web cam
    web_cam = cv2.VideoCapture(0)

    # loop to keep the web cam open
    while (True):
        # start time
        start = time.time()

        # detect the objects in this frame
        ret, frame = web_cam.read()
        results = model(frame)

        # print to the console the detected objects
        print("\nDetected Objects:")
        for pred in results.pred:
            pred_values = list(pred[: , -1].cpu().data.numpy())
            [print(names[int(x)]) for x in pred_values]
        frame = np.squeeze(results.render())
        cv2.imshow("Web Cam", frame)

        # calculate the time every frame
        print("\nSpeed:\t", end='')
        print(time.time() - start)

        # close the program
        if cv2.waitKey(50) & 0xFF == ord('q') or cv2.waitKey(50) & 0xFF == ord('Q'):
            break
    web_cam.release()
    cv2.destroyAllWindows()

# main program
if __name__ == '__main__':
    detect_web_cam()
