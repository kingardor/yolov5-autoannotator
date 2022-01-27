import os
import sys
import traceback
import numpy as np
import cv2
import argparse
from tqdm import tqdm

from yolov5 import yolov5

class Annotator:

    def __init__(self):
        self.opt = self.argparser()

        # Load the model
        self.yolov5 = yolov5(
            self.opt.onnx_path,
            self.opt.labels_path,
            self.opt.width,
            self.opt.height,
            self.opt.confThreshold,
            self.opt.nmsThreshold
        )

        # Instantiate viewer
        if self.opt.viewmode:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        
        # Make directory
        if not os.path.exists(self.opt.annodir):
            os.makedirs(self.opt.annodir)

    def __del__(self):
        del self.opt
    
    def annotate(self):
        images = os.listdir(self.opt.imgdir)
        for image in tqdm(images):
            try:
                if '.jpg' in image or '.png' in image or '.jpeg' in image:
                    frame = cv2.imread(
                        os.path.join(self.opt.imgdir, image)
                    )

                    h,w = frame.shape[:2]

                    preds = self.yolov5.detect(frame)
                    
                    if len(preds) > 0:

                        # Create text file
                        f = open(os.path.join(self.opt.annodir, image.split('.')[0] + '.txt'), 'a+') 

                        for pred in preds:
                            id = pred['classId']
                            x1 = pred['x1']
                            y1 = pred['y1']
                            x2 = pred['x2']
                            y2 = pred['y2']

                            x_cen = (x1 - (w/2)) / w
                            y_cen = (y1 - (h/2)) / h
                            obj_w = (x2 - x1) / w
                            obh_h = (y2 - y1) / h

                            f.write('{} {} {} {} {} \n'.format(id, x_cen, y_cen, obj_w, obh_h))

                            if self.opt.viewmode:
                                cv2.rectangle(
                                    frame, 
                                    (x1, y1), 
                                    (x2, y2),
                                    (0, 255, 0),
                                    2
                                )

                        f.close()

                    if self.opt.viewmode:
                        cv2.imshow('image', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except:
                traceback.print_exc()
                break

    def argparser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--viewmode', action='store_true', help='Toggle View Mode')
        parser.add_argument('--imgdir', type=str, help='Directory of images')
        parser.add_argument('--annodir',type=str, help='Directory of annotations')
        parser.add_argument('--confThreshold', default=0.3, type=float, help='Class confidence')
        parser.add_argument('--nmsThreshold', default=0.5, type=float, help='NMS threshold')
        parser.add_argument('--width', default=640, type=int, help='Width of network input')
        parser.add_argument('--height', default=640, type=int, help='Height of network input')
        parser.add_argument('--onnx_path', type=str, help='Path to onnx file')
        parser.add_argument('--labels_path', type=str, help='Path to labels file')
        return parser.parse_args()

if __name__ == '__main__':
    annotator = Annotator()
    annotator.annotate()
