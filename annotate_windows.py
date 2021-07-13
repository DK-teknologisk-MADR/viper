import numpy as np
import glob
import cv2
import os
import json
import sys

class Annotator():
    def __init__(self,folder,user_input,pics):
        self.folder = folder #'./windows/'
        self.user_input = user_input #{'d': 'discard', 'g': 'good'}
        self.files = glob.glob(self.folder + '/*.jpg')
        self.files.sort()

    def run(self):
        for fname in self.files:

            img = cv2.imread(fname)
            while True:
                cv2.imshow(fname, img)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    sys.exit('You pressed \'q\' - exiting!')
                if k == ord('s'):
                    print(fname + ' - skipping!')
                    cv2.destroyAllWindows()
                    break
                annotation = None
                for key, value in self.user_input.items():
                    if k == ord(key):
                        annotation = value
                        break
                if annotation is None:
                    print('Oops - did not recognize user input - try again!')
                else:
                    print(fname + ' - annotation = ' + str(annotation))
                    cv2.destroyAllWindows()
                    break

            data = {}
            data['imagePath'] = os.path.basename(fname)
            data['imageHeight'] = img.shape[0]
            data['imageWidth'] = img.shape[1]
            data['shapes'] = []
            for key, value in self.user_input.items():
                shape = {}
                shape['label'] = value
                shape['shape_type'] = 'point'  # doesn't matter?
                if annotation == value:
                    shape['points'] = [1]
                else:
                    shape['points'] = [0]
                data['shapes'].append(shape)

            with open(self.folder + os.path.splitext(os.path.basename(fname))[0] + '.json', "w") as fout:
                json.dump(data, fout, indent=2)




