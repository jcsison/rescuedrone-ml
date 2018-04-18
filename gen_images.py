import os
import sys

import cv2
import numpy as np
import random

def generate(file_path, save_path='input/images/', class_name='rc', identifier='00'):
    def next_seed():
        blur = round(1.5*random.random())*2 + 1
        tilt = round(360*random.random())
        erode = round(3*random.random()) + 1
        dilate = round(3*random.random()) + 1
        return blur, tilt, erode, dilate

    def process(image):
        blur, tilt, erode, dilate = next_seed()
        # img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        # channels = cv2.split(img_ycrcb)
        # channels[0] = cv2.equalizeHist(channels[0])
        # img_ycrcb = cv2.merge(channels)
        # image = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
        image = cv2.GaussianBlur(image, (blur, blur), 0)
        (h, w) = image.shape[:2]
        image = cv2.copyMakeBorder(image, 500, 500, 500, 500, cv2.BORDER_WRAP)
        (h2, w2) = image.shape[:2]
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((w2 / 2, h2 / 2), tilt, 1.0), (w2, h2))
        image = image[int(h2//2 - h//2):int(h2//2 + h/2), int(w2//2 - w/2):int(w2//2 + w/2)]
        image = cv2.erode(image, np.ones((erode, erode)))
        image = cv2.dilate(image, np.ones((dilate, dilate)))
        return image

    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    if not os.path.exists(save_path + file_name + '/'):
        os.makedirs(save_path + file_name + '/')

    index = 0;
    while True:
        # blur, tilt, erode, dilate = next_seed()
        ret, frame = cap.read()
        if index % fps == 0:
            if not ret:
                sys.exit(0)
            frame = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
            image_path = save_path + '{0}/{1}_{2}_{3}.jpg'.format(file_name, class_name, identifier, index // fps)
            cv2.imwrite(image_path, frame)
            print('{0} created'.format(image_path))
            # if pp:
            #     frame = process(frame)
            #     image_path = save_path + '{0}/{1}_{2}_{3}_p.jpg'.format(file_name, class_name, identifier, index // fps)
            #     cv2.imwrite(image_path, frame)
            #     print('{0} created'.format(image_path))
        index += 1

if __name__ == '__main__':
    if len(sys.argv) > 2:
        generate(sys.argv[1], identifier=sys.argv[2])
    elif len(sys.argv) > 1:
        print('gen_images.py: missing identifier operand')
        print('usage: python3 gen_images.py [file] [identifier]')
    else:
        print('gen_images.py: missing file operand')
        print('usage: python3 gen_images.py [file] [identifier]')
