import os
import sys

import cv2

def generate(file_path, save_path='input/images/', class_name='rc', identifier='00'):
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    if not os.path.exists(save_path + file_name + '/'):
        os.makedirs(save_path + file_name + '/')

    index = 0;
    while True:
        ret, frame = cap.read()
        if index % fps == 0:
            if not ret:
                sys.exit(0)
            frame = cv2.resize(frame, (0, 0), fx=1/3, fx=1/3)
            image_path = save_path + '{0}/{1}_{2}_{3}.jpg'.format(file_name, class_name, identifier, index // fps)
            cv2.imwrite(image_path, frame)
            print('{0} created'.format(image_path))
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
