import os
import sys

import cv2

def generate(file_path, save_path='input/images/'):
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
            image_path = save_path + '{0}/{0}_{1}{2}'.format(file_name, index // fps, '.jpg')
            cv2.imwrite(image_path, frame)
            print('{0} created'.format(image_path))
        index += 1

if __name__ == '__main__':
    if len(sys.argv) > 1:
        generate(sys.argv[1])
    else:
        print('gen_images.py: missing file operand')
        print('usage: python3 gen_images.py [file1]')
