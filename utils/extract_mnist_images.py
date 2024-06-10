r"""
File to extract csv images from csv files for mnist dataset.
"""

import os
import cv2
from tqdm import tqdm
import numpy as np
import _csv as csv

def extract_images(save_dir, csv_fname, dataset_name):
    assert os.path.exists(save_dir), "Directory {} to save images does not exist".format(save_dir)
    assert os.path.exists(csv_fname), "Csv file {} does not exist".format(csv_fname)
    with open(csv_fname) as f:
        reader = csv.reader(f)
        if not os.path.exists(os.path.join(save_dir, dataset_name)):
            os.mkdir(os.path.join(save_dir, dataset_name))
        label_file = open(os.path.join(save_dir, dataset_name, 'labels.csv'), 'w')
        
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            label_file.write(os.path.join('{}.png'.format(idx)))
            label_file.write(',' + row[0])
            label_file.write('\n')
            im = np.zeros((784))
            im[:] = list(map(int, row[1:]))
            im = im.reshape((28,28))
            # if not os.path.exists(os.path.join(save_dir, dataset_name)):
            #     os.mkdir(os.path.join(save_dir, dataset_name))
            cv2.imwrite(os.path.join(save_dir, dataset_name, '{}.png'.format(idx)), im)
            if idx % 1000 == 0:
                print('Finished creating {} images in {}'.format(idx+1, save_dir))

        label_file.close()
            
            
if __name__ == '__main__':
    extract_images('data/train/', 'data/mnist_train.csv', 'mnist')
    extract_images('data/test/', 'data/mnist_test.csv', 'mnist')