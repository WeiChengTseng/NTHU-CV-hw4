import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from tqdm import tqdm

# you can implement your own data augmentation functions

def get_positive_features(train_path_pos, feature_params, reload=True):
    """
    FUNC: This function should return all positive training examples (faces)
        from 36x36 images in 'train_path_pos'. Each face should be converted
        into a HoG template according to 'feature_params'. For improved performances,
        try mirroring or warping the positive training examples.
    ARG:
        - train_path_pos: a string; directory that contains 36x36 images
                          of faces.
        - feature_params: a dict; with keys,
                          > template_size: int (probably 36); the number of
                            pixels spanned by each train/test template.
                          > hog_cell_size: int (default 6); the number of pixels
                            in each HoG cell.
                          Template size should be evenly divisible by hog_cell_size.
                          Smaller HoG cell sizez tend to work better, but they
                          make things slower because the feature dimenionality
                          increases and more importantly the step size of the
                          classifier decreases at test time.
    RET:
        - features_pos: (N,D) ndarray; N is the number of faces and D is the
                        template dimensionality, which would be,
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
    """
    #########################################
    ##          you code here              ##
    #########################################
    if os.path.isfile('features_pos.npy') and reload:
        features_pos = np.load('features_pos.npy')
        return features_pos

    features_pos = []
    for img in tqdm(os.listdir(train_path_pos)):
        path = os.path.join(train_path_pos, img)
        img = imread(path).astype(np.float32)
        hog_f = hog(img, cell_size=feature_params['hog_cell_size'])
        features_pos.append(hog_f.reshape(-1))
    features_pos = np.array(features_pos)

    np.save('features_pos', features_pos.copy())
    #########################################
    ##          you code here              ##
    #########################################

    return features_pos 
