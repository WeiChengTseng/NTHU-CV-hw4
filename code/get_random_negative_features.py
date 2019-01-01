import numpy as np
import os
import random
from cyvlfeat.hog import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from tqdm import tqdm

# you may implement your own data augmentation functions

def get_random_negative_features(non_face_scn_path, feature_params, num_samples, reload=True):
    '''
    FUNC: This funciton should return negative training examples (non-faces) from
        any images in 'non_face_scn_path'. Images should be converted to grayscale,
        because the positive training data is only available in grayscale. For best
        performance, you should sample random negative examples at multiple scales.
    ARG:
        - non_face_scn_path: a string; directory contains many images which have no
                             faces in them.
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
        - features_neg: (N,D) ndarray; N is the number of non-faces and D is 
                        the template dimensionality, which would be, 
                        (template_size/hog_cell_size)^2 * 31,
                        if you're using default HoG parameters.
        - neg_examples: TODO
    '''
    #########################################
    ##          you code here              ##
    #########################################
    import glob
    template_size = feature_params['template_size']
    hog_cell_size = feature_params['hog_cell_size']
    
    neg_train_image = glob.glob(non_face_scn_path+'/*.jpg')

    # reload = False
    if os.path.isfile('features_neg.npy') and reload:
        features_neg = np.load('features_neg.npy')
        if len(features_neg) == num_samples:
            print('-> Restore negative features from {}'.format('features_neg.npy'))
            return features_neg, len(neg_train_image)
    
    features_neg = []
    n_sam = int(np.ceil(num_samples/len(neg_train_image)))
    
    for i in tqdm(range(len(neg_train_image))):
        img = imread(neg_train_image[i], as_gray=True)
        h, w = img.shape[0], img.shape[1]
        
        if min(h, w) - template_size < n_sam:
            sample_per_image = min(h, w)-template_size
        else:
            sample_per_image = n_sam
        
        h_random = np.random.choice(h-template_size, sample_per_image, replace=False)
        w_random = np.random.choice(w-template_size, sample_per_image, replace=False)
        
        for j in range(sample_per_image):
            img_n = img[h_random[j]: template_size+h_random[j], w_random[j]: template_size+w_random[j]]
            hog_feats = hog(img_n, hog_cell_size).flatten()
            features_neg.append(hog_feats)
        
    if len(features_neg) > num_samples:
        index = np.random.choice(len(features_neg), num_samples, replace=False)
        features_neg = np.asarray(features_neg)[index]
    else:
        features_neg = np.asarray(features_neg)
    neg_examples = len(neg_train_image)
    
    np.save('features_neg', features_neg.copy())
    print('-> Save negative features to {}'.format('features_neg.npy'))
    #########################################
    ##          you code here              ##
    #########################################
            
    return features_neg, neg_examples

