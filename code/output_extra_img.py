# Sliding window face detection with linear SVM
import numpy as np
import os
import pdb
from cyvlfeat.hog import hog
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import draw, exposure

from get_positive_features import get_positive_features
from get_random_negative_features import get_random_negative_features
from svm_classify import svm_classify
from report_accuracy import report_accuracy
from run_detector import run_detector
from skimage.io import imread
import pickle
import cv2


def output_extra_img(bboxes, confidences, image_ids, test_scn_path, fig_path):
    print(image_ids)
    for img_id in np.unique(image_ids):
        cur_test_image = imread(os.path.join(test_scn_path, img_id))
        # cur_test_image[:, :, 1], cur_test_image[:, :, 2] = cur_test_image[:, :, 2], cur_test_image[:, :, 1] 
        # cur_test_image[:, :, 2], cur_test_image[:, :, 0] = cur_test_image[:, :, 0], cur_test_image[:, :, 2] 
        cur_detections = [idx for idx, image_id in enumerate(image_ids) if img_id == image_id[0]]
        cur_bboxes = bboxes[cur_detections, :]
        cur_confidences = confidences[cur_detections]
        fig = plt.figure(15)
        plt.imshow(cur_test_image)

        num_detections = len(cur_detections)
        x_pos = [0, 2, 2, 0, 0]
        y_pos = [1, 1, 3, 3, 1]

        for j in range(num_detections):
            bb = cur_bboxes[j, :]
            plt.plot(bb[x_pos], bb[y_pos], color='r', linewidth=2, linestyle='--')

        plt.axis('image')
        plt.axis('off')
        plt.title('image: "{}"'.format(img_id, 'interpreter', 'none'))

        # plt.show()
        # plt.pause(0.1)  # let's ui rendering catch up
        fig.savefig('{}/detections_{}'.format(fig_path, img_id))
        plt.close()



def main():
    data_path = '../data'
    train_path_pos = os.path.join(data_path, 'caltech_faces/Caltech_CropFaces')
    non_face_scn_path = os.path.join(data_path, 'train_non_face_scenes')
    test_scn_path = os.path.join(data_path, 'extra_test_scenes')
    fig_path = '../results/extra/'

    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    feature_params = {'template_size': 36,
                      'hog_cell_size': 6}

    # features_pos = get_positive_features(train_path_pos, feature_params)
    # num_negative_examples = 10000
    # features_neg, neg_examples = get_random_negative_features(non_face_scn_path, feature_params, num_negative_examples)

    # ## Step 2. Train classifier
    # features_total = np.concatenate([features_pos,features_neg], axis=0)
    # labels = np.concatenate([np.ones((features_pos.shape[0], 1)), -np.ones((features_neg.shape[0], 1))],
    #                         axis=0)

    # model = svm_classify(features_total, labels)

    if os.path.isfile('SVM_linear_3.pkl'):
        print('-> Restore exist model from {}'.format('SVM_linear_3.pkl'))
        with open('SVM_linear.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        print('-> Please run proj4.py first.')
        exit(0)

    # print('Initial classifier performance on train data:')
    # confidences = model.decision_function(features_total)
    # label_vector = labels
    # tp_rate, fp_rate, tn_rate, fn_rate = report_accuracy(confidences, label_vector)

    # non_face_confs = confidences[label_vector.ravel()<0]
    # face_confs = confidences[label_vector.ravel()>0]
    # fig2 = plt.figure(2)
    # # plt.hold(True)
    # plt.plot(np.arange(non_face_confs.size), np.sort(non_face_confs), color='g')
    # plt.plot(np.arange(face_confs.size), np.sort(face_confs), color='r')
    # plt.plot([0, non_face_confs.size], [0,0], color='b')
    # plt.hold(False)
    
    bboxes, confidences, image_ids = run_detector(test_scn_path, model, feature_params, is_single=True)
    # threshold = 0.0
    # print(confidences)
    # filtered_idx = np.argwhere(confidences>threshold)
    # filtered_idx = filtered_idx[:, 0]
    # print('filtered_idx: ', filtered_idx.shape)
    # print('bboxes.shape: ', bboxes.shape)
    # bboxes = bboxes[filtered_idx]
    # confidences = confidences[filtered_idx]
    # image_ids = image_ids[filtered_idx]
    # print('bboxes.shape: ', bboxes.shape)

    output_extra_img(bboxes, confidences, image_ids, test_scn_path, fig_path)


if __name__=="__main__":
    main()