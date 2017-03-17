import cv2
import glob
import time
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from detection_functions.feature_extraction import *

def train_classifier(learn_new_classifier=False, svc_pickle='./detection_functions/svc_pickle.pickle', X_scaler_pickle='./detection_functions/X_scaler_pickle.pickle'):

    try:
        if learn_new_classifier:
            raise Exception('Flag for new learning is True')
        svc = open(svc_pickle, 'rb')
        svc = pickle.load(svc)
        X_scaler = open(X_scaler_pickle, 'rb')
        X_scaler = pickle.load(X_scaler)

    except:
        # Read in notcars
        images = glob.glob('./data/non-vehicles/*/*.png')
        notcars = []
        for image in images:
            notcars.append(image)

        # Read in cars
        images = glob.glob('./data/vehicles/*/*.png')
        cars = []
        for image in images:
            cars.append(image)

        ### TODO: Tweak these parameters and see how the results change.
        color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 12  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32  # Number of histogram bins
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off


        car_features = extract_features(cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(notcars, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = SVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()

        classifier_values = {'svc': svc, 'X_scaler': X_scaler}
        with open(svc_pickle, 'wb') as file:
            pickle.dump(svc, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(X_scaler_pickle, 'wb') as file:
            pickle.dump(X_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

    return svc, X_scaler

if __name__ == "__main__":
    train_classifier()