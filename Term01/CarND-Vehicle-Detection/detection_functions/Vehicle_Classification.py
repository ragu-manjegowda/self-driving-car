import glob
import time
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

from detection_functions.feature_extraction import *

class Vehicle_Classification():
    def __init__(self):
        self.color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 12  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = '0'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 64# Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off

        self.classifier = None
        self.X_scaler = None

    def train_classifier(self, learn_new_classifier=False, classifier_pickle='./detection_functions/svc_pickle.pickle', X_scaler_pickle='./detection_functions/X_scaler_pickle.pickle'):
        try:
            if learn_new_classifier:
                raise Exception('Flag for new learning is True')
            self.classifier = open(classifier_pickle, 'rb')
            self.classifier = pickle.load(self.classifier)
            self.X_scaler = open(X_scaler_pickle, 'rb')
            self.X_scaler = pickle.load(self.X_scaler)

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

            car_features = extract_features(cars, color_space=self.color_space,
                                            spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                            orient=self.orient, pix_per_cell=self.pix_per_cell,
                                            cell_per_block=self.cell_per_block,
                                            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                            hist_feat=self.hist_feat, hog_feat=self.hog_feat)
            notcar_features = extract_features(notcars, color_space=self.color_space,
                                               spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                               orient=self.orient, pix_per_cell=self.pix_per_cell,
                                               cell_per_block=self.cell_per_block,
                                               hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                               hist_feat=self.hist_feat, hog_feat=self.hog_feat)

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

            print('Using:', self.orient, 'orientations', self.pix_per_cell,
                  'pixels per cell and', self.cell_per_block, 'cells per block')
            print('Feature vector length:', len(X_train[0]))
            # Use a SVC
            svc = LinearSVC()
            # Check the training time for the SVC
            t = time.time()
            svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t = time.time()


            with open(classifier_pickle, 'wb') as file:
                pickle.dump(svc, file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(X_scaler_pickle, 'wb') as file:
                pickle.dump(X_scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

            self.classifier = svc
            self.X_scaler = X_scaler