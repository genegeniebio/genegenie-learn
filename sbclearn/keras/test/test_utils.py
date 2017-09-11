'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
import unittest

from keras.utils import to_categorical
from sklearn import metrics, model_selection
from sklearn.datasets.samples_generator import make_blobs
# from sklearn.metrics import confusion_matrix

from sbclearn.keras.utils import Classifier
import numpy as np


class TestClassifier(unittest.TestCase):
    '''Tests the Classifier class.'''

    def test_classify(self):
        '''Tests the classify method.'''
        centers = 5

        x_data, y_data = make_blobs(n_samples=1000,
                                    centers=centers,
                                    n_features=3,
                                    cluster_std=1.0,
                                    random_state=0)

        y_data = to_categorical(y_data, num_classes=centers)

        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x_data, y_data, test_size=0.2)

        classifier = Classifier(x_train, y_train)
        classifier.train()

        y_pred = classifier.predict(x_test)
        y_pred = np.array([[round(val) for val in pred] for pred in y_pred])

        # print confusion_matrix(y_test, y_pred)

        self.assertTrue(metrics.accuracy_score(y_test, y_pred) > 0.9)
