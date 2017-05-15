'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
import unittest

from sklearn import datasets, metrics, model_selection
from sklearn.datasets.samples_generator import make_blobs

from sbclearn.theanets import theanets_utils


class TestClassifier(unittest.TestCase):
    '''Tests the Classifier class.'''

    def test_classify(self):
        '''Tests the classify method.'''
        x_data, y_data = make_blobs(n_samples=1000, centers=5, n_features=3,
                                    cluster_std=1.0, random_state=0)

        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(x_data, y_data, test_size=0.2)

        classifier = theanets_utils.Classifier(x_train, y_train)
        classifier.train()
        y_pred = classifier.predict(x_test)

        self.assertTrue(metrics.accuracy_score(y_test, y_pred) > 0.9)


class TestRegressor(unittest.TestCase):
    '''Tests the Regressor class.'''

    def test_predict(self):
        '''Tests the predict method.'''
        dataset = datasets.load_diabetes()

        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(dataset.data, dataset.target,
                                             test_size=0.2)

        regressor = theanets_utils.Regressor(x_train, y_train)
        regressor.train()
        y_pred = regressor.predict(x_test)

        self.assertTrue(metrics.r2_score(y_test, y_pred) > 0.3)
