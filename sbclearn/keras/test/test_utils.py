'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
import unittest

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics, model_selection
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sbclearn.keras.utils import Classifier
import numpy as np
import pandas as pd


# from sklearn.metrics import confusion_matrix
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


class TestRegressor(unittest.TestCase):
    '''Tests the Regressor class.'''

    def test_regression(self):
        '''Tests the regression method.'''
        df = pd.read_csv('housing.csv', delim_whitespace=True,
                         header=None)
        dataset = df.values
        x_data = dataset[:, 0:-1]
        y_data = dataset[:, -1]

        # create model

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        # evaluate model with standardized dataset
        regressor = KerasRegressor(
            build_fn=get_model, nb_epoch=100, batch_size=5)

        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_val_score(regressor, x_data, y_data, cv=kfold)
        print "Results: %.2f (%.2f) MSE" % (results.mean(), results.std())

        self.assertTrue(results.mean() > 50)


def get_model():
    '''Gets model.'''
    model = Sequential()
    model.add(Dense(64,
                    input_dim=13,
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
