'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author: neilswainston
'''
import os
import unittest

from sbclearn.tensorflow import classifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas


class Test(unittest.TestCase):

    def test_logistic_regression(self):
        '''Tests logistic_regression method.'''
        data = _get_data()
        predict = classifier.logistic_regression(data)
        self.assertGreater(accuracy_score(predict, data[3]), 0.5)

    def test_linear_classifier(self):
        '''Tests linear_classifier method.'''
        data = _get_data()
        predict = classifier.linear_classifier(data)
        self.assertGreater(accuracy_score(predict, data[3]), 0.5)

    def test_dnn_classifier(self):
        '''Tests dnn_classifier method.'''
        data = _get_data()
        predict = classifier.dnn_classifier(data)
        self.assertGreater(accuracy_score(predict, data[3]), 0.5)

    def test_dnn_tanh(self):
        '''Tests dnn_tanh method.'''
        data = _get_data()
        predict = classifier.dnn_tanh(data)
        self.assertGreater(accuracy_score(predict, data[3]), 0.5)


def _get_data(test_size=0.2):
    '''Reads data and splits into training and test.'''
    directory = os.path.dirname(os.path.realpath(__file__))
    train = pandas.read_csv(os.path.join(directory, 'titanic_train.csv'))
    x, y = train[['Age', 'SibSp', 'Fare']].fillna(0), train['Survived']
    return train_test_split(x, y, test_size=test_size)

if __name__ == "__main__":
    unittest.main()
