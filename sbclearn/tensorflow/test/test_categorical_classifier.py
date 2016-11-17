'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author: neilswainston
'''
import os
import unittest

from sbclearn.tensorflow import categorical_classifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from tensorflow.contrib import learn
import numpy as np
import pandas


class Test(unittest.TestCase):

    def test_categorical_model(self):
        '''Tests categorical_model method.'''
        data = _get_data()
        predict = categorical_classifier.classify(
            data, categorical_classifier.get_categ_model(data[4]))
        self.assertGreater(accuracy_score(predict, data[3]), 0.5)

    def test_one_hot_categorical_model(self):
        '''Tests one_hot_categorical_model method.'''
        data = _get_data()
        predict = categorical_classifier.classify(
            data, categorical_classifier.get_one_hot_categ_model(data[4]))
        self.assertGreater(accuracy_score(predict, data[3]), 0.5)


def _get_data():
    '''Gets data.'''
    directory = os.path.dirname(os.path.realpath(__file__))
    data = pandas.read_csv(os.path.join(directory, 'titanic_train.csv'))
    x = data[['Embarked']]
    y = data['Survived']
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        x, y, test_size=0.2)

    cat_processor = learn.preprocessing.CategoricalProcessor()
    x_train = np.array(list(cat_processor.fit_transform(x_train)))
    x_test = np.array(list(cat_processor.transform(x_test)))

    n_classes = len(set([val for vals in x_train for val in vals]))

    return x_train, x_test, y_train, y_test, n_classes

if __name__ == "__main__":
    unittest.main()
