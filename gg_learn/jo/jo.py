'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=wrong-import-order
import sys

from keras.utils import to_categorical
from sklearn import metrics, model_selection
from sklearn.metrics import confusion_matrix

from gg_learn.keras import Classifier
import numpy as np
import pandas as pd


def get_data(filename):
    '''Gets data.'''
    df = pd.read_csv(filename, sep='\t')
    x_data = np.array(sbclearn.get_aa_props(df['seq'].tolist()))
    return (x_data, df['bin'])


def classify(x_data, y_data):
    '''Runs the classify method.'''
    y_data = to_categorical(y_data, num_classes=len(set(y_data)))

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x_data, y_data, test_size=0.05)

    classifier = Classifier(x_train, y_train)
    classifier.train(learn_rate=0.001, epochs=200)

    y_pred = classifier.predict(x_test)
    y_pred = np.array([[round(val) for val in pred] for pred in y_pred])

    print(confusion_matrix([np.argmax(y) for y in y_test],
                           [np.argmax(y) for y in y_pred]))
    print(metrics.accuracy_score(y_test, y_pred))


def regression(x_data, y_data):
    '''Runs the regression method.'''
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x_data, y_data, test_size=0.05)

    classifier = Classifier(x_train, y_train)
    classifier.train(learn_rate=0.001, epochs=200)

    y_pred = classifier.predict(x_test)
    y_pred = np.array([[round(val) for val in pred] for pred in y_pred])

    print(confusion_matrix([np.argmax(y) for y in y_test],
                           [np.argmax(y) for y in y_pred]))
    print(metrics.accuracy_score(y_test, y_pred))


def main(args):
    '''main method.'''
    x_data, y_data = get_data(args[0])
    classify(x_data, y_data)


if __name__ == '__main__':
    main(sys.argv[1:])
