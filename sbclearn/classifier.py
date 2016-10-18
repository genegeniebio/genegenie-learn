'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author: neilswainston
'''
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.contrib import framework, layers, learn
import pandas
import tensorflow


def logistic_regression(data):
    '''Perform logistic regression.'''
    log_res = LogisticRegression()
    log_res.fit(data[0], data[2])
    print(accuracy_score(log_res.predict(data[1]), data[3]))


def linear_classifier(data, learning_rate=0.05, batch_size=128, steps=500):
    '''Perform linear classifier.'''
    classes, feat_col, optimizer = _get_classifier_params(data, learning_rate)

    classifier = learn.LinearClassifier(n_classes=classes,
                                        feature_columns=feat_col,
                                        optimizer=optimizer)

    _classify(data, classifier, batch_size, steps)


def dnn_classifier(data, learning_rate=0.1, hidden_units=[10, 20, 10],
                   batch_size=128, steps=500):
    '''Perform 3 layer neural network with rectified linear activation.'''
    classes, feat_col, optimizer = _get_classifier_params(data, learning_rate)

    classifier = learn.DNNClassifier(hidden_units=hidden_units,
                                     n_classes=classes,
                                     feature_columns=feat_col,
                                     optimizer=optimizer)

    _classify(data, classifier, batch_size, steps)


def dnn_tanh(data, batch_size=128, steps=100):
    '''Perform 3 layer neural network with hyperbolic tangent activation.'''
    classifier = learn.Estimator(model_fn=_dnn_tanh)
    _classify(data, classifier, batch_size, steps)


def _dnn_tanh(features, target):
    '''Perform 3 layer neural network with hyperbolic tangent activation.'''
    target = tensorflow.one_hot(target, 2, 1.0, 0.0)
    logits = layers.stack(features, layers.fully_connected, [10, 20, 10],
                          activation_fn=tensorflow.tanh)
    prediction, loss = learn.models.logistic_regression(logits, target)
    train_op = layers.optimize_loss(loss,
                                    framework.get_global_step(),
                                    optimizer='SGD',
                                    learning_rate=0.05)
    return tensorflow.argmax(prediction, dimension=1), loss, train_op


def _get_classifier_params(data, learning_rate):
    '''Gets the classifier parameters.'''
    classes = len(set(list(data[2]._values) + list(data[3]._values)))
    feat_col = learn.infer_real_valued_columns_from_input(data[0])
    optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate)

    return classes, feat_col, optimizer


def _classify(data, classifier, batch_size, steps):
    '''Perfoms classification.'''
    classifier.fit(data[0], data[2], batch_size=batch_size, steps=steps)
    print(accuracy_score(classifier.predict(data[1]), data[3]))


def _get_data(test_size=0.2):
    '''Reads data and splits into training and test.'''
    train = pandas.read_csv('../data/titanic_train.csv')
    x, y = train[['Age', 'SibSp', 'Fare']].fillna(0), train['Survived']
    return train_test_split(x, y, test_size=test_size)


def main():
    '''main method.'''
    data = _get_data()
    logistic_regression(data)
    linear_classifier(data)
    dnn_classifier(data)
    dnn_tanh(data)

if __name__ == '__main__':
    main()
