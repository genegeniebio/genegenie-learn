from sklearn import metrics, cross_validation
from tensorflow.contrib import layers
from tensorflow.contrib import learn
from tensorflow.contrib.learn import models
import pandas

import numpy as np
import tensorflow as tf


def classify(data, model_fn, steps=1000):
    classifier = learn.Estimator(model_fn=model_fn)
    classifier.fit(data[0], data[2], steps=steps)
    return classifier.predict(data[1])


def _get_categorical_model(n_classes):
    def _categorical_model(features, target):
        '''Perform categorical model.'''
        target = tf.one_hot(target, 2, 1.0, 0.0)
        features = learn.ops.categorical_variable(features, n_classes,
                                                  embedding_size=3,
                                                  name='embarked')
        prediction, loss = models.logistic_regression(tf.squeeze(features,
                                                                 [1]),
                                                      target)
        train_op = layers.optimize_loss(loss,
                                        tf.contrib.framework.get_global_step(),
                                        optimizer='SGD',
                                        learning_rate=0.05)
        return tf.argmax(prediction, dimension=1), loss, train_op

    return _categorical_model


def _get_one_hot_categorical_model(n_classes):
    def _one_hot_categorical_model(features, target):
        '''Perform one hot model.'''
        target = tf.one_hot(target, 2, 1.0, 0.0)
        features = tf.one_hot(features, n_classes, 1.0, 0.0)
        prediction, loss = models.logistic_regression(tf.squeeze(features,
                                                                 [1]),
                                                      target)
        train_op = layers.optimize_loss(loss,
                                        tf.contrib.framework.get_global_step(),
                                        optimizer='SGD',
                                        learning_rate=0.01)
        return tf.argmax(prediction, dimension=1), loss, train_op

    return _one_hot_categorical_model


def _get_data():
    '''Gets data.'''
    data = pandas.read_csv('../data/titanic_train.csv')
    x = data[['Embarked']]
    y = data['Survived']
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        x, y, test_size=0.2)

    cat_processor = learn.preprocessing.CategoricalProcessor()
    x_train = np.array(list(cat_processor.fit_transform(x_train)))
    x_test = np.array(list(cat_processor.transform(x_test)))

    return x_train, x_test, y_train, y_test


def main():
    '''main method.'''
    data = _get_data()
    n_classes = len(set([val for vals in data[0] for val in vals]))

    predict = classify(data, _get_categorical_model(n_classes))
    print 'Accuracy: {0}'.format(metrics.accuracy_score(predict, data[3]))
    print 'ROC: {0}'.format(metrics.roc_auc_score(predict, data[3]))

    predict = classify(data, _get_one_hot_categorical_model(n_classes))
    print 'Accuracy: {0}'.format(metrics.accuracy_score(predict, data[3]))
    print 'ROC: {0}'.format(metrics.roc_auc_score(predict, data[3]))

if __name__ == '__main__':
    main()
