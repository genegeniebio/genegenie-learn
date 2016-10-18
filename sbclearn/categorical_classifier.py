'''
synbiochem (c) University of Manchester 2015

synbiochem is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author: neilswainston
'''
from tensorflow.contrib import layers, learn
from tensorflow.contrib.learn import models
import tensorflow as tf


def classify(data, model_fn, steps=1000):
    classifier = learn.Estimator(model_fn=model_fn)
    classifier.fit(data[0], data[2], steps=steps)
    return classifier.predict(data[1])


def get_categ_model(n_classes):
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


def get_one_hot_categ_model(n_classes):
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
