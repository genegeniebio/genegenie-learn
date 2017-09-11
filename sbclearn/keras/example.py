'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=no-member
from keras.utils import to_categorical

from sbclearn.keras.utils import classify
import numpy as np


def main():
    '''main method.'''
    num_classes = 10
    x_data = np.random.random((1000, 20))
    y_data = to_categorical(np.random.randint(num_classes, size=(1000, 1)),
                            num_classes=num_classes)

    print classify((x_data, y_data))


if __name__ == '__main__':
    main()
