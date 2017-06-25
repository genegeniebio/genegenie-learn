'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=ungrouped-imports
import json
import sys
import urllib

from sklearn.metrics import classification_report, confusion_matrix

from sbclearn.theanets import theanets_utils
import numpy as np


def get_data(project_id):
    '''Gets data.'''
    url = 'http://debrief.synbiochem.co.uk/data/' + project_id
    response = urllib.urlopen(url)
    data = json.loads(response.read())

    data = [(mutation['b_factors'], mutation['active'])
            for mutation in data['mutations']
            if 'b_factors' in mutation]

    return [np.array(arr) for arr in zip(*data)]


def main(args):
    '''main method.'''
    results = theanets_utils.k_fold_cross_valid(get_data(args[0]),
                                                regression=False)

    results = zip(*results)
    labels = [False, True]
    print confusion_matrix(results[0], results[1], labels)
    print classification_report(results[0], results[1], labels)


if __name__ == '__main__':
    main(sys.argv[1:])
