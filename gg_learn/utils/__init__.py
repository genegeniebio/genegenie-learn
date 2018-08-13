'''
sbclearn (c) University of Manchester 2017

sbclearn is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
from scipy import stats


def coeff_corr(i, j):
    '''coeff_correlation.'''
    _, _, r_value, _, _ = stats.linregress(i, j)
    return r_value**2
