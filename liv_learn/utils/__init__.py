'''
(c) University of Liverpool 2019

All rights reserved.

@author: neilswainston
'''
from scipy import stats


def coeff_corr(i, j):
    '''coeff_correlation.'''
    _, _, r_value, _, _ = stats.linregress(i, j)
    return r_value**2
