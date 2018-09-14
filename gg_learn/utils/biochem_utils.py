'''
(c) GeneGenie Bioinformatics Ltd. 2018

Licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
from Bio import SeqIO
from Bio.PDB.Polypeptide import aa1, d1_to_index
from rdkit.Chem import AllChem
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd


def fasta_to_df(filename):
    ''' Read fasta file, return DataFrame.'''
    data = [[rec.id, rec.name, rec.description, str(rec.seq)]
            for rec in SeqIO.to_dict(SeqIO.parse(filename, 'fasta')).values()]

    return pd.DataFrame(data, columns=['id', 'name', 'description', 'seq'])


def get_onehot_seq(seqs):
    '''Encode an amino acid sequence as a tensor by concatenating one-hot
    encoding up to desired depth.'''
    # Pad sequences at beginning to ensure consistent length:
    max_seq_len = max([len(seq) for seq in seqs])
    X = [seq.rjust(max_seq_len) for seq in seqs]

    # Encode 20 amino acids plus a default for non-amino acid:
    encoder = OneHotEncoder()
    encoder.fit([[idx] for idx in _aa_index(aa1 + ' ')])
    return encoder.fit_transform(X).toarray()


def get_ordinal_seq(seqs):
    '''Encode an amino acid sequence as a tensor by concatenating ordinal
    encoding up to desired depth.'''
    return np.array([_aa_index(seq) for seq in seqs])


def get_ordinal_seq_padded(seqs):
    '''Encode an amino acid sequence as a tensor by concatenating ordinal
    encoding up to desired depth.'''
    # Left pad seqs with spaces to ensure consistent length,
    # and return as indexes:
    max_seq_len = max([len(seq) for seq in seqs])
    seqs = [seq.rjust(max_seq_len) for seq in seqs]
    return get_ordinal_seq(seqs)


def get_tensor_chem(mols, fingerprint_size, depth):
    '''Encode a chemical as a tensor by concatenating fingerprints
    up to desired depth.'''
    X = np.zeros((len(mols), fingerprint_size, depth))

    for i, mol in enumerate(mols):
        for k in range(0, depth):
            fpix = _chem_fp(mol, fingerprint_size, k + 1, k + 1)
            X[i, :, k] = fpix

    return X.reshape(X.shape[0], -1)


def get_tensor_reac(reacs, fingerprint_size, depth, token_size=1, min_path=1,
                    max_path=5):
    '''Encode a reaction as a tensor by concatenating fingerprints
    up to desired depth.'''
    X = np.zeros((len(reacs), fingerprint_size, depth * token_size))

    for i, reac in enumerate(reacs):
        fpix = _reac_fp(reac, min_path, max_path)

        for l in range(0, len(fpix)):
            for k in range(0, depth):
                X[i, l, fpix[l + k] + token_size * k] = 1

    return X


def _aa_index(seq):
    '''Convert amino acid to numerical index.'''
    return [d1_to_index.get(aa, -1) + 1 for aa in seq]


def _chem_fp(mol, fingerprint_size, min_path=1, max_path=5):
    '''Get chemical fingerprint.'''
    fpix = AllChem.RDKFingerprint(
        mol, minPath=min_path, maxPath=max_path, fpSize=fingerprint_size)

    return [int(x) for x in fpix.ToBitString()]


def _reac_fp(reac, min_path=1, max_path=5):
    ''' Reaction fingerprint '''
    left, right = reac

    left = [_chem_fp(m, min_path, max_path) for m in left]
    right = [_chem_fp(m, min_path, max_path) for m in right]

    lfp = left[0]

    for m in left:
        lfp = lfp | m

    rfp = right[0]

    for m in right:
        rfp = rfp | m

    rfp = lfp ^ rfp

    return rfp
