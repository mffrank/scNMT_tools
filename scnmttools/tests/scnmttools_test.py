import pytest
import os
import numpy as np

from scnmttools import io

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# print(os.getcwd())

filenames = [
    'E6.5_Plate3_G6.tsv.gz',
    'E7.5_Plate1_A3.tsv.gz',
    'E7.5_Plate3_F5.tsv.gz',
    'E7.5_Plate4_B9.tsv.gz'
]

def test_data_avail():
    for f in filenames:
        assert os.path.exists(os.path.join('data/', f)), 'Example data %s not found'%f

def test_read_tsv():
    f = io.read_tsv(os.path.join('data/', filenames[0]))
    assert f.shape == (49, 4), 'Expected input dimensions (49, 4) but got (%d,%d)'%f.shape

def test_make_genomic_index():
    assert np.all(io.make_genomic_index([1,1,2,2,4,6]) == np.array([0, 0, 1, 1, 2, 3]))

def test_collapse_strands():
    met = io.read_tsv(os.path.join('data/', filenames[2]))
    met_collapsed = io.collapse_strands(met)
    assert met_collapsed.shape == (58, 4), 'Expected collapsed dimensions (58, 4) but got (%d,%d)'%met_collapsed.shape
    assert met_collapsed.iloc[11,2] == 2, 'Expected 2 in position [11,2] but got %d'%met_collapsed.iloc[11,2]

