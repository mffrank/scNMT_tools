import pytest
import os

from scnmttools import *

print(os.getcwd())

filenames = [
    'E6.5_Plate3_G6.tsv.gz',
    'E7.5_Plate1_A3.tsv.gz',
    'E7.5_Plate3_F5.tsv.gz',
    'E7.5_Plate4_B9.tsv.gz'
]

def test_data_avail():
    for f in filenames:
        assert os.path.exists(f), 'Example data %s not found'%f
