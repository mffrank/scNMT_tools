import pytest
import os

# from scnmttools import *
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
