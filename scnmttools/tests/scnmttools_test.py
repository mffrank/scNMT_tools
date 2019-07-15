import pytest
import os
import numpy as np
import pandas as pd
import gzip
import re

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

def test_to_deepcpg_format():
    met = io.read_tsv(os.path.join('data/', filenames[2]))
    deepcpg_format = io.calculate_met_rate(
        met, binarize=True, enable_collapse_strands=False, drop_ambiguous=True, 
        drop_reads_columns=True)
    assert deepcpg_format.shape[1] == 3, 'Expected 3 columns in deepcpg format, but got %d' % deepcpg_format.shape[1]
    assert deepcpg_format.shape[0] == 62, 'Expected 62 columns to pass filtering, but got %d' % deepcpg_format.shape[0]

def test_to_bed_graph_format():
    met = io.read_tsv(os.path.join('data/', filenames[2]))
    nrow_before = met.shape[0]
    bedgraph = io.to_bed_graph_format(met)
    assert bedgraph.shape[1] == 4, 'Expected bed graph format to have 4 columns, but got %d' %  bedgraph.shape[1]
    assert bedgraph.shape[0] == nrow_before, \
        'Expected number of rows to be unchaged, but changed from %d to %d' % (nrow_before , bedgraph.shape[0])


def validate_bed_graph_file(filename, expected_nrow):
    assert os.path.exists(filename), 'Bedgraph file does not exist'
    with open(filename, 'r') as f:
        header = f.readline()
        assert header.startswith('track type=bedGraph'), 'Bedgraph file does not have appropriate header'
        bedgraph = pd.read_csv(filename, delimiter='\t', header=None, skiprows=1)
        assert bedgraph.shape[1] == 4, 'Expected bed graph format to have 4 columns, but got %d' %  bedgraph.shape[1]
        assert bedgraph.shape[0] == expected_nrow, \
            'Expected number of rows to be unchaged, but changed from %d to %d' % (expected_nrow, bedgraph.shape[0])
        assert np.all(bedgraph.columns[1] == bedgraph.columns[2]-1), 'Locations in bedgraph format are wrong. Can only be 1 apart.'


def test_write_bed_graph(tmpdir):
    met = io.read_tsv(os.path.join('data/', filenames[2]))
    nrow_before = met.shape[0]

    outfile = os.path.join(tmpdir, filenames[2]+'1.bedGraph')
    assert not os.path.exists(outfile), 'Temp output file already exists before test at %s' % outfile
    io.write_bed_graph(met, outfile, convert=True)
    validate_bed_graph_file(outfile, nrow_before)
    # Make sure it wasn't gzip compressed
    try:
        with gzip.open(outfile) as f:
            f.read()
        assert False, 'Output file should NOT be gzip compressed, but seems to have been.'
    except OSError:
        pass

    '''Now testing with compression (should infer gzip from filename)'''
    outfile = os.path.join(tmpdir, filenames[2]+'1.bedGraph.gz')
    assert not os.path.exists(outfile), 'Temp output file already exists before test at %s' % outfile
    io.write_bed_graph(met, outfile, convert=True)
    try:
        with gzip.open(outfile) as f:
            f.read()
    except OSError:
        assert False, 'Output file should be gzip compressed, but seems to not be that way.'

    '''Now testing with manual conversion'''
    outfile = os.path.join(tmpdir, filenames[2]+'2.bedGraph')
    assert not os.path.exists(outfile), 'Temp output file already exists before test at %s' % outfile
    met = io.to_bed_graph_format(met)
    io.write_bed_graph(met, outfile, convert=False)
    validate_bed_graph_file(outfile, nrow_before)



def validate_samtools_faidx_region_file(filename, expected_nrow):
    assert os.path.exists(filename), 'Region file does not exist'
    with open(filename, 'r') as f:
        prog = re.compile('^[^:]+:\\d+-\\d+$')
        for i in range(expected_nrow):
            line = f.readline()
            assert prog.match(line), 'Invalid format in %s' % line


def test_samtools_faidx_region(tmpdir):
    met = io.read_tsv(os.path.join('data/', filenames[2]))
    nrow_before = met.shape[0]

    outfile = os.path.join(tmpdir, filenames[2]+'_region.txt')
    assert not os.path.exists(outfile), 'Temp output file already exists before test at %s' % outfile
    io.write_samtools_faidx_region_file(met, outfile)
    validate_samtools_faidx_region_file(outfile, nrow_before)
