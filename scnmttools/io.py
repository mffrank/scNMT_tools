import pandas as pd
import numpy as np

def read_tsv(filename, header=True):
    '''Reads a standardized tsv file (Must have the following columns:
    Chromosome, Genomic position, Methylated reads, Unmethylated reads)

    Parameters:
    filename: Path to file
    header: Whether the file has a header row

    Returns:
    pd.DataFrame
    '''
    met = pd.read_csv(filename,
                      delimiter = '\t',
                      header = None,
                      skiprows= 1 if header else 0, # Skip the header row
                      usecols = [0,1,2,3],
                      names=['chr', 'location','met', 'unmet'],
                      dtype = {'chr':'category',
                               'location':np.int64,
                               'metfrac':np.int32,
                               'unmet': np.int32})
    return(met)

def make_genomic_index(location):
    '''Generate an index for a numpy array of redundant genomic locations
    Parameters:
    location: numpy array of genomic locations

    Returns:
    np.array
    '''
    d = np.diff(location)
    i = np.insert(np.cumsum(d != 0), 0, 0)
    return(i)
