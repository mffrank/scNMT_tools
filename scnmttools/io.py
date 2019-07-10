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
                      delimiter='\t',
                      header=None,
                      skiprows= 1 if header else 0, # Skip the header row
                      usecols = [0,1,2,3],
                      names=['chr', 'location','met_reads', 'nonmet_reads'],
                      dtype = {'chr':'category',
                               'location':np.int64,
                               'met_reads':np.int32,
                               'nonmet_reads': np.int32})
    return met

def collapse_strands(met):
    '''Sum reads of neighboring genomic locations.
    Caution: This function makes the assumption that both sister chromosomes in
    the cell are always methylated the same. It is recommended to use this if you
    are not interested in allele specific analyses.'''

    def collapse(met):
        locations = met.location.values
        # Find locations that have a neighbor
        neighbor_ind = met.index[np.append(np.diff(locations) == 1, False)]
        # Add the reads to the first of the two neighbors
        met.loc[neighbor_ind, 'met_reads'] += met.loc[neighbor_ind + 1, 'met_reads'].values
        met.loc[neighbor_ind, 'nonmet_reads'] += met.loc[neighbor_ind + 1, 'nonmet_reads'].values
        met.drop(neighbor_ind+1, inplace = True)
        # met.reset_index(inplace=True)
        return met

    met_grouped = met.groupby('chr')
    met = met_grouped.apply(lambda x: collapse(x))
    met.reset_index(drop = True, inplace = True)
    return met


def calculate_met_rate(met, binarize=True, collapse_strands=True, drop_ambiguous=True):
    '''Adds a rate column to methylation table

    Parameters:
    met: pd.DataFrame with columns met_reads, unmet_reads
    binarize: boolean, whether to return a binary rate
    collapse_strands: boolean, whether to sum reads from neighboring methylation sites
    drop_ambiguous: boolean, whether to drop sites with 0.5 methylation rate

    Returns:
    pd.DataFrame with added rate column
    '''
    if collapse_strands:
        met = collapse_strands(met)

    # Calculate the rate
    met['met_rate'] = met['met_reads'] / (met['met_reads'] + met['nonmet_reads'])
    if drop_ambiguous:
        met = met.loc[met['met_rate'] != 0.5]
        met.reset_index(inplace=True)
    if binarize:
        met['met_rate'] = (met['met_rate'] > 0.5) * 1
    return met

def make_genomic_index(location):
    '''Generate an index for a numpy array of redundant genomic locations
    Parameters:
    location: numpy array of genomic locations

    Returns:
    np.array
    '''
    d = np.diff(location)
    i = np.insert(np.cumsum(d != 0), 0, 0)
    return i
