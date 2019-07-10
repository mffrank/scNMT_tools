import pandas as pd
import numpy as np
import re
from scipy import sparse
import anndata

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

def collapse_str(met):
    '''Sum reads of neighboring genomic locations.
    Caution: This function makes the assumption that both sister chromosomes in
    the cell are always methylated the same. It is recommended to use this if you
    are not interested in allele specific analyses.'''

    def collapse(met):
        locations = met.location.values
        # Find locations that have a neighbor
        neighbor_ind = met.loc[np.append(np.diff(locations) == 1, False)].index
        # Add the reads to the first of the two neighbors
        met.loc[neighbor_ind, 'met_reads'] += met.loc[neighbor_ind + 1, 'met_reads'].values
        met.loc[neighbor_ind, 'nonmet_reads'] += met.loc[neighbor_ind + 1, 'nonmet_reads'].values
        met.drop(neighbor_ind+1, inplace = True)
        return met

    met_grouped = met.groupby('chr')
    met = met_grouped.apply(lambda x: collapse(x))
    met.reset_index(drop = True, inplace = True)
    return met


def calculate_met_rate(
        met, binarize=True, collapse_strands=True, drop_ambiguous=True, 
        drop_rate_columns=False):
    '''Adds a rate column to methylation table

    Parameters:
    met: pd.DataFrame with columns met_reads, unmet_reads
    binarize: boolean, whether to return a binary rate
    collapse_strands: boolean, whether to sum reads from neighboring 
                      methylation sites
    drop_ambiguous: boolean, whether to drop sites with 0.5 methylation rate
    drop_rate_columns: boolean, whether columns met_reads and nonmet_reads 
                       should be removed

    Returns:
    pd.DataFrame with added rate column
    '''
    if collapse_strands:
        met = collapse_str(met)

    # Calculate the rate
    met['met_rate'] = met['met_reads'] / (met['met_reads'] + met['nonmet_reads'])
    if drop_ambiguous:
        met = met.loc[met['met_rate'] != 0.5]
        met.reset_index(inplace=True, drop=True)
    if binarize:
        met['met_rate'] = (met['met_rate'] > 0.5) * 1
    if drop_rate_columns:
        met.drop(['met_reads', 'nonmet_reads'], inplace=True, axis=1)
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

def read_data(
        files,
        per_chromosome=False,
        chromosomes=[None],
        binarize=True,
        collapse_strands=True,
        drop_ambiguous=True,
        outfile=None):
    # Get cellnames from input files
    cellnames = [re.sub('\\.csv|\\.txt|\\.tsv|\\.gz','',os.path.basename(file)) for file in files]

    # Find which chromosomes to read if per chromosome
    if per_chromosome:
        if chromosomes[0] is None:
            # Read first file to find which chromosomes exist
            chromosomes = pd.unique(pd.read_csv(files[0],
                                                sep = '\t',
                                                usecols = [0],
                                                skiprows = 1,
                                                header = None).iloc[:,0])

    # Read Files and generate anndata formatted files
    for chromosome in chromosomes:
        allmet = read_cells(files, chromosome)
        allmet = [calculate_met_rate(met,
                                    binarize=binarize,
                                    collapse_strands=collapse_strands,
                                    drop_ambiguous=drop_ambiguous) for met in allmet]
        if verbose: print('Concatenating...')
        allmet = pd.concat(allmet, keys = cellnames, copy = False)
        allmet.sort_values(['chr', 'location'], inplace=True)
        allmet.met_rate = allmet.met_rate + 1

        print('Constructing reference index...')
        allmet['ind'] = make_genomic_index(allmet.location.values)
        rowcoord, rownames = pd.factorize(allmet.index.get_level_values(0)) # Numerical representation of the index objects
        print('Constructing sparse matrix...')
        obs = pd.DataFrame(index = rownames)
        var = allmet.drop_duplicates(subset='ind')[['chr', 'location']].reset_index(drop = True)
        metmat = sparse.coo_matrix((allmet.met_rate,
                                    (rowcoord, allmet['ind'])),
                                   dtype = np.int64,
                                   shape=(obs.shape[0], var.shape[0]))
        metmat = metmat.tocsc()
        print('Constructing anndata object...')
        a = anndata.AnnData(X = metmat, obs = obs, var = var, dtype = np.int32)
        if outfile is not None:
            print('Writing h5ad file')
            a.write(outfile)
            a.file.close()
        return(a)

def read_cells(files, chromosome = None, header = True, verbose = True):
    allmet = []
    if verbose: print('Reading files...')
    for cell in files:
        try:
            met = read_tsv(cell, header=header)
            if verbose: print(cell + ": Coverage: " + str(met.shape[0]))
            if chromosome is not None:
                met = met.loc[met.chr == chromosome]
                if verbose: print(cell +
                                  ": Coverage of Chromosome %s: "%chromosome +
                                  str(met.shape[0]))
            allmet.append(met)
        except:
            print('Could not read file %s. Skipping...'%cell)
    return allmet

