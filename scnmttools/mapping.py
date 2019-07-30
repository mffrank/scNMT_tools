import pandas as pd
import numpy as np
import re

def map_to_regions(cpg_locations, regions, nonevalue=-1, is_sorted=False):
    ''' Takes a list of cpg sites, and a list of genomic regions, and maps 
    the sites to the regions in linear time (O(n+m), where n is the number 
    of locations and m the number of regions). If lists are not presorted,
    O(n log(n)) time is required because we need to sort first.
    
    A few important prerequisites on the input: 
      * Both dataframes need to have the columns in the order
        specified below, since the function uses itertuples to iterate
        quickly. Column names are irrelevant.
      * Both dataframes need to be sorted by the chromosome column, such
        that we can compare the two. So, if they are both using strings
        for the chromosome, they should both be sorted lexographically.
        If you are unsure about this, set is_sorted to false, so this
        function will perform the correct sorting for you.
      * Both dataframes are allowed to have further columns, as long
        as the first few columns are as specified.
    
    Parameters:
      cpg_locations: pandas dataframe of the format:
        (chromosome, location, ...)
      regions: 
        (chromosome, start_location, end_location,...)
      nonevalue: what value to use for a "None" index. Since not all
        datatypes are nullable, you should choose whatever works best
        for your datatype. Default is -1 since this works with int64.
        Careful: If you provide "None" but the index datatype is not
        nullable, pandas will just convert it (to float, for example)
        and you will end up with indices that are not comparable.
        
    Returns:
      pd.Series: A panda series where the index is the same index as 
                 in cpg_locations, and the value is either the index
                 of the matching region, or None if the location is 
                 in no region
    '''

    region_membership = pd.Series(data=nonevalue, index=cpg_locations.index,
                                  dtype=regions.index.dtype)

    # Pre-sorting if necessary
    if not is_sorted:
        chromosomes = sorted(list(set(regions.iloc[:,0]).union(set(
            cpg_locations.iloc[:,0]))))
        regions.iloc[:,0] = pd.Categorical(regions.iloc[:,0], ordered=True, 
            categories=chromosomes)
        cpg_locations.iloc[:,0] = pd.Categorical(cpg_locations.iloc[:,0], 
            ordered=True, categories=chromosomes)
        regions = regions.sort_values([regions.columns.values[0], 
            regions.columns.values[1]])
        cpg_locations = cpg_locations.sort_values([
            cpg_locations.columns.values[0], cpg_locations.columns.values[1]])

    en_regions = enumerate(regions.itertuples())
    en_loc = enumerate(cpg_locations.itertuples())

    _, region = next(en_regions)
    _, loc = next(en_loc)
    
    try:
        '''
        When accessing the tuples, remember:
            loc[0]: cpg index
            loc[1]: chromosome
            loc[2]: location
            region[0]: region index
            region[1]: chromosome
            region[2]: start site
            region[3]: end site
         '''
        while True:
            ''' If different chromosome, check which one to spool '''
            while(loc[1] != region[1]):
                if loc[1] < region[1]:
                    _,loc= next(en_loc)
                if region[1] < loc[1]:
                    _,region = next(en_regions)

            ''' If location is behind region, spool location '''
            while(loc[2] < region[2]):
                _,loc= next(en_loc)

            ''' If location is past region, spool region '''
            while(loc[2] > region[3] and loc[1] == region[1]):
                _,region = next(en_regions)

            ''' Check all the constraints '''
            if loc[1] == region[1] and loc[2] >= region[2]  and loc[2] <= region[3]:
                region_membership[loc[0]] = region[0]
            else:
                ''' This happens if we spooled the region past the location '''
                pass

            ''' Get next cpg location '''
            _,loc= next(en_loc)

    except StopIteration:
        pass

    return region_membership


def compute_ratio_from_sparse_matrix(met_matrix):
    return (met_matrix==2).sum(axis=1) / (met_matrix>0).sum(axis=1)

def aggregate_continuous_regions(data, group_by='gene', 
        fn_aggregate=compute_ratio_from_sparse_matrix, min_var_per_region = 100):
    '''This functions computes an aggregate over all variations with the same 
    annotation. It assumes that regions are continuous, i.e. that variables 
    from the same regions are next to each other. This is true, for example, if
    the variables are sorted by locations and mapped to unoque regions (like 
    genes). 

    Parameters:
        data: anndata object
        group_by: the key of the variable annotation of the regions 
            (default: 'gene')
        fn_aggregate: the aggregation function. The default aggregation function
                      assumes that 0 are missing values, 1 is non-methylated, 2
                      is methylated, and then computes the methylation ratio. 
                      The function receives 1 argument, which is a matrix with
                      the variables of one region as the columns, and the 
                      observations as rows.
        min_var_per_region: regions with fewer variables than this will be 
                            ignored. (default: 100)

    Return:
        pd.DataFrame: a data frame where the row indices are the row indices
                      of the observations in the input dataset, and the column
                      indices are the regions. The values contain the aggregate
                      values for each region.
    '''
    aggregated = pd.DataFrame(index=data.obs.index)

    i = 0
    j = 0
    cur_region =  data.var[group_by][i]

    while j < data.shape[1]:
        region = data.var[group_by][j]
        if region!=cur_region:
            region_matrix = data[:,i:j].X
            # Check if it has been flattened (older anndata versions will do that)
            if len(region_matrix.shape)==2: 
                if region_matrix.shape[1] > min_var_per_region:
                    aggregated[cur_region] = fn_aggregate(region_matrix)
            i=j
            cur_region=region
        j+=1
    return aggregated
        
def read_gtf_file(filename, feature='gene', index_attribute='gene_id'):
    ''' Simplified function to read a single attribute from a GTF file 
    and return a pandas DataFrame.
     
    Parameters:
        filename: the filename of the gtf file. may be .gz file
        feature: which feature (third column in GTF) should be kept
                 (default: 'gene')
        index_attribute: which attribute from the attributes column should
                         be kept. This will be used as an index for the
                         returne dataframe. (default: 'gene_id')
    Returns:
        pd.DataFrame with the columns 'chr', 'start', 'end' and the
            index is determined by the parameter index_attribute
    '''

    gtf = pd.read_csv(filename, sep='\t', header=None, comment="#", 
        usecols=[0,2,3,4,8],names=['chr','feature','start','end', 'anno'], 
        dtype={'chr':'category', 'feature':'category','begin':np.int64, 
            'end':np.int64, 'anno':'object'})

    # Remove lines that are not from the feature of interest 
    gtf = gtf[gtf['feature'] == feature]

    # Extract index attributes
    attr_pattern = re.compile('%s "([^"]*)"' % index_attribute)
    gtf[index_attribute] = gtf['anno'].map(lambda x: attr_pattern.match(x).group(1))
    gtf = gtf.set_index(index_attribute)

    gtf = gtf.drop(['feature','anno'], axis=1)

    return gtf
