"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176
"""

"""
This is an implementation of the Histogram interface (in AproxBLUE) that takes public Census products as input.
"""

from AproxBLUE import Histogram
import itertools, numpy
from collections import defaultdict
import pandas as pd
from itertools import chain, combinations
from functools import reduce
from sf1extract import get_true_data

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class NMFHistogram(Histogram):
    """
    Attributes
    ----------
    var_levels : int or list of int
        List of levels corresponding to each variable in histogram's `var_list`, or one level for each variable
    num_children : int 
        Number of children for each geocode
    num_vars : int
        Number of variables
    max_depth : int
        Maximum depth of histogram tree
    max_detail : int
        Maximum number of marginal crosses within histogram
    var_list : list of str
        List of variable names
    true_data : `np.ndarray`
        List of true data (no noise)
    noised_data : `np.ndarray`
        List of data after noise
    variance : `np.ndarray`
    weights : `np.ndarray`
    data : `np.ndarray`
        List of data after noise
    categories_by_length : dict
        Dictionary where keys are the length of cateogory (number of variables)
        and values are the cateogory representation of the given length
    categories : dict
        Dictionary where keys are the categories (i.e. (A, B, C)) and the values are a list of 
        indices of variables that are under the category (i.e. (1, 1, 1, 0))
    all_vars : list
        List of all variables types
    variables : `np.ndarray`
        List of all variables inside the histogram 
    data : `np.ndarray`
        List of all data inside histogram, indexing corresponds to `variables` list
    geocodes : `np.ndarray`
        List of all geocodes inside histogram, indexing corresponds to `variables` list
        Note: each copy of all_vars within `variables` consitutes another geocode
    depths : `np.ndarray`
        List of all levels inside histogram, indexing corresponds to `variables` list
    parents : `np.ndarray`
        List of parent geocode, indexing corresponds to `variables` list
        Note: root depth nodes (geocode=1) has `0` parent geocode
    """
    
    
    """
    Create a Histogram from a Census NMF Parquet file. Expects a `detailed_dpq` Parquet entry at minimum.
    
    This Histogram will contain ALL possible crosses, up to `max_dummy_combo_size`, that can be formed from the marginals,
    whether they are in the Parquet records or not. Cross that are not supplied are filled in as 0-value and given a
    placeholder variacne value supplied by the missing_variance_value argument (this should be a huge value like 1e100, or
    a signal value like -1 so it can be filtered out as needed).
    
    missing_variance_value - a placeholder to use when a table is absent (-1 works well)
    noisy_path - path to the NMF parquet file
    true_path - optional; an SF1 filepath template like "ri000{0}2010.sf1" where {0} can be replaced by two digits to access SF1 files by number
    marginal_names - list of strings of the variable names; these are in the order used by the table crosses
                      (marginals [A,B,C,D] will make the cross (A,B), never (B,A))
    geocode - optinal; limits reading in records pertaining to this geocode only (Parquet files may contain entries for many geocodes)
    max_dummy_combo_size - limit dummy data to crosses up to this many variables
    """
    def __init__(self, missing_variance_value, noisy_path, true_path, marginal_names, geocode=None, max_dummy_combo_size=3, continuous=False):
        nmf = pd.read_parquet(noisy_path)
        if geocode is not None:
            nmf = nmf.loc[nmf.geocode == geocode]
        self.all_vars = []
        self.data = []
        self.noised_data = []
        self.variance = []
        self.categories = {}
        self.var_list = {}
        self.categories_by_length = defaultdict(list)
        self.var_levels = nmf[nmf.query_name == "detailed_dpq"].query_shape.iloc[0].tolist()
        self.geocodes = []
        self.depths = []
        self.parents = []
        self.true_data = []
        self.input_queries = []
        self.input_true_queries = []
        self.continuous = continuous

        self.num_vars = len(nmf.iloc[0]["query_shape"])
        
        # get all marginals
        # construct all crosses of those marginals
        # for each cross,
        #    see if there is noisy data
        #      if yes, populate it with associated variance
        #      if no, populate it with zeroes
        #    see if there is true data
        #      if yes, populate true data
        #      if no, populate zeroes
        all_combos = powerset(marginal_names)

        geocodes = list(nmf['geocode'].unique())

        for geocode in geocodes:
            if true_path is not None:
                true_dict = get_true_data(geocode, file_template = true_path)
            else:
                true_dict = {}
        
            # enumerate all crosses that can be formed from the marginals
            for combo in all_combos:
            
                # this is a patch to fix an out-of-order cross in DHC
                reversed_cross_name = ""
                
                if combo == tuple(marginal_names):
                    cross_name = "detailed"
                elif combo == tuple():
                    cross_name = "total"
                else:
                    cross_name = " * ".join(combo)
                    reversed_cross_name = " * ".join(reversed(combo))

                shape = list(map(lambda n: 1 if n[1] not in combo else self.var_levels[n[0]], enumerate(marginal_names)))
                count = reduce(lambda a,b: a*b, shape)

                # for each cross, populate hist with noisy and true data, if available 
                noisy_row = nmf.loc[((nmf["query_name"] == cross_name+"_dpq") | ((nmf["query_name"] == reversed_cross_name+"_dpq"))) & (nmf["geocode"] == geocode)]
                # if noisy data is available, add it
                if len(noisy_row) > 0:
                    noisy_row.apply(self.consume_row, axis=1)
                    self.input_queries.append(combo)
                    
                    # if true data is available, add it
                    if cross_name in true_dict:
                        self.true_data += true_dict[cross_name]
                        self.input_true_queries.append(combo)
                    # otherwise add dummy-None true data
                    else:
                        self.true_data += [None] * count
                else:
                    # noised data is not available, add dummy noised and true data
                    if max_dummy_combo_size is None or len(combo) <= max_dummy_combo_size:
                        self.consume_row(pd.Series({ "variance":missing_variance_value, "value":[0] * count, "query_shape":shape,
                                                    "query_name": cross_name+"_dpq", "geocode":geocode }))
                        
                        # if true data is available, add it
                        if cross_name in true_dict:
                            self.true_data += true_dict[cross_name]
                            self.input_true_queries.append(combo)
                        # otherwise add dummy-None true data
                        else:
                            self.true_data += [None] * count
                
        self.variables = numpy.array(self.all_vars)
        self.var_list = list(dict(sorted(self.var_list.items())).values())

    """
    Function to read in a single row of an NMF Parquet file and populate the histogram with its data
    
    self - this histogram
    row - a Pandas row read from the Parquet file
    """
    def consume_row(self, row):
        vars = row["query_name"][:-4]
        if vars == "detailed":
            vars = tuple(self.var_list.values())
        elif vars == "total":
            vars = tuple()
        else:
            vars = tuple(vars.split(" * "))
            
            # DHC "hispanic * sex" query is out of order -- we expect "sex * hispanic"
            if vars == ('hispanic', 'sex'):
                vars = ('sex','hispanic')

        # if this is a single-var query,
        # update var_list with this var name using the non-1 position in query_shape
        if len(vars) == 1 and vars[0] != "detailed" and vars[0] != "total":
            position = [i for i, x in enumerate(list(row.query_shape)) if x != 1][0]
            self.var_list[position] = vars[0]

        self.categories_by_length[len(vars)].append(vars)
        noisy_values = list(row["value"])
        old_length = len(self.data)
        new_length = old_length + len(noisy_values)
        
        # NOTE: this is a simple hack to ensure longer geocodes lie below shorter ones
        depth = len(row.geocode)
        self.depths += [depth] * len(noisy_values)

        self.categories[vars] = list(range(old_length, new_length))
        self.data += noisy_values
        self.noised_data += noisy_values
        self.variance += [row.variance] * len(noisy_values)
        self.geocodes += [row.geocode] * len(noisy_values)
        self.all_vars += enumerate_shape(row.query_shape)

"""
Given a list of variable sizes, enumerate all combinations of those varaible levels

e.g., [2,3] enumerates (1,1) (1,2) (1,3) (2,1) (2,2) (2,3)

shape - list of integers
"""
def enumerate_shape(shape):
    # use a single 0 for a len==1 dimension, otherwise use a [1...len] range
    all_values = map(lambda v: [0] if v==1 else range(1,v+1), shape)
    combos = itertools.product(*all_values)
    return list(combos)

if __name__ == "__main__":
    hist = NMFHistogram(-1, "nmf_sample.parquet")
    #print(hist.all_vars)
    #print(hist.variables)
    #print(hist.categories)
    #print(hist.var_levels)
    #print(hist.var_list)
