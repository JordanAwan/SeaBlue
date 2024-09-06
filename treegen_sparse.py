"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176
"""
import math
import time
import os
import re
import psutil
import tracemalloc
import traceback
import argparse

import numpy as np
import pandas as pd

# attempt to fix an error on HPC
import scipy.sparse as sp
import scipy.linalg as la
import scipy.stats as st

import scipy
import itertools

def check_skip(category, children, var_list):
    """
    Determines which first-level children variables (if any) can be skipped given the current category
    If the category variable is out of ordr with respect to the sorted variable list, skip the first level

    Parameters
    ----------
    category : tuple, current category 
    children : list of tuples, children categories of this current category
    var_list : list of str, all variables within the histogram (sorted)

    Returns
    -------
    list of tuples, tuples are the child category and any first-level variables to skip
    """
    skip_children = []
    for child in children: 
        skip_var = []
        for i in range(len(category)):
            if category[i] != child[i]:
                skip_var += [var_list.index(category[i])]
        skip_children += [(child, skip_var)]
    return skip_children

# 'a function to make the A matrix
# '
# '---General notes:
# 'The 2-way interaction case may be better thrown in with the "most" block than the "corner" block
# 'It will probably be faster to make a "temp levels" list object, and just reference the temp levels of interest
# 'rather than remaking the temp levels every time you want to make a constraint.
def A_make(hist, var_list, verbose=False):
    """
    Creates a full rank constraint matrix
    Parameters
    -----------
    hist : histogram object
    var_list : list of variable names

    Returns
    -------
    `scipy.sparse.csr_matrix`, a full rank constraint matrix
    """
    # empty constraint matrix
    num_geo, num_int, _ = hist.num_constraints()
    A = sp.lil_matrix((num_geo, len(hist)), dtype=np.int8)
    curr_row = 0

    lengths = list(hist.categories_by_length.keys())
    lengths.sort()
    
    maxes = hist.categories_by_length[lengths.pop(-1)]
    missing = np.setdiff1d(hist.var_list, np.array(maxes).flatten())

    var_sort = pd.DataFrame({"vars": hist.var_list,
                             "max_level": [0]*hist.num_vars}, index=hist.var_list)

    # extract all of the n-way marginals that includes the highest order for each variable
    while len(missing): 
        length = lengths.pop(-1)
        cats = hist.categories_by_length[length]
        vars_added = np.intersect1d(missing, np.array(cats).flatten())
        if len(vars_added):
            maxes += [*cats]
            missing = np.setdiff1d(missing, vars_added)
            for var in vars_added:
                var_sort.loc[var, "max_level"] = length

    var_sort.sort_values(by=["max_level"], ascending=[False])
    var_sort = list(var_sort["vars"])
    for cat in maxes: 
        skip_var = [var for var in cat if var_sort.index(var) != cat.index(var)]

        ind = np.array(hist.categories[cat], dtype=int)
        table = pd.DataFrame(ind, columns=["ind"]).set_index(ind)
        table["geocodes"] = hist.geocodes[ind]
        table["parents"] = hist.parents[ind]
        table["depths"] = hist.depths[ind]
        table["variables"] = [tuple(var) for var in hist.variables[ind, :]]

        for var in table.variables.unique():
            sub = table[table.variables == var]
            skip_first = [hist.var_list[v] for v in range(hist.num_vars) if var[v] == 1]
            skip = set(skip_var) & set(skip_first)

            for i in sub.ind: 
                if sub.depths[i] == hist.max_depth:
                    continue

                children = (sub.parents == sub.loc[i, "geocodes"])
                children = np.where(children)[0]
                
                if not skip:
                    indices = sub.ind.iloc[children]
                    A.rows[curr_row] = [i] + list(indices)
                    A.data[curr_row] = [1] + ([-1] * len(indices))
                    curr_row += 1

    sub = np.where(hist.geocodes == 1)[0]
    num_nodes = len(sub)
    
    all_vars = pd.DataFrame(hist.all_vars, columns=var_list)

    temp_levels = pd.DataFrame() # maybe make it an array??
    for var in var_list: 
        missing = all_vars.drop(var, axis=1)
        # add by column (missing var), not by rows 
        temp_levels[var] = [tuple(row) for row in missing.itertuples(index=False)]
    
    internal = sp.lil_matrix((num_int, num_nodes), dtype=np.int8)
    curr_row = 0

    # ensure all one way marginals:
    for var in hist.categories_by_length[1]: 
        ind = [i for i in hist.categories[var] if i in sub]
        internal.rows[curr_row] = [0] + ind
        internal.data[curr_row] = [1] + ([-1] * len(ind)) # indicates total = the sum of all the variable children
        curr_row += 1

    # for every category (not the total or a max_detail category)
    for category in hist.categories: 
        length = len(category)
        if length == 0 or length == hist.max_detail:
            continue
        # find all categories of one greater marginal cross that contains all variables in current category
        children = [cat for cat in hist.categories_by_length[length + 1] if not (set(category) - set(cat))]
        if not children: 
            continue
        # determine which first level variables to skip (if any)
        children = check_skip(category, children, var_list)
        for child, var in children:
            # extract the variable indices from the root nodes
            cat_sub = np.array(list(itertools.takewhile(lambda i: i in sub, hist.categories[child])), dtype=int)
            
            # create mask to skip variables
            levels = hist.variables[cat_sub]
            mask = np.zeros(len(cat_sub), dtype=bool) 
            # determine the missing variable in the child category 
            miss = [i for i, val in enumerate(var_list) if val in child and val not in category]
            # NOTE: categories that are missing the same variable will result in duplicate constraints
            # determine which variables are NOT missing, and filter out the duplicates
            there = [i for i in range(hist.num_vars) if i not in miss]
            mask[np.unique(levels[:, there], return_index=True, axis=0)[1]] = True
            
            # if there is a variable out of order (a variable to skip)
            if var: 
                ind = np.where(levels[:, var] == 1)[0]
                mask[ind] = False

            # update constraints
            for j in cat_sub[mask]:
                ind = np.where(temp_levels.iloc[:, miss] == temp_levels.iloc[j, miss])[0]
                internal.rows[curr_row] = list(ind)
                internal.data[curr_row] = [1] + ([-1] * (len(ind)-1))
                curr_row += 1

    # need to attach internal to georaphic and make A block diagonal
    num_geocodes = len(set(hist.geocodes))
    internal_diag = sp.kron(sp.identity(num_geocodes).tocsc(), internal.tocsc(), format="csc")

    A = sp.vstack([A.tocsc(), internal_diag])

    del all_vars
    del temp_levels
    del table

    return A

"""
Create constraint matrix from histogram detailed query
"""
def A_make_from_detailed(hist, var_list):
    A = []
    detailed_idxs = hist.categories[tuple(var_list)]
    number_of_idxs = len(hist.noised_data)

    A = sp.dok_array((number_of_idxs - len(detailed_idxs), number_of_idxs))

    for n, idx in enumerate([idx for idx in range(number_of_idxs) if idx not in detailed_idxs]):
        A[n, idx] = 1
        current_level_set = hist.variables[idx]
        # for this position, consider each detailed position 
        # and add a -1 to each detailed position for which
        # the detailed levels match the nonzero levels of the current position
        for detailed_idx in detailed_idxs:
            detailed_item_level_set = hist.variables[detailed_idx]
            if all(left == right or left == 0
                   for left, right in zip(current_level_set, detailed_item_level_set)):
                A[n, detailed_idx] = -1

    return A

# Confidence interval calculation
def mean_confidence_interval(m, v, true_data, position=None, confidence=0.95):
    """
    Confidence interval calculation

    Parameters
    ----------
    m : 
    v : 
    true_data : true data from Histogram
    position : None, 
    confidence : float, 
    """
    sd = np.sqrt(v)
    h = sd * st.norm.ppf((1 + confidence) / 2.)
    # if position:
    #     if v[position] == 0:
    #         m[position] = true_data[position]
    #         h[position] = 0
    return m, m - h, m + h


# Confidence interval calculation with integer clipping
def mean_confidence_interval_integer_rounding(m, v, true_data, position=None, confidence=0.95):
    """
    Confidence interval calculations with integer rounding

    Parameters
    ----------
    m : 
    v : 
    true_data : true data from Histogram
    position : None, 
    confidence : float, 
    """
    sd = np.sqrt(v)
    h = sd * st.norm.ppf((1 + confidence) / 2.)
    # if position:
    #     if v[position] == 0:
    #         m[position] = true_data[position]
    #         h[position] = 0

    return m, np.ceil(m - h), np.ceil(m + h)
