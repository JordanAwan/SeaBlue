"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176
"""

from itertools import combinations, product, chain
from functools import reduce
import numpy as np
import pandas as pd
import random
import math
from scipy.stats import norm, t
import numpy
from collections import defaultdict
from pympler.asizeof import asizeof
from pympler import tracker
from discretegauss import discretegauss
from sys import getsizeof

"""
Tree structure to represent the histogram of counts
"""
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']
class Histogram: 
    """
    Histogram structure

    Parameters
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
    var_list : list of str, optional
        List of variable names
    continuous : bool, default True
        Used for data (noise) generation
    p_zero : float, default 0.5
        Used for determining number of variables at the maximum depth level to zero out

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
    def __init__(self, 
                 var_levels, 
                 num_children, 
                 num_vars, 
                 max_depth,
                 max_detail,
                 var_list = None, 
                 continuous = True,
                 p_zero = 0.5):

        # constants
        self.size = 1000
        self.ds = 5
        self.continuous = continuous
        self.p_zero = p_zero

        self.var_levels = var_levels if type(var_levels) == list else [var_levels] * num_vars # number of levels per variable corresponds to index of var_list
        self.num_children = num_children # number of children in geography
        self.num_vars = num_vars # number of individual variables, not marginals
        self.max_depth = max_depth # maximum number of levels
        self.max_detail = max_detail

        # store/create crossed variables "A", "B", "C" for default
        if var_list == None: 
            self.var_list = [LETTERS[i] for i in range(num_vars)]
        else: 
            self.var_list = var_list

        self.true_data = []
        self.noised_data = []
        self.variance = []
        self.weights = []

        self.categories_by_length = {}
        self.categories = {}
        self.all_vars = []
        self.variables = []
        self.data = []
        self.geocodes = []
        self.depths = []
        self.parents = []

        self.bottom_up_creation()
        # self.top_down_creation()

    def __len__(self):
        return len(self.variables)

    def num_overconstraints(self):
        """
        Returns the number of geographical and internal constraints if there were no optimization (no skipping)
        Note: internal constraints only account for one geocode, full internal constraints are generated through diagonalizing block matrix

        Returns
        -------
        tuple of int
        """
        L = sum([self.num_children ** x for x in range(0, self.max_depth - 1)])
        var_levels = max(self.var_levels)

        geo_num = math.comb(self.num_vars, self.max_detail) * (var_levels ** self.max_detail) * L

        int_num = sum([math.comb(self.num_vars, i) * (var_levels ** i) * i for i in range(2, self.max_detail + 1)]) + self.num_vars
        
        return geo_num, int_num

    def num_constraints(self):
        """
        Returns the number of geographical and internal constraints after optimization (for full rank)
        Note: internal constraints only account for one geocode, full internal constraints are generated through diagonalizing block matrix

        Returns
        -------
        tuple of int
        """
        
        # number of geocodes that are parents to other geocodes within this histogram
        # i.e., number of places that must enforce geography consistentcy with its children
        # we subtract 1 because the root node has a null parent (currently identified by "0" but there must be some null root "parent" to filter out)
        L = len(set(self.parents)) - 1

        # get size of biggest variable
        var_levels = max(self.var_levels)
        
        detail_range = np.arange(self.max_detail + 1, dtype=int)

        if L == 0:
            geo_num = 0
        else:
            if self.num_vars == self.max_detail:
                gpl = var_levels ** self.num_vars
            else:
                f_num = np.vectorize(lambda x: math.comb(self.num_vars - self.max_detail - 1 + x, x)) 
                num = f_num(detail_range)

                f_defs = np.vectorize(lambda x: (var_levels ** (self.max_detail - x)) * ((var_levels - 1) ** x))
                defs = f_defs(detail_range)

                gpl = sum(num * defs)
            geo_num = gpl * L

        # number of geographies, i.e., places that must be internally consistent
        M = len(set(self.geocodes))

        ipl = 0
        for x in detail_range[1:]:
            f_ipl = np.vectorize(lambda z: (var_levels ** (x - z - 1)) * ((var_levels - 1) ** z))
            ipl += (sum(f_ipl(np.arange(x, dtype=int))) * math.comb(self.num_vars, x))
            
        int_num = ipl * M

        return geo_num, ipl, geo_num + int_num

    def get_category(self, variable):
        """
        Returns the category given the current variable

        Parameters
        ----------
        variable : tuple or list
            Variable representation is tuple of ints or list of ints
        
        Returns 
        -------
        tuple of str
        """
        return tuple(self.var_list[i] for i in range(self.num_vars) if variable[i])

    def bottom_up_creation(self):
        """
        Generates a histogram given the specifications upon initialization
        """
        # generate all possible categories
        # store categories as values within their respective lengths
        # initialize categories dictionary to be populated with variable indices 
        for i in range(1, self.max_detail + 1):
            # generate all possible combinations of variable names of length i
            combos = list(combinations(self.var_list, i))
            self.categories_by_length[i] = combos
            for combo in combos: 
                # initialize list of indicies for all variables of this cateogry type
                self.categories[tuple(combo)] = [] 
        # initialize the total category `()` key
        self.categories[()] = []

        # create variables table with all possible marginal crosses (absolute maximum detail)
        all_vars = list(product(*[[*range(var_level + 1)] for var_level in self.var_levels]))

        # create level table
        self.level_table = pd.DataFrame(np.array(all_vars), columns=self.var_list)
        self.level_table = self.level_table[(self.level_table).all(axis=1)].reset_index(drop=True)

        # create geocode 1 (root) variables 
        self.all_vars = [var for var in all_vars if var.count(0) >= (self.num_vars - self.max_detail)]
        n_counts = len(self.all_vars)

        # populate histogram with root variable nodes (with respective indices)
        for i in range(n_counts): 
            cat = self.get_category(self.all_vars[i])
            self.categories[cat] += [i]
        self.variables += self.all_vars
        self.geocodes += [1]*n_counts
        self.parents += [0]*n_counts
        self.data += [np.nan]*n_counts
        self.depths += [1]*n_counts
        # create high order geocode nodes

        # create a categories dictionary of `np.ndarrays` to keep track of index calculations
        categories_base = {cat: np.array(self.categories[cat], dtype=int) for cat in self.categories}

        # initialize current parent_geocode
        parent_geo = [1]
        max_geo = 1
        # generate higher order geocode nodes level by level
        for depth in range(2, self.max_depth + 1):
            # keep track of the geocodes that were generated at this level
            all_children = []
            # generate the current level by creating children for each parent
            for i in parent_geo: 
                max_geo = max(self.geocodes)
                # generate all children geocodes for this parent
                child_geo = range(max_geo + 1, max_geo + self.num_children + 1)
                # update histogram with new nodes
                for j in child_geo: 
                    self.variables += self.all_vars
                    self.geocodes += [j]*n_counts
                    self.parents += [i]*n_counts
                    self.depths += [depth]*n_counts
                    self.data += [np.nan]*n_counts
                    # update categories with new nodes
                    # NOTE: categories_base is np.ndarray, self.categories is list type.
                    for cat in categories_base: 
                        categories_base[cat] += n_counts
                        self.categories[cat] += list(categories_base[cat])
                all_children += list(child_geo)
            parent_geo = all_children

        # convert anything necessary into `np.ndarray``
        self.geocodes = np.array(self.geocodes, dtype=int)
        self.data = np.array(self.data)
        self.parents = np.array(self.parents, dtype=int)
        self.variables = np.array(self.variables, dtype=int)
        self.depths = np.array(self.depths, dtype=int)

        # collect all max depth geocodes: 
        num_details = len(self.level_table)
        block_ind = parent_geo
        for i in block_ind: 
            # create detailed level table query 
            detail = self.level_table

            # set all counts to 0
            detail['counts'] = 0

            # calcualte number of counts that should stay 0 NOTE: only work with histograms with same number of levels
            p_stay = self.p_zero ** (1 / (self.var_levels[0] ** (self.num_vars - self.max_detail)))
            gen = random.sample(range(num_details), int(num_details*(1-p_stay)))
            # median is 5 and we don't want 0s
            detail.loc[gen, "counts"] = np.random.poisson(4, len(gen)) + 1
            # create general sums table
            sums = pd.DataFrame({"counts": [sum(detail["counts"])]}, index=pd.MultiIndex.from_tuples([(0,)*self.num_vars]))
            # find the counts we want for rest of the table

            for category in self.categories: 
                if category == ():
                    continue

                # sum together columns of category from the detail table (find the variables)
                marginal = detail.groupby(list(category)).sum()

                # extract positions from the variable list of the category 
                positions = [i for i in range(self.num_vars) if self.var_list[i] in category]
                
                variable = np.zeros(self.num_vars, dtype=int)
                def index_to_column(row):
                    variable = np.zeros(self.num_vars, dtype=int)
                    variable[positions] = row.name
                    return variable
                # for the index of the marginals
                marginal["named"] = marginal.apply(lambda row: index_to_column(row), axis = 1)
                marginal.index  = pd.MultiIndex.from_tuples(marginal.named)
                sums = pd.concat([sums,marginal]).filter(["counts"]) 
                    
            # fill in counts for the general sum from this geocode
            geo_ind = np.where(self.geocodes == i)[0]
            self.data[geo_ind] = sums.loc[self.all_vars, "counts"]
        # fill in counts for the rest of the geocodes in reverse 
        for i in np.where(self.depths != self.max_depth)[0][::-1]:
            children = (self.parents == self.geocodes[i]) & np.all(self.variables == self.variables[i], axis=1)
            ind = np.where(children)[0]
            self.data[i] = sum(self.data[ind])
        # update data with noise
        self.true_data = np.copy(self.data)
        self.noised_data = np.copy(self.data)

        self.variance = np.zeros(len(self.data)) + 2
        self.weights = 1 / self.variance

        self.create_noised_data_from_true_and_variance()

    def create_noised_data_from_true_and_variance(self):
        """
        Populate the `noised_data` property using the `true_data` and `variance` properties
        Can be continuous or dicrete, based on `continuous` boolean property
        """
        if self.continuous:
            noise = [np.random.normal(scale=math.sqrt(var), size = 1)[0] if var >= 0 else 0 for var in self.variance]
        else:
            noise = [discretegauss.sample_dgauss(var) if var >= 0 else 0 for var in self.variance]

        self.noised_data = [true + noise for true, noise in zip(self.true_data, noise)]
        #self.data = self.true_data + noise


    def subhist(self, predicate, starting_indexes=None):
        """
        Get a list of indexes from this histogram that match the provided predicate
        
        predicate - a function that takes an index number and return a boolean
        starting_indexes - a list of candidate indexes to filter on (if missing, uses the full histogram)
        """
        indexes = starting_indexes or range(len(self.data))
        return [idx for idx in indexes if predicate(idx)]
        
    def sub_select(self, predicate, starting_indexes=None):
        """
        Creates a new sub-histogram from this, filtered based on a predicate.
        
        Same as subhist, but actually generates a new histogram, with all histogram properties, for the resulting indexes from the parent

        predicate - a function that takes an index number and return a boolean
        starting_indexes - a list of candidate indexes to filter on (if missing, uses the full histogram)
        """
        
        target_indexes = starting_indexes or range(len(self.data))
        target_indexes = [idx for idx in target_indexes if predicate(idx)]
        
        new_hist = Histogram(num_vars = self.num_vars, var_levels = self.var_levels, num_children = self.num_children, max_detail = self.max_detail, max_depth=self.max_depth, continuous=self.continuous)
        
        new_hist.true_data = [d for i,d in enumerate(self.true_data) if i in target_indexes]
        new_hist.noised_data = [d for i,d in enumerate(self.noised_data) if i in target_indexes]
        new_hist.variance = [d for i,d in enumerate(self.variance) if i in target_indexes]
        new_hist.weights = [d for i,d in enumerate(self.weights) if i in target_indexes]
        new_hist.variables = [d for i,d in enumerate(self.variables) if i in target_indexes]
        new_hist.data = [d for i,d in enumerate(self.data) if i in target_indexes]
        new_hist.geocodes = [d for i,d in enumerate(self.geocodes) if i in target_indexes]
        new_hist.depths = [d for i,d in enumerate(self.depths) if i in target_indexes]
        new_hist.parents = [d for i,d in enumerate(self.parents) if i in target_indexes]
        
        """
        List comprehension for selecting new_hist.categories:
        
        target_indexes - list of all indexes that we're limiting the new Histogram down to
        category_idxs - as we loop through categories, the indexes of each category
        idx - each value of category_idxs
        target_indexes.index(idx) - this maps the old category index values to their position in the target_indexes
            so if we're limiting hist down to [2,4,6,8] and some category used to be [4,6]
            that cat will map to [1,2] (because 4 and 6 are positions 1 and 2 of the new hist)
            also: category idxs that aren't in target_indexes are dropped
        """
        new_hist.categories = { cat: [target_indexes.index(idx) for idx in category_idxs if idx in target_indexes] for cat, category_idxs in self.categories.items() }
        
        return new_hist

# reciprocal of input
def inverse(number):
    return 1/number

# list of all subsets of the given iterable
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# get number of nonzero elements in a tuple
def cardinality(var_combo):
    return sum(1 for i in var_combo if i != 0) 


def create_projection_matrix(variance_matrix, A, estimates_size, track_memory=False):
    """
    Create projection matrix from variance and constraints
    
    variance_matrix - matrix with variances on the diagonal
    A - constraint matrix
    estimates_size - integer, number of estiamtes
    track_memory - boolean to track memory usage
    """

    memory = 0
    
    if track_memory:
        tr = tracker.SummaryTracker()

    ident = np.identity(estimates_size)
    '#Matrix Multiplies A by variance_matrix then matrix multiplies that by A transposed,and takes the inverse of that step'
    inner_inverse_sigma = np.linalg.inv((A @ variance_matrix) @ A.T)
    
    '#Matrix multiplies variance_matrix by A transposed, then matrix multiplies that by our inner_inverse_sigma, then finally again by A'
    projection_sigma = ((variance_matrix @ np.transpose(A)) @ inner_inverse_sigma) @ A
    projection_sigma = ident - projection_sigma
    
    if track_memory:
        memory = sum(item[2] for item in tr.diff()) + projection_sigma.nbytes
    return  projection_sigma, memory


def one_way_cross(hist,current_var, track_memory=False):
    """
    Takes up-pass results for a marginal and projects them to be consistent with the total

    hist - input histogram of counts
    current_var - string name of the current variable under consideration
    track_memory - boolean to track memory consuption
    """
    memory = 0

    # Collects Keys for Associated Variable
    AttributeKeys = hist.categories[(hist.var_list[current_var],)]
    
    Estimates = []
    Variance = []
    v = []
    Solution = hist.final_estimate[0] / len(AttributeKeys) #Current solution is (total / number of estimates)
    '# Adds Estimates,Variances and Solution to associated arrays'
    for i in AttributeKeys:
        Estimates.append(hist.final_estimate[i])
        Variance.append(hist.final_variances[i])
        v.append(Solution)
    A = np.matrix(np.ones(len(AttributeKeys)))
    
    projection_sigma, proj_memory = create_projection_matrix(np.diag(Variance), A, len(Estimates), track_memory)
    estimate_results = projection_sigma @ Estimates
    
    v_results = projection_sigma @ v
    Final = estimate_results + v - v_results
    
    if track_memory:
        memory = proj_memory + A.nbytes + v_results.nbytes + estimate_results.nbytes
    
    return Final, memory
    
def n_way_order(hist, combo):
    """
    Final-var by second-to-final-var... iteration of an n-way cross

    Returns a sorted list of histogram idexes pertaining to this cross

    hist - input histogram of counts
    combo - tuple of strings of the variable names under consideration
    """
    levels = { idx: tuple(hist.variables[idx]) for idx in hist.categories[combo] }
    #creates a sorted level table in the order of, e.g., Third Var, Second var, First var
    sortedlevels = {key: val for key, val in sorted(levels.items(),
                       key=lambda x: tuple(map(lambda letter: x[1][hist.var_list.index(letter)], reversed(combo))))}

    return list(sortedlevels.keys())
    

def unequal_variance_two_way_cross(hist, two_way_combo, final_counts, track_memory=False):
    """
    Takes up-pass results for a two-way table and projects those results to be consistent with its marginals
    Since marginals have already been made consistent with the total, this two-way becomse consistent with total as well

    hist - input histogram of counts
    two_way_combo - tuple of strings of the variable names under consideration
    """
    sortedlevels = n_way_order(hist, two_way_combo)
    return unequal_variance_n_way(hist, two_way_combo, final_counts, sortedlevels, track_memory)

def two_way_cross(hist, estimate1, estimate2, table_tuple, correctedCounts, track_memory=False):
    """
    Creates estimates for a two-variable category

    Takes up-pass results for a two-way table and projects those results to be consistent with its marginals
    Since marginals have already been made consistent with the total, this two-way becomes consistent with total as well
    This is a fast implementation requires all variances to be internally equal within each table
    (i.e., for each table, each count in that table is equal, but tables don't have to be equal to other tables)

    Doesn't perform a matrix inverse
    Expects vars in a particular order

    See unequal_variance_two_way_cross

    hist - full histogram
    estimate1 - estimate for the first variable as a marginal
    estimate1 - estimate for the second variable as a marginal
    table_tuple - tuple of strings describing the two variables in the category being estimated
    """

    memory = 0
    
    # cardinality of first and second vars
    i = len(estimate1)
    j = len(estimate2)

    Identity = np.identity(i)
    '#Creates M and iterates over it in the correct shape by adding 1s by index'
    M = np.zeros(shape = (j+i-1,i*j))
    column = 1
    column2 = 0
    for index in range(i):
        for x in range(j):
            M[index][x*i+column2] = 1
        column2+=1
            
    for row in range(j-1):
        for x in range(i):
           M[row+i][x+(column*i)]=1
        column+=1

    # for future non-matmul approach, constants for on- and off-diagonal
    A1 = ( i, (j-1) )
    A2 = ( i, -1 )

    #Creates each section of the MMTinverse Matrix to be slotted in
    A = (1/j)*(Identity + (j-1)/i * np.ones(shape =(i,i)))  
    B = (-1/i)*np.ones(shape =(i,j-1))
    C = np.transpose(B)
    D = (1/i) *( np.eye(j-1) + np.ones(shape =(j-1,j-1)))

    #Inserts Sections created above into the associated spot on the matrix
    MMTinverse = np.zeros(shape = (j+i-1,j+i-1))
    MMTinverse[0:i,0:i] = A
    MMTinverse[0:i,i:i+j-1] = B
    MMTinverse[i:i+j-1,0:i] = C
    MMTinverse[i:i+j-1,i:i+j-1] = D

    Identity = np.identity(i*j)
    # Projection is M matrix multplied by MMTinverse which is then Matrix Multipled by M transposed ( M.T(MM.T)^-1 M )
    almost_projection = (M.T @ (MMTinverse @ M))

    Projection = Identity - almost_projection

    # Gets Associated Estimates and Variance for current two way down pass
    # AttributeKeys is the list indexes pertaining to this category
    AttributeKeys = hist.categories[table_tuple]

    # levels is a dict whose keys are histogram indexes and values are a tuple with the levels of each variable
    levels = { idx: tuple(hist.variables[idx]) + tuple([hist.geocodes[idx]]) for idx in AttributeKeys }
    # sortedlevels is the same content as levels but items are sorted by each tuple's second var name
    #  x[1] gets the tuple value, then sort by the var level number in the position belonging to the second variable of the category being processed
    #  e.g., if ('A','C'), we find that 'C' belongs to var-level slot [2], and so sort tuples by the value in index 2
    if len(table_tuple) == 2:
        secondVarPosition = hist.var_list.index(table_tuple[1])
    else:
        secondVarPosition = hist.num_vars
    sortedlevels = {key: val for key, val in sorted(levels.items(), key = lambda x: (x[1][secondVarPosition]))}
    # capture estimates and variances in same order as sortedlevels
    sorted_estimates = []
    for key in sortedlevels:
        sorted_estimates.append(hist.final_estimate[key]) #NC Result of Up Pass

    #creates the V vector of length i*j
    TotalLevels = j*i
    total = sum(estimate1)
    End = (1/(TotalLevels)*total) # Calculated before loop as it remains the same
    v = [None] * TotalLevels
    for y in range(j):
        for x in range(i):
            v[i*y+x] = ((1/j)*estimate1[x])+((1/i)*estimate2[y]) - End
            
    # calculates U by taking Estimates from aggregation step, matrix multiplied by projection, adds v and then subtracts v matrix multiplied by our projection
    corrected_estimates = Projection @ sorted_estimates
    corrected_estimates += v
    
    if track_memory:
        memory = M.nbytes + Projection.nbytes + corrected_estimates.nbytes
    
    return corrected_estimates, memory

def three_way_cross(hist, three_way_combo, final_counts, track_memory=False, use_matmul=False):
    """
    Perform three-way cross projection

    Takes up-pass results for a three-way table and projects those results to be consistent with its marginals and two-way subtables
    Since marginals and two-ways have already been made consistent with the total, this three-way becomes consistent with total as well
    This is a fast implementation requires all variances to be internally equal within each table
    (i.e., for each table, each count in that table is equal, but tables don't have to be equal to other tables)

    Doesn't perform a matrix inverse
    Expects vars in a particular order

    See unequal_variance_three_way_cross

    hist - input histogram
    three_way_combo - a tuple of strings what vars are in this 3-way cross, e.g., ('A','B','C')
    final_counts - dictionary of previously computed marginal and two-way answers (indexed by tuple of cross)
    track_memory - boolean to produce memory usage results
    use_matmul - do projection by matrix multiplation, or else do by summation and multiplcation
    """

    memory = 0

    '#creates a sorted level table in the order of Third Var, Second var, First var'
    sortedlevels = n_way_order(hist, three_way_combo)
    '# Gets estimates & variance in the order indicated by sorted level table'
    sorted_estimates = []
    for key in sortedlevels:
        sorted_estimates.append(hist.final_estimate[key])

    '# Calculates Sizes of i , j , k'
    i = hist.var_levels[hist.var_list.index(three_way_combo[0])]  # Size A
    j = hist.var_levels[hist.var_list.index(three_way_combo[1])]  # Size B
    k = hist.var_levels[hist.var_list.index(three_way_combo[2])]  # Size C

    '#Collects corrected counts i , j , k'
    correctedCount_i = final_counts[(three_way_combo[0],)]  # e.g.(A)
    correctedCount_j = final_counts[(three_way_combo[1],)]  # e.g.(B)
    correctedCount_k = final_counts[(three_way_combo[2],)]  # e.g.(C)
    
    '#Collects corrected counts jk , ik , ij'
    correctedCount_j_k = final_counts[(three_way_combo[1], three_way_combo[2])]  # e.g.(BC)
    correctedCount_i_k = final_counts[(three_way_combo[0], three_way_combo[2])]  # e.g.(AC)
    correctedCount_i_j = final_counts[(three_way_combo[0], three_way_combo[1])]  # e.g.(AB)

    levels_j_k = j * k  # size BC
    levels_i_k = i * k  # size AC
    levels_i_j = i * j  # size AB
    
    '#Implementation of Pauls Solution V'
    end = (1 / (i * j * k)) * (sum(correctedCount_i))
    v = []
    for third in range(k):
        for second in range(j):
            for first in range(i):
                v.append(((1 / k) * correctedCount_i_j[second * i + first]) +
                         ((1 / j) * correctedCount_i_k[third * i + first]) +
                         ((1 / i) * correctedCount_j_k[third * j + second]) -
                         ((1 / levels_j_k) * correctedCount_i[first]) -
                         ((1 / levels_i_k) * correctedCount_j[second]) -
                         ((1 / levels_i_j) * correctedCount_k[third]) + end)

    if not use_matmul:
        """
        Fast non-matmul A1/A2/B1/B2 implementation
        
        The 3-way cross projection involves constructing a projection matrix out of blocks called A1/A2/B1/B2.
        This code replaces that matrix multiplcation by multiplying the components of those A1/A2/B1/B2 against
        rows of the input data. For each "row", we multiply A1/A2/B1/B2 constants against the values of the row
        and the sum of the row
        
        This code pulls this 3-way sub-histogram from the aggregation step and "paginates" the data into rows.
        A row consists of the elements for which two variables remain constant and the other one is enumerated entirely.
        Paginating by the first var would make rows like
            [(1,1,1) (2,1,1) (3,1,1) ...]
            [(1,2,1) (2,2,1) (3,2,1) ...]
            [(1,3,1) (3,3,1) (3,3,1) ...]
            ...
            [(1,9,8) (2,9,8) (3,9,8) ...]
            [(1,9,9) (2,9,9) (3,9,9) ...]
            
        Each successive row choses a new second and third variable combo, and then includes all possible first-var values
        under that combo. Alternatively, we can paginate on the second position, which would make a first row of (1,*,1)
            [(1,1,1) (1,2,1) (1,3,1) ...]
        
        Computationally, it is most efficeint to paginate on the largest variable, creating as few rows as possible
        that are as long as possible. This means we must
          1. decide which variable to pagnate on (i.e., the one with highest cardinality)
          2. enumerate each combination of the other two variables
          3. select the elements into those rows
          
        Therefore, for each row we have some "first_var" which is the variable that will be represented exhaustively,
        alongside "second_var" and "third_war" which will both be constant internal to each row.
        
        The forumlas for A1/A2/B1/B2 use `i`, `j`,and `k` meaning the length of the first, second and third variables.
        The formulas below use `first_var`, `second_var`, and `third_var` as the respective variable lengths. This ordering
        is based on the rule of making the largest variable the first, and the remaining two vars are used in order
        """

        # answer_sums will hold the final answer
        answer_sums = np.array([0.0] * (i*j*k))

        # if i is biggest, paginate by i
        if i>=j and i>=k:
            rows_of_jk_levels = [list(range(idx,idx+i)) for idx in range(0, i*j*k, i)]
            paginated_rows = rows_of_jk_levels
            first_var, second_var, third_var = i, j, k
        # if j is biggest, paginate by j
        elif j>=k and j>=i:
            rows_of_ik_levels = []
            for k_offset in range(0, i*j*k, i*j):
                for idx in range(0, i):
                    rows_of_ik_levels.append(list(range(k_offset+idx,k_offset+i*j,i)))
            paginated_rows = rows_of_ik_levels
            first_var, second_var, third_var = j, i, k
        # else, k is the biggest, so paginate by k
        else:
            rows_of_ij_levels = []
            for offset in range(0,i*j):
                rows_of_ij_levels.append(list(range(offset, i*j*k, i*j)))
            paginated_rows = rows_of_ij_levels
            first_var, second_var, third_var = k, i, j

        # create A1/A2/B1/B2 constants, using `first_var` for `i`
        shifted_r = (first_var * second_var) + (first_var * third_var) + (second_var * third_var) - (first_var + second_var + third_var - 1)
        constants = {}
        constants["A1"] = ( (shifted_r - (third_var - 1) * (second_var - 1)), (second_var - 1) * (third_var - 1) )
        constants["A2"] = ( (third_var - 1) * first_var, 1 - third_var )
        constants["B1"] = ( (second_var - 1) * first_var, 1 - second_var )
        constants["B2"] = ( -first_var , 1)

        # numpy array of values to do 3-way cross on
        target_numbers = np.array(sorted_estimates)
        
        # enumerate rows of first_var-values that share same second_ & third_
        for which_row, row in enumerate(paginated_rows):
            # capture list of values and the sum of all values
            row_estimates = target_numbers[row]
            row_sum = sum(row_estimates)
            
            # get second_ and third_ of this row
            this_second_var = which_row % second_var + 1
            this_third_var = int(which_row / second_var) + 1
        
            # dict of A1/A2/B1/B2 sums for this row
            sums = {}
            
            # enumerate A1,A2,B1,B2 matrices and multiply/sum them against the row values
            for is_a in [True, False]:
                for is_1 in [True, False]:
                    letter = "A" if is_a else "B"
                    number = "1" if is_1 else "2"
                    id = letter + number
                    # multiply first constant against each value and sum the results
                    sum1 = constants[id][0] * np.array(row_estimates)
                    # multiply second constant against (the row total * length of row)
                    sum2 = np.array([constants[id][1] * row_sum] * len(row))
                    # store answer for retrieval in next step
                    sums[(is_a, is_1)] = sum1 + sum2

            # enumerate all same-second+third rows as target idxs for augmenting the running sums with this row's reults
            for target_row_idx, target_row in enumerate(paginated_rows):
                target_second_var = target_row_idx % second_var + 1
                target_third_var = int(target_row_idx / second_var) + 1
                # does this storage target share a second and/or third var value with the current source value (of outer loop)?
                match_second_var = target_second_var == this_second_var
                match_third_var = target_third_var == this_third_var
                # pull out the A1/A2/B1/B2 product-sum that corresponds to this kind of second/third (mis)match
                target_sum = sums[(match_third_var, match_second_var)]
                # store the result in each index held by the target row
                answer_sums[target_row] += target_sum

        # divide sums by ijk and add general solution
        answer_sums = answer_sums/(i*j*k) + v
        
        memory = answer_sums.nbytes + asizeof(paginated_rows) + asizeof(sums) + sum1.nbytes + sum2.nbytes
        estimates = answer_sums
    else:
        '#Calculations for A1 &A2 which are then inserted into the A matrix'
        r = (i * j) + (i * k) + (j * k) - (i + j + k - 1)
        P = np.zeros(shape=(i * j * k, i * j * k))
        A = np.zeros(shape=(i * j, i * j))
        A1 = (r - (k - 1) * (j - 1)) * np.identity(i) + (j - 1) * (k - 1) * np.ones(shape=(i, i))
        A2 = (k - 1) * (i * np.identity(i) - np.ones(shape=(i, i)))
        for x in range(0, j * i, i):
            for y in range(0, j * i, i):
                if x == y:
                    A[x:x + i, y:y + i] = A1
                else:
                    A[x:x + i, y:y + i] = A2
                    
        '#Calculations for B1 &B2 which are then inserted into the B matrix'
        B = np.zeros(shape=(i * j, i * j))
        B1 = (j - 1) * (i * np.identity(i) - np.ones(shape=(i, i)))
        B2 = np.ones(shape=(i, i)) - i * np.identity(i)
        for x in range(0, j * i, i):
            for y in range(0, j * i, i):
                if x == y:
                    B[x:x + i, y:y + i] = B1
                else:
                    B[x:x + i, y:y + i] = B2
                    
        '#Inserts A and B matricies in the correct places on the P matrix'
        for x in range(0, j * i * k, i * j):
            for y in range(0, j * i * k, i * j):
                if x == y:
                    P[x:x + i * j, y:y + i * j] = A
                else:
                    P[x:x + i * j, y:y + i * j] = B

        P = (1 / (i * j * k)) * P
        Identity =np.identity(i*j*k)
        ProjP = Identity - P
            
        estimates = (ProjP @ sorted_estimates) + v
        
        if track_memory:
            memory = ProjP.nbytes + estimates.nbytes + Identity.nbytes

    return estimates, memory


def unequal_variance_three_way_cross(hist, three_way_cross, final_counts, track_memory=False):
    """
    Takes up-pass results for a three-way table and projects those results to be consistent with its marginals
    Since marginals and two-ways have already been made consistent with the total, this two-way becomes consistent with total as well

    hist - input histogram of counts
    two_way_combo - tuple of strings of the variable names under consideration
    """
    sortedlevels = n_way_order(hist, three_way_cross)
    return unequal_variance_n_way(hist, three_way_cross, final_counts, sortedlevels, track_memory)

def find_sorted_position_of_level(level_tuple, degrees):
    """
    Given an n-tuple and the sizes of the variables of its table,
    gives the position to find this variable level in a sorted list (by n_way_order())
    """
    degrees = [1] + degrees
    return sum([(value-1)*reduce(lambda a,b: a*b, degrees[0:pos+1]) for pos, value in enumerate(level_tuple)])

def find_sorted_positions_across_zeros(level_tuple, degrees):
    """
    Given an n-tuple and the sizes of the variables of its table,
    finds the lists positions of iterating over the zeros variable level in a sorted list (by n_way_order())
    """
    which_zeros = [pos for pos, value in enumerate(level_tuple) if value==0]
    first_nonzero_match = tuple(1 if level == 0 else level for level in level_tuple)
    answers = [find_sorted_position_of_level(first_nonzero_match, degrees)]

    degrees_prefixed_one = [1] + degrees
    
    for zero_pos in which_zeros:
        temp = list(answers)
        this_product = reduce(lambda a,b: a*b, degrees_prefixed_one[0:zero_pos+1])
        addends = range(this_product, degrees[zero_pos]*this_product, this_product)
        answers += [answer+addend for answer in answers for addend in addends]
        
    return answers


def create_general_solution(hist, table_tuple, target_counts, organized_by_table=True, track_memory=False):
    """
    Given a table and constraint values (either from aggregation, or down pass solutions for less complex tables),
    constructs an evenly-spaced solution by evenly distributing contraint values into the structure of this table.

    Constraints can be structured either as
     * a dictionary whose keys are variable level tuples and whose values are numbers { (1,3,4): 75, ... }
     * a dictionary whose keys are table string-tuples and whose values are lists of numbers { ('A','C'): [45, 31, ...], ... }
    Which structure is in use is declared by the `organized_by_table` boolean argument

    hist - histogram with the table being solved
    table_tuple - tuple of strings (variable names) of the table being solved
    target_counts - dictionary of constraints used in solution (see above)
    organized_by_table - dictates the constraint data structure (see note above)
    track_memory - boolean to track memory usage
    """
    memory = 0
    one_pass_memory = 0
    
    # sizes of each variable in table_tuple
    degrees = [hist.var_levels[hist.var_list.index(varname)] for varname in table_tuple]
    # positions of each variable in table_tuple in the larger histogram
    table_tuple_positions = [pos for pos, varname in enumerate(hist.var_list) if varname in table_tuple]
    number_of_target_vars = len(table_tuple)
    
    # create blank general solution
    cross_size = 1
    for degree in degrees:
        cross_size *= degree
    general_solution = numpy.array([0.0] * cross_size)

    levels_of_target_table = [tuple(hist.variables[idx]) for idx in hist.categories[table_tuple]]

    all_subsets = list(powerset(table_tuple))
    all_subsets.remove(table_tuple)
    
    if track_memory:
        memory += getsizeof(general_solution) + getsizeof(levels_of_target_table) + getsizeof(all_subsets)

    # for each subset of this three-way / n-way
    for varset in all_subsets:
        # get indices under this subset of vars
        number_of_vars = len(varset)
        #n_way_sort = hist.categories[varset]
        n_way_sort = n_way_order(hist, varset)
        
        if organized_by_table and len(varset) > 0:
            varset_counts = target_counts[varset]
        idxs = range(len(hist.categories[varset]))
        
        one_pass_memory = 0
        for idx in idxs:
            # get the numeric levels of this index
            target_levels = tuple(hist.variables[n_way_sort[idx]])
            pruned_target_levels = tuple(value for pos, value in enumerate(target_levels) if pos in table_tuple_positions)
            
            # get the indexes of levels with matching non-zero levels (e.g., from [1,0,2] get idxs of [1,1,2], [1,2,2] [1,3,2])
            # find all level-sets for which each level either matches the target or the target level is a 0
            idxs_of_nonzero_matches = find_sorted_positions_across_zeros(pruned_target_levels, degrees)
            
            if not organized_by_table:
                # target counts are organize by a dict of precise levels; grab the value at this level set
                value = target_counts[target_levels]["count"]
            else:
                if len(varset) == 0:
                    value = hist.final_estimate[0]
                else:
                    value = varset_counts[idx]
                    
            # divide the value at this idx across all nonzero matches
            value_fragment = value / len(idxs_of_nonzero_matches)
            
            # add/subtract based on whether the difference in cardinaliry between the target and the current is odd/even
            pos_neg_coefficient = 1 if (number_of_target_vars - number_of_vars) % 2 else -1
            
            # add/subtract the fragment to all positions with matching nonzero levels
            general_solution[idxs_of_nonzero_matches] += pos_neg_coefficient * value_fragment
            
            if track_memory:
                this_pass_memory = getsizeof(idxs_of_nonzero_matches) + getsizeof(n_way_sort)
                one_pass_memory = max(one_pass_memory, this_pass_memory)

    memory += one_pass_memory

    return general_solution, memory


def unequal_variance_n_way(hist, table_tuple, final_counts, sortedlevels, track_memory=False):
    """
    General N-way algorithm to get a general solution, constraints, and projection answer for a given table type

    hist - histogram object
    table_tuple - tuple of strings indicating what table to compute for, .e.g, ('A', 'C', 'D')
    final_counts - dictionary of lists that holds down-pass answers for smaller-degree tables (keys are string tuples)
    sortedlevels - list of histogram indexes
    """
    memory = 0

    # capture estimates and variances in same order as sortedlevels
    sorted_estimates = [hist.final_estimate[key] for key in sortedlevels]
    sorted_Variance = [hist.final_variances[key]  for key in sortedlevels]    

    # STEP 1: compute general solution for table_tuple's table
    v, gen_memory = create_general_solution(hist, table_tuple, final_counts, sortedlevels)

    # STEP 2: make constrint matrix for N ways
    # table_tuple is target var list, .e.g., ("A", "B", "C")
    # one_removed is each table with one variable removed from table_tuple, e.g., ("A","B"), ("A","C"), ("B","C")
    A = []
    for var_to_remove in table_tuple:
        table_tuple_idxs = hist.categories[table_tuple]
        table_tuple_levels = [hist.variables[idx] for idx in table_tuple_idxs]
        one_removed = tuple(var_name for var_name in table_tuple if var_name != var_to_remove)
        one_removed_idxs = hist.categories[one_removed]
        one_removed_var_positions = [hist.var_list.index(var_name) for var_name in one_removed]
        
        # in determining which level-1 vars to skip
        # consider if the variable in the one-removed table name is in the same position as in the target table_tuple table name
        # e.g., A,B positions A and B the same as in A,B,C
        #     -- A,C positions A the same but not C (second slot vs. third slot)
        #     -- B,C positions both and B and C differently
        # so skip the first level of C in A,C; skip first level of B and C in B,C

        # logical columns for each subtable var, each item is: (is this var same-positioned?) OR (this level != 1)
        # for each row for which these logical column values are all TRUE,
        #  make a constraint for this level ensuring that each var of the subtable is a level-match for the target table
        
        # consider the enumeration of level-sets of the one-removed table
        for idx in one_removed_idxs:
            candidate_row_levels = hist.variables[idx]
            # if each variable's level in this level-set is either:
            #   * of a variable that is same-positioned with its position in the target variable set, or
            #   * is a level with a not-1 value
            if all(one_removed.index(variable_name) == table_tuple.index(variable_name)
                   or candidate_row_levels[hist.var_list.index(variable_name)] != 1
                   for variable_name in one_removed):
                # this candidate is now approved to be made into a row
                # for each position in this row,
                #   enumerate through the combos of the target table
                #   put a 1 whenever the values of the candidate levels all equal those positions in the target table
                A.append([1 if all(table_tuple_level[position] == candidate_row_levels[position] for position in one_removed_var_positions) else 0 for table_tuple_level in table_tuple_levels])

    A = np.array(A)
    
    # STEP 3: projection
    projection_sigma, creation_memory = create_projection_matrix(np.diag(sorted_Variance), A, len(sorted_estimates), track_memory)
    estimate_results = (projection_sigma @ sorted_estimates)

    #v_results, _ , proj2_memory = projection(np.diag(sorted_Variance), v, A, track_memory)
    v_results = (projection_sigma @ v)

    Final = estimate_results + v - v_results
    
    if track_memory:
        memory = gen_memory + creation_memory + v_results.nbytes + A.nbytes
    
    return Final, memory


def uniform_vairance_down_pass(hist, means, variances, max_complexity=None, track_memory=False):
    """
    Down pass based on multiple per-table general solutions

    hist - histogram object
    dataframe - aggregation answers as Pandas dataframe
    max_complexity - the maximum number of variables to consider at once
    track_memory - tabulate memory usage information
    """
    memory = 0
    # order means and variances by histogram variable level, for use in down pass
    hist.final_estimate = list(map(lambda v: means[v], list(map(tuple, hist.variables))))
    hist.final_variances = list(map(lambda v: variances[v], list(map(tuple, hist.variables))))

    # dictionary for holding running answers
    down_pass_answers = {}
    # add the aggregation total to answers right away (not affected by down pass)
    down_pass_answers[tuple()] = [hist.final_estimate[0]]

    one_pass_memory = 0

    # get all subtables that can be formed by the histogram's variables
    table_tuples = powerset(hist.var_list)
    
    # iterate through tables in increasing complexity
    for table_tuple in table_tuples:
        # if this table is more complex than max complexity, skip it
        if max_complexity is not None and len(table_tuple) > max_complexity:
            continue

        # get all sums that can be tabulated from this table
        sum_dict = subtable_sums_from_table(hist, table_tuple, hist.final_estimate, target_cardinality = max_complexity, for_aggregation=False)
        # compute general solution for this table based on sums
        z_acute, first_memory = create_general_solution(hist, table_tuple, sum_dict, organized_by_table=False, track_memory=track_memory)
        # compute general solution for this table based previous answers
        z_hat, second_memory = create_general_solution(hist, table_tuple, down_pass_answers, organized_by_table=True, track_memory=track_memory)
        
        # get aggregation answers for this table
        aggregate_answer_for_table = numpy.array([hist.final_estimate[idx] for idx in n_way_order(hist,table_tuple)])
        
        # adjust aggregation answer based on general solutions for this table and save it
        down_pass_answers[table_tuple] = aggregate_answer_for_table - z_acute + z_hat
        
        # tabulate and store max memory so far
        if track_memory:
            this_pass_memory = second_memory + getsizeof(z_acute) + getsizeof(aggregate_answer_for_table)
            one_pass_memory = max(one_pass_memory, this_pass_memory)
        
    memory += getsizeof(down_pass_answers) + one_pass_memory
    
    return down_pass_answers, memory


def down_pass(hist, means, variances, track_memory=False, max_complexity=None, use_matmul=False):
    """
    Executes the SEA down pass algorithm.

    Returns a dictionary of lists whose keys are tuples of variable names (i.e., cross table identifiers) and whose values are lists of final mean estiamtes

    {
      (): [500.7],
      ('A','B','D'): [1, 5, 7, 8, 2, 2, ...],
      ...
    }

    Table value-lists are ordered by last variable rotating first: (1,1,1), (1,1,2), (1,1,3) ... (1,2,1) (1,2,2) (1,2,3) ...

    hist- histogram being down-passed
    means - aggregation means, dict keyed by variable-level tuples
    variances - aggregation variances, dict keyed by variable-level tuples
    track_memory - boolean to track memory usage
    max_complexity - maximum number of variable interactions to consider at once
    use_matmul - for 3-way unequal-variance table case, use the matrix multipication approach (versus the pagination approach)
    """
    memory = 0
    
    # order means and variances by histogram variable level, for use in down pass
    hist.final_estimate = list(map(lambda v: means[v], list(map(tuple, hist.variables))))
    hist.final_variances = list(map(lambda v: variances[v], list(map(tuple, hist.variables))))
    
    # determine if each category has uniform vairance
    has_uniform_variance = all(len(set(hist.variance[i] for i in category_idxs)) == 1 for category_idxs in hist.categories.values())
    
    # if each table has the same variance for each of its entries, use the uniform-variance approach
    if has_uniform_variance:
        return uniform_vairance_down_pass(hist, means, variances, max_complexity, track_memory)
    
    # if each table does NOT have the same variance for each of its entries, use more complicated approach
    one_way_memory = 0
    two_way_memory = 0
    three_way_memory = 0

    Final_Results = {}
    variances = [0]
    
    #Loops through all One way interactions calculating down pass Estimates & Variance
    for index in range(hist.num_vars):
        marginal_result, new_one_way_memory = one_way_cross(hist, index, track_memory)
        one_way_memory = max(new_one_way_memory, one_way_memory)
        Final_Results[(hist.var_list[index],)] = marginal_result.tolist()[0]
    
    #Loops through all Two way interactions calculating down pass Estimates & Variance
    two_way_tables = [table for table in hist.categories.keys() if len(table) == 2]
    for item in two_way_tables:
        two_way_result, new_two_way_memory = unequal_variance_two_way_cross(hist, item, Final_Results, track_memory)
        two_way_memory = max(new_two_way_memory, two_way_memory)
        Final_Results[item] = two_way_result.tolist()
    
    #Loops through all Three way interactions calculating down pass Estimates & Variance
    three_way_tables = [table for table in hist.categories.keys() if len(table) == 3]
    for item in three_way_tables:
        three_way_results, new_three_way_memory = unequal_variance_three_way_cross(hist, item, Final_Results, track_memory)
        three_way_memory = max(new_three_way_memory, three_way_memory)
        Final_Results[item] = three_way_results.tolist()

    if track_memory:
        memory = max(one_way_memory, two_way_memory, three_way_memory)

    return Final_Results, memory

def quick_analysis(final_estimates, hist, max_complexity=None, track_memory = False):
    """
    Puts down pass answers in a dict of same-orded list dataframe. Not part of the SEA algorithm but used to order and collate results.

    final_estimates - down_pass output: dict with var-level tuple keys and lists of numbers as values
    hist - histogram to collate data from
    max_complexity - only render results from tables at this complexity or below
    track_memory - boolean to track memory usage
    """
    #GroupedFinals = pd.DataFrame(columns=["Table", "True_Value", "Final_Estimate", "Aggregation_Estimate", "Variance", "Level"]).astype({
    #    "Table":"object", "True_Value":"float", "Final_Estimate":"float", "Aggregation_Estimate":"float", "Variance":"float", "Level":"object"
    #});
    GroupedFinals = { "Final_Estimate":[], "True_Value":[], "Combo":[] }
    for table_tuple, idxs in hist.categories.items():
        if max_complexity and len(table_tuple) > max_complexity:
            continue
        # get which variables this table pertains to
        table_tuple_positions = [pos for pos, varname in enumerate(hist.var_list) if varname in table_tuple]
        degrees = [hist.var_levels[pos] for pos in table_tuple_positions]
        for idx in idxs:
            # get the level at this position 
            level = tuple(hist.variables[idx])
            pruned_level = tuple(value for pos, value in enumerate(level) if pos in table_tuple_positions)
            position_within_table = find_sorted_position_of_level(pruned_level, degrees)
            # get the estimate for this position (or NaN if this level was never computed)
            if table_tuple in final_estimates:
                final_estimate_value = final_estimates[table_tuple][position_within_table]
            else:
                final_estimate_value = float("nan")
            # add a new entry for SEA answer, true answer, and var level
            GroupedFinals["Final_Estimate"].append(final_estimate_value)
            GroupedFinals["True_Value"].append(hist.true_data[idx])
            GroupedFinals["Combo"].append(tuple(hist.variables[idx]))
            #GroupedFinals.loc[len(GroupedFinals.index)] = [table_tuple, hist.true_data[idx], final_estimate_value, hist.final_estimate[idx], hist.final_variances[idx], level]
    return GroupedFinals, hist, 0


def analysis(final_estimates, hist, track_memory = False):
    """
    Puts down pass answers in a Pandas dataframe. Not part of the SEA algorithm but used to order and collate results.

    final_estimates - down_pass output: dict with var-level tuple keys and lists of numbers as values
    hist - histogram to collate data from
    track_memory - boolean to track memory usage
    """
    GroupedFinals = pd.DataFrame(columns=["Type", "True_Value", "Final_Estimate", "Aggregation_Estimate" , "Variance", "Combo"])
    hist.collected_true_data = hist.true_data
    hist.collected_noisy_data = hist.noised_data

    OneWayAttributes = {k: v for k, v in hist.categories.items() if len(k) == 1}
    TwoWayAttributes = {k: v for k, v in hist.categories.items() if len(k) == 2}
    ThreeWayAttributes = {k: v for k, v in hist.categories.items() if len(k) == 3}

    GroupedFinals.loc[len(GroupedFinals.index)] = ["Total",hist.collected_true_data[0],hist.final_estimate[0],hist.collected_noisy_data[0],hist.final_variances[0],tuple()]
    #One Way Table
    testindex = 0
    varianceIndex = 1
    value = OneWayAttributes.get(list(OneWayAttributes.keys())[testindex])
    order = []
    for y in OneWayAttributes.values():
        order.extend(y)
    for y in TwoWayAttributes.values():
        order.extend(y)
    for y in ThreeWayAttributes.values():
        order.extend(y)

    marginal_dict = { k[0]: v for k, v in final_estimates.items() if len(k) == 1 }
    for var_name, x in marginal_dict.items():
        index = 0 
        for y in x:
            GroupedFinals.loc[order[varianceIndex-1]] = ["Marginal", hist.collected_true_data[order[varianceIndex-1]],x[index],hist.final_estimate[order[varianceIndex-1]],hist.final_variances[order[varianceIndex-1]], (var_name,)]
            index +=1
            varianceIndex+=1
        testindex+=1
        
    two_way_dict = { k: v for k, v in final_estimates.items() if len(k) == 2 }
    for combo, positions in two_way_dict.items():
        left_pos = hist.var_list.index(combo[0])
        right_pos = hist.var_list.index(combo[1])
        i=hist.var_levels[left_pos]
        j=hist.var_levels[right_pos]
        reorder = []
        current = 0
        for y in range(i):
            for x in range(j):
                reorder.append(current)
                current = current+ i
            current = current - i*j +1

        for index, y in enumerate(positions):
            entry = ["Two Way",
                     hist.collected_true_data[order[varianceIndex-1]],
                     positions[reorder[index]],
                     hist.final_estimate[order[varianceIndex-1]],
                     hist.final_variances[order[varianceIndex-1]],
                     combo]
            GroupedFinals.loc[order[varianceIndex-1]] = entry
            varianceIndex += 1
        
    three_way_dict = { k: v for k, v in final_estimates.items() if len(k) == 3 }
    for combo, positions in three_way_dict.items():
        left_pos = hist.var_list.index(combo[0])
        middle_pos = hist.var_list.index(combo[1])
        right_pos = hist.var_list.index(combo[2])
     
        i=hist.var_levels[left_pos]
        j=hist.var_levels[middle_pos]
        k=hist.var_levels[right_pos]
        reorder = []
        current = 0
        #breakpoint()
        for z in range(i):
            for y in range(j):
                for x in range(k):
                    reorder.append(current)
                    current = current+ j*i
                current = current - k*j*i + i
            current = current - j*i + 1

        for index, y in enumerate(positions):
            GroupedFinals.loc[order[varianceIndex-1]] = ["Three Way",
                                                         hist.collected_true_data[order[varianceIndex-1]],
                                                         positions[reorder[index]],
                                                         hist.final_estimate[order[varianceIndex-1]],
                                                         hist.final_variances[order[varianceIndex-1]],
                                                         combo]
            varianceIndex += 1
    memory = 0

    return GroupedFinals, hist, memory

def run_aggregation_step(hist, track_memory = False, target_cardinality=None):
    """
    Executes the collection pass.
    Return values: means, variances, memory.

    Returns two dictionaries whose keys are variable level tuples and whose values are mean & variance values:

    {
      (0,0,0): 500,
      (0,0,1): 247,
      ...
    }

    Also returns memory usage, which is meaningfully populated if track_memory is True.

    hist - histogram to run aggregation
    track_memory - boolean to track memory usage
    target_cardinality - maximum number of variables considered in interactions
    """
    if track_memory:
         tr = tracker.SummaryTracker()

    running_cs = defaultdict(lambda: 0)
    running_ds = defaultdict(lambda: 0)

    # initalize each C and D entry
    for var_combo, noisyData, variance in zip(list(map(tuple, hist.variables)), hist.noised_data, hist.variance):
        # if this is a blank entry, skip it
        if variance == -1:
            continue
        
        running_cs[var_combo] = noisyData / variance
        running_ds[var_combo] = 1 / variance

    # for each category, add all the elements of this category to the input dict
    for category in hist.categories:
        category_answers = subtable_sums_from_table(hist, category, hist.noised_data, target_cardinality, for_aggregation=True)
        
        for var_combo, answer in category_answers.items():
            running_cs[var_combo] += answer["count"] / answer["variance"]
            running_ds[var_combo] += 1 / answer["variance"]

    # gets means and vairances from C and D values
    means = {}
    variances = {}
    for k,d in running_ds.items():
        means[k] = running_cs[k] / d
        variances[k] = 1 / d

    if track_memory:
        memory = sum(item[2] for item in tr.diff())
    else:
        memory = 0
          
    return means, variances, memory

def create_subtable_sums(category_mask, elems_in_category, var_levels, final_answers, target_cardinality, target_cardinality_count, max_target_cardinality):
    """
    Recursively collapse tables whose variables are selected by non-zero positions in the category_mask

    category_mask - tuple of 1s and 0s indicating which variables are to be considered for subtable creation
    elems_in_category - list of {idx,var_combo,count,variance} dicts for postions from the histogram within this variable category
                        (literally, information about all hist positions from a single list-value out of the hist.categories dict)
    var_levels - list that tells the cardinality of each variable; smae as hist.var_levels
    final_answers - dict for storing answers whose keys are var-level tuples and values are {count,variance} dicts
                     e.g., {(1,0,2): { "count":20, "variance": 2 }, ... }
    target_cardinality - an integer capping the number of variable interactions we want in answers (e.g., 3 will leave out 4-way answers like ABCD)
    target_cardinality_count - number of combinations we have handled that are of the max cardinality
    max_target_cardinality - number of maximum-cardinality combinations we can admit (n-choose-k result for "target_cardinality choose cardinality(initial category_mask)")
    """
    # if this subtable is of target cardinality
    if target_cardinality is not None and cardinality(category_mask) >= target_cardinality:
        if max_target_cardinality is not None and target_cardinality_count == max_target_cardinality:
            return target_cardinality_count

    estimate_sums_by_table = {}
    list_of_vars_to_eliminate = [n for n,v in enumerate(category_mask) if v!=0]

    # sort eliminations by largest to smallest var, to do expensive elminations first
    list_of_vars_to_eliminate = sorted(list_of_vars_to_eliminate, key=lambda idx: var_levels[idx], reverse=True)

    # for each var under consideration, find summations of subtables with this variable removed
    for nth_target_var in list_of_vars_to_eliminate:

        # create dict of lists of the form { (0,1,1):[{}, {}, ...], (0,1,2): [{}, {}, ...] }
        # list elements are {idx,data,variance,var_combo} for every combo under the zeroed-out type
        #   (1,1,0)-indexed list contains entries for (1,1,1) (1,1,2) (1,1,3) ...
        elems_by_var_with_nth_removed = defaultdict(list)
        for elem in elems_in_category:
            # categorize each element by variable level with an extra zero replaced in
            var_combo_with_nth_removed = elem["var_combo"][:nth_target_var] + (0,) + elem["var_combo"][nth_target_var+1:]
            # only include reductions for which we don't already have an answer
            if var_combo_with_nth_removed not in final_answers.keys():
                elems_by_var_with_nth_removed[var_combo_with_nth_removed].append(elem)
                
        # add up subtables with a variable removed and put it in the answer dict
        for var_combo, elems in elems_by_var_with_nth_removed.items():
            if var_combo not in estimate_sums_by_table:
                estimate_sums_by_table[var_combo] = {
                    "var_combo": var_combo,
                    "count": sum(elem["count"] for elem in elems),
                    "variance": sum(elem["variance"] for elem in elems)
                }
        final_answers.update(estimate_sums_by_table)

        var_combos = elems_by_var_with_nth_removed.keys()
        # flatten all var combos to 1 & 0 for nonzero and zero levels
        var_level_masks = set(tuple(1 if x else 0 for x in combo) for combo in var_combos)

        new_card_match_count = len(list(filter(lambda m: cardinality(m) == target_cardinality, var_level_masks)))
        target_cardinality_count += new_card_match_count

        # recurse for other combos
        for var_combo in var_level_masks:
            target_cardinality_count = create_subtable_sums(var_combo, estimate_sums_by_table.values(), var_levels, final_answers, target_cardinality, target_cardinality_count, max_target_cardinality)

    return target_cardinality_count


def subtable_sums_from_table(hist, table_tuple, counts, target_cardinality = None, for_aggregation=True):
    """
    Given a table tuple, sum up as many collapses of that table as possible
    e.g., from ('A','B','C'), create sums for ('A','B') ('A','C') ('B','C') ('A',) ('B,') ('C',) ()

    hist - histogram whose tables are being summed
    table_tuple - tuple of strings (variable names of the cross to collapse)
    counts - list of values in this table to sum up into collapsed subtables
    target_cardinality - upper bound for complexity of sums
    for_aggregation - if this is called from the aggregation step versus the down pass
                      (slightly different behavior, for missing tables)
    """
    # for each category, add all the elements of this category to the input dict
    idxs = hist.categories[table_tuple]
    
    category_answers = {}
    
    # if this is a blank table, skip it
    if for_aggregation and hist.variance[idxs[0]] == -1:
        return category_answers
    
    elems_in_category = list(map(lambda idx: {
        "idx":idx,
        "var_combo": tuple(hist.variables[idx]),
        "count": counts[idx],
        "variance": hist.variance[idx]
    }, idxs))

    var_mask = tuple(1 if (name in table_tuple) else 0 for name in hist.var_list)

    current_cardinality = cardinality(var_mask)
    # how many var combos we can make of the target cardinality from our input combo
    if target_cardinality is None:
        target_cardinality_count = float("inf")
    else:
        target_cardinality_count = math.comb(current_cardinality, target_cardinality)

    create_subtable_sums(var_mask, elems_in_category, hist.var_levels, category_answers, target_cardinality, 0, target_cardinality_count)
        
    return category_answers

def generate_error_df(projection_result, hist, limit=None):
    """
    Given a projection answer array and a Histogram object, build a Pandas dataframe aligning projection
    answers and Histogram true values. Optionally limited to tables at or under a cardinality limit.
    """
    # get and sort all values for 3-way, 2-way, and marginal reults, for calculating error against true data
    if limit:
        list_of_lists = [indexes for category, indexes in hist.categories.items() if len(category) <= limit]
    else:
        list_of_lists = [indexes for category, indexes in hist.categories.items()]
    index_list = [item for sublist in list_of_lists for item in sublist]
    index_list.sort()
    final_estimates = projection_result.tolist()
    marginal_names = hist.var_list
    levels = [hist.all_vars[i] for i in index_list]
    combos = [tuple(marginal_names[idx] for idx,val in enumerate(level) if val!=0) for level in levels]
    selected_finals = [final_estimates[i] for i in index_list]
    selected_trues = [hist.true_data[i] for i in index_list]
    proj_df = pd.DataFrame(columns=["True_Value", "Final_Estimate", "Combo"])
    proj_df["Combo"] = combos
    
    proj_df["Final_Estimate"] = selected_finals
    proj_df["True_Value"] = selected_trues
    return proj_df


def calculate_error(GroupedFinals, query_selection=None):
    """
    Given a Pandas dataframe produced by analysis(), compute mean square error and mean absolute error
    between the Final_Estimate and True_Value columns. Optionally limited by a list of table tuples

    GroupedFinals - a Pandas dataframe as produced by analysis(), with True_Value, Final_Estimate, Combo columns
    query_selection - a list of tuples of strings, e.g. [("A","B"), ("C","D","E")], limiting which tables to tabulate the error for 
    """
    GroupedFinals = GroupedFinals.sort_index(axis = 0).loc[~GroupedFinals["True_Value"].isna()]
    if query_selection is not None:
        GroupedFinals = GroupedFinals.loc[GroupedFinals["Combo"].isin(query_selection)]
    
    MSE = ((GroupedFinals["True_Value"] - GroupedFinals["Final_Estimate"])**2).mean()
    MAE = (abs((GroupedFinals["True_Value"] - (GroupedFinals["Final_Estimate"])))).mean()
    return MSE , MAE


def variance_generation(hist, max_table_size = None, track_memory = False):
    """
    Creates an array of variance answers. If possible, does the faster approach of solving each table once, rather than each individual value.

    Returns a list of vairances in histogram order.
    """
    # determine if each category has uniform vairance
    has_uniform_variance = all(len(set(hist.variance[i] for i in category_idxs)) == 1 for category_idxs in hist.categories.values())

    if has_uniform_variance:
        return tablewise_variance_generation(hist, max_table_size, track_memory)
    else:
        return individual_variance_generation(hist, track_memory)


def tablewise_variance_generation(hist, max_table_size=None, track_memory = False):
    """
    Compute variance for a histogram wherein each table has internally equal variance.
    This works by computing variance for a single element from each table, then applying
    that answer to each entry in the table.

    hist - histogram to compute variance for
    max_table_size - maximum table size to compute variance for
    track_memory - record memory usage
    """
    # stash actual histogram before we repurpose it with variance values
    original_noisy = hist.noised_data
    original_true = hist.true_data
    original_final_estimate = hist.final_estimate

    total_memory = 0

    variances = [None]* len(hist.noised_data)
    # since all entries in a category have the same variance,
    # consider only one entry per category to calculate variance for the whole category
    
    # run SEA repeatedly using a variance from each table as input
    # Memory tracking: one execution of analysis, plus total storage of `variances` list
    for category, index_list in hist.categories.items():
        # only compute variances for 1-, 2-, and 3-way tables
        if max_table_size is not None and len(category) > max_table_size:
            continue
    
        first_index = index_list[0]
        col = [0] * len(hist.variance)
        
        # replace missing variances with high values
        if hist.variance[first_index] == -1:
            input_variance = 1e100
        else:
            input_variance = hist.variance[first_index]
        
        col[first_index] = input_variance
        hist.noised_data = col
        hist.true_data = col
        
        # run SEA; only log memory if we're tracking memory AND this is the first iteration
        var_estimate, _, agg_mem = run_aggregation_step(hist, target_cardinality=max_table_size, track_memory=(track_memory and total_memory == 0))
        final_estimates, down_pass_memory = down_pass(hist, var_estimate, _, max_complexity=max_table_size, track_memory=(track_memory and total_memory == 0))

        answer = final_estimates[category][0]

        # save computed variance for entire category
        for index in index_list:
            variances[index] = answer
            
        # add in first-iteration analysis and agg memory usage
        total_memory += agg_mem + down_pass_memory
    total_memory = asizeof(variances) + total_memory
    
    variances = [v for v in variances if v is not None]
    
    # restore actual histogram
    hist.noised_data = original_noisy
    hist.true_data = original_true
    hist.final_estimate = original_final_estimate
    
    return variances, hist, total_memory


def individual_variance_generation(hist, track_memory = False):
    """
    In the case that there are counts whose variances differ within a table, we must compute the variance for each count individually.

    Reutrns a list of variances in histogram order.
    """
    # stash actual histogram before we repurpose it with variance values
    original_noisy = hist.noised_data
    original_true = hist.true_data
    original_final_estimate = hist.final_estimate

    total_memory = 0
    
    variance_matrix = []
    variances = [None] * len(hist.noised_data)
    variance_order = []

    # run SEA repeatedly using each variance as input
    for varindex, variance in enumerate(hist.variance):
                
        # replace missing variances with high values
        if variance == -1:
            variance = 1e100
    
        col = [0] * len(hist.variance)
        col[varindex] = variance**(1/2)
        
        hist.noised_data = col
        hist.true_data = col
        
        # run SEA; only log memory if we're tracking memory AND this is the first iteration
        var_estimate, _, agg_mem = run_aggregation_step(hist, track_memory=(track_memory and total_memory == 0))
        final_estimates, down_pass_memory = down_pass(hist, var_estimate, _, track_memory=(track_memory and total_memory == 0))
        var_estimate, hist, analysis_mem = analysis(final_estimates, hist, (track_memory and total_memory == 0))
        variance_order.append(int(var_estimate.iloc[varindex].name))
 
        if track_memory and total_memory == 0:
            tr = tracker.SummaryTracker()
        
        var_estimate["Identity"] = [hist.variables[idx] for idx, val in var_estimate.iterrows()]
        variance_matrix.append(list(var_estimate["Final_Estimate"]))

    variance_matrix_transpose = np.matrix(variance_matrix)
    variance_matrix = variance_matrix_transpose.transpose()
    
    for i,row in enumerate(variance_matrix):
        matmul_answer = (row @ row.transpose())
        
        # store first matmul in total_memory
        if track_memory and total_memory == 0:
            total_memory = sum(item[2] for item in tr.diff())

        output_idx = variance_order[i]
        variances[output_idx] = (matmul_answer[0,0])
            
        # add in first-iteration analysis and agg memory usage
        total_memory += analysis_mem + agg_mem
        
    total_memory = asizeof(variances) + total_memory

    hist.noised_data = original_noisy
    hist.true_data = original_true
    hist.final_estimate = original_final_estimate

    return variances, hist, total_memory


def calculate_confidence_interval(means, variances, confidence, is_estimated=False, degrees_of_freedom=None):
    """
    Compute the lower and upper bounds within the specified confience percentage, given means and variances.

    means - mean answers
    variances - variance answers
    confidence - value between 0 and 1
    is_estimated - boolean; was this variance estimated? (otherwise it was calculated)
    degrees_of_freedom - if estimated, how many estimation passes were used
    """
    # find a symmetric range on standard norm that will contain a confidence proportion of the values
    # i.e., the integral of standard normal under (-critical_stat, +critical_stat) is equal to `confidence`
    
    if not is_estimated:
        critical_stat = norm.ppf(1 - (1 - confidence) / 2)
    else:
        critical_stat = t.ppf(1 - (1 - confidence) / 2, degrees_of_freedom)
    
    # multiply crit stat by stdev and add/subtract product from means to gte upper/lower bounds
    variance_products = [critical_stat * (math.sqrt(v) if v!=-1 else 1e100) for v in variances]
    lower_bounds = [m - v for m,v in zip(means, variance_products)]
    upper_bounds = [m + v for m,v in zip(means, variance_products)]
    
    return lower_bounds, upper_bounds
    
def variance_estimation(hist, num_iterations=20, confidence=0.95, max_complexity=None):
    """
    Estimate variance by repeatedly running SEA collection and down pass on noise from input variances
    
    Returns
      1. estimated variances for each histogram position, and
      2. order-statistic-based selection of std dev for each position
    """
    hist.true_data = [0] * len(hist.true_data)
    results = []
    for n in range(num_iterations):
        hist.create_noised_data_from_true_and_variance()

        # do aggregation
        means, variances, aggregation_memory = run_aggregation_step(hist, target_cardinality = max_complexity)
        # compute down pass
        final_estimates, down_pass_memory = down_pass(hist, means, variances, max_complexity)
        hist_final, hist, analysis_memory = quick_analysis(final_estimates, hist)
        results.append(hist_final["Final_Estimate"])
        
    mses = [sum([sample**2 for sample in samples]) / len(samples) for samples in zip(*results)]
    
    ratio = 1 - (1 - confidence) / 2
    which_sample = math.ceil((num_iterations - 1) * ratio)
    stdev_percentiles = [sorted(abs(sample) for sample in samples)[which_sample] for samples in zip(*results)]
    return mses, stdev_percentiles
    
def clip_confidence_range(upper, lower):
    """
    Clip confidence range to nonnegative integer range
    """
    clipped_upper = []
    clipped_lower = []

    # note: if too slow, rewrite as list comprehension
    for upper_item, lower_item in zip(upper, lower):
        if upper_item < 0:
            clipped_upper.append(None)
            clipped_lower.append(None)
        else:
            clipped_upper.append(math.floor(upper_item))
            # note: might be faster to make all negative lower values 0 and skip max in loop
            clipped_lower.append(max(0, math.ceil(lower_item)))
    return clipped_upper, clipped_lower
    
    