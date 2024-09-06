"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176
"""

from AproxBLUE import Histogram, run_aggregation_step, down_pass, analysis, variance_generation, calculate_error, calculate_confidence_interval, create_projection_matrix, generate_error_df
from treegen_sparse import A_make_from_detailed
import numpy, pandas
from NMFHistogram import NMFHistogram

#####
# Set up varaibles to configure sample histogram
#####

# Number of geographic levels
# e.g., depth of 3 could mean Nation, State, County
depth = 1

# Vector of possible levels for variables
# e.g., "sex * age * hispanic" could have shape [2, 110, 2]
levels = [4,4,4,4]

# Number of variables
# e.g., "sex * age * hispanic" has 3 varaibles
numvars = len(levels)

# maximum level of interaction given
# e.g., even if there are 5 variables (A,B,C,D,E), this can cap how big the crosses are:
#       detail = 3 would make only 3-way crosses and below (A*B, B*C, etc., A*B*C, B*C*D, etc.),
#         and exclude the 5-way cross A*B*C*D*E and 4-way crosses A*B*C*D, A*B*C*E, etc.
#         and exclude the 5-way cross A*B*C*D*E and 4-way crosses A*B*C*D, A*B*C*E, etc.
detail = numvars


# Number of geographic children per parent
# e.g., children=3 means a nation with 3 states, each state with 3 counties, each county with 3 tracts, etc.
children = 0

# Probability of zeros in the most detailed greographic leaf nodes
# Makes geographic leaves have fully-detailed zeroes more or less frequent
# Probability that everything the goes int othe detailed query are zero
# e.g. if detail=3 and levels=[3,3,3,3,3], then Prob(zero in detailed query) = p_zero^(1 / 3^2 )
p_zero = 0.5

#####
# STEP 0: create simulated histogram using settings
#####
hist = Histogram(num_vars = numvars, var_levels = levels, num_children = children, max_detail = detail, max_depth=1)

# alternative approach: making a histogram from PL-94 NMF parquet file, instead of simulated data
"""
hist = NMFHistogram(-1,
                    "ri_block_nmf.parquet",
                    "ri2010.sf1\\ri000{0}2010.sf1",
                    ["hhgq", "votingage", "hispanic", "cenrace"],
                    '0441000700491024400701070233001')
"""

original_true = hist.true_data

#####
# STEP 1: perform projection approach
#####

# create a full rank constraint matrix
A = A_make_from_detailed(hist, hist.var_list)

# make diagonal vairance matrix
# -1 variance values signal that this count was not present in the input and should be skipped
variance_diag = numpy.diag([v if v!=-1 else 1e100 for v in hist.variance])

# perform projection
projection_sigma, projection_creation_memory = create_projection_matrix(variance_diag, A, len(hist.noised_data))

# get projection-based variance via matrix multiplication
entire_variance_estimate = projection_sigma @ variance_diag
projection_result = projection_sigma @ hist.noised_data

# generate dataframe of final estimates against true values, for 3-way and below
proj_df = generate_error_df(projection_result, hist, limit=3)

# compute projection error
proj_MSE, proj_MAE = calculate_error(proj_df)
print("projection MSE: "+ str(proj_MSE))
print("projection MAE: "+ str(proj_MAE))

#####
# STEP 2: SEA BLUE
#####

# do aggregation
means, agg_variances, aggregation_memory = run_aggregation_step(hist)

# compute down pass and organize returned data into dataframe format
final_estimates, down_pass_memory = down_pass(hist, means, agg_variances)

# set up dataframe for result displays
#df = setup_dataframe(hist, means, agg_variances)
hist_final, hist, analysis_memory = analysis(final_estimates, hist)

# compute variance
variances, hist, variance_memory = variance_generation(hist)

# compute ratio of true values within projection mean/variance result
lower_bounds, upper_bounds = calculate_confidence_interval(projection_result, numpy.diag(entire_variance_estimate), 0.95, False, None)
confidence_answers = [upper > true > lower for upper, lower, true in zip(upper_bounds, lower_bounds, original_true) if true!=None]
print("Projection coverage:", sum(confidence_answers) / len(confidence_answers))


# Compute error for final estimates against true data
MSE , MAE = calculate_error(hist_final)
print("SEA MSE: "+ str(MSE))
print("SEA MAE: "+ str(MAE))
# compute ratio of true values within SEA mean/variance result
lower_bounds, upper_bounds = calculate_confidence_interval(hist_final["Final_Estimate"], variances, 0.95, False, None)
confidence_answers = [upper > true > lower for upper, lower, true in zip(upper_bounds, lower_bounds, hist_final["True_Value"]) if true!=None]
print("SEA coverage:", sum(confidence_answers) / len(confidence_answers))

