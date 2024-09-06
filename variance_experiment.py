"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176
"""
import os
from AproxBLUE import Histogram, down_pass, quick_analysis , run_aggregation_step , analysis , individual_variance_generation , calculate_error , generate_error_df, cardinality, variance_generation
from operator import attrgetter
import pickle, sys
import pandas as pd
import time
import numpy
import gc
from NMFHistogram import NMFHistogram
import numpy as np
import math

is_continuous = True

smallest_decreases = []
count = 10

for _ in range(count):
    hist = Histogram(num_vars = 3, var_levels = [3,3,3], num_children = 0, max_detail = 3, max_depth=1, continuous=is_continuous)

    hist.variance = [[4,50*count][i%2] for i,v in enumerate(hist.variance)]

    #three_way_and_under_idxs = list(map(lambda p: p[0], filter(lambda v: cardinality(v[1]) < 4, enumerate(hist.variables))))
    #three_way_and_under_variances = [v for idx,v in enumerate(hist.variance) if idx in three_way_and_under_idxs]

    means, middle_variances, memory = run_aggregation_step(hist, track_memory = False, target_cardinality = 3)
    ordered_middle_variances = list(map(lambda v:middle_variances[v], hist.all_vars))

    # compute down pass
    final_estimates, down_pass_memory = down_pass(hist, means, middle_variances, max_complexity = 3, track_memory = False)
    

    # compute variance
    final_variances, hist, total_memory = variance_generation(hist, max_table_size = 3, track_memory = False)

    decreases = [post - final for post,final in zip(ordered_middle_variances, final_variances)]
    print(min(decreases))
    smallest_decreases += [min(decreases)]

    #Generates the calculated variance 
    #variances, hist, variance_memory = experimental_covariance_variance_generation(hist, track_memory)
    #

    #final_three_way_and_under_variances = [v for idx,v in enumerate(variances) if idx in three_way_and_under_idxs]

    #print((hist_final["True_Value"] - hist_final["Final_Estimate"]) ** 2)

    #hist_final, hist = quick_analysis(final_estimates, hist, variances, max_complexity=None)

    #final_squared_errors.append((hist_final["True_Value"] - hist_final["Final_Estimate"]) ** 2)
    #agg_squared_errors.append((hist_final["True_Value"] - hist_final["Aggregation_Estimate"]) ** 2)

    #

#avg_err = sum(final_squared_errors) / count
#avg_agg = sum(final_squared_errors) / count

print("Smallest decreases (if negative, variance has increased during down pass):", smallest_decreases)

#print("Average error:", avg_err)
#print(list(avg_err))

    #print("Final sum", sum(three_way_and_under_variances))
    #print(three_way_and_under_variances)
    #print(middle_variances)
    #print(final_three_way_and_under_variances)
    #print(list(zip(hist.all_vars, final_three_way_and_under_variances)))
    #print(list(map(lambda a: a[0] - a[1], zip(middle_variances, final_three_way_and_under_variances))))
    #print(hist.all_vars)
