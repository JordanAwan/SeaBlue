"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176'
"""

"""
Runs batched time and memory trials of SEA BLUE and projection BLUE.

For a list of options, run `python testing_doc.py --help`
"""
from AproxBLUE import Histogram, run_aggregation_step, down_pass, quick_analysis, variance_generation, variance_estimation, clip_confidence_range, calculate_error, calculate_confidence_interval, create_projection_matrix, generate_error_df
from treegen_sparse import A_make, A_make_from_detailed
import pandas as pd
import time, os
import numpy
import argparse
from NMFHistogram import NMFHistogram
from pympler import tracker
from sys import getsizeof
import numpy as np
import math

var_results = pd.DataFrame(columns=["size", ""])

#'number of geographic levels
depth = 1
#' number of children per parent
children = 0
#' probability of zeros in the most detailed leaf nodes
p_zero = 0.5 #Default 0.5

"""
Run a single time trial on a single histogram

hist - histogram to be timed
track_memory - boolean to track or ignore memory use
do_projection - perform projection BLUE approach
do_sea - perform SEA BLUE approach
"""
def run_trial(hist, track_memory, do_projection, do_sea, do_confidence, max_table_size):
    result = {}
    result["size"] = f"{hist.var_levels}"

    # projection approach first
    if do_projection:
        blueT0 = time.time()
        
        try:
            # create a full rank constraint matrix
            if track_memory:
                tr = tracker.SummaryTracker()
            A = A_make_from_detailed(hist, hist.var_list)
            if track_memory:
                A_memory = sum(item[2] for item in tr.diff()) + getsizeof(A)
            else:
                A_memory = 0
        except Exception as e:
            print(e)
            A_memory = 0
            
        blueT1 = time.time()

        blueT2 = time.time()
        # create projection matrix
        try:
            variance_diag = numpy.diag([v if v!=-1 else 1e100 for v in hist.variance])
            projection_sigma, projection_creation_memory = create_projection_matrix(variance_diag, A, len(hist.noised_data), track_memory)
            projection_creation_memory += variance_diag.nbytes
        except Exception as e:
            print("ERROR in projection creation", e)
            projection_creation_memory = 0
            
        blueT3 = time.time()
            
        # perform projection to get means
        try:
            projection_result = projection_sigma @ hist.noised_data
            if track_memory:
                projection_mean_execution_memory = projection_result.nbytes
            else:
                projection_mean_execution_memory = 0
        except Exception as e:
            print("ERROR in projection execution", e)
            projection_mean_execution_memory = 0
            
        blueT4 = time.time()
            
        # perform projection to get variances
        try:
            entire_variance_estimate = projection_sigma @ variance_diag
            if track_memory:
                projection_variance_execution_memory = entire_variance_estimate.nbytes
            else:
                projection_variance_execution_memory = 0
        except Exception as e:
            print("ERROR in projection execution", e)
            projection_vairance_execution_memory = 0
            
        blueT5 = time.time()

        print("constraint time:", blueT1 - blueT0)
        print("projection creation time:", blueT3 - blueT2)
        print("projection mean execution time:", blueT4 - blueT3)
        print("projection variance execution time:", blueT5 - blueT4)
        print("constraint memory:", A_memory)
        print("projection size:", projection_sigma.nbytes)
        print("projection creation memory:", projection_creation_memory)
        print("projection mean execution memory:", projection_mean_execution_memory)
        print("projection variance execution memory:", projection_variance_execution_memory)

        result["constraint_time"] = blueT1 - blueT0
        result["projection_creation_time"] = blueT3 - blueT2
        result["projection_mean_execution_time"] = blueT4 - blueT3
        result["projection_variance_execution_time"] = blueT5 - blueT4
        result["proj_creation_memory"] = projection_creation_memory
        result["proj_mean_execution_memory"] = projection_mean_execution_memory
        result["proj_variance_execution_memory"] = projection_variance_execution_memory
        result["const_memory"] = A_memory

        # get error metrics
        proj_df = generate_error_df(projection_result, hist, 3)
        full_proj_df = generate_error_df(projection_result, hist)
        
        del projection_result
        del projection_sigma

        try:
            proj_MSE, proj_MAE = calculate_error(proj_df)
            result["proj_MSE"] = proj_MSE
            result["proj_MAE"] = proj_MAE
        except:
            result["proj_MSE"] = 0
            result["proj_MAE"] = 0

    if do_sea:    
        t0 = time.time()
        # collection method
        means, agg_variances, aggregation_memory = run_aggregation_step(hist, track_memory, target_cardinality=max_table_size)
        t1 = time.time()
        
        print("aggregation time:", t1 - t0)
        
        t2 = time.time()
        # execute down pass
        final_estimates, down_pass_memory = down_pass(hist, means, agg_variances, max_complexity=max_table_size, track_memory=track_memory)
        t3 = time.time()

        print("downpass time:", t3 - t2)
        result["agg_time"] = t1 - t0
        result["downpass_time"] = t3 - t2
        
        
        t4 = time.time()
        # Generates the calculated variance 
        variances, hist, variance_memory = variance_generation(hist, max_table_size=max_table_size, track_memory=track_memory)
        t5 = time.time()

        print("variance generation time:", t5 - t4)

        
        result["var1_time"] = t5 - t4
        result["aggregation_memory"] = aggregation_memory
        result["downpass_memory"] = down_pass_memory
        result["variance_memory"] = variance_memory
        
        if do_confidence:
            # finalize and order dataframe for answer reporting (not part of the algorithm, not timed)
            # create dataframe for answer reporting (not related to algorithm)
            hist_final, hist, analysis_mem = quick_analysis(final_estimates, hist, max_table_size, track_memory)
            
            initial_lower_bounds, initial_upper_bounds = calculate_confidence_interval(hist.noised_data, hist.variance, 0.95, False, None)
            width_amounts = [upper - lower for upper, lower in zip(initial_upper_bounds, initial_lower_bounds) if upper - lower < 1e90]
            width_sums = sum(width_amounts) / len(width_amounts)
            result["initial_confidence_width"] = width_sums
            
            num_iterations = 20
            confidence = 0.95
            
            lower_bounds, upper_bounds = calculate_confidence_interval(hist_final["Final_Estimate"], variances, confidence, False, None)
            # get answers where true value is non NaN but checking true==true (i.e., it is present in the input table)
            # and where variance is nonzero (i.e. it hasn't been skipped for too-high complexity)
            # and decide if true value is in the lower/upper bounds of the confidence interval
            # (filter true==true to filter out NaNs of absent true tables)
            confidence_answers = [upper >= true >= lower
                                  for upper, lower, true, variance in zip(upper_bounds, lower_bounds, hist_final["True_Value"], variances)
                                  if true==true and true is not None and variance != 0]
            result["sea_confidence_coverage"] = sum(confidence_answers) / len(confidence_answers)
            print("SEA confidence coverage", sum(confidence_answers) / len(confidence_answers))
            
            result["sea_confidence_width"] = sum(upper - lower for upper, lower in zip(upper_bounds, lower_bounds)) / len(upper_bounds)
            clipped_upper, clipped_lower = clip_confidence_range(upper_bounds, lower_bounds)
            result["sea_confidence_width_clipped"] = sum(upper - lower
                                                         for upper, lower in zip(clipped_upper, clipped_lower)
                                                         if upper is not None) / len(upper_bounds)
            
            # run MC simulations
            estimated_variances, stdevs_selected = variance_estimation(hist, num_iterations, confidence, max_table_size)
            
            # get normal MC answers
            lower_bounds, upper_bounds = calculate_confidence_interval(hist_final["Final_Estimate"], estimated_variances, confidence, is_estimated=True, degrees_of_freedom=num_iterations)
            estimated_confidence_answers = [upper >= true >= lower
                                            for upper, lower, true, estimate, original_variance
                                            in zip(upper_bounds, lower_bounds, hist_final["True_Value"], hist_final["Final_Estimate"], variances)
                                            if true==true and true is not None and estimate==estimate and estimate is not None and original_variance != 0]
            result["normal_monte_carlo_coverage"] = sum(estimated_confidence_answers) / len(estimated_confidence_answers)
            print("normal monte carlo confidence coverage", sum(estimated_confidence_answers) / len(estimated_confidence_answers))
            
            result["normal_monte_carlo_width"] = sum(upper - lower for upper, lower in zip(upper_bounds, lower_bounds)) / len(upper_bounds)
            filtered_lower_bounds = [lower for lower, original_variance in zip(lower_bounds, variances) if original_variance != 0]
            filtered_upper_bounds = [upper for upper, original_variance in zip(upper_bounds, variances) if original_variance != 0]
            result["normal_monte_carlo_width_filtered"] = sum(upper - lower for upper, lower in zip(filtered_upper_bounds, filtered_lower_bounds)) / len(filtered_upper_bounds)
            
            clipped_upper, clipped_lower = clip_confidence_range(upper_bounds, lower_bounds)
            clipped_filtered_upper, clipped_filtered_lower = clip_confidence_range(filtered_upper_bounds, filtered_lower_bounds)
            result["normal_monte_carlo_width_clipped"] = sum(upper - lower for upper, lower in zip(clipped_upper, clipped_lower) if upper is not None) / len(upper_bounds)
            result["normal_monte_carlo_width_clipped_filtered"] = sum(upper - lower for upper, lower in zip(clipped_filtered_upper, clipped_filtered_lower) if upper is not None) / len(filtered_upper_bounds)
            
            # get order-seleted MC answers
            lower_bounds = [mean - stdev_selected for mean, stdev_selected in zip(hist_final["Final_Estimate"], stdevs_selected)]
            upper_bounds = [mean + stdev_selected for mean, stdev_selected in zip(hist_final["Final_Estimate"], stdevs_selected)]
            mc_confidence_answers = [upper >= true >= lower
                                  for upper, lower, true, variance, original_variance in zip(upper_bounds, lower_bounds, hist_final["True_Value"], [stdev**2 for stdev in stdevs_selected], variances)
                                  if true==true and true is not None and original_variance != 0]
            result["monte_carlo_coverage"] = sum(mc_confidence_answers) / len(mc_confidence_answers)
            print("order monte carlo confidence coverage",sum(mc_confidence_answers) / len(mc_confidence_answers))
            result["monte_carlo_width"] = sum(upper - lower for upper, lower in zip(upper_bounds, lower_bounds)) / len(upper_bounds)
            filtered_lower_bounds = [lower for lower, original_variance in zip(lower_bounds, variances) if original_variance != 0]
            filtered_upper_bounds = [upper for upper, original_variance in zip(upper_bounds, variances) if original_variance != 0]
            result["monte_carlo_width_filtered"] = sum(upper - lower for upper, lower in zip(filtered_upper_bounds, filtered_lower_bounds)) / len(filtered_upper_bounds)
            
            clipped_upper, clipped_lower = clip_confidence_range(upper_bounds, lower_bounds)
            clipped_filtered_upper, clipped_filtered_lower = clip_confidence_range(filtered_upper_bounds, filtered_lower_bounds)
            result["monte_carlo_width_clipped"] = sum(upper - lower for upper, lower in zip(clipped_upper, clipped_lower) if upper is not None) / len(upper_bounds)
            result["monte_carlo_width_clipped_filtered"] = sum(upper - lower for upper, lower in zip(clipped_filtered_upper, clipped_filtered_lower) if upper is not None) / len(filtered_upper_bounds)
   
    return result

"""
Takes arguments form command line and runs timing tests
"""
def loop_runs(batch_name, histogram_types, test_set, do_projection, do_sea, do_confidence, num_iterations=10, max_table_size=None, nmf_path="", sf1_path="", nmf_geocode=""):
    for i in histogram_types:

        for name, is_continuous, predicate, track_memory in test_set:

            results = pd.DataFrame(columns=["size", "agg_time", "matmul_time", "downpass_time", "var1_time", "aggregation_memory", 
            "downpass_memory", "variance_memory", "constraint_time", "projection_creation_time", 
            "projection_mean_execution_time", "projection_variance_execution_time", "proj_creation_memory", "proj_mean_execution_memory",
            "proj_variance_execution_memory", "const_memory", "MSE", "MAE", "proj_MSE", "proj_MAE", "SEA_vs_Proj_MSE", 
            "hhgq_votingage_MSE", 
            "sea_confidence_coverage", "normal_monte_carlo_coverage", "monte_carlo_coverage",
            "initial_confidence_width",
            "sea_confidence_width", "sea_confidence_width_clipped",
            "normal_monte_carlo_width", "normal_monte_carlo_width_clipped", "normal_monte_carlo_width_filtered", "normal_monte_carlo_width_clipped_filtered",
            "monte_carlo_width", "monte_carlo_width_clipped", "monte_carlo_width_filtered", "monte_carlo_width_clipped_filtered"])

            for j in range(num_iterations):
                
                print("running trial #",j)
                
                if i == "uneven":
                    hist = Histogram(num_vars = 5, var_levels = [11,4,4,4,4], num_children = children, max_detail = 5, max_depth=depth, continuous=is_continuous)
                
                elif i == "pl94":
                    hist = NMFHistogram(-1, nmf_path, sf1_path,
                                        ["hhgq", "votingage", "hispanic", "cenrace"],
                                        nmf_geocode, max_table_size)

                elif i == "dhc":
                    hist = NMFHistogram(-1, nmf_path, None,
                                        ["relgq", "sex", "age", "hispanic", "cenrace"],
                                        nmf_geocode, max_table_size)

                # if not a known test-type, treat as a number and create an NxN histogram (N vars of N depth)
                else:
                    i = int(i)
                    hist = Histogram(num_vars = i, var_levels = [i]*i, num_children = children, max_detail = i, max_depth=depth, continuous=is_continuous)

                # store result of this run
                result = run_trial(hist, track_memory, do_projection, do_sea, do_confidence, max_table_size)
                results.loc[len(results)] = result

            output_path=f"{batch_name}_{i} {name}.csv"
            results.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


if __name__== "__main__":
    parser = argparse.ArgumentParser(
                        prog='SEA BLUE time testing',
                        description='Tests and logs time for executing the SEA BLUE algorithm')

    parser.add_argument('batch_name', help="a human-readable name used in saving results to files")
    parser.add_argument('histogram_types', help='comma-separated list of types of histograms to run, which can include: "uneven", "dhc", "pl94", and numeric values for N variable levels of N size')
    parser.add_argument('--nmf', help='a file path to a Census noisy mesaurement file in Parquet format')
    parser.add_argument('--sf1', help='a template file path to SF1 files to read, replacing 3 digits with {0}')
    parser.add_argument('--geocode', help='geocode to read from the NMF file')
    parser.add_argument('-n', type=int, default=10, help='number of iterations to run')
    parser.add_argument('--tablesize', type=int, default=None, help='max variable interactions to consider')
    parser.add_argument('--sea', action='store_true', help="perform SEA")
    parser.add_argument('--proj', action='store_true', help="perform projection")
    parser.add_argument('--mem', action='store_true', help="test memory use")
    parser.add_argument('--time', action='store_true', help='test run time')
    parser.add_argument('--confidence', action='store_true', help='compute SEA confidence numbers')

    args = parser.parse_args()
    
    if args.sea is None and args.proj is None:
        parser.error("at least one of --sea or --proj is required: you must measure SEA, projetion, or both")
        
    if args.time is None and args.mem is None:
        parser.error("at least one of --time or --mem is required: you must measure time, memory, or both")
    
    histogram_types = args.histogram_types.split(',')
    histogram_types = list(map(lambda s:s.strip(), histogram_types))

    # set of tests to run:
    # fileoutput name, is continuous?, "mutate variance?" predicate, track_memory
    test_set = []
    if args.mem:
        test_set.append(["trackmem", False, lambda key: False, True])
    if args.time:
        test_set.append(["tracktime", False, lambda key: False, False])

    loop_runs(args.batch_name, histogram_types, test_set, args.proj, args.sea, args.confidence, num_iterations=args.n, max_table_size=args.tablesize, nmf_path=args.nmf, sf1_path=args.sf1, nmf_geocode=args.geocode)
