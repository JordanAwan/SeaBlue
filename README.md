# Approximate BLUE / SEA BLUE Python

This code implements the SEA BLUE algorithm, and contains code to compare its results to projection.

It was developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

MITRE rights notice: Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176

## File manifest

* `AproxBLUE.py` - contains implementations of the SEA BLUE algorithm
* `testing_doc.py` - a script for testing time and memory performance of SEA BLUE versus projetion
* `Demo_Doc.py` - a sample file demonstrating how to use the SEA Python interface
* `variance_experiment.py` - a sanbox file for testing the effects of differing input variance on the resulting collection variance versus down-pass variance
* `NMFHistogram.py` - an interface-compatible implementation of the Histogram object in `AproxBLUE.py` that is constructed from noisy measurement files in Parquet format
* `sf1extract.py` - code for extracting counts out of SF1 data files
* `treegen_sparse.py` - code for creating contraint matrices for projection
* `discretegauss.py` - thrid-party implementation of discrete Gauss sampler, licensed under Apache
* `requirements.txt` - dependency file

## Runnable Example

To get started, you can inspect and run `python Demo_Doc.py`, which executes BLUE for all the SEA and projection steps, for a 4-variable histogram.

For timing and memory performance, you can use `testing_doc.py` to batch-execute tests. This tool has build-in argparse command-line documentation with `python testing_doc.py --help`.

Some example runs of `testing_doc` could be:

    python testing_doc.py MyBatchID pl94 --sea --time -n=5 --nmf=~/ri_block_nmf.parquet --sf1="~/ri2010.sf1/ri{0}2010.sf1" --geocode=0441000700491024400701070233001

    python testing_doc.py SomeOtherBatchID dhc --sea --time -n=10 --nmf="~/some_dhc.parquet" --geocode=044
    
    python testing_doc.py MyBatchID 3,5,uneven --sea --time -n=1
    

## Histogram and NMF Reader

This code uses a `Histogram` class, defined in `AproxBLUE.py` that represents a histogram of counts under some set of variables.

The `Histogram` class creates dummy data, but the `NMFHistogram` is an interface-compatible implementation that reads in real Noisy Measurement File (NMF) data  from Census Parquet files and (optionally) SF1 data for true comparison. Information on how to read SF1 data and collate it into tables is in `sf1extract.py`.

Trials so far have used Parquet-formatted NMF files of person counts, e.g., in the `DPQuery` folders at [the 2010 PL 94 noised deomstration product](https://app.globus.org/file-manager?origin_id=c89b5c48-ca65-11ed-8cfb-f9fa098153fc&origin_path=%2F) and [the DHC noised demonstration product](https://app.globus.org/file-manager?origin_id=ab444649-2203-4f65-94d3-62a3268bcdca). Future work could modify the `NMFHistogram` reader to ingest any kind of data.

The `NMFHistogram` ingest process **requires** a `detailed_dpq` row in the Parquet data, and it uses that row to look for smaller tables. Tables with variables names that are not among the detailed marginal variables are omitted. (This is not related to the SEA BLUE methodology, but is only the process for how the reader selects data from Census products for ingest into the SEA BLUE algorithm.)

## Approach / Functions

The SEA approach has three primary steps: aggregation, down pass, and variance calculation.

Aggregation:

    means, variances, memory = run_aggregation_step(hist, track_memory = False, target_cardinality = 3)

`means` and `variances` answers are presented as dictionaries whose keys are variable-level tuples and whose keys are numbers.

Down Pass:

    final_estimates, down_pass_memory = down_pass(hist, means, agg_variances, max_complexity = 3, track_memory = False)

`final_estimates` is a dictionary whose keys are tables as string-tuples (like `("A","B","C")`) and whose values are lists of numbers.

Tables are ordered by last variable rotating first: (1,1,1), (1,1,2), (1,1,3) ... (1,2,1) (1,2,2) (1,2,3) ...

Variance Generation:

    variances, hist, total_memory = variance_generation(hist, max_table_size = 3, track_memory = False)

Computes vairances for the histogram, either by tablewise batch solving (if all the table have internally uniform variance) or solving each variance individually.


## Down Pass Approaches

There are multiple ways to compute (identical) down pass answers.

**Two general solution approach** - This is the default approach for cases where each table has internally-matching variances. It works by computing and combining two general solutions (one based on sum-ups of aggregation, one from smaller-order tables so far) for each table. It is implemented in `uniform_vairance_down_pass` which is called by `down_pass` when all tables contain uniform variance.

**Matrix multiplication** - Uses an A1/A2/B1/B2 matrix multiplication to compute the down pass. Due to C implementation of matmul, this is practically fastest for small examples. It is performed in `two_way_cross`, and in `three_way_cross` when `use_matmul=True`. Currently this approach has been solved only two- and three-way crosses, unlike the the general-solution approach, which can operate on an unbounded complexity of crosses.

**Paginated** - This approach mimics the matrix multiplication by performing the same multiplicaitons and additions in a flat structure (made possible because the matmul involves a lot of zeroes). The implementation is correct, but might yield results in a different order. (This bug hasn't been resolved because the general-solution approach has proven superior at scale, and matmul is superior for small examples, so this hasn't been needed.)

**Uneven variance** - A much slower approach that works on cases with differing variances within tables. Relies on finding a matrix inverse, which is very slow.
