"""
Code developed to implement the methodology of Best Linear Unbiased Estimate from Privatized Histograms
by
Jordan Awan; Department of Statistics, Purdue University
Adam Edwards, Paul Bartholomew, Andrew Sillers; The MITRE Corporation

Approved for Public Release; Distribution Unlimited. Public Release Case Number 24-2176'
"""

"""
This file contains utilities for extracting values from 2010 Census SF1 data products
and organizing them into tables used for true data to compare against noisy PL94 data.

Note: True 2010 PL94 data is available, but it is much less detailed than the tables included
in the noisy demonstration product (e.g., HHGQ information is totally absent in the true PL94,
but present in the noisy demo. For this reason, the code builds the true tables from SF1 and
shapes them to match the noisy PL94 data.

We can build the following PL94 marginals and crosses using these SF1 tables (P#):

total (P1)
cenrace (P8)
hispanic (P4)
votingage (p1, p10)
hhgq (p1 - p42)
hispanic * cenrace (P8 - P9)
votingage * cenrace (P8 - P10)
votingage * hispanic (P11 - P4)
hhgq * votingage (P43)

IMPOSSIBLE under SF1:
hhgq * cenrace

(this cross is also absent from true 2010 PL94)
"""

import pandas

def lookup_cells_from_df(df, target_logrec, offset, cells):
    """
    Given an SF1 file dataframe and a logrec ID to select, select the cells at the specified relative positions starting at offset

    This is used in conjunction with the positional SF1 information to extract cells from particular tables.
    e.g., For Rhode Island, the P8 Race table begins at cell 55 on file #3; to get the first 4 cells, read file #3 into a df and pass in
            (file_3_df, some_logrec, offset = 55, cells = [1,2,3,4])
            
    Cells are one-indexed to match the counting nomenclature used in the SF1 docs, e.g., P8 begins with cell P0080001
    """
    # get the SF1 row with the specified logrec
    target_row = df.loc[df.iloc[:,4] == target_logrec]
    # select specified cells from that row
    target_cells = target_row[[idx + offset - 1 for idx in cells]]
    # return values as list
    return list(target_cells.iloc[0])

def get_logrec_from_geocode(id, file_template):
    """
    Given a Census geocode, return the associated SF1 logrecno. Must be supplied a file path to the appropriate state's SF1 geo file.

    Geo fixed-width spec is in Figure 2-5 of SF1 spec, page 2-72.
    https://www2.census.gov/programs-surveys/decennial/2010/technical-documentation/complete-tech-docs/summary-file/sf1.pdf

    id - geocode whose SF1 logrecno we search for
    geofile - file template to the SF1 geo record file to search against
    """
    county = None
    block = None
    block_group = None
    tract = None
    
    state = int(id[1:3])
    
    if len(id) > 3:
        county = int(id[5:8])
    if len(id) >= 31:
        block = int(id[-4:])
        block_group = int(id[-5:-4])
        tract = int(id[-11:-5])

    
    geodata = pandas.read_fwf(file_template.format("geo"),
                widths=[6,2,3,2,3,2,7,1,1,2,3,2,2,5,2,2,5,2,2,6,1,4],
                names=["FILEID","STUSTAB","SUMLEV","GEOCOMP","CHARITER","CIFSN","LOGRECNO","REGION","DIVISION","STATE","COUNTY","COUNTYCC","COUNTYSC","COUSUB","COUSUBCC","COUSUBSC","PLACE","PLACECC","PLACESC","TRACT","BLKGRP","BLOCK"])

    if not county:
        item = geodata.loc[(geodata.STATE == state) & geodata.COUNTY.isna() & (geodata.GEOCOMP == "00")].iloc[0]
    elif not block:
        item = geodata.loc[(geodata.STATE == state) & (geodata.COUNTY==county) & geodata.TRACT.isna() & (geodata.GEOCOMP == "00")].iloc[0]
    else:
        item = geodata.loc[(geodata.STATE == state) & (geodata.COUNTY==county) & (geodata.TRACT==tract) & (geodata.BLKGRP==block_group) & (geodata.BLOCK==block) & (geodata.GEOCOMP == "00")].iloc[0]
    
    return item.LOGRECNO

def get_true_data(target_geocode, file_template="ri{0}2010.sf1"):
    """
    Given a geocode and a templated file path, extract the true SF1 data and combine them into the necessary tables for PL94 comparison.

    Needs a filepath template string which points to SF1 data files like `ri{0}2010.sf1` (e.g., for Rhode Island) where
    non-year digits are replaced with the sequence {0}.

    target_geocode - geocode to get SF1 true data for
    file_template - filepath template string -- SF1 filenames with the sequence numbers replaced with {0}
    """
    target_logrec = get_logrec_from_geocode(target_geocode, file_template)

    # the identify the offsets of the start of each SF1 table within their respective files
    # e.g., table P1 begins at the 5th value with its file
    p1_offset = 5
    p4_offset = 13
    p8_offset = 55
    p9_offset = 126
    p10_offset = 5
    p11_offset = 76
    p42_offset = 168
    p43_offset = 178
    pct20_offset = 161
    pct22_offset = 200
    pct20h_offset = 69
    pct22h_offset = 26

    # cells we want from P8 to make 63-levels of cenrace
    race_idxs = [3,4,5,6,7,8, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25, 27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46, 48,49,50,51,52,53,54,55,56,57,58,59,60,61,62, 64,65,66,67,68,69, 71]

    # read in files 1, 3, 4, 6, etc. to get necessary data
    f1 = pandas.read_csv(file_template.format("00001"), header=None)
    f3 = pandas.read_csv(file_template.format("00003"), header=None)
    f4 = pandas.read_csv(file_template.format("00004"), header=None)
    f6 = pandas.read_csv(file_template.format("00006"), header=None)
    f38 = pandas.read_csv(file_template.format("00038"), header=None)
    f39 = pandas.read_csv(file_template.format("00039"), header=None)

    # extract various tables from various files to build necessary PL94 crosses
    total = lookup_cells_from_df(f1, target_logrec, p1_offset, [1])
    
    # voting age: over/under 18
    over18 = lookup_cells_from_df(f4, target_logrec, p10_offset, [1])
    under18 = total[0] - over18[0]
    votingage = [under18] + over18
    
    # hisp / non-hisp
    hisp = lookup_cells_from_df(f3, target_logrec, p4_offset, [2,3])

    # cenrace
    race = lookup_cells_from_df(f3, target_logrec, p8_offset, race_idxs)

    # get non-group-quarters count and all GQ counts:
    p42 = lookup_cells_from_df(f6, target_logrec, p42_offset, [1,3,4,5,6,8,9,10])
    
    # non-GQ is total pop minus GQ total
    non_gq = total[0] - p42[0]
    # then collate them to together to make HH & GQ
    hhgq = [non_gq] + p42[1:]

    # hispanic * race
    not_hisp_race_idxs = list(map(lambda i: i+2, race_idxs))
    not_hisp_race = lookup_cells_from_df(f3, target_logrec, p9_offset, not_hisp_race_idxs)
    yes_hisp_race = list(map(lambda x: x[0]-x[1], zip(race, not_hisp_race)))
    hisp_x_race = not_hisp_race + yes_hisp_race

    # votingage * race
    yes_18_race = lookup_cells_from_df(f4, target_logrec, p10_offset, race_idxs)
    not_18_race = list(map(lambda x: x[0]-x[1], zip(race, yes_18_race)))
    votingage_x_race = not_18_race + yes_18_race

    # votingage * hisp 
    yes_18_hisp = lookup_cells_from_df(f4, target_logrec, p11_offset, [3,2])
    no_18_hisp = [hisp[0] - yes_18_hisp[0], hisp[1] - yes_18_hisp[1]]
    votingage_x_hisp = no_18_hisp + yes_18_hisp

    # get male/female under 18 and sum GQs by sex
    gq_under_18 = lookup_cells_from_df(f6, target_logrec, p43_offset, [5,6,7,8,10,11,12, 36,37,38,39,41,42,43])
    gq_under_18 = list(map(sum, zip(gq_under_18[:7], gq_under_18[7:])))

    # hhgq * votingage
    hh_under_18 = votingage[0] - sum(gq_under_18)
    hhgq_under_18 = [hh_under_18] + gq_under_18
    hhgq_over_18 = list(map(lambda x: x[0]-x[1], zip(hhgq, hhgq_under_18)))
    # interleave for we have under/over-18 pairs for each HHGQ category
    hhgq_x_votingage = [val for pair in zip(hhgq_under_18, hhgq_over_18) for val in pair]

    # votingage * hisp * race
    yes_18_not_hisp_race = lookup_cells_from_df(f4, target_logrec, p11_offset, not_hisp_race_idxs)
    yes_18_yes_hisp_race = list(map(lambda x: x[0]-x[1], zip(yes_18_race, yes_18_not_hisp_race)))
    not_18_not_hisp_race = list(map(lambda x: x[0]-x[1], zip(not_hisp_race, yes_18_not_hisp_race)))
    not_18_yes_hisp_race = list(map(lambda x: x[0]-x[1]-x[2]+x[3], zip(race, not_hisp_race, yes_18_race, yes_18_not_hisp_race)))
    votingage_x_hisp_x_race = not_18_not_hisp_race + not_18_yes_hisp_race + yes_18_not_hisp_race + yes_18_yes_hisp_race

    
    try:
        # some SF1 data is not available smaller than tract level; try reading that data now with a try-catch fallback
        
        # hhgq * hisp (TRACT AND LARGER ONLY)
        yes_hisp_gq = lookup_cells_from_df(f38, target_logrec, pct20h_offset, [3,10,14,15,22,23,26])
        yes_hisp_hh = hisp[1] - sum(yes_hisp_gq)
        yes_hisp_hhgq = [yes_hisp_hh] + yes_hisp_gq
        not_hisp_hhgq = list(map(lambda x: x[0]-x[1], zip(hhgq, yes_hisp_hhgq)))
        hhgq_x_hisp = [val for pair in zip(not_hisp_hhgq, yes_hisp_hhgq) for val in pair]

        # hhgq * votingage * hisp ((TRACT AND LARGER ONLY))
        yes_18_yes_hisp_gq_by_sex = lookup_cells_from_df(f39, target_logrec, pct22h_offset, [4,5,6,7,9,10,11, 14,15,16,17,19,20,21])
        yes_18_yes_hisp_gq = list(map(sum, zip(yes_18_yes_hisp_gq_by_sex[:7], yes_18_yes_hisp_gq_by_sex[7:])))
        yes_18_yes_hisp_hh = yes_18_hisp[1] - sum(yes_18_yes_hisp_gq)
        yes_18_yes_hisp_hhgq = [yes_18_yes_hisp_hh] + yes_18_yes_hisp_gq

        not_18_yes_hisp_gq = list(map(lambda x: x[0]-x[1], zip(yes_hisp_gq, yes_18_yes_hisp_gq)))
        not_18_yes_hisp_hh = no_18_hisp[1] - sum(not_18_yes_hisp_gq)
        not_18_yes_hisp_hhgq = [not_18_yes_hisp_hh] + not_18_yes_hisp_gq

        yes_18_not_hisp_gq = list(map(lambda x: x[0]-x[1], zip(hhgq_over_18[1:], yes_18_yes_hisp_gq)))
        yes_18_not_hisp_hh = yes_18_hisp[0] - sum(yes_18_not_hisp_gq)
        yes_18_not_hisp_hhgq = [yes_18_not_hisp_hh] + yes_18_not_hisp_gq

        # not_18_not_hisp_hhgq = hhgq - hhgq_over_18 - yes_hisp_hhgq + yes_18_yes_hisp_hhgq
        # (subtract 18+ and yes-hisp counts, then add back in their intersection because it's been subtracted twice)
        not_18_not_hisp_hhgq = list(map(lambda x: x[0]-x[1]-x[2]+x[3], zip(hhgq, hhgq_over_18, yes_hisp_hhgq, yes_18_yes_hisp_hhgq)))

        hhgq_x_votingage_x_hisp = [val for group in zip(not_18_not_hisp_hhgq, not_18_yes_hisp_hhgq, yes_18_not_hisp_hhgq, yes_18_yes_hisp_hhgq) for val in group]

        return {
            "total": total,
            "hhgq": hhgq,
            "votingage": votingage,
            "hispanic": hisp,
            "cenrace": race,
            "hispanic * cenrace": hisp_x_race,
            "hhgq * votingage": hhgq_x_votingage,
            "hhgq * hispanic": hhgq_x_hisp,
            "votingage * cenrace": votingage_x_race,
            "votingage * hispanic": votingage_x_hisp,
            "hhgq * votingage * hispanic": hhgq_x_votingage_x_hisp,
            "votingage * hispanic * cenrace": votingage_x_hisp_x_race
        }
    except:
        # if lower-than-tract data is not available, fallback to reduced tables
        return {
            "total": total,
            "hhgq": hhgq,
            "votingage": votingage,
            "hispanic": hisp,
            "cenrace": race,
            "hispanic * cenrace": hisp_x_race,
            "hhgq * votingage": hhgq_x_votingage,
            "votingage * cenrace": votingage_x_race,
            "votingage * hispanic": votingage_x_hisp,
            "votingage * hispanic * cenrace": votingage_x_hisp_x_race
        }
