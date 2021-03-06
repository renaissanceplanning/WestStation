# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 07:31:01 2020

@author: Alex Bell


This file contains a collection of functions used in modeling travel behavior
using emma data structures and processing idioms. These functions are imported
into the West Station Area scripts for analysis and mapping purposes.

"""


#%% IMPORTS

# Temporarily add the path to the emma library so the scripts can work.
#  Later, we need to have emma "installed" as part of an environment 
import sys
sys.path.append(r"K:\Tools\RP\emma\scripts")


# Main imports
import emma
from emma import labeled_array as lba
from emma import ipf
import numpy as np
import pandas as pd
import os


#%% GLOBALS
PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]

# For each purpose, accessibility scores focus on different destination types
PURP_ACTIVITIES = {
    "HBW": "Total Emp",
    "HBO": "Retail",
    "HBSch": "TotEnroll",
    "NHB": "Total Emp"
    }

# Modes
MODE_DICT = {
    "non-motorized": ["walk", "bike"],
    "motorized": ["auto", "transit"],
    "auto": ["driver", "passenger"],
    "transit": ["WAT", "DAT"]
    }
MODES = ["walk", "bike", "driver", "passenger", "WAT", "DAT"]
DAT_SUBMODES = ["BT", "CR", "LB", "RT"]


# Analysis scales
SCALES = ["block", "TAZ"]

# Access scores are stored in csv files that are systematically generated.
#  Depending on what geography is needed for indexing, use the appropriate
#  key column, which is recorded in the global variables below.
BLOCK_ACCESS_KEY = "GEOID10"
TAZ_ACCESS_KEY = "ID"





#%% SKIM IMPORTS



#%% ACCESS ANALYSIS



#%% HANDLING ACCESS SCORES






def maxTransitScore(scen, purpose, direction, activity, index_cols,
                    match_axis=None, match_level=None, include_wat=True):
    """
    Process transit scores by each access mode and submode to get the
    best access scores for each zone.
    
    Parameters
    -----------
    scen: String
    purpose: String
    direction: String
    activity: String
    index_cols: String or [String,...]
    match_axis: LbAxis
    match_level: String
    include_wat: Boolean, default=True
        If True, max transit scores are returned including all DAT submodes
        and the WAT submode. If False, the max among DAT submodes only is
        returned.
    
    See Also
    ---------
    fetchAccessScores
    """
    # Fetch data vectors
    dat_vectors = []
    for submode in DAT_SUBMODES:
        fetch_mode = f"transit_da_{submode}"
        dat_scores = fetchAccessScores(scen, fetch_mode, purpose, direction,
                                       activity, index_cols,
                                       match_axis=match_axis,
                                       match_level=match_level)
        
        dat_vectors.append(dat_scores)
        
    # Get the max DAT score through concatenation
    max_val = pd.concat(dat_vectors, axis=1).max(axis=1)
    
    if include_wat:
        # Fetch the WAT scores
        wat_scores = fetchAccessScores(scen, "transit", purpose, direction, activity, 
                                   index_cols, match_axis=match_axis,
                                   match_level=match_level)
        
        # Get the max transit score (WAT or DAT)
        max_val = pd.concat([max_val, wat_scores], axis=1).max(axis=1)
    max_val.name = activity
    
    return pd.DataFrame(max_val)


class ScaleReference():
    """
    A collection of attributes for simplifying the fetching of access sores
    based on different scales of anlaysis (block vs TAZ, e.g.).
    
    Parameters
    -----------
    index_cols: String or [String,...]
    suffix: String
    match_axis: LbAxis
    match_level: String
    imped_tag: String
    array: LbArray
        If a specific labeled array is associated with work being done at
        this scale, provide the array object here.
    dim_name: String
        If specifying a labeled array, provide the name of the dimension
        to which values at this scale are related.
    
    See Also
    ---------
    fetchAccessScores
    """
    def __init__(self, index_cols, suffix, match_axis, match_level, imped_tag,
                 array, dim_name):
        self.index_cols = index_cols
        self.suffix = suffix
        self.match_axis = match_axis #var_axis
        self.match_level = match_level #var_level
        self.imped_tag = imped_tag
        self.array = array
        self.dim_name = dim_name
        

#%% TRIP GEN



#%% MODE CHOICE

def summarizeTripsByMode(scen, scale, dimensions, **criteria):
    """
    Using a trips by mode labeled array (generated by the mode choice script),
    summarize trips along selected axes.

    Parameters
    ----------
    scen : String
    scale : String ("block", "taz", e.g.)
    dimensions : String or [String,...]
        The name(s) of the dimension(s) along which to summarize the trips by
        mode table. For example, ["Purpose", "End"] will summarize all trips
        by purpose (HBW, HBO, e.g.) and trip end (production, attraction).
        Note that the "Mode" dimension is not implied so must be included
        to keep the modal breakdown.
    **criteria : kwargs
        Keyword arguments for selecting axes and labels from the labeled
        array prior to summarization.

    Returns
    -------
    LbArray
        A labeled array with summarized trips

    """
    # Connect to trips by mode tables
    trips_file = r"scen\{}\Trips_by_{}.h5".format(scen, scale)
    trips = lba.openLbArray_HDF(trips_file, "/trips")
    return trips.take(**criteria).sum(dimensions)
    

def tripsToShares(trips_array, shares_by_dimension):
    """
    Convert a table of trips by mode into a table of mode shares.

    Parameters
    ----------
    trips_array : LbArray
        The labeled array with trips by mode
    shares_by_dimension : String
        The name of the axis along which shares will be calculated 
        (the "Mode" axis, e.g.)

    Returns
    -------
    LbArray
        A labeled array with dimensions matching `trip_array` but values
        converted from trips to shares.

    """
    dims = trips_array.axisNames()
    sum_dims = [d for d in dims if d != shares_by_dimension]
    total = trips_array.sum(sum_dims)
    total_cast = total.cast(trips_array[shares_by_dimension])
    total_cast = lba.alignAxes(total_cast, trips_array)
    return trips_array.impress(fill_with=trips_array.data/total_cast.data)


#%% DISTRIBUTION
def inheritFactors(factors, block_axis, block_level="block_id"):
    """
    A simple function for succinctly merging data frames so that taz-level
    trip balancing factors are applied to their constituent block features.
        
    The `block_axis` is assumed to be a MultiIndex object with one level
    identifying blocks and the other identifying TAZs.
    
    Parameters
    -------------
    taz_scores: pandas data frame
    block_axis: LbAxis
    block_level: String, default="block_id"  
    
    Returns
    -------
    pd.DataFrame
    """
    idx_df = block_axis.to_frame()    
    df = idx_df.merge(taz_scores, how="left", left_index=True,
                            right_index=True)
    df.fillna(0.0, inplace=True)
    df.set_index([block_level, taz_level], inplace=True)
    return df.reindex(block_axis.labels)
