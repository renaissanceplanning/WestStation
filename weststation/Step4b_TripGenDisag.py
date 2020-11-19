"""
Created 2020

@author: Alex Bell

Step 4b: Trip Generation Disaggregation
==========================================
This script estimates block-level trips by household type and non-home 
activities based on TAZ-level trip estimates and block-level activity
estimates. This is accomplished in 3 major steps:
    
    1. Estimate activity-based block trip gen propensity by purpose
        - Propensities are estimated such that every block has an extremely
          small starting propensity, and this grows based on block activities.
          This is to alleviate potential discontinuities between block-level
          and TAZ-level activity estimates.
          
           - Home-based propensity is informed by HH types
           
           - Non-home-based propensity is informed by job types and enrollments
           
    2. Normalize the activity-based propensities such that they sum to 1.0
       within each TAZ.
       
        - Summarize the activity-based propensities to yield the total
          trip-making propensity for each TAZ. The resultsing score is the
          total trip gen propensity (either for P's or A's) by purpose.
          
        - Normalize block-level, activity-specific propensities based on its
          parent TAZ's total trip-making propensity.
          
        - When each block's trip total is estimated, these normalized shares
          determine trips by activity type.
        
    3. Multiply TAZ-level trip estimates by purpose and trip-end to block-
       level, activity-based normalized propensities. This yields activity-
       based trips by block.
"""
# -*- coding: utf-8 -*-

#%% IMPORTS
#import emma
from emma import ipf, lba
import pandas as pd
import numpy as np
import os
from wsa import tg
import logging
import yaml

# All reading/writing taks place within a root directory
#  and analysis is run for a specific land use configuration.
#  Updating the `lu_config` paramter will apply trip gen disaggregation logic 
#  to the specified configuration
root = r"K:\Projects\MAPC\FinalData"
os.chdir(root)
lu_config = "FEIR_MAX"

# Setup logging
logger = logging.getLogger("EMMA")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"lu\{}\log_trip_gen_disag.log".format(lu_config), mode="w")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('-------------\n%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#%% GLOBALS & ASSUMPTIONS (TRIP PROPENSITY WEIGHTS)
PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]
HB_PURPS = [p for p in PURPOSES if p!= "NHB"]
LEVELS=["block_id", "TAZ", "INFOCUS", "INWINDOW"]
ENDS = ["P", "A"]

# Generalized trip-making propensity assumptions are used to guide the
#  trip gen disaggregation process. Trip totals are governed by the regional
#  TAZ-level estimates, but these are pushed down to blocks based on the
#  block-level activities and their relative trip-making propensities.
#  Define the trip making propensities below.
BASE_PROP = 0.0001

HHSIZE_PROP = {
    "HHSize1": 10,
    "HHSize2": 20,
    "HHSize3": 30,
    "HHSize4p": 35
    }

INCOME_PROP = {
    "Income1": 10,
    "Income2": 20,
    "Income3": 30,
    "Income4": 40
    }

WORKER_PROP ={
    "worker0": 5,
    "worker1": 10,
    "worker2": 20,
    "worker3p": 30
    }

VEHICLE_PROP = {
    "Veh0": 10,
    "Veh1": 20,
    "Veh2": 30,
    "Veh3p": 40
    }

HH_PROP = {
    "HHSize": HHSIZE_PROP,
    "Income": INCOME_PROP,
    "Workers": WORKER_PROP,
    "VehOwn": VEHICLE_PROP
    }

# Home
HH_DIMS = ["HHSize", "Income", "VehOwn", "Workers"]
HH_LABELS = [
    ["HHSize1", "HHSize2", "HHSize3", "HHSize4p"],
    ["Income1", "Income2", "Income3", "Income4"],
    ["Veh0", "Veh1", "Veh2", "Veh3p"],
    ["worker0", "worker1", "worker2", "worker3p"]
]
HH_CRIT = dict(zip(HH_DIMS, HH_LABELS))
NH_CRIT = dict(zip(HH_DIMS, ["-" for _ in HH_DIMS]))

# Non-home
ATTR_DIM = "GenSector"
ATTR_VALS = "tot_emp"
ATTR_ACTIVITIES = ["Basic", "Retail", "Service", "EnrollPreK", "EnrollK12",
                   "EnrollCU"]
# Attraction propensities lists are specified by purpose below. Order of
#  propensity values corresponds to order of activities in ATTR_ACTIVITIES.
ATTR_PROP = {
    "HBW": {
        ATTR_VALS: dict(zip(ATTR_ACTIVITIES, [40, 40, 40, 1, 1, 1]))
    },
    "HBO": {
        ATTR_VALS: dict(zip(ATTR_ACTIVITIES, [5, 40, 20, 1, 1, 1]))
    },
    "HBSch": {
        ATTR_VALS: dict(zip(ATTR_ACTIVITIES, [1, 1, 1, 10, 30, 40]))
    },
    "NHB": {
        ATTR_VALS: dict(zip(ATTR_ACTIVITIES, [40, 40, 40, 10, 10, 20]))
    }
}

logger.info(f"\nHome-based-trip propensities:\n{yaml.dump(HH_PROP)}")
logger.info(f"\nNon-home-trip propensities:\n{yaml.dump(ATTR_PROP)}")

#%% READ INPUT DATA
# For trip disaggregation, we need the following data sets
#  - Block level activity data
#     - Households by type
#     - Jobs by type and school enrollments
#  - A lookup table of blocks-to-TAZs and which blocks are in the window and
#     focus areas
print("READ INPUT DATA")
hh_df, nh_df, wb_df = tg.readBlocks(lu_config=lu_config,
                                    block_hh_table="Household_Types_by_Block.csv",
                                    block_emp_table="Jobs_Enroll_by_Block.csv",
                                    hh_id_field="block_id",
                                    emp_id_field="block_id",
                                    window_blocks=r"input\window_blocks.csv",
                                    wb_id="GEOID10")

# - TAZ-level trip generation estimates
taz_tg_file = r"lu\{}\trips_by_taz.h5".format(lu_config)
taz_tg = lba.openLbArray_HDF(taz_tg_file, "/trips")

# Log trips in window
taz_sum = taz_tg.take(TAZ={'INWINDOW': 1}).sum(["Purpose", "End"])
logger.info(f"\nTrips in window (TAZ scale):\n{taz_sum}\n{taz_sum.axes}")

#%% PROCESS INPUT DATA
# hh_df contains rows for all blocks in the window. Some blocks have
#  no HH information (NaN values in dimension columns). We want to
#  avoid have NaN's show up as axis labels in a labeled array (created
#  below), and we want to ensure that all axis labels are represented.
#  This loop "tiles in" axis labels for all na values to ensure there
#  are no NaN labels. In theory, it is still possible for a given axis
#  value to go missing if it never occurs in the block-level HH data
#  and there are fewer missing NaN's in hh_df than there are distinct
#  labels in the dimension where the missing value occurs. However,
#  this is highly unlikely to occur in application.
print("MAKE HH BY TYPE ARRAY")
hh_array = tg.dfToLabeledArray_tg(tg_df=hh_df,
                                  ref_array=taz_tg,
                                  dims=HH_DIMS,
                                  act_col="Households",
                                  block_id="block_id",
                                  excl_labels=[],
                                  levels=LEVELS)

#%% MAKE OUTPUT CONTAINERS
# Write two new H5 files to hold block-level data
#  - Save the households by type data to an on-disk labeled array
#  - Create a new on-disk labeled array to store block-level trip estimates
print("CREATE OUTPUT CONTAINERS")
# HHs by type
hdf_store = r"lu\{}\HHs_by_type_block.h5".format(lu_config)
node_path = "/"
name = "households"
# Use the `copy` method to store it on disk
hh_output = hh_array.copy(hdf_store, node_path, name, overwrite=True)

# Trip container
#  The array for block-level trip data has the same organization and 
#  dimensions as the trip array for TAZs, except the TAZ `dimension` is
#  replaced with a `block_id` dimension.
hdf_store = r"lu\{}\trips_by_block.h5".format(lu_config)
node_path = "/"
name = "trips"
# Create an impression to set dimensions
block_imp = taz_tg.cast(hh_array.block_id, copy_data=False, drop="TAZ")
# Fill the impression to initialize all values to zero on-disk
block_tg = block_imp.fill(0.0, hdf_store, node_path, name, overwrite=True)

#%% INITIALIZE PROPENSITY ARRAY
# Make an impression of the taz_tg array, dropping the taz dimension.
#  This leaves activity axes (HHSize, Income, VehOwn, Workers), Purpose,
#  and End. A basic propensity matrix for these dimensions is assembled based
#  on the relative weights for each label in each dimension. Non-home
#  propensities vary by activity type, but non-home trip generators are
#  accounted for only by the label "-" in each activity axis. Thus, in this
#  basic propensity matrix, the non-home label is always filled with 1.0.
#  Home-based labels are populated with propensity estimates through
#  cross-multiplication of propensities across all potential HH types.
print("INITIALIZE PROPENSITY ARRAY")
basic_prop_array = taz_tg.impress(drop="TAZ", fill_with=1.0)

#%% HOME-BASED TOTAL PROPENSITY
# Calcualte general trip-making propensities for cross-classified HH types.
# Then apply these propensities to each block's HH type estimates.
print("CALCULATE HOME-BASED TRIP PROPENSITY")
prods_prop = tg.calcPropensity_HB(basic_prop_array, HH_PROP, end_axis="End",
                                  prod_label="P")
basic_prop_array.put(prods_prop.data, End="P")

# Extend the basic propensity array into the `block_id` dimension (i.e.,
#  repeat the basic propensities for every block)
block_prop_array = basic_prop_array.cast(block_tg.block_id)
block_prop_array.data = block_prop_array.data.copy()

# Weight propensity by number of households by type
# Push the results to the propensity array
print("APPLY HOME-BASED PROPENSITIES TO BLOCK HOUSEHOLDS")
for purpose in PURPOSES:
    for end in ENDS:
        crit = {"Purpose": purpose, "End": end}
        block_prop_array.put(
            block_prop_array.take(squeeze=True, **crit).data * hh_array.data,
            **crit
            )

#%% NON-HOME-BASED TOTAL PROPENSITY
# Non-home activities by type are used to build up purpose-specific
# propensity estimates.
print("CALCULATE NON-HOME-BASED TRIP PROPENSITY")
nh_array = tg.calcPropensity_NH(nh_df=nh_df, 
                                lbl_col=ATTR_DIM, 
                                act_col=ATTR_VALS, 
                                purposes=PURPOSES, 
                                propensities=ATTR_PROP,
                                base_prop=BASE_PROP, 
                                wb_df=wb_df, 
                                block_id="block_id",
                                wb_block_id="block_id",
                                levels=LEVELS)
lba.alignAxisLabels(
    nh_array, "Purpose", block_prop_array.Purpose, inplace=True)

print("UPDATE BLOCK PROPENSITY ARRAY FOR NON-HOME ACTIVITIES")
for end in ENDS:
    crit = {"End": end}
    crit.update(NH_CRIT)
    block_prop_array.put(
        block_prop_array.take(squeeze=True, **crit).data * nh_array.data,
        **crit
        )
    
#%% SET BASELINE PROPENSITY
# To ensure all TAZs have their trips disaggergated a minute base propensity
#  is applied for any case where propensity = 0. In blocks with activities,
#  the estimated propensities should be significantly larger than the base
#  propensity so that a negligible number of trips are disaggregate to blocks
#  with no estimated propensity. However, in TAZs where all blocks have no
#  estimated propensity, the base propensity ensures that non-zero values are
#  available for disaggregateion.
block_prop_array.data[block_prop_array.data == 0] = BASE_PROP

#%% NORMALIZE ACTIVITY PROPENSITIES
# Once all activity-based propensities are in place, we calculate each block's
#  share of total TAZ-level trip-making propensity, by activity type.
print("NORMALIZE TRIP PROPENSITY BY BLOCK, ACTIVITIY")
tg.normalizePropensity(block_prop_array, block_prop_array.block_id, 
                       taz_level="TAZ", block_level="block_id", ends=ENDS,
                       purposes=PURPOSES)


#%% APPLY TAZ TRIP ESTIMATES TO BLOCKS
# Now, the TAZ level trip totals can be applied to block/actvity propensity
#  shares to get total trips produced or attracted by each block by activity
#  type.
print("APPLY TAZ TRIP ESTIMATES TO NORMALIZED BLOCKS, ACTIVITIES")
taz_sum = taz_tg.sum(["TAZ", "Purpose", "End"])
tg.applyTAZTrips(taz_sum, taz_sum.TAZ, "TAZ", 
                 block_prop_array, block_prop_array.block_id, "block_id",
                 ends=ENDS, purposes=PURPOSES)

#%% CHECK RESULTS
block_sum = block_prop_array.sum(["Purpose", "End"])
logger.info(f"\nTrips in window (block scale):\n{block_sum}\n{block_sum.axes}")

#%% PUSH TO OUTPUT CONTAINER
# Write detailed ouptuts
block_tg.data[:] = block_prop_array.data

# Summarize for csv tables
by_block_csv = r"lu\{}\trips_by_block.csv".format(lu_config)
block_tg_sum = block_tg.sum(["block_id", "Purpose", "End"])
block_df = block_tg_sum.to_frame("trips").reset_index()
block_df[LEVELS] = pd.DataFrame(
    block_df["block_id"].to_list(), index=block_df.index)
block_df.to_csv(by_block_csv, index=False)


