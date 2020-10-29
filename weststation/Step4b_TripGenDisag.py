"""
Created 2020

@author: Alex Bell

Step 4b: Trip Generation Disaggregation
==========================================
This script estimates block-level trips by household type and non-home 
attractions based on TAZ-level trip estimates and block-level activity
estimates. This is accomplished in 5 major steps:
    
    1. Estimate activity-based block trip gen propensity by purpose
        - Propensities are estimated such that every block has an extremely
          small starting propensity, and this grows based on block activities.
          This is to alleviate potential discontinuities between block-level
          and TAZ-level activity estimates.
           - Production propensity is informed by HH types
           - Attraction propensity is informed by job types and enrollments
        
    2. Summarize total block trip-making propensity by purpose
        - Summarize the activity-based propensities to yield the total
          trip-making propensity for the block.The resultsing score is the
          total trip gen propensity (either for P's or A's) by purpose.
        - Summarize the block-level totals to TAZs. This will be used to
          determine each block's share of total trip-making propensity.
          
    3. Normalize the activity-based propensities such that they sum to 1.0
       within each block.
        - When each block's trip total is estimated, these shares will
          determine trips by activity type.
    
    4. Summarize block-level total trip propensity to TAZ and estimate block-
        level shares of TAZ totals (by purpose and trip-end) by activity.
        
    5. Multiply TAZ-level trip estimates by purpose and trip-end to block-
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
lu_config = "FEIR"

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

# Generalized trip-making propensity assumptions are used to guide the
#  trip gen disaggregation process. Trip totals are governed by the regional
#  TAZ-level estimates, but these are pushed down to blocks based on the
#  block-level activities and their relative trip-making propensities.
#  Define the trip making propensities below.
BASE_PROP = 0.0001

# HH Size: as household size increases, trip-making increases (there are 4
#          size classes -  1 person, 2 persons, 3 persons, 4+ persons)
HHSIZE_PROP = [10, 20, 30, 35]

# Income: as income increases, trip-making increases (there are 4 income 
#         classes - less than $35k, $35-75K, $75-125K, $125K+)
INCOME_PROP = [10, 20, 30, 40]

# Workers: as the number of workers increases, trip-making increases (there
#          are 4 worker classes - no workers, 1 worker, 2 workers, 3+ workers)
WORKER_PROP = [5, 10, 20, 30]

# Vehicle Ownership: as the number of vehicles increases, trip-making 
#                    goes up slightly (there are 4 vehicle ownership classes - 
#                    no vehicles, 1 vehicle, 2 vehicles, 3+ vehicles)
VEHICLE_PROP = [10, 15, 20, 20]

# Each activity's purpose-specific propensity is stored in a dictionary.
#  Propensity values are stored in nested dictionaries:
# {Purpose: {
#    Axis_name: {
#       Axis_label: propensity_weight
#       }
#    }
# }

# Home
HH_DIMS = ["HHSize", "Income", "VehOwn", "Workers"]
HH_LABELS = [
    ["HHSize1", "HHSize2", "HHSize3", "HHSize4p"],
    ["Income1", "Income2", "Income3", "Income4"],
    ["Veh0", "Veh1", "Veh2", "Veh3p"],
    ["worker0", "worker1", "worker2", "worker3p"]
]
ALL_HH_LABELS = sum(HH_LABELS, [])
_hh_props_ = [dict(zip(HH_LABELS[0], HHSIZE_PROP)), 
              dict(zip(HH_LABELS[1], INCOME_PROP)),
              dict(zip(HH_LABELS[2], VEHICLE_PROP)),
              dict(zip(HH_LABELS[3], WORKER_PROP))
]
_hh_props_mid_ = dict(zip(HH_DIMS, _hh_props_))
HH_PROP = {
    "HBW": _hh_props_mid_,
    "HBO": _hh_props_mid_,
    "HBSch": _hh_props_mid_
}

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
                                  excl_labels="-",
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

#%% PRODUCTION-END DISAG
print("DISAGGREGATE HB PRODUCTIONS")
tg.disaggregateTrips_hb(taz_trips=taz_tg,
                        block_trips=block_tg,
                        act_array=hh_array, 
                        act_dims=HH_DIMS,
                        purposes=HB_PURPS,
                        end="P",
                        propensities=HH_PROP,
                        base_prop=BASE_PROP,
                        block_axis="block_id",
                        block_id="block_id",
                        block_taz_id="TAZ",
                        taz_axis="TAZ",
                        taz_id="TAZ",
                        logger=logger)

# Log block-level trip generation estimates (home-based)
block_sum = block_tg.sum(["Purpose", "End"])
logger.info(f"\nTrips in window (block scale):\n{block_sum}\n{block_sum.axes}")


# %% ATTRACTION-END: 
print("DISAGGREGATE ATTRACTIONS")
tg.disaggregateTrips_nh(taz_trips=taz_tg,
                        block_trips=block_tg,
                        nh_df=nh_df,
                        lbl_col=ATTR_DIM,
                        act_col=ATTR_VALS,
                        purposes=PURPOSES,
                        end="A",
                        propensities=ATTR_PROP,
                        base_prop=BASE_PROP,
                        wb_df=wb_df,
                        block_axis="block_id",
                        block_id="block_id",
                        block_taz_id="TAZ",
                        taz_axis="TAZ",
                        taz_id="TAZ",
                        wb_block_id="block_id",
                        act_crit="-",
                        levels=LEVELS,
                        act_dims=HH_DIMS,
                        logger=logger)

# Log block-level trip generation estimates (with attractions)
block_sum = block_tg.sum(["Purpose", "End"])
logger.info(f"\nTrips in window (block scale):\n{block_sum}\n{block_sum.axes}")

# %% NHB PRODUCTIONS:
print("DISAGGREGATE NHB PRODUCTIONS")
tg.disaggregateTrips_nh(taz_trips=taz_tg,
                        block_trips=block_tg,
                        nh_df=nh_df,
                        lbl_col=ATTR_DIM,
                        act_col=ATTR_VALS,
                        purposes=["NHB"],
                        end="P",
                        propensities=ATTR_PROP,
                        base_prop=BASE_PROP,
                        wb_df=wb_df,
                        block_axis="block_id",
                        block_id="block_id",
                        block_taz_id="TAZ",
                        taz_axis="TAZ",
                        taz_id="TAZ",
                        wb_block_id="block_id",
                        act_crit="-",
                        levels=LEVELS,
                        act_dims=HH_DIMS,
                        logger=logger)

# Log block-level trip generation estimates (with NHB prods)
block_sum = block_tg.sum(["Purpose", "End"])
logger.info(f"\nTrips in window (block scale):\n{block_sum}\n{block_sum.axes}")
