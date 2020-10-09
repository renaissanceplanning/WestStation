# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:20:47 2020

@author: Alex Bell

This script models trips distribution in #### steps:

1. Creates empty trip tables for each purpose. Each table has an axis
   for each travel mode in addition to `From` and `To` axes.

2. Seeds each purpose's trip tables using production and attraction
   estimates produced for each mode by the mode choice model and the
   `weightedInteractions` method in emma. In this step, period factors
   are applied to adjust daily trip estimates down to period estimates.

3. Setup shadowing - this forges a relationship between the block-level
   trip table and the TAZ-level trip table. Factors applied to trips at
   TAZ level are mimicked at the block level. This allows block-level
   detail to influence mode shares and distribution to reflect travel
   opportunities beyond the window area.

4. Apply k factors - these factors adjust emma's raw mode choice estimates
   to better match an observed or modeled condition in a specific scenario.
   A preferred set of factors should be establsihed and used for all
   other scenarios for meaningful cross-scenario comparisons.

5. Balance the trip tables using iterative proporational fitting (IPF).
   The balancing process focuses on total person trip productions and 
   attractions in the selected travel period. Trips by mode can fluctuate
   during the balancing process.

After the balancing step is complete, results are reported in csv files.
"""

#%% IMPORTS
# Main imports
import emma
from emma import lba, ipf
import numpy as np
import pandas as pd
import os
from wsa import distribution as ds

import logging

# All reading/writing taks place within a root directory
#  and analysis is run for a specific scenario.
#  The scenario consists of a combination of land use and network
#  configurations.  Specify which `scen`, `net_config`, and `lu_config`
#  will be analyzed. Resulting accessibility scores will reflect land 
#  development in the named `lu_config` and travel costs in the named
#  `net_config`.  Outputs are stored in the `scen` subfolder.
root = r"K:\Projects\MAPC\FinalData"
os.chdir(root)

scen = "RV_gc_parking_tt"
lu_config = "FEIR"
net_config = "RV_gc_parking_tt"

# Setup logging
logger = logging.getLogger("EMMA")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"scen\{}\log_distribution.log".format(scen), mode="w")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#%% GLOBALS
PERIOD = "AM"
PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]
MODES = ["walk", "bike", "driver", "passenger", "WAT", "DAT"]
NONMOTOR_MODES = ["walk", "bike"]
DECAY_REFS = {
    "driver": "auto",
    "passenger": "auto",
    "WAT": "transit",
    "DAT": "transit_da_gen",
    "bike": "bike",
    "walk": "walk"}
DECAY_NODE = "/pdf"

# TODO: can we be sure the ordering of purposes and modes is consistent with
#  labeled array axis labels?
PERIOD_FACTORS = {
    "AM": [0.246, 0.178, 0.122, 0.178],
    "MD": [0.218, 0.565, 0.372, 0.565],
    "PM": [0.274, 0.148, 0.216, 0.148],
    "NT": [0.262, 0.109, 0.290, 0.109]
    }

CONVERGES_AT = 1e-3
MAX_ITERS = 100
TOLERANCE = 1e-5

#K_FACTORS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Naive
#K_FACTORS = [0.81, 0.81, 1.09, 1.09, 0.56, 0.98] # Survey region
#K_FACTORS = [0.85, 0.85, 1.04, 1.04, 0.98, 0.26] # CTPS Base region
#K_FACTORS = [0.70, 0.70, 1.18, 1.18, 0.49, 0.22] # CTPS LRTP region
#K_FACTORS = [0.99, 0.99, 1.01, 1.01, 1.01, 0.42] # CTPS Base window
K_FACTORS = [0.83, 0.83, 1.41, 1.41, 0.57, 0.24] # CTPS LRTP window
out_csv = r"scen\{}\trips_by_mode_and_purpose_dist.csv".format(scen)

# %% DISTRIBUTION
for purpose in PURPOSES:
    # Initialize the trip table
    trips_taz, trips_block = ds.initTripTable(scen,
                                              purpose,
                                              PERIOD,
                                              logger=logger)
    # Get trip ends by mode
    trips_df_taz, trips_df_block = ds.tripEndsByMode(scen=scen,
                                                     purpose=purpose,
                                                     period=PERIOD,
                                                     period_factors=PERIOD_FACTORS,
                                                     taz_zone_dim="TAZ",
                                                     block_zone_dim="block_id",
                                                     mode_dim="Mode",
                                                     purp_dim="Purpose",
                                                     end_dim="End",
                                                     block_taz_level="TAZ",
                                                     logger=None)
    # Seed matrix
    # -- TAZs
    ds.seedTripTable(net_config=net_config,
                    purpose=purpose,
                    trip_table=trips_taz,
                    trips_df=trips_df_taz,
                    decay_refs=DECAY_REFS,
                    mode_col="Mode",
                    trips_col="trips",
                    end_col="End",
                    id_col="TAZ",
                    level="TAZ",
                    scale="TAZ",
                    modes=MODES,
                    nonmotor_modes=NONMOTOR_MODES,
                    decay_node=DECAY_NODE,
                    logger=None)
    # -- Blocks
    ds.seedTripTable(net_config=net_config,
                    purpose=purpose,
                    trip_table=trips_block,
                    trips_df=trips_df_block,
                    decay_refs=DECAY_REFS,
                    mode_col="Mode",
                    trips_col="trips",
                    end_col="End",
                    id_col="TAZ",
                    level="TAZ",
                    scale="TAZ",
                    modes=MODES,
                    nonmotor_modes=NONMOTOR_MODES,
                    decay_node=DECAY_NODE,
                    logger=None)
    
    # Create shadow
    sx_to = lba.ShadowAxis(trips_block.To, trips_taz.To, "TAZ", "TAZ")
    sx_from = lba.ShadowAxis(trips_block.From, trips_taz.From, "TAZ", "TAZ")
    shadow = lba.Shadow(trips_block, trips_taz, [sx_to, sx_from])
    
    # Apply K factors (pre-dist)
    trips_taz.data[:] = ds.applyKFactors(
                            trips_taz, K_FACTORS, mode_axis="Mode").data
    trips_block.data[:] = ds.applyKFactors(
                            trips_block, K_FACTORS, mode_axis="Mode").data
    
    # Set person trip targets by zone for this purpose
    targets_p, targets_a = ds.tripTargetsByZone(trips_df=trips_df_taz,
                                                zone_col="TAZ",
                                                trips_col="trips",
                                                end_col="End",
                                                logger=None)
    m_targets = [targets_p, targets_a]
    
    # Balance table
    key_dims = ["From", "To"]
    print(f"Running IPF for {purpose} trip table")
    trips_taz = ipf.IPF(trips_taz,
                        m_targets,
                        key_dims=key_dims,
                        converges_at=CONVERGES_AT,
                        max_iters=MAX_ITERS,
                        tolerance=TOLERANCE,
                        report_convergence=False,
                        shadows=[shadow],
                        logger=logger,
                        log_axes=["Mode"])

# %% REPORTING
# Levels
print("REPORTING")
model_area_zones = pd.read_csv("input\model_area_zones.csv")
level_cols = ["TAZ", "INWINDOW", "INFOCUS"]
header=True
m = "w"
# Trips by mode 
for purpose in PURPOSES:
    hdf_f = r"scen\{}\trip_dist_taz_{}_{}.h5".format(scen, PERIOD, purpose)
    trips_taz = emma.od.openSkim_HDF(hdf_f, "/dist")
    # trip dimensions: Mode, From, To
    # Iterate over mode
    for mode in MODES:
         # dump to data frame
         trips_df = trips_taz.take(Mode=mode).to_frame("trips").reset_index()
         # Join window, focus columns
         for dim in ["From", "To"]:
             merge_df = trips_df.merge(model_area_zones[level_cols],
                                       how="inner", left_on=dim,
                                       right_on="TAZ")
             # Summarize
             merge_df["Purpose"] = purpose
             merge_df["Direction"] = dim
             merge_df["Scenario"] = scen
             group_cols = ["Scenario", "Purpose", "Mode", "Direction",
                           "INWINDOW", "INFOCUS"]
             sum_df = merge_df.groupby(group_cols).sum()["trips"].reset_index()
             # Append to csv output
             sum_df.to_csv(out_csv, mode=m, header=header)
             header=False
             m="a"

#TODO: report trip lengths
