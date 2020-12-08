# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:20:47 2020

@author: Alex Bell

This script models trip distribution in five steps:

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

  - **trips_by_mode_and_purpose_dist_{period}.csv**: Summarizes trips by mode
  and pupose to and from the key broad reporting geographies: focus area,
  window area, remainder area.

  - **trip_len_dist_{period}.csv**: A summary of trips, person miles of
  travel, and average trip length to and from each TAZ. Can be summarized
  for arbitray collections of TAZs, including built-in groupings for zones
  in the focus and window areas. When summarizing, average trip length should
  be calculated based on the sum of person miles of travel divided by the
  sum of trips.

  -  **trip_dur_dist_{period}.csv**: Same as `trip_len_dist.csv`, but focused on
  trip duration (in minutes).

   **trip_cost_dist_{period}.csv**: Same as `trip_len_dist.csv`, but focused on
  trip generalized costs (in dollars).
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

scen = "BRT_Scen_B"
lu_config = "LRTP"
net_config = "BRT_Scen_B"

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
LEVELS = ["INWINDOW", "INFOCUS"]
MODES = ["walk", "bike", "driver", "passenger", "WAT", "DAT"]
NONMOTOR_MODES = ["walk", "bike"]
DECAY_REFS = {
    "driver": "auto",
    "passenger": "auto",
    "WAT": "transit",
    "DAT": "transit_da",
    "bike": "bike",
    "walk": "walk"}
DECAY_NODE = "/pdf"
# For each mode, what skim details (file, node, axis, label) are used for 
#  distance estimates?
DISTANCE_SKIMS = {
    "driver": ["auto", "/costs", "Impedance", "Distance"],
    "passenger": ["auto", "/costs", "Impedance", "Distance"],
    "WAT": ["auto", "/costs", "Impedance", "Distance"],
    "DAT": ["auto", "/costs", "Impedance", "Distance"],
    "bike": ["auto", "/costs", "Impedance", "Distance"],
    "walk": ["auto", "/costs", "Impedance", "Distance"]
    }
# For each mode, what skim details (file, node, axis, label) are used for 
#  duration estimates?
TIME_SKIMS = {
    "driver": ["auto", "/costs", "Impedance", "TravelTime"],
    "passenger": ["auto", "/costs", "Impedance", "TravelTime"],
    "WAT": ["transit", "/costs", "Impedance", "TotalTime"],
    "DAT": ["transit_da", "/costs", "Impedance", "TotalTime"],
    "bike": ["auto", "/costs", "Impedance", "BikeTime"],
    "walk": ["auto", "/costs", "Impedance", "WalkTime"]
    }

# For each mode, what skim details (file, node, axis, label) are used 
#  for generalized cost estimates?
COST_SKIMS = {
    "driver": ["auto", "/gc_by_purpose", "Purpose", "{get purp}"],
    "passenger": ["auto", "/gc_by_purpose", "Purpose", "{get purp}"],
    "WAT": ["transit", "/costs", "Impedance", "GenCost"],
    "DAT": ["transit_da", "/costs", "Impedance", "BestGC"],
    "bike": ["auto", "/costs", "Impedance", "BikeTime"],
    "walk": ["auto", "/costs", "Impedance", "WalkTime"]
    }

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

WORK_TIME_VALUE = 0.45
NONWORK_TIME_VALUE = 0.25
P_FACTORS = {
    "HBW": WORK_TIME_VALUE,
    "HBO": NONWORK_TIME_VALUE,
    "HBSch": WORK_TIME_VALUE,
    "NHB": NONWORK_TIME_VALUE
    }

# %% DISTRIBUTION
for pi, purpose in enumerate(PURPOSES):
    # Initialize the trip table
    ###
    ### strip levels so the trip table From/To is not MultiIndex?
    ### OR make sure lba.reindexDf
    ###
    trips_taz, trips_block = ds.initTripTable(scen,
                                              purpose,
                                              PERIOD,
                                              logger=logger)
    # Get trip ends by mode
    period_factor = PERIOD_FACTORS[PERIOD][pi]
    trips_df_taz, trips_df_block = ds.tripEndsByMode(scen=scen,
                                                     purpose=purpose,
                                                     period_factor=period_factor,
                                                     taz_zone_dim="TAZ",
                                                     block_zone_dim="block_id",
                                                     mode_dim="Mode",
                                                     purp_dim="Purpose",
                                                     end_dim="End",
                                                     block_taz_level="TAZ",
                                                     logger=logger)
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
                     logger=logger)
    # -- Blocks
    ds.seedTripTable(net_config=net_config,
                     purpose=purpose,
                     trip_table=trips_block,
                     trips_df=trips_df_block,
                     decay_refs=DECAY_REFS,
                     mode_col="Mode",
                     trips_col="trips",
                     end_col="End",
                     id_col="block_id",
                     level="block_id",
                     scale="block",
                     modes=MODES,
                     nonmotor_modes=NONMOTOR_MODES,
                     decay_node=DECAY_NODE,
                     logger=logger)
    
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

# %% REPORTING - TRIPS BY MODE/PURPOSE
# Levels
print("REPORTING - TRIPS BY MODE AND PURPOSE (DIST)")
model_area_zones = pd.read_csv("input\model_area_zones.csv")
level_cols = ["TAZ", "INWINDOW", "INFOCUS"]
out_csv = r"scen\{}\trips_by_mode_and_purpose_dist_{}.csv".format(scen, PERIOD)
header=True
m = "w"
# Trips by mode 
for purpose in PURPOSES:
    print(f" -- {purpose}")
    hdf_f = r"scen\{}\trip_dist_taz_{}_{}.h5".format(scen, PERIOD, purpose)
    trips_taz = emma.od.openSkim_HDF(hdf_f, "/dist")
    # trip dimensions: Mode, From, To
    # Iterate over mode
    for mode in MODES:
        # dump to data frame
        trips_df = trips_taz.take(Mode=mode).to_frame("trips").reset_index()
        # Unpack multiIndex
        for dim in ["From", "To"]:
            dim_df = trips_df.copy()
            if isinstance(trips_taz.zones, pd.MultiIndex):
                levels = trips_taz.zones.names
                dim_df[levels] = pd.DataFrame(dim_df[dim].to_list(),
                                              index=trips_df.index)
            else:
                dim_df = dim_df.merge(model_area_zones[level_cols],
                                      how="inner", left_on=dim, right_on="TAZ")
            # Summarize
            dim_df["Purpose"] = purpose
            dim_df["Direction"] = dim
            dim_df["Scenario"] = scen
            group_cols = ["Scenario", "Purpose", "Mode", "Direction",
                          "INWINDOW", "INFOCUS"]
            sum_df = dim_df.groupby(group_cols).sum()["trips"].reset_index()
            # Append to csv output
            sum_df.to_csv(out_csv, mode=m, header=header)
            header=False
            m="a"

# %% REPORTING - TRIP LENGTH/DURATION
print("REPORTING - TRIP LENGTH AND DURATION")
trip_len_csv = r"scen\{}\trip_len_dist_{}.csv".format(scen, PERIOD)
trip_dur_csv = r"scen\{}\trip_dur_dist_{}.csv".format(scen, PERIOD)
trip_cost_csv = r"scen\{}\trip_cost_dist_{}.csv".format(scen, PERIOD)
group_cols = ["INWINDOW", "INFOCUS"]
header=True
m="w"

# Trips by mode 
for purpose in PURPOSES:
    print(f" -- {purpose}")
    # Fetch trips by mode
    hdf_f = r"scen\{}\trip_dist_taz_{}_{}.h5".format(scen, PERIOD, purpose)
    trips_taz = emma.od.openSkim_HDF(hdf_f, "/dist")
    # trip dimensions: Mode, From, To
    # Iterate over mode
    for mode in MODES:
        factor = 1.0
        miles = ds.summarizeTripAttributes(trips_taz, mode, net_config, 
                                           DISTANCE_SKIMS, "Miles",
                                           sum_dims=["From", "To"],
                                           factor=factor)
        minutes =  ds.summarizeTripAttributes(trips_taz, mode, net_config,
                                              TIME_SKIMS, "Minutes",
                                              sum_dims=["From", "To"])
        # Cost summaries
        if COST_SKIMS[mode][-1] == "{get purp}":
            p_spec_label = True
            COST_SKIMS[mode][-1] = purpose
        else:
            p_spec_label = False
        factor = P_FACTORS[purpose]
        costs = ds.summarizeTripAttributes(trips_taz, mode, net_config,
                                           COST_SKIMS, "Cost",
                                           sum_dims=["From", "To"],
                                           factor=factor)
        if p_spec_label:
            COST_SKIMS[mode][-1] = "{get purp}"
        
        # Tag with mode, purpose, scenario
        miles["Mode"] = mode
        minutes["Mode"] = mode
        costs["Mode"] = mode
        miles["Purpose"] = purpose
        minutes["Purpose"] = purpose
        costs["Purpose"] = purpose
        miles["scen"] = scen
        minutes["scen"] = scen
        costs["scen"] = scen
        # Export
        miles.to_csv(trip_len_csv, mode=m, header=header)
        minutes.to_csv(trip_dur_csv, mode=m, header=header)
        costs.to_csv(trip_cost_csv, mode=m, header=header)
        header=False
        m = "a"
        
        



