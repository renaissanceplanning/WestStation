# -*- coding: utf-8 -*-
"""
Created 2020

@ Author: Aaron Weinstock, Alex Bell

This script estimates the number of TNC trips by TAZ based on a trip table,
modal skims (to understand costs by the estimated mode and estimate costs
for a potential TNC trip), and *alpha* values. The probability for taking a
TNC trip given its generalized cost must exceed the probability of taking the
same trip by the original estimated mode by a factor that exceeds the alpha
value set for that mode and travel purpose.

This is a post-processing step that ingests ouptuts from previous steps to
provide insight into the number of TNC trips made and what those trips
imply for mode shifts.

TNC trip cost parameters are specified as global parameters: `TNC_BASE_FARE`,
`TNC_SERVICE_FEE`, `TNC_COST_PER_MILE`.

Alpha values may be calculated when the script is run to calibrate them to
targets for aggregate TNC trips by purpose and mode replaced. Once calibrated
to a base condition, the same values should be run for alternative scenarios
or alternative TNC cost constructions to gauge the impact on TNC utilization.

The script summarizes TNC estimates in two csv tables:

- **TNC_flow_summary_{period}.csv**: reports TNC trips by purpose and mode
replaced for generalized reporting geographies (focus, window, and remainder
areas). Summarizes TNC trips, total trips (in the original trip tables), and
percent switchted to TNC.

- **TNC_zone_summary_{period}.csv**: similar to `TNC_flow_summary`, but
broken down on a TAZ-by-TAZ basis.

"""

# %% IMPORTS
import emma
from emma import lba, ipf
from wsa import TNC
import pandas as pd
import numpy as np
import os

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

scen = "RailVision"
lu_config = "FEIR"
net_config = "RailVision"

# Setup logging
logger = logging.getLogger("EMMA")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"scen\{}\log_TNC.log".format(scen), mode="w")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %% GLOBALS
ESTIMATE_TNC_COSTS = False
PERIOD = "AM"
USE_UNITS = "Dollars"
CALC_ALPHAS = False
# Cutoff values if not estimaing on-the-fly, dictionary by purpose - order
#  of values reflects order of modes listed in trip table arrays.
ALPHAS = {
    "HBW": [20.250, 1.952, 3.679, 3.454, 0.900, 1.479],
    "HBO": [35.046, 2.609, 7.294, 2.913, 1.057, 2.275],
    "HBSch": [41.365, 2.462, 40.364, 6.298, 1.002, 1.611],
    "NHB": [40.604, 2.349, 1.800, 1.721, 3.415, 10.639]
    }

PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]
MODES = ["walk", "bike", "driver", "passenger", "WAT", "DAT"]
MODE_DICT = {
    "walk": "nonmotorized",
    "bike": "nonmotorized",
    "driver": "auto",
    "passenger": "auto",
    "WAT": "transit",
    "DAT": "transit_da"
    }
DECAY_REFS = {
    "driver": "auto",
    "passenger": "auto",
    "WAT": "transit",
    "DAT": "transit_da",
    "bike": "auto",
    "walk": "auto"}
TNC_MODE_ORDER = ["walk", "bike", "WAT", "DAT", "driver", "passenger"]

# Mode impedance = {Mode: (node, axis, label, units)}
MODE_IMPEDANCES = {
    "driver": ("/gc_by_purpose", "Purpose", PURPOSES, "minutes"),
    "passenger": ("/gc_by_purpose", "Purpose", PURPOSES, "minutes"),
    "WAT": ("/costs", "Impedance", "GenCost", "dollars"),
    "DAT": ("/costs", "Impedance", "BestGC", "dollars"),
    "bike": ("/costs", "Impedance", "BikeTime", "minutes"),
    "walk": ("/costs", "Impedance", "WalkTime", "minutes")
    }

AUTO_IMPEDANCE_NODE = "/costs"
AUTO_IMPEDANCE_AXIS = "Impedance"
AUTO_IMPEDANCE_TIME_LBL = "TravelTime"
AUTO_IMPEDANCE_DIST_LBL = "Distance"

VALUE_OF_TIME = {
    "HBW": 27.10,
    "HBSch": 27.10,
    "HBO": 15.20,
    "NHB": 15.20
    }

TNC_BASE_FARE = 5.80
TNC_SERVICE_FEE = 0.20
TNC_COST_PER_MILE = 1.17

# TNC DECAY PARAMS: {Purpose: {units: value}}
TNC_DECAY_MU = {
    "HBW": {"minutes": 3.730, "dollars":  2.934},
    "HBSch": {"minutes": 3.730, "dollars":  2.934},
    "HBO": {"minutes": 4.134, "dollars":  2.761},
    "NHB": {"minutes": 4.134, "dollars":  2.761}
    }
TNC_DECAY_SIGMA = {
    "HBW": {"minutes": 0.209, "dollars":  0.209},
    "HBSch": {"minutes": 0.209, "dollars":  0.209},
    "HBO": {"minutes": 0.208, "dollars":  0.208},
    "NHB": {"minutes": 0.208, "dollars":  0.208}
    }

# %% RUN TNC ESTIMATION
if ESTIMATE_TNC_COSTS:
    for purpose in PURPOSES:
        print(purpose)
        logger.info(f"\n {purpose} \n-------------------------------")
        # Initialize TNC Arrays
        tnc_skim_f = r"scen\{}\TNC_costs_{}_{}.h5".format(
            scen, PERIOD, purpose)
        # -- Costs
        tnc_cost_skim = TNC.initTNCCostArray(scen, purpose, PERIOD,
                                         hdf_store=tnc_skim_f,
                                         node_path="/", name="costs",
                                         overwrite=True)
        # -- Probabilities
        tnc_ratio_skim = TNC.initTNCRatioArray(scen, purpose, PERIOD,
                                         hdf_store=tnc_skim_f,
                                         node_path="/", name="ratios",
                                         overwrite=True)
        # Estimate TNC costs
        value_of_time = VALUE_OF_TIME[purpose]
        # Open auto skim
        auto_f = r"net\{}\auto.h5".format(net_config)
        auto_skim = emma.od.openSkim_HDF(auto_f, AUTO_IMPEDANCE_NODE)
        TNC.estimateTNCCosts(auto_skim,
                            purpose=purpose,
                            tnc_cost_skim=tnc_cost_skim,
                            value_of_time=value_of_time,
                            tnc_base_fare=TNC_BASE_FARE,
                            tnc_service_fee=TNC_SERVICE_FEE,
                            tnc_cost_per_mile=TNC_COST_PER_MILE,
                            tnc_decay_mu=TNC_DECAY_MU[purpose],
                            tnc_decay_sigma=TNC_DECAY_SIGMA[purpose],
                            imp_axis=AUTO_IMPEDANCE_AXIS,
                            time_label=AUTO_IMPEDANCE_TIME_LBL,
                            dist_label=AUTO_IMPEDANCE_DIST_LBL,
                            logger=logger)
        
        # Estimate TNC prob ratio/likelihood
        TNC.estimateTNCProb(net_config=net_config,
                            purpose=purpose,
                            tnc_ratio_skim=tnc_ratio_skim,
                            tnc_cost_skim=tnc_cost_skim,
                            decay_refs=DECAY_REFS,
                            mode_dict=MODE_DICT,
                            mode_impedances=MODE_IMPEDANCES,
                            use_units=USE_UNITS,
                            all_purposes=PURPOSES,
                            logger=logger)
    
# %% APPLY TNC ESTIMATION
for purpose in PURPOSES:
    print(purpose)
    logger.info(f"\n {purpose} \n-------------------------------")
    tnc_skim_f = r"scen\{}\TNC_costs_{}_{}.h5".format(scen, PERIOD, purpose)
    tnc_ratio_skim = emma.od.openSkim_HDF(tnc_skim_f, "/ratios")
    # Setup estimation
    trip_table_f = r"scen\{}\Trip_dist_TAZ_{}_{}.h5".format(
        scen, PERIOD, purpose)
    trip_table = emma.od.openSkim_HDF(trip_table_f, "/dist")
    tnc_hdf = r"scen\{}\TNC_trips_{}_{}.h5".format(scen, PERIOD, purpose)
    # Calc alphas if needed
    if CALC_ALPHAS:
        alphas = []
        # Read ipf seed
        seed_xl = r"input\TNC_target_setting.xlsx"
        seed_array = TNC.fetchTNCSeed(seed_xl, named_range="ipf_seed",
                                      purp_col="Purpose", modes=MODES)
        # Read mode targets
        mode_targets = TNC.fetchTNCModeTargets(seed_xl,
                                               named_range="mode_targets",
                                               modes=TNC_MODE_ORDER)
        # Align targets with seed array axes
        purp_targets = seed_array.sum("Purpose").data
        mode_targets = mode_targets.reindex(seed_array.Mode.labels).values
        # Apply IPF
        mp_targets = ipf.IPF(seed_array, [purp_targets, mode_targets])
        for mode in trip_table.Mode.labels:
            alpha = TNC.estimateAlphas(trip_table, tnc_ratio_skim, mp_targets,
                                       mode, purpose)
            print(f".. alpha for {mode}: {alpha}")
            logger.info(f".. alpha for {mode}: {alpha}")
            alphas.append(alpha)
    else:
        alphas = ALPHAS[purpose]

    # -- run probability ratios
    tnc_prob = TNC.applyTNCProbRatio(trip_table, tnc_ratio_skim, alphas,
                                     hdf_store=tnc_hdf, node_path="/",
                                     name="ProbRatio", overwrite=True,
                                     logger=logger)
    
# %% REPORTING
print("RERPORTING RESULTS")

out_csv = r"scen\{}\TNC_flow_summary_{}.csv".format(scen, PERIOD)
out_csv2 = r"scen\{}\TNC_zone_summary_{}.csv".format(scen, PERIOD)
header=True
m = "w"
for purpose in PURPOSES:
    print(f"-- {purpose}")
    # Open trip table
    trip_table_f = r"scen\{}\Trip_dist_TAZ_{}_{}.h5".format(
        scen, PERIOD, purpose)
    trip_table = emma.od.openSkim_HDF(trip_table_f, "/dist")
    # Open TNC probability table
    tnc_hdf = r"scen\{}\TNC_trips_{}_{}.h5".format(scen, PERIOD, purpose)
    tnc = lba.openLbArray_HDF(tnc_hdf, "/ProbRatio")
    
    # FLOW SUM
    # Summarize number of trips that switch by mode
    diss_axes = ["From", "To"]
    diss_labels = ["INWINDOW", "INFOCUS"]
    switch_by_mode = tnc.dissolve(
        diss_axes, [diss_labels, diss_labels], np.sum)
    tnc_df = switch_by_mode.to_frame("tnc_trips")
    # Summarize total number of trips
    trips_by_mode = trip_table.dissolve(
        diss_axes, [diss_labels, diss_labels], np.sum)
    trips_df  = trips_by_mode.to_frame("total_trips")
    # Join tables
    trips_join = tnc_df.join(trips_df)
    trips_join["Pct_Switch"] = trips_join.tnc_trips/trips_join.total_trips
    # Add other details
    trips_join.reset_index(inplace=True)
    for dx in diss_axes:
        new_cols = [f"{dx}_{dl}" for dl in diss_labels]
        trips_join[new_cols] = pd.DataFrame(
            trips_join[dx].to_list(), index=trips_join.index)
    trips_join.drop(columns=diss_axes, inplace=True)
    trips_join["Purpose"] = purpose
    trips_join["Period"] = PERIOD
    trips_join["scen"] = scen
    
    # ZONE SUM
    # Summarize number of trips that switch by mode from/to each zone
    stack = []
    sum_axes = ["From", "To"]
    for axis in sum_axes:
        sum_by = ["Mode", axis]
        tnc_df = tnc.sum(sum_by).to_frame("tnc_trips", unpack_indices=True)
        trips_df = trip_table.sum(sum_by).to_frame(
            "total_trips", unpack_indices=True)
        # Join tables
        join_fields = ["Mode", f"{axis}_TAZ", f"{axis}_INWINDOW", f"{axis}_INFOCUS"]
        renames = ["Mode", "TAZ", "INWINDOW", "INFOCUS"]
        trips_join = tnc_df.merge(trips_df, how="inner", on=join_fields)
        trips_join["Pct_Switch"] = trips_join.tnc_trips/trips_join.total_trips
        trips_join.rename(
            dict(zip(join_fields, renames)), axis=1, inplace=True)
        trips_join["FT"] = axis[0]        
        # Add to the stack
        stack.append(trips_join)
    zone_sums = pd.concat(stack)        
    # Add other details
    zone_sums["Purpose"] = purpose
    zone_sums["Period"] = PERIOD
    zone_sums["scen"] = scen
    
    # Write results
    trips_join.to_csv(out_csv, mode=m, header=header, index=False)
    zone_sums.to_csv(out_csv2, mode=m, header=header, index=False)
    header=False
    m = "a"
