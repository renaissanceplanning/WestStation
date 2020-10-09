# -*- coding: utf-8 -*-
"""
Created 2020

@ Author: Aaron Weinstock, Alex Bell

What does it do?
For an individual trip, obtain the ratio of TNC probability / observed trip probability

Parameters:
1. purpose: trip purpose -- "HBW", "HBO", "HBSch", or "NHB" (character)
2. mode: mode of the trip -- "Walk", "Bike", "Driver", "Passenger", "WAT", or "DAT" (character)
3. obs_cost: the observed generalized cost of the trip (numeric)
4. distance: distance of the trip in miles (numeric)
5. duration: duration of the trip in minutes (numeric)
6. cost_per_mile: cost per mile of a TNC trip (number, default 14/12 from Fare Choices)
7. method: units for TNC GC construction -- "time" or "dollars" (character, default "time")

Returns:
A dictionary with the following elements:
1. constructed TNC cost
2. TNC trip probability
3. Observed trip probability
4. Ratio value of TNC probability / Observed trip probability

TNC_array = estimated TNC cost and trip probability by purpose

"""

# %% IMPORTS
import emma
from emma import lba, ipf
from wsa import TNC
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

scen = "RV_gc_parking_tt"
lu_config = "FEIR"
net_config = "RV_gc_parking_tt"

# Setup logging
logger = logging.getLogger("EMMA")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"scen\{}\log_TNC.log".format(scen), mode="w")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %% GLOBALS
ESTIMATE_TNC_COSTS = True
PERIOD = "AM"
USE_UNITS = "Dollars"
CALC_ALPHAS = False
# Cutoff values if not estimaing on-the-fly, dictionary by purpose - order
#  of values reflects order of modes listed in trip table arrays.
ALPHAS = {
    "HBW": [75.627, 2.067, 4.156, 1.714, 1.095, 1.333],
    "HBO": [79.838, 2.481, 8.322, 2.914, 1.311, 1.660],
    "HBSch": [130.170, 2.417, 6.635, 5.493, 1.235, 0.374],
    "NHB": [7.456, 1.305, 3.266, 2.860, 1.387, 4.226]
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
    "DAT": ("/costs", None, None, "dollars"),
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
                                     name="ProbRatio", overwrite=True)
