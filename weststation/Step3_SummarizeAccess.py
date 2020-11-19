"""
Created 2020

@author: Alex Bell

This script reads modal impedances from skim files, specifies decay
rates by purpose, and calculates purpose-specific OD decay factors.
It then summarizes access scores using zonal activity data and the
calculated decay factors.

General steps
  - Read data
      - Travel cost skims
      - Zonal activity data
  - Specify decay rates
  - Calculate decay factors
  - Summarize access scores
  - Export results
"""
# -*- coding: utf-8 -*-
#%% IMPORTS
# Main imports
import emma
from emma import labeled_array as lba
import numpy as np
import pandas as pd
import os
from wsa import access

# All reading/writing taks place within a root directory
#  and analysis is run for a specific scenario.
#  The scenario consists of a combination of land use and network
#  configurations. Specify which `scen`, `net_config`, and `lu_config`
#  will be analyzed. Resulting accessibility scores will reflect land
#  development in the named `lu_config` and travel costs in the named
#  `net_config`. Outputs are stored in the `scen` subfolder.
root = r"K:\Projects\MAPC\FinalData"
os.chdir(root)
scen = "BRT_Scen_B"
lu_config = "FEIR"
net_config = "BRT_Scen_B"

#%% GLOBALS
MODES = ["auto",
         "transit",
         "bike",
         "walk",
         #"transit_da_BT",
         #"transit_da_CR",
         #"transit_da_LB",
         #"transit_da_RT",
         "transit_da"
         ]

PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]

# If decay skims have already been calculated (if, for example, only land
#  use changes are being tested), set the `CALC_DECAY` parameter to False.
#  Decay calculations can take some time, so this step can be skipped if
#  it isn't needed.
CALC_DECAY = True

#%% READ INPUT DATA
# The accessibility summarization procedures utilize the following datasets
#  - TAZ activity data (for auto and transit)
#  - Block activity data (for walking and biking)
#  - Skims
print("READ INPUT DATA")
# see wsa.summarize_access.loadInputZones function for table names, etc.
#  override if needed
taz_df, block_hh_df, block_emp_df = access.loadInputZones(lu_config)

# Connect to skim files and store in a list
#  For most modes, the skim node to connect is `costs`, but for auto
#  use `gc_by_purpose`
cast_from_node = {"auto": "gc_by_purp"}
skim_objs = {}
for mode in MODES:
    if mode == "auto":
        node_path = "/gc_by_purpose"
    else:
        node_path = "/costs"
    skim_file = r"net\{}\{}.h5".format(net_config, mode)
    skim = emma.od.openSkim_HDF(skim_file, node_path)
    skim_objs[mode] = skim

#%% REFORMAT ZONE DATA
# Block-level activity data are stored in long tables that need to be
#  transformed to a wide format. Then, total activities (households, jobs,
#  e.g.) are calculated.
print("REFORMAT ZONE DATA")
# Pivot block employment data to wide form
idx_col = "block_id"
cols = "GenSector"
vals = "tot_emp"
block_emp = pd.pivot_table(block_emp_df, index=idx_col, columns=cols, 
                           values=vals).fillna(0)

# Pivot block household data to wide form. Two pivots are needed:
#  -- Income
cols = "Income"
vals = "Households"
block_inc = pd.pivot_table(block_hh_df, index=idx_col, columns=cols,
                           values=vals).fillna(0)
#  -- Vehicle ownership
cols = "VehOwn"
block_veh = pd.pivot_table(block_hh_df, index=idx_col, columns=cols,
                           values=vals).fillna(0)

# Calculate total emp, total enrollment
job_fields = ["Basic", "Retail", "Service"]
enrollment_fields = ["EnrollPreK", "EnrollK12", "EnrollCU"]
block_emp["Total Emp"] = block_emp[job_fields].sum(axis=1)
block_emp["TotEnroll"] = block_emp[enrollment_fields].sum(axis=1)

# Join hh data and calculate total households
inc_fields = ["Income1", "Income2", "Income3", "Income4"]
block_inc["TotalHH"] = block_inc[inc_fields].sum(axis=1)
block_hh= block_veh.merge(block_inc, how="inner", 
                          left_index=True, right_index=True)

#%% SPECIFY DECAY RATES
# Decay rates define the affect that travel costs (time, monetary, etc.) have
#  on the value of accessible activities. Higher costs generally degrade the
#  relevance of activities between zones.
# Two types of decay curves are calculated:
#  - A "cumulative" decay form that expresses a general willingness to travel
#     in light of travel costs. This is used to evaluate accessibility.
#  - A probability density form that defines typical trip distributions based
#     on travel costs. This is used in trip distribution.
print("SPECIFY DECAY RATES")
decay_table = r"input\decay_specs.csv"
# CDF curves
crit_a = {"Mode": "auto", "Distribution": "cumulative"}
crit_t = {"Mode": "transit", "Distribution": "cumulative"}
crit_d = {"Mode": "transit_da", "Distribution": "cumulative"}
crit_n = {"Mode": "nonmotorized", "Distribution": "cumulative"}
auto_cdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_a)))
trans_cdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_t)))
trans_da_cdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_d)))
nm_cdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_n)))

# PDF curves
crit_a = {"Mode": "auto", "Distribution": "density"}
crit_t = {"Mode": "transit", "Distribution": "density"}
crit_d = {"Mode": "transit_da", "Distribution": "density"}
crit_n = {"Mode": "nonmotorized", "Distribution": "density"}
auto_pdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_a)))
trans_pdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_t)))
trans_da_pdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_d)))
nm_pdf = dict(zip(PURPOSES, access.decaysFromTable(decay_table, **crit_n)))

# Assemble all decays in dictionaries:
cdf_decays = {
    "auto": auto_cdf,
    "transit": trans_cdf,
    "walk": nm_cdf,
    "bike": nm_cdf,
    "transit_da_BT": trans_da_cdf,
    "transit_da_CR": trans_da_cdf,
    "transit_da_LB": trans_da_cdf,
    "transit_da_RT": trans_da_cdf,
    "transit_da": trans_da_cdf
    }

pdf_decays = {
    "auto": auto_pdf,
    "transit": trans_pdf,
    "walk": nm_pdf,
    "bike": nm_pdf,
    "transit_da_BT": trans_da_pdf,
    "transit_da_CR": trans_da_pdf,
    "transit_da_LB": trans_da_pdf,
    "transit_da_RT": trans_da_pdf,
    "transit_da": trans_da_pdf
    }

#%% CONNECT/CREATE DECAY SKIMS
# Decay rates are used to calculate decay factors based on travel costs.
#  If calculating decay due to new/updated skims, decay skim files will
#  be created in this step by casting travel costs into purpose dimensions
#  and applying decay factors. The decay skim files will have two nodes:
#    - cdf (cumulative decay factor)
#    - pdf (probability density factor)
#  If only land use changes are being modeled, this step simply connects
#  to existing decay skim files.
print("CONNECT/CREATE DECAY SKIMS")
# store cdf skim objects in a dictionary for use later in this script.
cdf_skims = {}

if CALC_DECAY:
    #To calculate decay factors, OD costs are copied into a new skim.
    # These values are then modified in-place using the decay objects
    # defined above. The criteria below define which costs are copied.
    # For the auto mode, purpose-specific costs from the `gc_by_purpose`
    # node are used. The `Purpose` dimension is fully copied into the
    # new decay skim.
    # For other modes, the `cost` node is referenced, and a single cost
    # estimate from the `Impedance` dimension is broadcast into the 
    # new decay skim.
    cast_criteria = {
        "auto": {"Purpose": PURPOSES},
        "transit": {"Impedance": "GenCost"},
        "transit_da_BT": {"Impedance": "GenCost"},
        "transit_da_CR": {"Impedance": "GenCost"},
        "transit_da_LB": {"Impedance": "GenCost"},
        "transit_da_RT": {"Impedance": "GenCost"},
        "transit_da": {"Impedance": "BestGC"},
        "walk": {"Impedance": "Time"},
        "bike": {"Impedance": "Time"}
        }
    
    # Cast the decay skim for each mode and calculate decay factors by
    #  purpose.
    for mode in MODES:
        print("anaylysing decay for", mode, "and connecting to decay skim")
        # Connect to the cost skim and look up field, rate references
        skim = skim_objs[mode]
        criteria = cast_criteria[mode]
        
        # Set up the decay skim 
        decay_file = r"net\{}\{}_decay.h5".format(net_config, mode)
        node_path = "/"
        cdf_name = "cdf"
        pdf_name = "pdf"
        new_axis = lba.LbAxis("Purpose", PURPOSES)
        
        # Cast cost data into the decay skim  
        #  For the auto mode, this is just copying the purpose-specific
        #  generalized costs. For other modes, use the skim.cast() method
        if mode == "auto":
            cdf_skim = skim.copy(hdf_store=decay_file, node_path=node_path,
                                 name=cdf_name, overwrite=True)
            pdf_skim = skim.copy(hdf_store=decay_file, node_path=node_path,
                                 name=pdf_name, overwrite=True)
        else:
            cdf_skim = skim.cast(new_axis, hdf_store=decay_file,
                                   node_path=node_path, name=cdf_name,
                                   overwrite=True, squeeze=True, **criteria)
            pdf_skim = skim.cast(new_axis, hdf_store=decay_file,
                                 node_path=node_path, name=pdf_name,
                                 overwrite=True, squeeze=True, **criteria)
        
        #Apply decay rates by purpose
        for purpose in PURPOSES:
            cdf_rate = cdf_decays[mode][purpose]
            pdf_rate = pdf_decays[mode][purpose]
            #cdf application
            cdf_skim.put(
                cdf_rate.apply(cdf_skim.Purpose[purpose]),
                Purpose=purpose)
            #pdf application
            pdf_skim.put(
                pdf_rate.apply(pdf_skim.Purpose[purpose]),
                Purpose=purpose)
        
        # Add skims to dictionaries
        cdf_skims[mode] = cdf_skim

else:
    for mode in MODES:
        # Point to the existing decay skim
        decay_file = r"net\{}\{}_decay.h5".format(net_config, mode)
        cdf_path = "/cdf"
        pdf_path = "/pdf"
        
        # Create the skim connection
        cdf_skim = emma.od.openSkim_HDF(decay_file, cdf_path)
        pdf_skim = emma.od.openSkim_HDF(decay_file, pdf_path)
        
        # Add skims to dictionaries
        cdf_skims[mode] = cdf_skim

#%% SUMMARIZE ACCESS TO DESTINATIONS
# Summarize access to five destination activities:
#  - Total jobs (Total Emp)
#  - Retail jobs
#  - Service jobs
#  - Basic-sector jobs
#  - School enrollments (TotEnroll)
print("ACCESS TO DESTINATIONS")
sum_fields = ["Total Emp", "Retail", "Service", "Basic", "TotEnroll"]

# Iterate over modes and purposes
for mode in MODES:
    print(mode)
    # Fetch the decay skim object
    cdf_skim = cdf_skims[mode]
    for purpose in PURPOSES:
        print("...", purpose)
        # Run the `summarizeAccess` function
        #  - Walk and bike use block-level activities
        #  - Other modes use TAZ-level activities
        if mode in ["bike", "walk"]:
            access_to_jobs = emma.od.summarizeAccess(block_emp,
                                 sum_fields, cdf_skim, key_level="block_id", 
                                 Purpose=purpose)
        else:
            access_to_jobs = emma.od.summarizeAccess(taz_df.set_index("TAZ"),
                                 sum_fields, cdf_skim, key_level="TAZ",
                                 Purpose=purpose)
        
        # Export access results to a csv file
        out_csv = r"scen\{}\access_to_jobs_{}_{}.csv".format(
            scen, mode, purpose)
        access_to_jobs.to_csv(out_csv)
        
#%% SUMMARIZE ACCESS FROM HOUSEHOLDS
# Analyzing access in the "from" direction summarizes the number of origin-
#  end activities that can travel to each zone.

# Summarize access from households by income group and by vehicle ownership.
#  Also summarize access from total households
print("ACCESS FROM ORIGINS")
sum_fields_hh = ["Income1", "Income2", "Income3", "Income4", 
                 "Veh0", "Veh1", "Veh2", "Veh3p", "TotalHH"]

for mode in MODES:
    print(mode)
    # Fetch the decay skim object
    cdf_skim = cdf_skims[mode]
    for purpose in PURPOSES:
        print("...", purpose)
        # Run the `summarizeAccess` function
        #  - Walk and bike use block-level activities
        #  - Other modes use TAZ-level activities
        # The `access_to_dests` arg can be set to False to summarize access
        #  in the appropriate direction
        if mode in ["bike", "walk"]:
            access_from_hh = emma.od.summarizeAccess(block_hh,
                                 sum_fields_hh, cdf_skim, key_level="block_id", 
                                 Purpose=purpose, access_to_dests=False)
        else:
            access_from_hh = emma.od.summarizeAccess(taz_df.set_index("TAZ"),
                                 sum_fields_hh, cdf_skim, key_level="TAZ",
                                 Purpose=purpose, access_to_dests=False)
        
        # Export access results to a csv file
        out_csv = r"scen\{}\access_from_hh_{}_{}.csv".format(
            scen, mode, purpose)
        access_from_hh.to_csv(out_csv)

#%% TAZ-LEVEL WALK AND BIKE
# All of the patterns established above are appled again to estimate
#  walking and biking accessibility at the TAZ scale.
#  - Calculate decay rates or connect to existing decay skim.
#  - Define access fields
#  - Summarize access and export outputs
print("TAZ-LEVEL WALK/BIKE")
# Calculate or connect to decay factors skim
# Point to the decay file
decay_file = r"net\{}\nonmotor_decay_TAZ.h5".format(net_config)

if CALC_DECAY:
    # Fetch the basic auto skim that has walk and bike time estimates at
    #  the TAZ level
    ref_file = r"net\{}\auto.h5".format(net_config)
    ref_skim = emma.od.openSkim_HDF(ref_file, "/costs")
    
    # Create the decay skim file
    node_path = "/"
    name = "cdf"
    name_p = "pdf"
    new_axis = lba.LbAxis("Purpose", PURPOSES)
    
    # Cast the skim into new dimensions
    taz_walk_bike = ref_skim.cast(new_axis, hdf_store=decay_file, 
                                  node_path=node_path, name=name,
                                  overwrite=True,
                                  Impedance=["WalkTime", "BikeTime"])
    taz_walk_bike_p = ref_skim.cast(new_axis, hdf_store=decay_file,
                                    node_path=node_path, name=name_p,
                                    overwrite=True, 
                                    Impedance=["WalkTime", "BikeTime"])
    
    # Apply non-motorized decay reates to walk and bike impedances
    #  We can just refer to the `walk` decay since both modes use the same
    #  rate.
    for purpose in PURPOSES:
        cdf_rate = cdf_decays["walk"][purpose]
        pdf_rate = pdf_decays["walk"][purpose]
        #Only summarize non-motorized trip opportunities within 45 minutes
        cdf_rate.max_impedance = 45
        # Apply the decay rate - both modes' time estimates will be run
        #  simultaneously
        taz_walk_bike.put(
            cdf_rate.apply(taz_walk_bike.Purpose[purpose]),
            Purpose=purpose)
        taz_walk_bike_p.put(
            pdf_rate.apply(taz_walk_bike_p.Purpose[purpose]),
            Purpose=purpose)
else:
    taz_walk_bike = emma.od.openSkim_HDF(decay_file, "/cdf")
    taz_walk_bike_p = emma.od.openSkim_HDF(decay_file, "/pdf")
    
# Summarize accessibility for non-motorized trips
nm_modes = ["walk", "bike"]
nm_costs = ["WalkTime", "BikeTime"]
for mode, cost in zip(nm_modes, nm_costs):
    for purpose in PURPOSES:
        print(purpose)
        # Access to jobs for this purpose
        access_to_jobs = emma.od.summarizeAccess(
            taz_df.set_index("TAZ"), sum_fields, taz_walk_bike, 
            key_level="TAZ", Purpose=purpose, Impedance=cost
            )
    
        # Export jobs access results
        out_csv = r"scen\{}\access_to_jobs_{}_{}_TAZ.csv".format(
            scen, mode, purpose)
        access_to_jobs.to_csv(out_csv)
    
        # Access from households for this purpose
        access_from_hh = emma.od.summarizeAccess(
            taz_df.set_index("TAZ"), sum_fields_hh, taz_walk_bike, 
            key_level="TAZ", Purpose=purpose, Impedance=cost, 
            access_to_dests=False
            )
       
        # Export hh access results
        out_csv = r"scen\{}\access_from_hh_{}_{}_TAZ.csv".format(
            scen, mode, purpose)
        access_from_hh.to_csv(out_csv)