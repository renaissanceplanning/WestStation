# -*- coding: utf-8 -*-
"""
Created 2020

@author: Alex Bell

This script estimates trip generation for home-based trip productions and
non-home-based trip attractions.

Home-based productions are stratified by household type. 
  - Reads estimates of households in each zone across 4 dimensions
      - HH size (1, 2, 3, 4+)
      - Number of workers (0, 1, 2, 3+)
      - Income group (1, 2, 3, 4)
      - Vehicle ownership (0, 1, 2, 3+)
  - Reads trip generation rates and seed data for household cross-class
  - Cross-classifies households into 256 HH types using IPF

Attractions are estimated based on destination activities (jobs by type,
school enrollments, etc.)

Non-home-based productions are estimated based on household variables but
redistributed to match the spatial allocation implied by non-home generators.

This script needs to be run for each land use configuration. Results are
stored in the "lu" folder corresponding to each land use configuration.
Results generated include: 
    
    - **hh_by_type_sum.csv**: csv
        Summary table of households by type by focus and window area 
        inclusion (for high-level QA of results).
        
    - **trip_gen_summary.csv**: csv
        Summary table of trips by purpose and end by focus and window area
        inclusion.
        
    - **trips_by_zone.csv**: csv
        TAZ-level tabulation of trips by purpose and end.
    
    - **HHs_by_type_taz.h5**: LbArray
        cross-classified households by type in each taz.
    
    - **trips_by_taz.h5**: LbArray
        trip productions and attractions by purpose and household type in
        each taz.
"""

#%% IMPORTS
# Main imports
from emma import lba, ipf
import numpy as np
import pandas as pd
import os
from wsa import tg

# All reading/writing taks place within a root directory
#  and analysis is run for a specific land use configuration.
#  Updating the `lu_config` paramter will apply trip gen logic to the 
#  specified configuration
root = r"K:\Projects\MAPC\FinalData"
os.chdir(root)
lu_config = "FEIR_MAX"

#%% ASSUMPTIONS/GLOBALS
# The travel purposes to analyze
PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]

# Define simple vehicle ownership rates as a global variable
# Since the HH cross-classificaiton seed matrices do not include vehicle the
# HH vehicles dimension, it is added based on regional averages (specified
# here) to create a full 4-dimensional seed. The 4D seed is then used in IPF
# to factor household cross-classes to obtain marginal distributions
# that appoximate zonal target estimates.
VEH_DF = pd.DataFrame({"Vehicles": ["Veh0", "Veh1", "Veh2", "Veh3p"],
                       "VehRate": [0.122, 0.358, 0.351, 0.169]})

#%% READ INPUT DATA
print("READ INPUT")
# Socio-economic and demographic data for zones in model
taz_df = tg.readTAZs(
    lu_config, taz_table="MAPC_TAZ_data.xlsx", taz_sheet="Zdata",
    taz_id="TAZ", zones_in_model=r"input\model_area_zones.csv", zim_id="ID")

# Trip generation rates based on household types
rates_csv = r"input\trip_gen_4purpose.csv"
rates_df = pd.read_csv(rates_csv)

# IPF seeds
seed_file = r"input\TAZ_PUMA_Seed_Lookup_forMAPC.xlsx"
seed_sheet = "PUMA_Seed"
seed_df = pd.read_excel(seed_file, seed_sheet)
# Make a "long table" version of the seed data
long_seed = tg.prepSeed(seed_df)

# PUMA lookups to associate zones with seed tables
lookup_sheet = "TAZ_PUMA_lookup"
lookup_df = pd.read_excel(seed_file, lookup_sheet)

#%% CREATE IPF PROBLEM
print("CREATE IPF PROBLEM")
# Prepare dimensional columns for IPF
#  (the IPF seed does not address vehicle ownership,
#   so this column is added separately)
dimensions = ["Size", "Income", "Workers"]
dim_cols = [sorted(long_seed[dc].unique()) for dc in dimensions]
dim_cols.append(["Veh0", "Veh1", "Veh2", "Veh3p"])

# Create the ipf problem
#  (this is an object that facilitates running IPF in series,
#   where targets are read based on data frame row values
#   and seed matrices can be looked up dynamically)
ipf_prob = ipf.IPF_problem_series(taz_df, "TAZ", dim_cols)

#%% SOLVE IPF FOR ALL ZONES
print("SOLVE IPF FOR ALL ZONES")
# Solve the ipf problem
#   (pass a lambda referece to `fetchSeedArray()` to look up the appropriate
#    seed when solving for each TAZ)
xclass_df = ipf_prob.solve(
    lambda zone: tg.fetchSeedArray(zone, lookup_df, long_seed, VEH_DF),
    max_iters=500,
    report_convergence=True
)

#%% JOIN LAND USE DATA
print("JOIN LAND USE DATA")
# Melt the cross-classified df to a long form
value_vars = ipf.flattenDimLabels(dim_cols)
melted_df = xclass_df.melt(id_vars="TAZ", value_vars=value_vars,
                           var_name="HHType", value_name="Households")

# Trip gen rates vary based on a land use parameter, which is joined
#  from the original TAZ data frame 
ready_df = melted_df.merge(taz_df[["TAZ", "LU"]], how="inner", on="TAZ")

#%% APPLY HOME-BASED TRIP GEN RATES
print("APPLY HOME-BASED TRIP GEN RATES")
# Create purpose columns by trip end as containers for trip estimates
purpose_cols = []
for purpose in PURPOSES:
    for end in ["P", "A"]:
        purpose_cols.append(f"{purpose}_{end}")

# Join rates and calc trips
tg_home_df = ready_df.merge(rates_df, how='inner', on=["HHType", "LU"])
for col in purpose_cols:
    tg_home_df[col] = tg_home_df["Households"] * tg_home_df[col]

#%% APPLY NON-HOME-BASED TRIP GEN RATES
print("APPLY NON-HOME-BASED TRIP GEN RATES")
# Elongate non-home trip generators
nh_cols = ["Basic", "Retail", "Service",
           "K12_Emp", "College_Emp", "Other_Service_Emp",
           "EnrollK12", "EnrollCU", "EnrollPreK", "Dorm_Pop"]
long_nh_df = taz_df.melt(id_vars="TAZ", value_vars=nh_cols,
                         var_name="HHType", value_name="Qty")

# Join rates and calc trips
tg_nh_df = long_nh_df.merge(rates_df, how='inner',
                            left_on="HHType", right_on="HHType")
for col in purpose_cols: 
    tg_nh_df[col] = tg_nh_df["Qty"] * tg_nh_df[col]

#%% COMBINE RESULTS
print("COMBINE HOME/NON-HOME RESULTS")
group_cols = ["TAZ", "HHSize", "VehOwn", "Income", "Workers"]
keep_cols = group_cols + purpose_cols
tg_sum_df = pd.concat([tg_home_df[keep_cols], tg_nh_df[keep_cols]])

#%% REFACTOR TRIP ATTRACTIONS TO MATCH PRODUCTIONS
print("REFACTOR ATTRACTIONS TO MATCH PRODUCTIONS")
for purpose in PURPOSES:
    prods = tg_sum_df[f"{purpose}_P"].sum()
    attrs = tg_sum_df[f"{purpose}_A"].sum()
    factor = prods/attrs
    tg_sum_df[f"{purpose}_A"] *= factor

#%% REALLOCATE NON-HOME-BASED TRIP PRODUCTIONS
print("REALLOCATE NON-HOME-BASED TRIP PRODUCTIONS")
# In effect, the number of NHB trips is determined by household variables
# but they begin and end in the locations defined by the non-home activities
tg_sum_df["NHB_P"] = tg_sum_df["NHB_A"]

#%% DEFINE OUTPUT DIMENSIONS AND CONTAINER
print("PREP OUTPUTS")
# Melt the tg_sum_df
id_vars = [c for c in tg_sum_df.columns if c not in purpose_cols]
tg_sum_melted = tg_sum_df.melt(id_vars, value_vars=purpose_cols,
                               var_name="Purpose", value_name="Trips")

# Determine purpose and trip end
tg_sum_melted["End"] = tg_sum_melted["Purpose"].str.split("_").str.get(1)
tg_sum_melted["Purpose"] = tg_sum_melted["Purpose"].str.split("_").str.get(0)


# Summarize results
dimension_columns = id_vars + ['Purpose', "End"]
tg_sum_melted = tg_sum_melted.groupby(dimension_columns).sum().reset_index()

#%% EXPORT RESULTS - HOUSEHOLDS BY TYPE
print("EXPORT HH BY TYPE")
# Setup output H5 file
out_file = r"lu\{}\HHs_by_type_taz.h5".format(lu_config)
node = "/"
name = "households"

# Use the tg_home_df data frame to build the labeled array on disk
hh_array = lba.dfToLabeledArray(tg_home_df, id_vars, "Households",
                                fill_value=0.0)
# Add levels to zone index
hh_array.TAZ.addLevel(
    taz_df[["TAZ", "INWINDOW", "INFOCUS"]], left_on="index", right_on="TAZ")
hh_array.TAZ.levels = ["TAZ", "INWINDOW", "INFOCUS"]
# Save to disk
hh_array = hh_array.copy(hdf_store=out_file, node_path=node, name=name,
                         overwrite=True)
# Export summaries
sum_csv = r"lu\{}\hh_by_type_sum.csv".format(lu_config)
sum_fields = ["INFOCUS", "INWINDOW", "HHType"]
tg_home_out = tg_home_df.merge(taz_df[["TAZ", "INWINDOW", "INFOCUS"]],
                              how="inner", on="TAZ")
hh_sum = tg_home_out.groupby(sum_fields).sum()["Households"]
hh_sum.to_csv(sum_csv)


#%% EXPORT RESULTS - TRIPS BY ACTIVITY
print("EXPORT TRIPS BY PURPOSE AND END")
# Setup output H5 file
out_file =  r"lu\{}\trips_by_taz.h5".format(lu_config)
node = "/"
name = "trips"

# Use the melted array to build the labeled array on disk
trips_array = lba.dfToLabeledArray(tg_sum_melted, dimension_columns,
                                   "Trips", fill_value=0.0)
# Add levels to zone index
trips_array.TAZ.addLevel(
    taz_df[["TAZ", "INWINDOW", "INFOCUS"]], left_on="index", right_on="TAZ")
trips_array.TAZ.levels = ["TAZ", "INWINDOW", "INFOCUS"]
# Save to disk
trips_array = trips_array.copy(hdf_store=out_file, node_path=node, name=name,
                               overwrite=True)

# Export summaries
trips_csv = r"lu\{}\trip_gen_summary.csv".format(lu_config)
tg_out = tg_sum_melted.merge(taz_df[["TAZ", "INWINDOW", "INFOCUS"]],
                             how="inner", on="TAZ")
sum_fields = ["INFOCUS", "INWINDOW", "Purpose", "End"]
tg_sum_out = tg_out.groupby(sum_fields).sum()["Trips"]
tg_sum_out.to_csv(trips_csv)

by_zone_csv = r"lu\{}\trips_by_zone.csv".format(lu_config)
sum_fields.insert(0, "TAZ")
by_zone_out = tg_out.groupby(sum_fields).sum()["Trips"]
by_zone_out.to_csv(by_zone_csv)
