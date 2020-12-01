"""
Created 2020

@author: Alex Bell

This script reads skim data from csv files into the emma.od.Skim format
for use in accessibility and downstream travel modeling analyses.

Modes of travel for which skim data are provided include:
    - Auto (SOV)
    - Transit
        - Walk access (all technologies)
        - Drive access (various technologies)
            - Commuter rail
            - Ferry
            - Local bus
            - Rapid transit
    - Bicyling
    - Walking

The typical procedure for translating skim data from tables to Skim matrices
includes:
    - Identifying the csv data source
    - Defining the origin_id and destination_id columns in the csv
    - Defining the Skim zones index and axis labels
    - Define how value columns in the csv relate to axes
    - Initialize the skim object (pointing to an HDF store if desired)
    - Import data
"""
# -*- coding: utf-8 -*-
#%% IMPORTS
import emma
from emma import lba
import numpy as np
import pandas as pd
import os
from wsa import impfuncs

# All reading/writing taks place within a root directory
#  and analysis is run for a specific network configuration.
#  Updating the `net_config` paramter will ensure the skims written
#  by this script are stored in the appropriate folder reflecting
#  each network config.
#  The `lu_config` parameter is just to pull parking charges from an
#  approriate land use configuration.
root = r"K:\Projects\MAPC\FinalData"
os.chdir(root)
net_config = "Base"
lu_config = "Base"

#%% GLOBALS
PURPOSES = ["HBW", "HBO", "HBSch", "NHB"]
USE_TERM_TIME = True

# Average walk/bike speeds in mph for TAZ-level estimates of walk/bike time
AVG_WALK_SPEED = 2.5
AVG_BIKE_SPEED = 6.5

# Prorate half-day parking charges to mintes
PARKING_FACTOR = 240.0 #assumes 1/2 day = 4 hours

# Money to time conversion factors (for every $x.yz in costs, add one
#  minute of travel time)
WORK_TIME_VALUE = 0.45
NONWORK_TIME_VALUE = 0.25
P_FACTORS = {
    "HBW": WORK_TIME_VALUE,
    "HBSch": WORK_TIME_VALUE,
    "HBO": NONWORK_TIME_VALUE,
    "NHB": NONWORK_TIME_VALUE
    }
SUBMODES = ["BT", "CR", "LB", "RT"]

#%% READ ZONE DATA FOR SKIM INDEXING
# Skims always include i and j dimeinsions reflecting orign and destination
#  zones. To create a skim and maintain consistency across all skim files,
#  zone lists are read in here. Walk and bike skims are recorded at the
#  block scale and are only needed for blocks in the "window area". All other
#  modes record OD data at the TAZ scale and apply across the entire model
#  area.

# TAZ's in model area
taz_file = r"input\model_area_zones.csv"
taz_df = pd.read_csv(taz_file)


# Blocks in window area
block_file = r"input\window_blocks.csv"
block_df = pd.read_csv(block_file, dtype={"GEOID10": str})

# Some renaming to help references be more consistent downstream
taz_df.rename({"ID": "TAZ"}, axis=1, inplace=True)
block_df.rename({"GEOID10": "block_id"}, axis=1, inplace=True)

#%% AUTO SKIMS - INITIALIZATION
# For the auto mode, the following impedances are used in accessibility
#  and travel modeling:
#   - Time (in congested conditions)
#   - Total cost (operating cost + tolls)
#   - Parking costs (derived from zonal parking charges at destination)
#   - Distance (used for coarse walk/bike analysis outside window area)
#   - Terminal times (derived from zonal terminal time estimates - optional)
print("AUTO - INITIALIZATION")
# Zone indices (column names in zone data frames) and impedance labels
index_fields = ["INWINDOW", "INFOCUS", "TAZ"]
impedance_labels = ["OpsTolls", "TravelTime", "ParkRate",
                    "Distance", "WalkTime", "BikeTime"]

# Create the Skim object, initializing values to -1.0 in the arrays.
#  OD pairs found in the source csv files will overwrite these negative values.
#  Any remaining negative values designate "missing" data, usually reflecting
#  OD pairs that have no reasonable interaction potential (within the mode
#  being analyzed). Downstream analysis will handle these negative values
#  appropriately..
hdf_auto = r"net\{}\auto.h5".format(net_config)
auto_costs = impfuncs.initImpSkim_wsa(
    taz_df, index_fields, impedance_labels, hdf_auto, "/", "costs",
    overwrite=True, desc="Travel times and costs by highway mode")

#%% AUTO SKIMS - IMPORT COSTS FROM CSV
# Import data
print("AUTO - IMPORT")
# Source data and origin/destination fields
auto_csv_time = r"input\Skims\{}\SOV_AM.csv".format(net_config)
auto_csv_dist = r"input\Skims\{}\Distances.csv".format(net_config)
zonal_imped_f = r"input\zones\{}\MAPC_TAZ_data.xlsx".format(lu_config)
zonal_imped_sheet = "Terminal Time Parking Cost"
o_field = "from_zone_id"
d_field = "to_zone_id"

# Define which csv columns correspond to which impedances
#  In this case, columns are only availble for three cost variables
#    - Time (`CongTime`, from the `auto_csv_time` table)
#    - Total cost (`Total_Cost` from the `auto_csv_time` table)
#    - Distance (`Distance` from the `auto_csv_dist` table)
#  All other cost elements are calculated or modified through functions
field_map_time = {
    "Total_Cost": {"Impedance": "OpsTolls"},
    "CongTime": {"Impedance": "TravelTime"}}
field_map_dist = {"Distance": {"Impedance": "Distance"}}

#  Since the skims use a multi-level index, we need to specify the `level`
#   parameter to determine which zones correspond to the values in `o_field`
#   and `d_field`.
print("-- times")
emma.od.loadOD_csv(auto_costs, auto_csv_time, o_field, d_field, 
                   field_map_time, level="TAZ")
print("-- distances")
emma.od.loadOD_csv(auto_costs, auto_csv_dist, o_field, d_field,
                   field_map_dist, level="TAZ",
                   names=["from_zone_id", "to_zone_id", "Distance"])

#%% AUTO SKIMS - ESTIMATE TAZ-LEVEL WALKING AND BIKING TIMES
# Using highway distances between zones, estimate non-motorized travel times
#  to support coarse TAZ-level walk and bike accessibility scores. These
#  provide a mechanism for applying mode choice models that use these scores
#  in areas outside the window.
print("AUTO - TAZ WALK/BIKE CALCS")
# Divide assumed travel speeds by 60.0 to convert from mph to miles per minute
auto_costs.put(auto_costs.Impedance["Distance"].data/(AVG_WALK_SPEED/60.0),
               Impedance="WalkTime")
auto_costs.put(auto_costs.Impedance["Distance"].data/(AVG_BIKE_SPEED/60.0),
               Impedance="BikeTime")

#%% AUTO SKIMS - EMBELLISH COSTS BASED ON ZONAL DATA
# Load parking charges and terminal times
zone_costs = pd.read_excel(zonal_imped_f, zonal_imped_sheet)
id_field = "TAZ_ID"
parking_col = "Parking_cost"
print("AUTO - PARKING COSTS")
# Look up parking charges and apply to `ParkRate` matrix based on destination
#  zone. Reset the `ParkRate` matrix to zeros instead of -1.0
auto_costs.put(0.0, Impedance="ParkRate")
impfuncs.addZonalCosts(auto_costs, "Impedance", "ParkRate", zone_costs,
              parking_col, origin_cost=False)

# Apply terminal times if needed
if USE_TERM_TIME:
    tt_fields = ["Hwy_P_TermTime", "Hwy_A_TermTime"]
    origin_cost = True
    for tt_field in tt_fields:
        impfuncs.addZonalCosts(auto_costs, "Impedance", "TravelTime",
                               zone_costs, tt_field, origin_cost=origin_cost)
        origin_cost = False
            
#%% AUTO SKIMS - CALCULATE PURPOSE-SPECIFIC COSTS
# Purpose-specific costs consist of
#   - TravelTime (including terminal times if included)
#   - OpsTolls (converted to minutes, rate varies by purpose)
#   - ParkingCharges (converted to minutes, varies by purpose): 
#       charges depend on duration of stay (1/2 day charge pro-rated)
print("AUTO - PURPOSE-SPECIFIC GEN COST (/gc_by_purpose)")
# Start by casting travel times into the purpsoes-specific cost array
purpose_axis = lba.LbAxis("Purpose", PURPOSES)
desc = "Purpose-specific gen cost for auto - time, op cost, tolls, parking"
if USE_TERM_TIME:
    desc = desc + ", terminal time"
    
auto_gc_purp = auto_costs.cast(purpose_axis, Impedance="TravelTime",
                               squeeze=True, hdf_store=hdf_auto,
                               node_path="/", name="gc_by_purpose",
                               overwrite=True, desc=desc)

# Estimate and apply monetary costs as time modifiers
#  - Estimate parking time based on time, destination-end changes, and purpose.
#  - Estimate parking charges based on parking time and charges (pro-rated)
#  - Convert the results into "additional minutes" (travel time modifier) and
#     add to travel time estimates
#  - Convert OpsTolls costs to time modifer and add to travel time estimates
for purp_i, purpose in enumerate(PURPOSES):
    # Estimate parking duration
    parking = impfuncs.estimateParkingDuration(
        auto_costs.take(Impedance="TravelTime", squeeze=True).data,
        auto_costs.take(Impedance="ParkRate", squeeze=True).data,
        purpose)

    # Estimate parking charges
    parking *= auto_costs.take(
        Impedance="ParkRate", squeeze=True).data/PARKING_FACTOR
    
    # Convert parking charges to time modifiers and add penalties to skim data
    time_conversion = P_FACTORS[purpose]
    auto_gc_purp.data[purp_i] += parking / time_conversion
    
    # Convert ops/tolls charges to time modifiers and add penalties
    auto_gc_purp.data[purp_i] += auto_costs.take(
        Impedance="OpsTolls", squeeze=True).data / time_conversion
    
#%% TRANSIT SKIMS (WAT) - INITIALIZATION
# For the transit (walk access) mode, the following impedances are used in
#  accessibility and travel modeling:
#   - TotalTime (in-vehicle + out-of-vehicle)
#   - In-vehicle time
#   - Out-of-vehicle time (walking, waiting, transfering)
#   - Fare
#   - GeneralizedCost (derived from IVTT, OVTT and Fare if not directly
#      imported)
print("TRANSIT (WAT) - INITIALIZATION")
# Zone indices and impedance labels
index_fields = ["INWINDOW", "INFOCUS", "TAZ"]
impedance_labels = ["TotalTime", "IVTT", "OVTT", "Fare", "GenCost"]

# Initialize data store
hdf_wat = r"net\{}\transit.h5".format(net_config)
wat_costs = impfuncs.initImpSkim_wsa(
    taz_df, index_fields, impedance_labels, hdf_wat, "/", "costs",
    overwrite=True, desc="Travel times and costs by transit (wat) mode")

#%% TRANSIT SKIMS (WAT) - IMPORT BASIC COSTS
# Import data
print("TRANSIT (WAT) - IMPORT")
# Source data and origing/destination fields
transit_csv = r"input\Skims\{}\WAT_AM.csv".format(net_config)
o_field = "from_zone_id"
d_field = "to_zone_id"

# Define which csv columns correspond to which impedances
#  Cost elements not found in the csv source are calculated downstream
field_map = {    
    "Total_IVTT": {"Impedance": "IVTT"},
    "Total_OVTT": {"Impedance": "OVTT"},
    "Fare": {"Impedance": "Fare"},
    "GenCost": {"Impedance": "GenCost"} 
    }
emma.od.loadOD_csv(wat_costs, transit_csv, o_field, d_field, 
                   field_map, level="TAZ")

#%% TRANSIT SKIMS (WAT) - CALCULATE TOTAL TIME
print("TRANSIT (WAT) - TOTAL TIME CALC")
wat_costs.put(wat_costs.take(Impedance="IVTT", squeeze=True).data + 
              wat_costs.take(Impedance="OVTT", squeeze=True).data,
              Impedance="TotalTime")

#%% TRANSIT SKIMS (DAT) - INITIALIZATION AND IMPORT
print("TRANSIT (DAT) - INITIALIZATION/IMPORT/CALCS")
# There are four submodes for the DAT mode. Each is imported into its own
#  skim object. Iterate over submodes...
for submode in SUBMODES:
    print(submode)
# For the transit (drive access) mode, the following impedances are used in
#  accessibility and travel modeling:
#   - TotalTime (in-vehicle + out-of-vehicle)
#   - In-vehicle time
#   - Out-of-vehicle time (walking, waiting, transfering)
#   - Fare
#   - Driving distance
#   - Parking cost
#   - Total cost
#   - GeneralizedCost (derived from IVTT, OVTT and monetary costs if not 
#      directly imported)
    
    # Zone indices and impedance labels
    index_fields = ["INWINDOW", "INFOCUS", "TAZ"]
    impedance_labels = ["TotalTime", "IVTT", "OVTT", "Fare",
                        "DriveDist", "ParkCost", "TotalCost", "GenCost"]

    # Output data store
    print('..initialize')
    hdf_dat = r"net\{}\transit_da_{}.h5".format(net_config, submode)
    skim_dat = impfuncs.initImpSkim_wsa(
            taz_df, index_fields, impedance_labels, hdf_dat, "/", 
            "costs", overwrite=True, 
            desc=f"Travel times and costs by transit (dat - {submode})")

    # Source data and origing/destination fields
    dat_csv = r"input\Skims\{}\DAT_{}_AM.csv".format(
        net_config, submode)
    o_field = "from_zone_id"
    d_field = "to_zone_id"
    
    # Define which csv columns correspond to which impedances
    #  Cost elements not found in the csv source are calculated downstream
    field_map = {
        "Total_IVTT": {"Impedance": "IVTT"},
        "Total_OVTT": {"Impedance": "OVTT"},
        "Fare": {"Impedance": "Fare"},
        "DriveDist": {"Impedance": "DriveDist"},
        "PNR_Parking_Cost": {"Impedance": "ParkCost"},
        "TotCost": {"Impedance": "TotalCost"},
        "GenCost": {"Impedance": "GenCost"}
        }
    
    # Import data
    print("...import")
    emma.od.loadOD_csv(skim_dat, dat_csv, o_field, d_field, 
                       field_map, level="TAZ")

    # Calculate total time
    print('...calculate total time')
    skim_dat.put(
        skim_dat.take(Impedance="IVTT", squeeze=True).data + 
        skim_dat.take(Impedance="OVTT", squeeze=True).data,
        Impedance="TotalTime")

#%% TRANSIT SKIMS (DAT) - GENERALIZE
# For trip distribution, we need generalized costs reflecting the best
# available DAT option, so we'll calculate it now
print("TRANSIT (DAT) - GENERALIZE TO BEST AVAILABLE SUBMODE (/BestGC)")
index_fields = ["INWINDOW", "INFOCUS", "TAZ"]
impedance_labels = ["BestGC", "TotalTime"]
desc = "Generalized transit (DAT) travel costs (best available)"
# Gen skim initialization
hdf_dat = r"net\{}\transit_da.h5".format(net_config)
dat_gen_gc = impfuncs.initImpSkim_wsa(
                taz_df, index_fields, impedance_labels, hdf_dat, "/",
                "costs", overwrite=True, init_val=np.inf)

# Iterate over submode skims
for submode in SUBMODES:
    ref_hdf = r"net\{}\transit_da_{}.h5".format(net_config, submode)
    hdf_node = f"/costs"
    sub_costs = emma.od.openSkim_HDF(ref_hdf, hdf_node)
    # Take the generalized costs and times
    od_gc = sub_costs.take(Impedance="GenCost", squeeze=True)
    od_time = sub_costs.take(Impedance="TotalTime", squeeze=True)
    # Mask out missing values and set to infinity
    mask = np.where(od_gc.data <= 0)
    od_gc.data[mask] = np.inf
    mask_time = np.where(od_gc.data < dat_gen_gc.data[0])
    mask_time_g = (np.ones_like(mask_time[0]), mask_time[0], mask_time[1])
    
    # Get minimum value bewteen the submode skim and prevailing gen skim
    dat_gen_gc.data[0, :] = np.minimum(dat_gen_gc.data[0, :], od_gc.data)
    # Keep times for this submode
    dat_gen_gc.data[mask_time_g] = od_time.data[mask_time]
    
#%% WALK - INITIALIZE AND IMPORT
# For the walk mode, the following impedances are used:
#   - Time
#   - Distance
# Walking and biking are analyzed at the block level in the window area
#  so the index fields and reference data frame are different for these modes
print("WALK - INITIALIZATION")
index_fields = ["INWINDOW", "INFOCUS", "TAZ", "block_id"]
impedance_labels = ["Time", "Distance"]
hdf_walk = r"net\{}\walk.h5".format(net_config)
skim_walk = impfuncs.initImpSkim_wsa(
                    block_df, index_fields, impedance_labels, hdf_walk, "/",
                    "costs", overwrite=True, desc="Walking costs in window")

# Source data and origin/destination fields
walk_csv = r"input\Skims\{}\WalkAllWindow.csv".format(net_config)
o_field = "from_geoid"
d_field = "to_geoid"

# Define which csv columns correspond to which impedances
field_map = {
        "Total_Minutes": {"Impedance": "Time"},
        "Total_Length": {"Impedance": "Distance"}
        }
print("WALK - IMPORT")
# Import data
#  Make sure block id's are read in as strings rather than numbers
emma.od.loadOD_csv(skim_walk, walk_csv, o_field, d_field,
                    field_map, level="block_id",
                    dtype={o_field: str, d_field: str})

#%% BIKE - SETUP AND IMPORT
# For the bike mode, the following impedances are used:
#   - Time
#   - Distance
#   - CyclewayDistance (length of shortest path that utilizes cycleways)
#   - StressfulDistance (lenght of shortest path that utilizes "stressful"
#      facilities - high speed, high traffic roads)
print("BIKE - INITIALIZATION")
index_fields = ["INWINDOW", "INFOCUS", "TAZ", "block_id"]
impedance_labels = ["Time", "Distance", "CyclewayDistance", "StressfulDistance"]
hdf_bike = r"net\{}\bike.h5".format(net_config)
skim_bike = impfuncs.initImpSkim_wsa(
                    block_df, index_fields, impedance_labels, hdf_bike, "/",
                    "costs", overwrite=True, desc="Biking costs in window")

# Source data and origin/destination fields
bike_csv = r"input\Skims\{}\BikeAllWindow.csv".format(net_config)
o_field = "from_geoid"
d_field = "to_geoid"

# Define which csv columns correspond to which impedances
field_map = {
        "Total_Minutes": {"Impedance": "Time"},
        "Total_Length": {"Impedance": "Distance"},
        "Total_LenCycleway": {"Impedance": "CyclewayDistance"},
        "Total_LenLimited": {"Impedance": "StressfulDistance"}
        }
print("BIKE - IMPORT")
# Import data
#  Make sure block id's are read in as strings rather than numbers
emma.od.loadOD_csv(skim_bike, bike_csv, o_field, d_field, 
                    field_map, level="block_id",
                    dtype={o_field: str, d_field: str})