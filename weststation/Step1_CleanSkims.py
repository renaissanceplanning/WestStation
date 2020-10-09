# -*- coding: utf-8 -*-
"""
Created 2020

@author: Alex Bell

This script is used to read data from source csv files representing OD skims
in long form. The skims are filtered and columns renamed on the fly to prepare
the tables to be imported in the emma `Skim` object.
"""

#%% IMPORT
from wsa import cs
import os
import logging

# All reading/writing taks place within a root directory
#  and analysis is run for a specific network configuration.
#  Updating the `net_config` paramter will ensure the skims written
#  by this script are pulled from and stored in the appropriate 
#  folder reflecting each network config.
root = r"K:\Projects\MAPC\FinalData"
os.chdir(root)
net_config = "RailVision"
os.chdir(r"input\skims\{}".format(net_config))

# Logger setup
logger = logging.getLogger("EMMA")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(r"log_clean_skims.log".format(net_config), mode="w")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#%% FILE IO
# Dict of files by mode {mode_name: [in_file, out_file]}
files_by_mode = {
    "WAT": ["AM_WAT_Skim_RailVision_Alt_6.csv", "WAT_AM.csv"],
    "DAT - BT": ["AM_DAT_BT_Skim_RailVision_Alt_6.csv", "DAT_BT_AM.csv"],
    "DAT - CR": ["AM_DAT_CR_Skim_RailVision_Alt_6.csv", "DAT_CR_AM.csv"],
    "DAT - LB": ["AM_DAT_LB_Skim_RailVision_Alt_6.csv", "DAT_LB_AM.csv"],
    "DAT - RT": ["AM_DAT_RT_Skim_RailVision_Alt_6.csv", "DAT_RT_AM.csv"]
}

# %% RENAMING
rename = {
    "Generalized_Cost": "GenCost",
    "GeneralizedCost": "GenCost",
    "Total_Cost": "TotCost",
    "TotalCost": "TotCost",
    "Access_Drive_Distance": "DriveDist",
    "Access_Drive_Dist": "DriveDist",
    "AccessDriveDist": "DriveDist"
}

#%% CRITERIA
criteria = [
    ("GenCost", "__lt__", 99999),
    ("GenCost", "__ne__", 0)
    ]

# %% CLEAN SKIMS
logger.info(files_by_mode)
logger.info(rename)
logger.info(criteria)
for mode in files_by_mode.keys():
    print(f"----------------------\n{mode}") 
    logger.info(f"----------------------\n{mode}") 
    in_file, out_file = files_by_mode[mode]
    print(cs.previewSkim(in_file, nrows=2, logger=logger))
    cs.cleanSkims(in_file, out_file, criteria, rename=rename, logger=logger)
    print(cs.previewSkim(out_file, nrows=2, logger=logger))
    print("\n----------------------") 
