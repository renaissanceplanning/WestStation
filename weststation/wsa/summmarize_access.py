from emma import decay
import numpy as np
import pandas as pd


def loadInputZones(lu_config, taz_table="MAPC_TAZ_data.xlsx", taz_sheet="Zdata",
                    block_table_hh="Household_Types_by_Block.csv",
                    block_table_emp="Jobs_Enroll_by_Block.csv",
                    taz_id="TAZ", block_id="block_id"):
    """
    Reads zone input tables from default locations.

    Parameters
    -----------
    lu_config: String
    taz_table: String, default="MAPC_TAZ_data.xlsx"
    taz_sheet: String, default="Zdata"
    block_table_hh: String, default="Household_Types_by_Block.csv"
    block_table_emp: String, default="Jobs_Enroll_by_Block.csv"
    taz_id: String, default="TAZ"
    block_id: String, default="block_id"

    Returns
    -------
    taz_df: pd.DataFrame
    block_hh_df: pd.DataFrame
    block_emp_df: pd.DataFrame
    """
    taz_file = r"input\Zones\{}\{}".format(lu_config, taz_table)
    taz_df = pd.read_excel(taz_file, taz_sheet)

    block_hh_file = r"input\Zones\{}\{}".format(lu_config, block_table_hh)
    block_hh_df = pd.read_csv(block_hh_file, dtype={block_id: str})

    block_emp_file = r"input\Zones\{}\{}".format(lu_config, block_table_emp)
    block_emp_df = pd.read_csv(block_emp_file, dtype={block_id: str})

    return taz_df, block_hh_df, block_emp_df


def decaysFromTable(decay_table, **selection_criteria):
    """
    Create Decay objects based on parameters specified in a csv file.

    Parameters
    -----------
    decay_table: String
        Path to a well-formed csv file with decay curve specifications.
    selection_criteria:
        Keyword arguments for selecting rows from the table when constructing
        decay objects (`Mode="auto"` will only construct auto decay curves,
        e.g.).
    """
    # Load the table
    decay_table = pd.read_csv(decay_table)
    # Filter the table
    if selection_criteria:
        fltrs = []
        for col_name in selection_criteria.keys():
            crit = selection_criteria[col_name]
            fltr = decay_table[col_name] == crit
            fltrs.append(fltr)
        whole_fltr = np.logical_and.reduce(fltrs)
        decay_table = decay_table[whole_fltr].copy()
    table_dict = decay_table.to_dict(orient="records")
    # Setup field specs
    req_field_headings = ["ConstMu", "CoefSig"]
    opt_field_headings = [
        "min_impedance",
        "max_impedance",
        "excl_less_than_min",
        "excl_greater_than_max",
        "lbound",
        "ubound"
    ]
    # Make each object in the table
    decay_objs = []
    for row in table_dict:
        params = row.keys()
        args=[]
        kwargs={}
        for param in params:
            param_val = row[param]
            if param in req_field_headings:
                if np.isnan(param_val):
                    raise ValueError(
                        f"Decay parameter {param} cannot be null")
                else:
                    args.append(param_val)
            elif param in opt_field_headings:
                kwargs[param] = param_val
            elif param == "Form":
                decay_form = param_val
            
        decay_obj = getattr(decay, decay_form)(*args, **kwargs)
        decay_objs.append(decay_obj)
    return decay_objs


