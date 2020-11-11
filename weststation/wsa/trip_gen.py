"""
TRIP GEN
=========

Suppporting functions for executing trip generation modeling for the
MAPC region. Developed as part of the West Station Area accessibility-
based modeling project.
"""

from emma import lba, ipf
import pandas as pd
import numpy as np
import yaml

def readTAZs(lu_config, taz_table="MAPC_TAZ_data.xlsx", taz_sheet="Zdata",
             taz_id="TAZ", zones_in_model=r"input\model_area_zones.csv",
             zim_id="ID"):
    """
    Reads zone input tables from default locations.

    Parameters
    -------------
    lu_config: String
    taz_table: String, default="MAPC_TAZ_data.xlsx"
    taz_sheet: String, default="Zdata"
    taz_id: String, default="TAZ"
    zones_in_model: String, default="input\\model_area_zones.csv"
    zim_id: String, default="ID"
        The field that identifies each TAZ in `zones_in_model`.

    Returns
    -------
    taz_df: pd.DataFrame
        TAZ records from `taz_table` filtered by values in `zones_in_model`
    """
    taz_file = r"input\Zones\{}\{}".format(lu_config, taz_table)
    taz_df = pd.read_excel(taz_file, taz_sheet)
    taz_df.rename({taz_id: "TAZ"}, axis=1, inplace=True)

    zones_in_model = pd.read_csv(zones_in_model)
    zones_in_model.rename({zim_id: "TAZ"}, axis=1, inplace=True)
    taz_df = zones_in_model.merge(taz_df, how="left", on="TAZ")
    taz_df.fillna(0.0, inplace=True)
    
    return taz_df


def readBlocks(lu_config, block_hh_table="Household_Types_by_Block.csv",
               block_emp_table="Jobs_Enroll_by_Block.csv", 
               hh_id_field="block_id", emp_id_field="block_id", 
               window_blocks=r"input\window_blocks.csv", wb_id="GEOID10"):
    """
    Reads block input tables from default locations.

    Parameters
    ----------
    lu_config: String
    block_hh_table: String, default="Household_Types_by_Block.csv"
    block_emp_table: String, default="Jobs_Enroll_by_Block.csv"
    hh_id_field: String, default="block_id"
    hh_field: String, default="Households"
    emp_id_field: String, default="block_id"
    emp_act_field: String, default="tot_emp"
    window_blocks: String, default="window_blocks.csv"
    wb_id: String, default="GEOID10"

    Returns
    -------
    hh_df: pd.DataFrame
        Block-level households by type. Rows reflect only the blocks listed
        in window_blocks.
    nh_df: pd.DataFrame
        Block-level non-home activities. Rows reflect only the blocks listed
        in window_blocks.
    wb_df: pd.DataFrame
        Block ids with TAZ ids. Rows reflect only the blocks listed
        in window_blocks.
    """
    block_hh_file = r"input\Zones\{}\{}".format(lu_config, block_hh_table)
    block_emp_file = r"input\Zones\{}\{}".format(lu_config, block_emp_table)
    hh_df = pd.read_csv(block_hh_file, dtype={hh_id_field: str})
    nh_df = pd.read_csv(block_emp_file, dtype={emp_id_field:str})
    wb_df = pd.read_csv(window_blocks, dtype={wb_id: str})
    hh_df.rename({hh_id_field: "block_id"}, axis=1, inplace=True)
    nh_df.rename({emp_id_field: "block_id"}, axis=1, inplace=True)
    wb_df.rename({wb_id: "block_id"}, axis=1, inplace=True)
    # Left join block HH data to the window blocks file to ensure all blocks
    #  in the window are included.
    hh_df = wb_df.merge(hh_df, how="left", on="block_id")
    nh_df = wb_df.merge(nh_df, how="left", on="block_id")
    
    return hh_df, nh_df, wb_df

def dfToLabeledArray_tg(tg_df, ref_array, dims, act_col,
                        block_id="block_id", excl_labels="-",
                        levels=["block_id", "TAZ", "INWINDOW", "INFOCUS"]):
    """
    Convert a data frame of block-level activities to a labeled array. Fill
    in label values based on a reference array (taz activities, e.g.) and
    enrich and align axes for clean downstream operations.

    Parameters
    -----------
    tg_df: pd.DataFrame
        A data frame with block-level activity data. WARNING! The data frame
        is modified in-place (NaN values are replaced by labels)
    ref_array: LbArray
        A labeled array with activities by TAZ. This provdes the basic form
        for the block-level array returned by the function.
    dims: [String,...]
        Axis names in `ref_array` corresponding to activity dimensions.
    act_col: String
        The column in `tg_df` that contains activity value estimates.
    block_id: String, default="block_id"
        The column in`tg_df` that identifies each block.
    excl_labels: String or [String,...], default="-"
        A collection of labels (in any axis) whose values are dropped when
        casting the block-level array returned by the function.
    levels: [String,...], default=["block_id", "TAZ", "INWINDOW", "INFOCUS"]
        The names of columns in `tg_df` that will be used to add levels
        to the block-level array returned by the function. These keep intact
        block-level nesting information.
    
    Returns
    ---------
    act_array: LbArray
        A block level array of activity data (based on `tg_df`) whose axis
        labels align with those in `ref_array`.
    """
    # Input clean up
    if isinstance(excl_labels, lba.string_types):
        excl_labels = excl_labels
    elif not isinstance(excl_labels, lba.Iterable):
        excl_labels = excl_labels

    dim_cols = [block_id] + [x_name for x_name in ref_array.axisNames() if
                             x_name in dims]

    # Propagate values and axes
    align_axes = []
    for dim in dims:
        # Lookup axis labels
        ref_axis = ref_array.getAxisByName(dim)
        fill_vals = [label for label in ref_axis.labels 
                        if label not in excl_labels]
                
        # Make a copy of the dimension columns and a mask array of
        #  missing values
        col_vals = tg_df[dim].values
        mask = pd.isna(tg_df[dim]).values
        # Tally the number of na's
        num_na = len(col_vals[mask])
        # Create tiled labels (repetitions)
        fill_series = np.tile(fill_vals, (num_na//len(fill_vals)) + 1)

        # Fill column values with tiled labels and update the data frame
        col_vals[mask] = fill_series[:num_na]
        tg_df[dim] = pd.Series(col_vals)

        # Create alignment axis
        align_axis = lba.LbAxis(dim, fill_vals)
        align_axes.append(align_axis)

    # Create a labeled array of activities by type in the window area.
    # Convert block-level activity details into a labeled array for the
    # window area.
    act_array = lba.dfToLabeledArray(tg_df.sort_values(by=block_id),
                                     dim_cols, act_col, fill_value=0.0)
    act_array[block_id].labels.name = block_id
    # Update axes
    #  - levels
    level_df = tg_df.groupby(levels).size().reset_index()[levels]
    act_array[block_id].addLevel(
        level_df, left_on=block_id, right_on=block_id)
    #  - alignment
    for align_axis in align_axes:
        lba.alignAxisLabels(act_array, align_axis.name, align_axis,
                            inplace=True)
    return act_array


def calcPropensity_HB(prop_array, hh_propensity_dict, end_axis="End",
                      prod_label="P"):
    """
    A function to calculate trip-making propensities for home-based trip
    generators. Home-based trip-making propensity is estimated based on 
    household types in each block and their respective generalized trip-making
    propensity factors. 
       
    Parameters
    ------------
    prop_array: LbArray
        A labeled array, assumed to be initialized to 1.0 and having axes 
        with labels corresponding to those named in `hh_propensity_dict`.
    hh_propensity_dict: Dict
        A nested dictionary whose keys correspond to axis names in
        `prop_array` and whose values are dictionaries of labels that are
        members of the named axis and propensity weights.
    end_axis: String, default="End"
        `prop_array` is assumed to cover all potential trips (home-based or
        otherwise) and both trip ends. This function only updates the 
    prod_label: String, default="P"
    
    Returns
    ---------
    prods: LbArray
        An estimate of productions propensity for different household types.
    
    See Also
    --------
    calcPropensity_NH
    normalizePropensity
    applyTAZTrips
    """    
    # estimate home-based production propensity based on hh_propensity_dict 
    prods = prop_array.take(**{end_axis: prod_label})
    for hh_dim in hh_propensity_dict:        
        dim_dict = hh_propensity_dict[hh_dim]
        axis = prop_array.getAxisByName(hh_dim)
        weights = [dim_dict.get(label, 1.0) for label in axis.labels.to_list()]
        # broadcast and multiply
        prods.data *= lba.broadcast1dByAxis(prods, axis.name, weights).data
    return prods
    
    
def calcPropensity_NH(nh_df, lbl_col, act_col, purposes, propensities,
                      base_prop, wb_df, block_id="block_id",
                      wb_block_id="block_id",
                      levels=["block_id", "TAZ", "INFOCUS", "INWINDOW"],
                      logger=None):
    """
    A function to calculate raw block-level trip-making propensities for
    non-home trip generators, by purpose. Non-home trip-making propensity
    is estimated based on various non-home activities in each block and their
    respective generalized trip-making propensity factors.

    Parameters
    -----------
   
    nh_df: pd.DataFrame
        A data frame of non-home activities by block.
    lbl_col: String
        The column in `nh_df` that identifies each block's activity by type.
    act_col: String
        The column in `nh_df` that stores the number of each activity.
    purposes: [String,...]
        The purposes for which trip propensities are estimated.
    propensities: {String: {String: {String: [numeric,...]}}}
        A nested dictionary of trip-making propensities. At the outer level,
        keys are trip purposes; at the middle level, keys are axis names;
        at the inner level, keys are axis labels and values are propensity
        weights.
    base_prop: numeric
        A baseline trip-making propensity. This is a small value that ensures
        all blocks have a nominal trip-making potential. Blocks with 
        activities will have significantly higher propensities based on the
        number of activities and the `propensities` dictionary.
    block_id: String, default="block_id"
        The column in `nh_df` that uniquely identifies each block.
    wb_df: pd.DataFrame
        The data frame listing the blocks in the window area with their TAZ
        parentage.
    wb_block_id: String, default="block_id"
        The column in `wb_df` that uniquely identifies each block.
    levels: [String,...], default=["block_id", "TAZ", "INFOCUS", "INWINDOW"]
        A list of column names in `nh_df` that defined indexing levels.    
    
    Returns
    --------
    nh_array: LbArray
        A labeled array with an axis storing block information (with `levels`
        recorded in a multi-index) and an axis differentiating trip 
        propensities by purpose.

    See Also
    --------
    calcPropensity_HB
    normalizePropensity
    applyTAZTrips
    """
    # Apply trip propensities by destination activity type to each block.
    #  This is done by multiplying purpose-specific propensity rates by
    #  the activity types in non-home activity data frame.
    props = []
    for purpose in purposes:
        prop_df = pd.DataFrame.from_dict(
            propensities[purpose]).stack().reset_index()
        prop_df.columns=["Sector", "Category", "Propensity"]
        prop_df["Purpose"] = purpose
        props.append(prop_df)
    prop_df = pd.concat(props)
    prop_df = pd.pivot(prop_df, index="Sector", columns="Purpose",
                       values="Propensity")
    nh_df_prop = nh_df.merge(prop_df, how="inner", left_on=lbl_col,
                            right_index=True)
    # Multiply the activities by propensities
    # Ensure each block activity has the BASE_PROP propensity at a minimum
    for p in purposes:
        nh_df_prop[p] = (nh_df_prop[p] * nh_df_prop[act_col]) + base_prop

    # Melt this data frame for conversion to an LbArray
    melt_cols = levels + ["Purpose"]
    nh_melted = nh_df_prop.melt(
        levels, purposes, "Purpose", "Propensity").groupby(
            melt_cols).sum().reset_index()
    # Merge melted data back to all window blocks and fill NA's
    #  This ensures all blocks in window are present for array-building
    nh_melted = wb_df.merge(nh_melted, how="left", left_on=wb_block_id,
                            right_on=block_id, suffixes=["", "_x"])
    nh_melted.Purpose.fillna(purposes[0], inplace=True)
    nh_melted.Propensity.fillna(base_prop, inplace=True)

    # Sum total propensity by block and purpose
    nh_melted = nh_melted.groupby([block_id, "Purpose"]).sum().reset_index()
    nh_array = lba.dfToLabeledArray(nh_melted.sort_values(by=block_id),
                                [block_id, "Purpose"], "Propensity",
                                fill_value=base_prop)
    # Add levels
    nh_array[block_id].addLevel(
        wb_df[levels], left_on="index", right_on=wb_block_id)
    nh_array[block_id].levels = levels
    
    return nh_array 


def normalizePropensity(block_prop_array, block_axis, taz_level="TAZ", 
                        block_level="block_id", ends=["P", "A"],
                        purposes=["HBW", "HBO", "HBSch", "NHB"]):
    """
    Estimate each block's share of total TAZ trip-making by activty type.   

    Parameters
    -------------
    block_prop_array: LbArray
        An array of total estimated trip-making propensity by block and
        activity type. Assumed to have axes "Purpose" and "End".
    block_axis: LbAxis
        The axis in `block_prop_array` that identifies each block. Assumed
        to be a MultiIndex object with one level identifying blocks and the 
        other identifying TAZs.
    taz_level: String, default="TAZ"
        The name of the level in `block_axis` that identifies the TAZ each
        block is in.
    block_level: String, default="block_id"  
        The name of the level in `block_axis` that uniquely identifes each
        block.
    ends: [String,...], default=["P", "A"]
        The names of values in the `End` axis of `block_prop_array`.
    purposes: [String,...], default=["HBW", "HBO", "HBSch", "NHB"]
        The names of values in the `Purpose` axis of `block_prop_array`.
    
    Returns
    -------
    None
        `block_prop_array` is updated inplace, such that its raw propensity
        estimates are normalized into shares of total TAZ-level trip-making.
    
    See Also
    ---------
    calcPropensity_HB
    calcPropensity_NH
    applyTAZTrips
    """
    # Convert the input block axis to a data frame
    block_df = block_axis.to_frame()
    
    # Within each purpose and for each trip end, what is each TAZ's total 
    # propensity?
    block_prop_sum = block_prop_array.sum([block_axis.name, "Purpose", "End"])
    taz_prop_sum = block_prop_sum.dissolve(block_axis.name, taz_level, np.sum)
    
    # Convert taz sum to data frame
    taz_prop_df = taz_prop_sum.to_frame("prop").reset_index()
    
    for end in ends:
        for purpose in purposes:
            # Get propensities for this end, purpose
            fltr = np.logical_and(
                taz_prop_df.Purpose == purpose, 
                taz_prop_df.End == end)
            purp_df = taz_prop_df[fltr]
            # Join to blocks by TAZ ID
            merge_df = block_df.merge(purp_df, how="left", left_on=taz_level,
                                      right_on=block_level, 
                                      suffixes=("", "_z"))
            merge_df.fillna(0.0, inplace=True)
            merge_df.set_index([block_level, taz_level], inplace=True)
            # Broadcast for application
            prop_array_sel = block_prop_array.take(
                Purpose=purpose, End=end, squeeze=True)
            prop_cast = lba.broadcast1dByAxis(
                prop_array_sel, block_axis.name, merge_df.prop)
            # Divide to proportion trip propensities
            prop_array_sel.data /= prop_cast.data
            # Write back to block_prop_array
            block_prop_array.put(
                prop_array_sel.data, Purpose=purpose, End=end
                )


def applyTAZTrips(taz_trips_array, taz_axis, taz_level, 
                  block_prop_array, block_axis, block_level,
                  ends=["P", "A"], purposes=["HBW", "HBO", "HBSch", "NHB"]):
    """
    Multiply normalized trip-making propensities at block level by total
    trip estimates (by purpose) at TAZ level.

    Parameters
    ----------
    taz_trips_array : LbArray
        A labeled array with trip estimates by TAZ, purpose, and end
        (production vs. attraction)
    taz_axis : LbAxis
        The axis in `taz_trips_array` that identifies each TAZ.
    taz_level : String
        The name of the level in `taz_axis` that uniquely identifies each TAZ.
    block_prop_array : LbArray
        A labeled array with normalized trip-making propensities by block,
        activity, purpose, and end.
    block_axis : LbAxis
        The axis in `block_prop-array` that identifies each block by its TAZ
        perentage (assumed to use the same name as `taz_level`)
    block_level: String
        The name of the level in `block_axis` that uniquely identifies each
        block.
    ends : [String,...], default=["P", "A"]
        The values in the `End` axis of both `taz_trips_array` and
        `block_prop_array`.
    purposes : [String,...], default=["HBW", "HBO", "HBSch", "NHB"]
        The values in the `Purpose` axis of both `taz_trips_array` and
        `block_prop_array`.

    Returns
    -------
    None
        `block_prop_array` is modified in place such that its normalized
        shares of TAZ-level trip-making are multiplied by total trips in each
        corresponding TAZ to yield total trips at the block level, by activity
        
    See Also
    ---------
    calcPropensity_HB
    calcPropensity_NH
    normalizePropensity
    """
    block_df = block_axis.to_frame()
    trips_df = taz_trips_array.to_frame("trips").reset_index()
    trips_df[taz_axis.levels] = pd.DataFrame(
        trips_df[taz_level].to_list(), index=trips_df.index)
    
    for end in ends:
        for purpose in purposes:
            # Get propensities for this end, purpose
            fltr = np.logical_and(
                trips_df.Purpose == purpose, 
                trips_df.End == end)
            purp_df = trips_df[fltr]
            # Join to blocks by TAZ ID
            merge_df = block_df.merge(purp_df, how="left", left_on=taz_level,
                                      right_on=taz_level, 
                                      suffixes=("", "_z"))
            merge_df.fillna(0.0, inplace=True)
            merge_df.set_index([block_level, taz_level], inplace=True)
            # Broadcast for application
            block_trips_sel = block_prop_array.take(
                Purpose=purpose, End=end, squeeze=True)
            taz_trip_cast = lba.broadcast1dByAxis(
                block_trips_sel, block_axis.name, merge_df.trips)
            # Divide to proportion trip propensities
            block_trips_sel.data *= taz_trip_cast.data
            # Write back to block_prop_array
            block_prop_array.put(
                block_trips_sel.data, Purpose=purpose, End=end
                )


def relabel(row, column, value_dict):
    """
    Supporting funciton to replace values based on a dictionary lookup.
    Called from DataFrame.apply().

    Parameters
    ----------
    row: Iterable
        A row in a pandas DataFrame
    column: String
        The column to relabel
    value_dict: {String: String, ...}
        A dictionary with keys reflecting original value at this row/column
        location and values reflecting the value to write to this location.
        If the original value is not in the dictionary, the original value
        remains in place.
    """
    value = row[column]
    return value_dict.get(value, value)


def prepSeed(seed_df):
    """
    Convert seed data to a long form data frame. Use when reading in household
    cross-classification seeds. Transforms the seed from its original format
    to a long format compatible with later components of the trip-generation
    process.

    Parameters
    -----------
    seed_df: pd.DataFrame

    Returns
    --------
    long_seed: pd.DataFrame
    """
    #reduce number of HH size categories to match TAZ data
    seed_df[seed_df.HHSize > 4] = 4
    
    #relable HHSizes    
    size_dict = {1:"HHSize1", 2:"HHSize2", 3:"HHSize3", 4:"HHSize4p"}
    seed_df["Size"] = seed_df.apply(
        lambda r: relabel(r, "HHSize", size_dict), axis=1)
    
    #relable incomes
    inc_dict = {1:"Income1", 2:"Income2", 3:"Income3", 4:"Income4"}
    seed_df["Income"] = seed_df.apply(
        lambda r: relabel(r, "HHInc", inc_dict), axis=1)
    
    #summarize by new fields and PUMA ID
    group_vars = ["PUMA_ID", "Size", "Income"]
    worker_vars = ["0Worker", "1Worker", "2Worker", "3pWorker"]
    sum_df = seed_df.groupby(group_vars).sum().reset_index()
    
    #melt the resulting table to have a long form
    long_seed = sum_df.melt(id_vars=group_vars, value_vars=worker_vars,
                           var_name = "Workers", value_name="HH")
    
    #relable worker values
    wrk_dict = dict(zip(worker_vars, ["worker0", "worker1", "worker2", 
                                      "worker3p"]))
    long_seed["Workers"] = long_seed.apply(
        lambda r: relabel(r, "Workers", wrk_dict), axis=1)
    
    return long_seed


def getSeedByTAZ(taz_id, lookup_df, seed_df, veh_df):
    """
    Lookup seed matrices based on the PUMA ID associated with each TAZ. Add
    the vehicle ownership dimension (not included in seed source) from values
    provided in a separate data frame.

    Parameters
    -----------
    taz_id: Numeric
        The taz for which to fetch the appropriate HH cross-classification 
        seed.
    lookup_df: pd.DataFrame
        A data frame in which taz's are related to PUMA id's so that the
        appropriate seed can be pulled for the provided `taz_id`.
    seed_df: pd.DataFrame
        A data frame with seed details by PUMA.
    veh_df: pd.DataFrame
        A simple data frame with assumed seed values by vehicle ownership
        categories.

    Returns
    --------
    seed_vo: pd.DataFrame
        The seed details for this taz (based on its PUMA location), with
        vehicle ownership seed estimated from regional averages.

    See Also
    --------
    fetchSeedArray
    """
    # lookup the seed for the given TAZ
    PUMA = lookup_df["PUMA_ID"].values[lookup_df["TAZ_ID"].values == taz_id][0]
    seed = seed_df[seed_df["PUMA_ID"] == PUMA]
    
    # create a cartesion product of all HH type labels
    names=["Size", "Income", "Workers", "Vehicles"]
    mi = pd.MultiIndex.from_product([seed["Size"].unique(), 
                                     seed["Income"].unique(), 
                                     seed["Workers"].unique(), 
                                     veh_df["Vehicles"]])
    labels = mi.to_frame(index=False, name=names)
    
    # join the seed and vehicles data
    seed_vo = labels.merge(seed, on=["Size", "Income", "Workers"])
    seed_vo = seed_vo.merge(veh_df, on="Vehicles")
    # calculate estimate HHs in all categories
    seed_vo["HH"] *= seed_vo["VehRate"]   
    
    return seed_vo


def fetchSeedArray(taz_id, lookup_df, long_seed, veh_df):
    """
    Convert a seed matrix to a labeled array. The seed matrix is first
    looked up based on TAZ ID and the lookup data frame.

    Parameters
    -----------
    taz_id: Numeric
        The TAZ for which to fetch the seed data.
    lookup_df: pd.DataFrame
        A data frame relating TAZs to PUMA ID's.
    long_seed: pd.DataFrame
        A long-form data frame with IPF seed values for a PUMA
    veh_df: pd.DataFrame
        A data frame with generalize vehicle ownership seed shares.

    Returns
    -------
    seed_array: LbArray

    See Also
    --------
    getSeedByTAZ
    """
    sd = getSeedByTAZ(taz_id, lookup_df, long_seed, veh_df)
    seed_array = lba.dfToLabeledArray(
        sd, ["Size", "Income", "Workers", "Vehicles"], "HH", dtype=np.float)
    return seed_array



