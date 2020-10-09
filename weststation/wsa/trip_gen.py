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
    
    # Update axes
    #  - levels
    level_df = tg_df.groupby(levels).size().reset_index()[levels]
    act_array[block_id].addLevel(
        level_df, left_on="index", right_on=block_id)
    #  - alignment
    for align_axis in align_axes:
        lba.alignAxisLabels(act_array, align_axis.name, align_axis,
                            inplace=True)
    return act_array


def _disagFromPropensityArray(taz_trips, block_trips, prop_array, act_dims,
                              purposes, end, block_axis, block_id,
                              block_taz_id, taz_axis, taz_id, logger,
                              **axis_crit):
    for p in purposes:
        if np.any(prop_array.data[:] == 0):
            print(f"FOUND BLOCKS WITH NO PROPENSITY (purpose: {p})")
            if logger is not None:
                logger.info(f"FOUND BLOCKS WITH NO PROPENSITY (purpose: {p})")
    axis_crit["End"] = end
    # STEP 2 - TOTAL BLOCK TRIP PROPENSITY
    # Summarize each block's total trip propensity using the .sum method.
    print('...step 2: sum total block propensity')
    block_prop = prop_array.sum(block_axis)
    
    # Dissolve the block-level sums based on TAZ ID to get each zone's total
    #  propensity
    if prop_array.ndim == 1:
        taz_prop_df = prop_array.to_frame.reset_index()
        taz_prop_df = taz_prop_df.reset_index()
        levels = prop_array[block_axis].levels
        taz_prop_df[levels] = pd.DataFrame(
            taz_prop_df[block_axis].tolist(), index=taz_prop_df.index)
        taz_prop_sum = taz_prop_df.groupby(block_taz_id).sum().reset_index()
        taz_prop = lba.dfToLabeledArray(
            taz_prop_sum, ["TAZ"], ["prop"], fill_value=0)
    else:
        taz_prop = prop_array.dissolve(
            block_axis, block_taz_id, np.sum).sum(block_axis)
    
    # STEP 3 - NORMALIZE ACTIVITY-BASED PROPENSITIES
    # For each block, normalize its activity-based propensities as shares of
    #  its total propensity
    print('...step 3: normalize propensities across block activities')
    prop_array.data /= lba.broadcast1dByAxis(prop_array, block_id,
                                       block_prop.data).data
    
    # Push these shares to the output container, casting into the specified
    #  purposes. Output array values are modified in-place downstream to
    #  reflect actual trips rather than just normalized propensity.
    for p in purposes:
        block_trips.put(prop_array.data, Purpose=p, **axis_crit)
    if logger is not None:
        logger.info("\nNormalized block propensities "
                    "(should sum to n_block * n_purposes): "
                    f"{block_trips.take(Purpose=purposes).data[:].sum()}")
    # STEP 4 - BLOCK SHARE OF TAZ TRIP PROPENSITY
    # Calculate each block's share of its TAZ's trips
    #  For each block with a propensity sum...
    #   - get the sum
    #   - grab the block data associated with the TAZ
    #   - and modify the data by dividing the block propensities
    #      by the TAZ total
    print('...step 4: calculate block share of TAZ trips (by activity)')
    for taz in taz_prop[block_axis].labels:
        total_prop = taz_prop.take(**{block_axis:taz}).data[0]
        crit = {block_axis: {block_taz_id: taz}}
        block_prop.put(
            block_prop.take(**crit).data/total_prop,
            **crit)
    # Multiply the block share of propensity by the activity-based normalized
    #  propensities. In this way, each block's activity-specific cell contains
    #  a number defining it's share of total TAZ productions.
    for p in purposes:
        purp_chunk = block_trips.take(Purpose=p, **axis_crit)
        block_trips.put(
            purp_chunk.data *
            lba.broadcast1dByAxis(
                purp_chunk, block_axis, block_prop.data).data,
            Purpose=p, 
            **axis_crit
            )
    if logger is not None:
        logger.info("\nProportioned block propensities "
                    "(should sum to n_tazs * n_purposes): "
                    f"{block_trips.take(Purpose=purposes).data[:].sum()}")
    #STEP 5 - ESTIMATE DETAILED TRIPS BY BLOCK AND ACTIVITY
    # Multiply TAZ-level trip estimates by block-level, activity-specific
    #  propensities
    print('...step 5: apply taz trip totals at block level')
    # Make an in-memory array for faster processing
    _block_trips_ = block_trips.copy()

    window_zones = prop_array[block_axis].labels.get_level_values(
        block_taz_id).unique()
    purp_alloc=dict(zip(purposes, [0 for _ in purposes]))
    for wz in window_zones:
        axis_crit[taz_axis] = {taz_id: wz}
        # Get trip total for each zone for each purpose
        taz_trips_by_purp = taz_trips.take(**axis_crit).sum(["Purpose"])
        # Multiply trips by purpose across relevant block features
        for p in purposes:
            # Trips
            trips = taz_trips_by_purp.take(Purpose=p).data[0]
            # Apply
            block_crit = {"Purpose": p, "End": end, 
                          block_axis: {block_taz_id: wz}}
            _block_trips_.put(
                _block_trips_.take(**block_crit).data * trips,
                **block_crit)
            purp_alloc[p] += float(trips)
    del axis_crit[taz_axis]
    block_trips.put(
        _block_trips_.take(Purpose=purposes, squeeze=True, **axis_crit).data,
        Purpose=purposes, **axis_crit
    )
    print(f"Allocated trips for {len(window_zones)} TAZs.")
    print(f"Amounts allocated:\n{yaml.dump(purp_alloc)}")
    if logger is not None:
        logger.info(f"\nAllocated trips for {len(window_zones)} TAZs.")
        logger.info(f"\nAmounts allocated:\n{yaml.dump(purp_alloc)}")


def disaggregateTrips_hb(taz_trips, block_trips, act_array, act_dims,
                         purposes, end, propensities, base_prop,
                         block_axis="block_id", block_id="block_id",
                         block_taz_id="TAZ", taz_axis="TAZ", taz_id="TAZ",
                         logger=None):
    """
    A function to disaggregate TAZ-level home-based trip estimates to block
    level based on activities in each block and trip-making propensity
    factors defined by activity dimension.

    Parameters
    -----------
    taz_trips: LbArray
        A labeled array of trips by TAZ. Expected axes include 'Purpose',
        'End', and one or more activity dimensions.
    block_trips: LbArray
        A labeled array that will contain trips by block (expected to be
        passed to the function with initial values of 0.0). Expected axes
        match those in `taz_trips`
    act_array: LbArray
        A labeled array of activities by block. Expected axes include
        activity dimensions that match those in `taz_trips`. WARNING!
        This array is modified in-place during processing.
    act_dims: [String,...]
        The name of the activity dimensions used in each labeled array.
    purposes: [String,...]
        The purposes for which trip disaggregation is conducted.
    end: String
        The trip end ('P' or 'A') for which trip disaggregation is conducted.
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
    block_axis: String, default="block_id"
        The name of the axis in `block_trips` and `act_array` that identifies
        block features.
    block_id: String, default="block_id"
        The level in `block_axis` that uniquely identifies each block.
    block_taz_id: String
        The level in `block_axis` that identifies the TAZ each block is
        nested in.
    taz_axis: String, default="TAZ"
        The axis in `taz_trips` that identifies TAZ features.
    taz_id: String, default="TAZ"
        The level in `taz_axis` that uniquely identifies each TAZ.
    logger: Logger, default=None
        An initialized logger object may be provided to log processing steps.
    
    Returns
    --------
    None 
        `block_trips` is modified in place.
    
    See Also
    --------
    disaggregateTrips_nh
    """
    # STEP 1 - ACTIVITY-BASED TRIP PROPENSITY
    # Apply trip propensities by activity dimensions to each block.
    #  This is done by broadcasting the propensity values to match the
    #  dimensions of `act_array` and updating the data through multiplication
    print('...step 1: calculate activty-based trip propensities')
    for dim in act_dims:
        for purpose in purposes:
            prop_dict = propensities[purpose][dim]
            prop_rates = []
            for label in act_array[dim].labels:
                prop_rates.append(prop_dict[label])
            act_array.data *= lba.broadcast1dByAxis(
                act_array, dim, prop_rates).data
    # Bump each propensity value up by the `BASE_PROP` constant
    act_array.data += base_prop
    axis_crit = dict([(x.name, list(x.labels)) for x in act_array.axes
                    if x.name in act_dims])
    _disagFromPropensityArray(taz_trips, block_trips, act_array, act_dims,
                              purposes, end, block_axis, block_id,
                              block_taz_id, taz_axis, taz_id, logger,
                              **axis_crit)

def disaggregateTrips_nh(taz_trips, block_trips, nh_df, lbl_col, act_col,
                         purposes, end, propensities, base_prop, wb_df,
                         block_axis="block_id", block_id="block_id",
                         block_taz_id="TAZ", taz_axis="TAZ", taz_id="TAZ",
                         wb_block_id="block_id", act_crit="-",
                         levels=["block_id", "TAZ", "INFOCUS", "INWINDOW"],
                         act_dims=["HHSize", "Income", "VehOwn", "Workers"],
                         logger=None):
    """
    A function to disaggregate TAZ-level non-home trip estimates to block
    level based on activities in each block and trip-making propensity
    factors defined by activity label.

    Parameters
    -----------
    taz_trips: LbArray
        A labeled array of trips by TAZ. Expected axes include 'Purpose',
        'End', and one or more activity dimensions.
    block_trips: LbArray
        A labeled array that will contain trips by block (expected to be
        passed to the function with initial values of 0.0). Expected axes
        match those in `taz_trips`
    nh_df: pd.DataFrame
        A data frame of non-home activities by block.
    lbl_col: String
        The column in `nh_df` that identifies each block's activity by type.
    act_col: String
        The column in `nh_df` that stores the number of each activity.
    act_dims: [String,...]
        The name of the activity dimensions used in each labeled array.
    purposes: [String,...]
        The purposes for which trip disaggregation is conducted.
    end: String
        The trip end ('P' or 'A') for which trip disaggregation is conducted.
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
    block_axis: String, default="block_id"
        The name of the axis in `block_trips` and `act_array` that identifies
        block features.
    block_id: String, default="block_id"
        The level in `block_axis` that uniquely identifies each block.
    block_taz_id: String
        The level in `block_axis` that identifies the TAZ each block is
        nested in.
    taz_axis: String, default="TAZ"
        The axis in `taz_trips` that identifies TAZ features.
    taz_id: String, default="TAZ"
        The level in `taz_axis` that uniquely identifies each TAZ.
    logger: Logger, default=None
        An initialized logger object may be provided to log processing steps.
    
    Returns
    --------
    None 
        `block_trips` is modified in place.

    See Also
    --------
    disaggregateTrips_hb
    """
    # STEP 1 - ACTIVITY-BASED TRIP PROPENSITY
    # Apply trip propensities by destination activity type to each block.
    #  This is done by multiplying purpose-specific propensity rates by
    #  the activity types in non-home activity data frame.

    # Since the ultimate application is to tabulate all trip attractions simply
    #  as 'non-home' activities ('-' is the axis label for activity dimensions),
    #  this work can be done quickly in a data frame to create block trip
    #  propensity vectors that can be converted to a labeled array for 
    #  subsquent steps.
    print('...step 1: calculate activty-based trip propensities')
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
    
    # Disag by purpose
    axis_crit = dict([(x.name, act_crit) for x in block_trips.axes
                      if x.name in act_dims])
    for purpose in purposes:
        prop_array = nh_array.take(Purpose=purpose, squeeze=False)
        _disagFromPropensityArray(taz_trips, block_trips, prop_array, [],
                                  [purpose], end, block_axis, block_id,
                                  block_taz_id, taz_axis, taz_id, logger,
                                  **axis_crit)


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



