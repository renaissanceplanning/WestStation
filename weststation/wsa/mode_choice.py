import emma
from emma import lba
import numpy as np
import pandas as pd


# FUNCTIONS
def genMCContainers(template, ref_array, zone_dim, out_hdf, node_path, name,
                    mode_axis="Mode", desc=None, copy_data=True, 
                    init_val=1.0):
    """
    Generate an on-disk labeled array to hold mode choice estimation outputs.

    Parameters
    -----------
    template: Impression
        A labeled array impression defining the dimensions of the output
        container. These dimensions will be extended to have zonal dimension
        matching that of `ref_array`.
    ref_array: LbArray
        A labeled array with a zone dimension that is used to augment the 
        dimensionality of `template` when creating the output array.
    zone_dim: String
        The axis name in `ref_array` that corresponds to geographic zones.
    out_hdf: String
        The location where the mode choice labeled array container will be
        stored on disk.
    node_path: String
        The node in `out_hdf` where the container will be stored.
    name: String
        The name of the output container within `node_path`.
    mode_axis: String, default="Mode"
        The name of the axis that corresponds to travel modes in `template`.
    desc: String, default=None
        A brief description of the contents of the output array.
    copy_data: Boolean, default=True
        If True, the data in `ref_array` are copied into the output container.
        If False, the output container is returned with a specified `init_val`.
    init_val: Numeric, default=1.0
        If `copy_data` is False, the output container has all cells set to
        this value.
    
    Returns
    -------
    LbArray
    """   
    # Create template, casting into the zonal dimension of the in_file
    mc_template = template.cast(ref_array.getAxisByName(zone_dim))
       
    # Cast data into the mode dimension
    mode_axis_ = template.getAxisByName(mode_axis)
    ref_array = ref_array.cast(mode_axis_, copy_data=copy_data)
        
    # Assure proper axis and label alignment
    ref_array = lba.alignAxes(ref_array, mc_template)
    ref_array = lba.alignAxisLabels(ref_array, ref_array.axes, mc_template.axes)
    
    # Save to h5 file
    if copy_data:
        return lba.LbArray(ref_array.data, ref_array.axes, desc=desc,
                           hdf_store=out_hdf, node_path=node_path, name=name,
                           overwrite=True)
    else:
        return lba.LbArray(init_val, ref_array.axes, desc=desc,
                           hdf_store=out_hdf, node_path=node_path, name=name,
                           overwrite=True)


def loadWalkTimeToTransit(block_taz_df, block_axis, taz_axis, prem_base,
                          non_prem_base, prem_scen, non_prem_scen,
                          block_level=None, taz_level=None,
                          block_id_field="BLOCKID", time_field="Total_Minutes",
                          df_block_col="BLOCK_ID", df_taz_col="TAZ"):
    """
    Several mode choice models use inverted walk access to transit times
    (calculated as 1/walk_minutes), so these vectors are needed to estimate
    mode selection. This function assembles these estimates for all zones.

    Starting with base condition walk access to transit times from across the
    state, patch in scenario-specific walk access times within the window
    area. Generalize the block-level times to the TAZ level. The "patching"
    process allows the skim development for the analysis scenario to focus
    just on the window area.
    
    Parameters
    -----------
    block_taz_df: pd.DataFrame
        A data frame listing each block and its parent taz in separate columns
    block_axis: LbAxis
        An axis object by which block-level walk time to transit estimates will
        be indexed (ensures proper ordering for broadcasting purposes)
    taz_axis: LbAxis
        An axis object by which taz-level walk time to transit estimates will
        be indexed
    prem_base: String
        Path to base year estimates (csv file) of walk time to premium transit
        (T, commuter rail)
    non_prem_base: String
        Path to base year estimates of walk time to non-premium transit (bus)
    prem_scen: String
        Path to scenario estimates of walk time to premium transit
    non_prem_scen: String
        Path to scenario estimates of walk time to non-premium transit
    block_id_field: String, default="BLOCKID"
        The field identifying each block feature (must be the same in all csv
        files)
    time_field: String, default="Total_Minutes"
        The field containing the minimum travel time from each block to a
        transit stop or station (must be the same in all csv files)
    df_block_col: String, default="BLOCK_ID"
        The column in `block_taz_df` that identifies each block
    df_taz_col: String, default="TAZ"
        The column in `block_taz_df` that identifies each block's parent TAZ.
    block_level: String, default=None
        If `block_axis` uses a MultiIndex, specify the level name that
        identifies each unique block.
    taz_level: String, default=None
        If `taz_axis` uses a MultiIndex, specify the level name that
        identifies each unique TAZ.

    Returns
    --------
    inv_walk_times: dict
        This function returns four vectors in a dictionary:
            - Inverted walk access times to premium transit (T, Commuter Rail)
            
                - Indexed to a specified block axis

                - Summaried and indexed to a specified taz axis
            
            - Inverted walk access times to non-premium transit (bus stops)
            
                - Indexed to a specified block axis
            
                - Summaried and indexed to a specified taz axis
    """
    usecols=[block_id_field, time_field]
    dt={block_id_field: str}
    suffix = "_patch"
    
    # Fetch the full kit of walk access times from the base config
    prem_all = pd.read_csv(prem_base, usecols=usecols, dtype=dt)
    non_prem_all = pd.read_csv(non_prem_base, usecols=usecols, dtype=dt)
    
    # Fetch the patch of walk acccess times
    prem_patch = pd.read_csv(prem_scen, usecols=usecols, dtype=dt)
    non_prem_patch = pd.read_csv(non_prem_scen, usecols=usecols, dtype=dt)
    
    # Patch the scores
    prem_merge = prem_all.merge(prem_patch, how="outer", on=block_id_field,
                                suffixes=["", suffix])
    non_prem_merge = non_prem_all.merge(non_prem_patch, how="outer", 
                                        on=block_id_field, suffixes=["", suffix])
    
    patch_col = f"{time_field}{suffix}"
    prem_mask = pd.notna(prem_merge[patch_col])
    non_prem_mask = pd.notna(non_prem_merge[patch_col])
    
    prem_merge[time_field][prem_mask] = prem_merge[patch_col][prem_mask]
    non_prem_merge[time_field][non_prem_mask] = \
        non_prem_merge[patch_col][non_prem_mask]
    
    # Invert times
    prem_merge[time_field] = np.select([prem_merge[time_field] < 1],
                                       [1.0],
                                       1/prem_merge[time_field])
    non_prem_merge[time_field] = np.select([non_prem_merge[time_field] < 1],
                                           [1.0], 
                                           1/non_prem_merge[time_field])
    
    # Average times to TAZ level
    prem_merge = prem_merge.merge(block_taz_df, how="inner",
                                  left_on=block_id_field,
                                  right_on=df_block_col)
    non_prem_merge = non_prem_merge.merge(block_taz_df, how="inner",
                                          left_on=block_id_field,
                                          right_on=df_block_col)
    keep_cols = [df_taz_col, time_field]
    prem_taz = prem_merge[keep_cols].groupby(df_taz_col).mean()
    non_prem_taz = non_prem_merge[keep_cols].groupby(df_taz_col).mean()
    
    # Reindex results
    if isinstance(block_axis.labels, pd.MultiIndex):
        prem_block = prem_merge.set_index(block_id_field).reindex(
            block_axis.labels.get_level_values(block_level))
        non_prem_block = non_prem_merge.set_index(block_id_field).reindex(
            block_axis.labels.get_level_values(block_level))
    else:
        prem_block = prem_merge.set_index(block_id_field).reindex(
            block_axis.labels)
        non_prem_block = non_prem_merge.set_index(block_id_field).reindex(
            block_axis.labels)
        
    if isinstance(taz_axis.labels, pd.MultiIndex):
        # TAZ column should already be index based on group by above
        #  so the only need is to reindex
        prem_taz = prem_taz.reindex(
            taz_axis.labels.get_level_values(taz_level))
        non_prem_taz = non_prem_taz.reindex(
            taz_axis.labels.get_level_values(taz_level))
    else:
        prem_taz = prem_taz.reindex(taz_axis.labels)
        non_prem_taz = non_prem_taz.reindex(taz_axis.labels) 
        
    # Get rid of any nulls
    for df in [prem_block, non_prem_block, prem_taz, non_prem_taz]:
        df.fillna(0.0, inplace=True)
    
    return {
        "prem": {
            "block": prem_block[time_field], 
            "taz": prem_taz[time_field]
                },
        "non_prem": {
            "block": non_prem_block[time_field],
            "taz": non_prem_taz[time_field]
            }
        }


def fetchAccessScores(scen, mode, purpose, direction, activity, index_cols,
                      suffix="", match_axis=None, match_level="TAZ",
                      imped_tag=""):
    """
    Pull acess scores for this scenario based on mode, purpose, etc. from
    csv output files.
    
    Parameters
    -----------
    scen: String
    mode: String
    purpose: String
    activity: String
        The name of the activity in column headings
    direction: String ("from", "to")
        If "from", use `access_from_hh...` access scores.
        If "to", use `access_to_jobs...` access scores.
    index_cols: String or [String,...]
        Columns in the access score csv to use as a new data frame index
        (this will generally be the zone id field). Scores are sorted to 
        match the order of index values in the `match_axis` parameter, 
        if given.
    suffix: String , default=""
        Access score outputs follow a consistent naming convention. File names
        for select outputs include a suffix (TAZ-level walk/bike scores, e.g.),
        whch can be specified here.
    match_axis: LbAxis, default=None
        If scores will be used in labeled_array applications, specify the
        axis they are associated with to ensure appropriate ordering of 
        access score values.
    match_level: String, default="TAZ"
        If `match_axis` uses a MultiIndex, specify the level on which to sort
        access scores.
    imped_tag: String, default=""
        If the activty column heading has been embellished due to the scores
        being generated using multiple axis references (different impedances,
        e.g.), specify the suffix attached to the column heading to be read.
        
    Returns
    -------
    pd.DataFrame
    """
    if direction == "from":
        access_f = r"scen\{}\access_from_hh_{}_{}{}.csv".format(
            scen, mode, purpose, suffix)
    elif direction == "to":
        access_f = r"scen\{}\access_to_jobs_{}_{}{}.csv".format(
            scen, mode, purpose, suffix)
    
    heading = f"{activity}-Purpose-{purpose}{imped_tag}"
    cols = index_cols + [heading]
    
    df = pd.read_csv(access_f, usecols=cols, index_col=index_cols)
    df.rename({heading: activity}, axis=1, inplace=True)
    
    if match_axis is None:
        return df
    else:
        if isinstance(match_axis.labels, pd.MultiIndex):
            return df.reindex(match_axis.labels.get_level_values(match_level))
        else:
            return df.reindex(match_axis.labels)


def inheritAccessScore(taz_scores, block_axis, taz_level="TAZ", 
                       block_level="block_id"):
    """
    A simple function for succinctly merging data frames so that taz-level
    access scores are applied to their constituent block features.
    
    TAZ scores are assumed to have been obtained using `fetchAccessScores`
    above. Therefore, the index of the `taz_scores` object is used in the 
    joining process. 
    
    The `block_axis` is assumed to be a MultiIndex object with one level
    identifying blocks and the other identifying TAZs.
    
    Parameters
    -------------
    taz_scores: DataFrame
    block_axis: LbAxis
    taz_level: String, default="TAZ"
    block_level: String, default="block_id"  
    
    Returns
    -------
    DataFrame
    
    See Also
    ---------
    fetchAccessScores
    """
    idx_df = block_axis.to_frame()[[block_level, taz_level]]  
    taz_df = taz_scores.reset_index(level=taz_level)
    df = idx_df.merge(taz_df, how="left", left_on=taz_level,
                      right_on=taz_level)
    df.fillna(0.0, inplace=True)
    df.set_index([block_level, taz_level], inplace=True)
    return df.reindex(block_axis.labels)


def applyModel(to_array, model_dict, intercept=0.0, logger=None):
    """
    Given an input array and a dictionary that relates dimensional axes to
    arrays of values and their coefficients, apply the model to calculate
    mode shares.

    Model application always follows the same pattern: 
        1. An array of the same shape and size as `to_array` is initialized
        with all values set to `intercept`. This is the embryonic
        `model_array`.

        2. For each axis in `model_dict`, values are multiplied by the
        appropriate coefficient and broadcast into the shape and size of
        the `model_array`

        3. Broadcasted values are accumulated in the `model_array` as a
        general linear expression.

        4. Shares are calcualted logistically, first by exponentiating the
        values in `model_array`. The resulting array's values are then divided
        by themselves plus 1.0 to yield mode share estimates.
    
    Parameters
    -----------
    to_array: LbArray
        An array containing mode choice model inputs in various axes
        (HHSize, Income axes, e.g.)
    model_dict: dict
        A nested dictionary whose primary keys refer to axis names in
        `to_array`. Subordinate keys are "values" and "coef". Subordinate
        values are one or more lists whose lengths equal that of the
        axis and a corresponding number of coefficients (floats).

        '
        {
            "DimA": {
                "values": value_set,
                "coef": 1.0
                },
            "DimB": {
                "values": [value_set1, value_set2],
                "coef": [-1.0, 2.5]
            ...
            }
        '

        For each dimension, multiple vectors and coefficients may be passed.
        If `values` is a simple list, it implies multiple vectors which are
        zipped together with coefficient values. If the `coef` value
        corresponding to a given vector is a list, it implies the values are
        to be treated as factors (each value is a category with a separate 
        coefficient)
    
    intercept: numeric, default=0.0
        The intercept value for the regression formula
    
    logger: Logger, default=None
        A logging object to record process information.
    
    Returns
    -------
    model_array: LbArray
        A collection of regression outputs. Model array contains mode share
        estimates for a specific binomial nest. 

    See Also
    ---------
    pushSharesToMCArray
    modeChoiceApply
    """
    # HANDLE FACTORS
    # Make temp array based on model dict
    axes = []
    order = []
    for k in model_dict.keys():
        x = to_array.getAxisByName(k)
        ai = to_array.axes.index(x)
        axes.append(k)
        order.append(ai)
    axes = np.array(axes)[np.argsort(order)]
    
    # Create temp array with intercept
    model_array = to_array.impress(fill_with=intercept)
    if np.any(np.isnan(model_array.data)):
        print(f"Found nan values for intercept")
        if logger is not None:
            logger.info(f"Found nan values for intercept")

    # Apply dimensions
    for axis in axes:
        x_dict = model_dict[axis]
        values = x_dict["values"]
        coef = x_dict["coef"]
        
        if type(values) is list:
            # There are multiple vectors for this dimension
            for v, c in zip(values, coef):
                if isinstance(v, lba.LbArray):
                    model_array.data += values.data * c
                # Apply coef - if coef is list, factors are applied in order
                model_array.data += lba.broadcast1dByAxis(
                    model_array, axis, v * c).data
                
        elif isinstance(values, lba.LbArray):
            model_array.data += values.data * coef
        else:
            model_array.data += lba.broadcast1dByAxis(
                model_array, axis, values * coef).data
        
        if np.any(np.isnan(model_array.data)):
            print(f"Found nan values on axis: {axis}")
            if logger is not None:
                logger.info(f"Found nan values on axis: {axis}")
            
    # Apply exp formula
    model_array.data = np.exp(model_array.data)/(1 + np.exp(model_array.data))
    return model_array


def pushSharesToMCArray(mc_array, model_array, this_mode, mode_dict,
                        logger=None, **kwargs):
    """
    Having estimated mode share estimates for a specific binomial choice,
    update and store mode share estimates in a generalized container for all
    mode choices. Update submodes as you go.

    Parameters
    -----------
    mc_array: LbArray
        A container for holding mode choice estimates for all modes.
    model_array: LbArray
        The estimates for a specificy binomial mode choice (generated using
        `applyModel`)
    this_mode: String
        The mode currently being estimated and updated.
    mode_dict: dict
        A dictionary that defines mode nesting. This allows mode choice
        estimates for general modes to automatically cascade into their
        submodes.
    logger: Logger, default=None
    kwargs:
        Keyword arguments specifying where to place values from `model_array` in
        `mc_array`

    Returns
    --------
    None
        `mc_array` is modified inplace

    See Also
    ---------
    applyModel
    modeChoiceApply
    """
    _modes = mode_dict.get(this_mode, None)
    if _modes is None:
        # Just update this_mode
        if logger is not None:
            logger.info(f"-- -- -- Adjusting shares for {this_mode} mode")
        mc_array.put(
            mc_array.take(
                squeeze=True, Mode=this_mode, **kwargs).data * model_array.data,
            Mode=this_mode, **kwargs)
    else:
        # Update estimates for child modes
        for _mode in _modes:
            pushSharesToMCArray(mc_array, model_array, _mode, mode_dict,
                                logger=logger, **kwargs)


def modeChoiceApply(array, intercept, model_dict, this_mode, complement,
                    mode_dict, crit, logger=None):
    """
    A convenience function that makes calls to both `applyModel` and
    `pushSharesToMCArray` in a standardized fashion.

    Parameters
    -----------
    array: LbArray
        A labeled array in which all mode share estimates are stored.
    intercept: Numeric
        The model intercept.
    model_dict: dict
        A dictionary of model parameters (see `applyModel`)
    this_mode: String
        The current mode to be estimated. Mode choice model application
        estimates the share of trips made by this mode.
    complement: String
        The complement of `this_mode`. One minus the mode share estimate
        for `this_mode` is the `complement` mode's share.
    mode_dict: dict
        A dictionary that defines mode nesting. This allows mode choice 
        estimates for general modes to automatically cascade into their
        submodes.
    crit: dict
        A dictionary that specifies which axes and labels in `array` to
        focus on for mode choice model application.
    logger: Logger, default=None

    Returns
    --------
    None
        `array` is updated in place

    See Also
    ---------
    applyModel
    pushSharesToMCArray
    """
    model_array = array.take(squeeze=True, **crit)
    model_array = applyModel(model_array, model_dict,
                             intercept=intercept, logger=logger)

    logger.info("  -- -- |Simple model results by decile: {}".format(
        np.quantile(model_array.data, np.linspace(0.1, 1, 10))))
    
    # Store the results
    del crit["Mode"]
    pushSharesToMCArray(array, model_array, this_mode, mode_dict,
                        logger=logger, **crit)
    model_array.data = 1 - model_array.data
    pushSharesToMCArray(array, model_array, complement, mode_dict,
                        logger=logger, **crit)


def mcInfo(logger, trips_taz_disk, trips_block_disk, mc_taz_disk, 
           mc_block_disk):
    """
    A convenience function for logging mode choice and trips by mode
    estimates.

    Parameters
    ----------
    logger: Logger
    trips_taz_disk: LbArray
    trips_block_dis: LbArray
    mc_taz_disk: LbArray
    mc_block_dis: LbArray
    """
    logger.info("TAZ trips by purpose and end:\n{}".format(
        str(trips_taz_disk.sum(["Purpose", "End"]))))
    logger.info("Block trips by purpose and end:\n{}".format(
        str(trips_block_disk.sum(["Purpose", "End"]))))
    logger.info("TAZ choice sums by purpose and end:\n{}".format(
        str(mc_taz_disk.sum(["Purpose", "End"]))))
    logger.info("Block choice sums by purpose and end:\n{}".format(
        str(mc_block_disk.sum(["Purpose", "End"]))))
    
def reportTripsByMode(trips_taz, trips_block, out_csv, taz_level="TAZ",
                      block_axis_name="block_id", block_level="block_id",
                      sum_cols=["Purpose", "End", "Mode"],
                      levels=["TAZ", "INWINDOW", "INFOCUS" ]):
    """
    ...

    Parameters
    ----------
    trips_taz : TYPE
        DESCRIPTION.
    trips_block : TYPE
        DESCRIPTION.
    out_csv : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print("... patching block-level trip estimates in window to TAZs")
    # TAZ summary
    _sum_cols = [taz_level] + sum_cols
    taz_sum = trips_taz.sum(_sum_cols)
    # Block summary
    _sum_cols = [block_level] + sum_cols
    block_sum = trips_block.sum(_sum_cols)
    # Dissolve blocks to TAZ level
    block_diss = block_sum.dissolve(block_axis_name, taz_level, np.sum)
    block_diss[block_axis_name].labels.name = taz_level
    
    # Dump to data frames
    taz_df = taz_sum.to_frame("trips").reset_index()
    taz_df[levels] = pd.DataFrame(
        taz_df[taz_level].to_list(), index=taz_df.index)
    window_df = block_diss.to_frame("trips").reset_index()
    
    # Merge
    taz_df_m = taz_df.merge(window_df, how="left", on=taz_level)
    print("... writing output")
    