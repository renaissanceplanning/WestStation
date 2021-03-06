import emma
from emma import lba, ipf
import numpy as np
import pandas as pd
import os

def applyKFactors(trip_table, factors, mode_axis="Mode", **crit):
    """
    Expand or contract trips in each mode based on factors that would
    bring modal estimates in closer alignment with an authoritative validation
    source (CTPS model, e.g.). Factors should not be scenario-specific but
    applied consistently across all scenarios.

    Parameters
    ----------
    trip_table : emma.od.Skim
        Trips in this Skim will be factored by mode.
    factors : array-like 1d
        Factor values to apply. The number and order of factors must
        correspond to the modes listed in the `trip_table`'s `mode_axis`.
    mode_axis : String, default="Mode"
        The name of the axis in `trip_table` that identifies each mode.
    **crit : keyword arguments
        Criteria for applying the factors only to specific portions of the
        `trip_table` (apply only to zone in the window area, e.g.)

    Returns
    -------
    factor_table : emma.od.Skim
    """
    print("Applying K factors")
    # Cast factors into appropriate dimensions
    factor_table = trip_table.take(squeeze=True, **crit)
    factors_cast = lba.broadcast1dByAxis(
        factor_table,
        mode_axis,
        factors)
    factor_table.data[:] *= factors_cast[:]
    
    return factor_table


def initTripTable(scen, purpose, period, logger=None):
    """
    Intialize a trip table for this scenario, purpose, and period. All trip
    interchanges are intialized to zero (0.0).

    Parameters
    ----------
    scen : String
    purpose : String
    period : String
    logger: Logger

    Returns
    -------
    taz_trip_table, block_trip_table:  (emma.od.Skim, emma.od.Skim)
    """
    print(f"Initializing trip tables -- {purpose}")
    
    # Connect to zonal trips by mode estimates
    trips_by_mode_taz_f =  r"scen\{}\Trips_by_taz.h5".format(scen)
    trips_by_mode_block_f = r"scen\{}\Trips_by_block.h5".format(scen)
    trips_by_mode_taz = lba.openLbArray_HDF(trips_by_mode_taz_f, "/trips")
    trips_by_mode_block = lba.openLbArray_HDF(trips_by_mode_block_f, "/trips")

    # Create empy Skim objects for distributions
    trip_table_taz_f = r"scen\{}\Trip_dist_taz_{}_{}.h5".format(
        scen, period, purpose)
    trip_table_block_f = r"scen\{}\Trip_dist_block_{}_{}.h5".format(
        scen, period, purpose)
    
    #-- Use zone axes for i, dims; mode and purpose axes as `axes_k`
    tazs = trips_by_mode_taz.TAZ
    blocks = trips_by_mode_block.block_id
    axes_k = [trips_by_mode_taz.Mode]
    
    if logger is not None:
        logger.info(f"""
            Initializing {purpose} trip tables:
                \n-- {trips_by_mode_taz_f}
                \n-- {trips_by_mode_block_f}""")
    
    trips_taz = emma.od.Skim(tazs, 0.0, axes_k, hdf_store=trip_table_taz_f,
                             node_path="/", name="dist", overwrite=True)
    trips_block = emma.od.Skim(blocks, 0.0, axes_k, 
                               hdf_store=trip_table_block_f, node_path="/",
                               name="dist", overwrite=True)
    
    logger.info("...trip tables initialized")
    
    return (trips_taz, trips_block)


def tripEndsByMode(scen, purpose, period_factor, taz_zone_dim="TAZ", 
                   block_zone_dim="block_id", mode_dim="Mode", 
                   purp_dim="Purpose", end_dim="End", block_taz_level="TAZ", 
                   logger=None):
    """
    Summarize trips by mode by zone for this purpose (factored into the 
    specified travel period) and return data frames of trips at TAZ and 
    block level for use in distribtion.
    
    TAZ-level trip estimates in the "window" area are patched in from the 
    block-level estimates, over-riding the original TAZ-level estimates for
    those zones.
    
    Parameters are mostly focused on identifying axis names for trips_by_mode
    labeled arrays.
    
    Parameters
    ----------
    scen : String
    purpose : String
    period_factor: numeric
        A factor for this period and purpose by which trip estimates are
        scaled (for all modes) from daily trips to period trips        
    taz_zone_dim : String, default="TAZ"
    block_zone_dim : String, default="block_id"
    mode_dim : String, default="Mode"
    purp_dim : String, default="Purpose"
    end_dim : String, default="End"
    block_taz_level : String, default = "TAZ"
    logger: Logger, default=None
    
    Returns
    -------
    trips_df_taz : pd.DataFrame
        For this scenario, purpose, and travel period, these are the estimates
        of trips by mode (productions and attractions listed separately) at
        the TAZ level
    trips_df_block : pd.DataFrame
        Same as above, with trips summarized at block level.
    """
    # print(f"Fetching trip productions and attractions -- {purpose}")
    # logger.info(f"Fetching trip productions and attractions -- {purpose}")
    # Connect to zonal trips by mode estimates
    trips_by_mode_taz_f = r"scen\{}\Trips_by_taz.h5".format(scen)
    trips_by_mode_block_f = r"scen\{}\Trips_by_block.h5".format(scen)
    trips_by_mode_taz = lba.openLbArray_HDF(trips_by_mode_taz_f, "/trips")
    trips_by_mode_block = lba.openLbArray_HDF(trips_by_mode_block_f, "/trips")
    
    # Summarize trips from trip table
    sum_dims = [mode_dim, end_dim]
    crit = {purp_dim: purpose}
    
    taz_trips = trips_by_mode_taz.take(squeeze=True, **crit).sum(
        [taz_zone_dim] + sum_dims)
    block_trips = trips_by_mode_block.take(squeeze=True, **crit).sum(
        [block_zone_dim] + sum_dims)
    
    # Factor by purpose
    taz_trips.data *= period_factor
    block_trips.data *= period_factor
    
    # Dump to data frame
    trips_df_taz = taz_trips.to_frame("trips").reset_index()
    if isinstance(taz_trips[taz_zone_dim].labels, pd.MultiIndex):
        levels = taz_trips[taz_zone_dim].levels
        trips_df_taz[levels] = pd.DataFrame(
            trips_df_taz[taz_zone_dim].to_list(), index=trips_df_taz.index)
    trips_df_block = block_trips.to_frame("trips").reset_index()
    trips_df_block[block_trips[block_zone_dim].levels] = pd.DataFrame(
        trips_df_block[block_zone_dim].tolist(), index=trips_df_block.index)
    
    # Patch blocks into tazs
    print(f"Patching window trips from block level to TAZ level -- {purpose}")
    if logger is not None:
        logger.info(
            f"Patching window trips from block level to TAZ level -- {purpose}")
    # -- summarize block trips
    group_by_fields = [block_taz_level, mode_dim, end_dim]
    patch_df = trips_df_block.groupby(group_by_fields).sum().reset_index()
    # for gb_field in group_by_fields:
    #     patch_df[gb_field] = patch_df[gb_field].astype(str)
    #     trips_df_taz[gb_field] = trips_df_taz[gb_field].astype(str)
    
    # -- merge taz trips with block sums to 
    trips_df_taz_p = trips_df_taz.merge(
        patch_df, how="left", on=group_by_fields, suffixes=("_taz", "_patch"))
    trips_df_taz_p["trips"] = np.select(
        [pd.isna(trips_df_taz_p.trips_patch)],
        [trips_df_taz_p.trips_taz],
        trips_df_taz_p.trips_patch
    )
    
    return (trips_df_taz_p, trips_df_block)


def seedTripTable(net_config, purpose, trip_table, trips_df, decay_refs,
                  mode_col="Mode", trips_col="trips", end_col="End",
                  id_col="TAZ", level=None, scale="TAZ",
                  modes=["walk", "bike", "driver", "passenger", "WAT", "DAT"],
                  nonmotor_modes=["walk", "bike"], decay_node="/pdf",
                  logger=None):
    """
    Uses origin and destination trip estimates and emma's 
    `weightedInteractions` method to seed a trip distribution matrix. Trip
    estimates are filtered by mode and the resulting seed is scaled such that
    total trips sum to productions.

    Parameters
    -----------
    net_config : String
        The net config from which to pull decay skim factors.
    purpose : String
    trip_table : emma.od.Skim
        An empty H5 file where the trip table is stored. This file's values 
        will be updated, such that trips by each mode are seeded with values
        based on the `weightedInteractions` output.
    trips_df : pd.DataFrame
        A data frame containing trips by mode and end for this purpose. The
        trip estimates are used in the `weightedInteractions` procedure to
        define modal productions and attractions.
    decay_refs: dict
        A dictionary with keys reflecting mode names in `modes` and values
        relating each node to an hdf skim file in the `net_config` folder
        where OD decay factors are stored.
    mode_col : String, default="Mode"
        The column in `trips_df` that identifies each trip estimate's mode.
    trips_col : String, default="trips"
        The column in `trips_df` that identifies the trip estimate.
    end_col : String, default="End"
        The column in `trips_df` that identifies each trip estimate's end 
        (production or attraction).
    id_col : String, default="TAZ"
        The column in`trips_df` that identifies the zone for each trip
        estimate.
    level : String, default=None
        If a decay skim uses a mult-level index for its `zone` attribute,
        specify the level in the index to be used for indexing `trips_df`
        when running `weightedInteractions`.
    scale : String, default="TAZ"
        Some steps vary depending on the current analysis scale ("TAZ" or
        "block"), so the scale is specified here.
    modes: [String,...], default=["walk", "bike", "driver", "passenger", "WAT", "DAT"]
    nonmotor_modes: [String,...], default=["walk", "bike"]
        Which modes in `modes` are nonmotorized modes.
    decay_node: String, default="/pdf"
        Name of the node in `decay_ref` skim file where decay rates are stored.
    logger: Logger, default=None

    Returns
    --------
    None
        Nothing is returned by the function. Rather, values in `trip_table`
        are updated based on the outcomes of the `weightedInteractions`
        function.
    """
    print(f"Seeding distribution for {purpose} ({scale})")
    if logger is not None:
        logger.info(f"Seeding distribution for {purpose} ({scale})")
    for mode in modes:
        # Get the decay skim for this mode
        decay_crit = {"Purpose": purpose}
        decay_ref = decay_refs[mode]
        if scale == "TAZ" and mode in nonmotor_modes:
            decay_skim_f = r"net\{}\nonmotor_decay_TAZ.h5".format(net_config)
            decay_crit["Impedance"] = f"{mode.title()}Time"
        else:
            decay_skim_f = r"net\{}\{}_decay.h5".format(net_config, decay_ref)
        # TODO: put in try/except block
        decay_skim = emma.od.openSkim_HDF(decay_skim_f, decay_node)
        
        # Setup filters
        mode_fltr = trips_df["Mode"] == mode
        p_fltr = trips_df[end_col] == "P"
        a_fltr = trips_df[end_col] == "A"
        # Get trip estimates by mode for each end by combining filters
        trips_p_fltr = np.logical_and(mode_fltr, p_fltr)
        trips_a_fltr = np.logical_and(mode_fltr, a_fltr)
        # Fetch trips
        trips_p = trips_df[trips_p_fltr].copy()
        trips_a = trips_df[trips_a_fltr].copy()
        
        # Summarize total Ps, as these will be used to "scale" the seed
        total_ps = trips_p[trips_col].sum()
        
        # Use "weighted interactions as the seed
        print(f"-- {mode}")
        if logger is not None:
            logger.info(f"-- {mode}")
        try:
            wtd_int = emma.od.weightedInteractions(
                decay_skim=decay_skim, origins_df=trips_p, dests_df=trips_a,
                origins_col=trips_col, dests_col=trips_col, origins_id=id_col,
                dests_id=id_col, level=level, weighting_factor=total_ps,
                **decay_crit)
        except KeyError:
            print("-- ... mode name not found")
            continue
        except:
            raise
        
        # Put the weighted interactions table in the trip table
        trip_table.put(wtd_int.data, Mode=mode)


def tripTargetsByZone(trips_df, zone_col="TAZ", trips_col="trips",
                      end_col="End", logger=None):
    """
    From a data frame of trips by zone, mode, and end, summarize trips by
    zone for each trip end.
    
    Parameters
    -----------
    trips_df: pd.DataFrame
    zone_col: String, default="TAZ"
    trips_col: String, default="trips"
    end_col: String, default="End"
    logger: Logger, default=None
    
    Returns
    --------
    targets_p, targets_a: 1d arrays
    """
    print("-- Fetching IPF targets")
    if logger is not None:
        logger.info("-- Fetching IPF targets")
    fltr_p = trips_df[end_col] == "P"
    fltr_a = trips_df[end_col] == "A"
    
    #Filter the trips and group by zone_col
    trips_p = trips_df[fltr_p]
    trips_a = trips_df[fltr_a]
    
    targets_p = trips_p.groupby(zone_col).sum()[trips_col].values
    targets_a = trips_a.groupby(zone_col).sum()[trips_col].values
    
    if logger is not None:
        logger.info(targets_p)
        logger.info(targets_a)
    
    return targets_p, targets_a


def summarizeTripAttributes(trips, mode, net_config, skim_ref, unit,
                            sum_dims=["From", "To"], factor=1.0):
    """
    A helper function to facilitate summarization of trip table data, such
    as person miles of travel, average trip length, etc.

    Parameters
    ----------
    trips : emma.od.Skim
        A matrix of trips by mode for a given travel purpose
    mode : String
        The mode in the `Mode` axis of `trips` to summarize
    net_config : String
        Sets the source directory from which to pull OD cost data
    skim_ref : Dict
        A dictionary with mode names as keys and skim file parameters
        (file, node, axis, label) as values to look up key impedance values
        (distance, time, e.g.) for trip summarization.
    unit : String
        The trip units being summarized ("miles", "minutes", etc.). The units
        appear in output column headings.
    sum_dims : [String,...], default=["From", "To"]
        The dimensions over which to summarize trip data.
    factor: numeric, default=1.0
        Trip sums may be factored for unit conversion if desired (meters to
        minutes, time to cost, e.g.).

    Returns
    -------
    trip_sum: DataFrome
        A data frame with row for each zone
    """
    m_trips = trips.take(Mode=mode, squeeze=True)
    f, node, axis, label = skim_ref[mode]
    skim_f = r"net\{}\{}.h5".format(net_config, f)
    skim = emma.od.openSkim_HDF(skim_f, node)   
    skim_crit = {axis: label}
    prod = m_trips.impress(
        fill_with=m_trips.data * skim.take(squeeze=True, **skim_crit).data
        )
    
    stack = []
    for sum_dim in sum_dims:
        trip_col = f"Trips_{sum_dim}"
        prod_col = f"Trip_{unit}_{sum_dim}"
        avg_col = f"{unit}_{sum_dim}"
        trip_sum = m_trips.sum(sum_dim)
        wtd_sum = prod.sum(sum_dim)
        trip_df = trip_sum.to_frame(trip_col)
        prod_df = wtd_sum.to_frame(prod_col)
        prod_df[prod_col] *= factor
        merge_df = trip_df.merge(prod_df, how="inner", left_index=True, 
                                 right_index=True)
        merge_df[avg_col] = merge_df[prod_col]/merge_df[trip_col]
        
        # Make a well-formed index
        merge_df = merge_df.reset_index()
        names = m_trips[sum_dim].levels
        merge_df[names] = pd.DataFrame(
            merge_df[sum_dim].to_list(), index=merge_df.index)
        merge_df.drop(columns=sum_dim, inplace=True)
        merge_df.set_index(names, inplace=True)   
        stack.append(merge_df)
    return pd.concat(stack, axis=1)
