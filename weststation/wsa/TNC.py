import emma
from emma import lba, ipf
import numpy as np
import pandas as pd
import xlrd


# FUNCTIONS
def fetchTNCSeed(excel_file, named_range="ipf_seed", purp_col="Purpose",
                 modes=["Walk", "Bike", "Transit", "Driver", "Passenger"]):
    """
    Read an IPF seed for estimating regional TNC trips by mode and purpose
    targets from an excel file. Return the seed as a labeled array.

    Parameters
    ----------
    excel_file : String
        Path to the excel file with the IPF seed information
    named_range : String, default="ipf_seed"
        A named range in `excel_file` where seed data may be found.
    purp_col : String, default="Purpose"
        The column heading in the ipf seed range that identifies trip purpose
    modes: [Syring,...], default=["Walk", "Bike", "Transit", "Driver", "Passenger"]
        The column headings in the ipf seed range that identify travel modes
        
    Returns
    -------
    seed_array : LbArray
        A labeled array with seeds in two dimensions: Purpose and Mode
    """
    with xlrd.open_workbook(excel_file) as workbook:
        ipf_obj = workbook.name_map[named_range.lower()][0]
        loc_txt = ipf_obj.formula_text
        sheet, cell_range = loc_txt.split("!")
        from_col, from_row, to_col, to_row = cell_range.split("$")[-4:]
        usecols = ":".join([from_col, to_col])
        skiprows = int(from_row[:-1])
        nrows = int(to_row) - skiprows
        skiprows -= 1
    targets_df = pd.read_excel(excel_file, sheet, usecols=usecols, 
                               skiprows=skiprows, nrows=nrows)
    targets_df = targets_df.melt(purp_col, modes, "Mode", "Seed")
    seed_array = lba.dfToLabeledArray(targets_df, [purp_col, "Mode"], "Seed")
    return seed_array


def fetchTNCModeTargets(excel_file, named_range="mode_targets",
                        modes=["Walk", "Bike", "Transit", "Driver", "Passenger"]):
    """
    Read a set of modal targets from an excel file reflecting the total TNC 
    trips switched from various modes. These are used in an IPF process to
    set mode and purpose-specific control totals and setting TNC probability
    alpha values.

    Parameters
    ----------
    excel_file : String
        Path to the excel file with the modal target information
    named_range : String, default="mode_targets"
        A named range in `excel_file` where the targets are found. The range
        is assumed to consist of a single row, with values ordered in direct
        correspondence to `modes`
    modes : [String,...], default=["Walk", "Bike", "Transit", "Driver", "Passenger"]
        The modes included in the `named_range` value set. Listed in order
        shown in the excel file.

    Returns
    -------
    targets : pd.Series
        A series of modal targets with mode names as the index.

    """
    with xlrd.open_workbook(excel_file) as workbook:
        ipf_obj = workbook.name_map[named_range.lower()][0]
        loc_txt = ipf_obj.formula_text
        sheet, cell_range = loc_txt.split("!")
        from_col, from_row, to_col, to_row = cell_range.split("$")[-4:]
        usecols = ":".join([from_col, to_col])
        skiprows = int(from_row[:-1])
        nrows = 1
        skiprows -= 1
    targets_df = pd.read_excel(excel_file, sheet, usecols=usecols, 
                               skiprows=skiprows, nrows=nrows)
    
    targets = targets_df.columns.to_series(index=modes)
    return targets


def estimateAlphas(trip_table, tnc_ratio_skim, targets_array, mode, purpose):
    """
    Given labeled arrays with trip estimates and TNC probability ratios, determine
    the probability ratio needed to attain a targeted number of trips for a
    given mode and purpose.

    Parameters
    ------------
    trip_table: LbArray
        A labeled array with trips estimated for each OD pair for this `mode`
        and `purpose`.
    tnc_ratio_skim: LbArray
        A labeled array with TNC probabilty estimates for each OD pair for
        this `mode` and `purpose`.
    targets_array: LbArray
        A labeled array with targets for TNC trip substitutions given this
        `mode` and `purpose`.
    mode: String
        The current travel mode whose trips could be candidates for TNC
        substituation.
    purpose: String
        The purpose of travel.

    Returns
    -------
    alpha: Float
        The TNC probability ratio needed to substitute the targeted number
        of trips for this `mode` and `purpose`. Once established, it should
        remain the same for other scenarios for cross-scenario comparisons.
    """
    # Get the target number of trips for this mode and purpose
    tgt = targets_array.take(Mode=mode, Purpose=purpose, squeeze=True).data
    # Trip table and tnc ratios are already purpose-specific
    #  Dump them to data frames and merge
    prob_df = tnc_ratio_skim.take(squeeze=True,
        Mode=mode, Components="TNC_prob_ratio").to_frame("prob")
    trip_df = trip_table.take(Mode=mode, squeeze=True).to_frame("trips")   
    merge_df = prob_df.merge(trip_df, how="inner", left_index=True, 
                             right_index=True)
    # Sort by prob ratio
    merge_df.sort_values("prob", ascending=False, inplace=True)
    # Calculate cumulative sum of trips
    merge_df["cum_trips"] = merge_df.trips.cumsum()
    # Set alpha as prob ratio where cum_trips meets or exceeds the target
    fltr = merge_df.cum_trips <= tgt
    alpha = np.min(merge_df[fltr].prob)
    return alpha


def initTNCCostArray(scen, purpose, period, hdf_store=None, node_path=None, 
                 name=None, overwrite=False, logger=None):
    """
    Initialize a new Skim object to record TNC cost components and estimated
    TNC total costs and trip probabilities.

    Parameters
    ----------
    scen : String
    purpose : String
    period : String
    hdf_store : String, default=None
    node_path : String, default=None
    name : String, default=None
    overwrite; Boolean, default=False
    logger: Logger, default=None

    Returns
    -------
    Skim
    """
    # Open reference file
    base_file = r"scen\{}\Trip_dist_TAZ_{}_{}.h5".format(scen, period, purpose)
    base = emma.od.openSkim_HDF(base_file, "/dist")
    
    print(f"initializing TNC cost skim from base array {base_file}")
    
    # Cast into new axes
    comp_axis = lba.LbAxis("Components", 
                           ["Distance", "Duration", 
                            "EstCostMinutes", "DecayMinutes", 
                            "EstCostDollars", "DecayDollars"])
    zeros = base.stamp(fill_with=0.0, drop="Mode")
    tnc_array = zeros.cast(comp_axis, hdf_store=hdf_store, node_path=node_path,
                          name=name, overwrite=overwrite)
    if logger is not None:
        logger.info("intialized TNC costs array")
    return tnc_array    


def initTNCRatioArray(scen, purpose, period, hdf_store=None, node_path=None,
                    name=None, overwrite=False, logger=None):
    """
    Initialize a new Skim object to record modal trip probabilities and TNC
    trip probability ratios.

    Parameters
    ----------
    scen : String
    purpose : String
    period : String
    hdf_store : String, default=None
    node_path : String, default=None
    name : String, default=None
    overwrite; Boolean, default=False
    logger: Logger, default=None

    Returns
    -------
    Skim
    """
    # Open reference file
    base_file = r"scen\{}\Trip_dist_TAZ_{}_{}.h5".format(scen, period, purpose)
    base = emma.od.openSkim_HDF(base_file, "/dist")
    
    print(f"initializing TNC ratio skim from base array {base_file}")
    
    # Cast into new axes
    comp_axis = lba.LbAxis("Components", 
                           ["Mode_trip_prob", "TNC_prob_ratio"])
    zeros = base.stamp(fill_with=0.0)
    tnc_array = zeros.cast(comp_axis, hdf_store=hdf_store, node_path=node_path,
                          name=name, overwrite=overwrite)
    if logger is not None:
        logger.info("intialized TNC ratio array")
    return tnc_array


def estimateTNCCosts(auto_skim, purpose, tnc_cost_skim, value_of_time,
                     tnc_base_fare, tnc_service_fee, tnc_cost_per_mile,
                     tnc_decay_mu, tnc_decay_sigma, imp_axis="Impedance",
                     time_label="TravelTime", dist_label="Distance",
                     logger=None):
    """
    Estimate TNC cost components. Pulls data for estimates of auto trip
    durations and distances, calculates TNC estimated costs (time and money)
    using global cost params. Estimates TNC decay factors for each potential
    OD pair.

    Parameters
    ----------
    auto_skim: Skim
        A skim with time and distance impedances for traveling by car. These
        are used to estimate TNC travel costs.
    purpose : String
    tnc_cost_skim : String
        A labeled array that will holds the dimensions of TNC costs and the
        final generalized cost estimate for TNCs for all OD pairs.
    value_of_time: Numeric
        A factor to convert between time and monetary units, expressed as
        dollars per hour.
    tnc_base_fare: Numeric
    tnc_service_fee: Numeric
    tnc_cost_per_mile: Numeric
    tnc_decay_mu: {"minutes": Numeric, "dollars": Numeric}
        A parameter in the lognormal decay expression used to estimate TNC
        trip propensities, `mu` varies for cost-based or time-based analyses.
        The `mu` values are therefor provided in a dictionary.
    tnc_decay_sigma: Numeric
        A parameter in the lognormal decay expression used to estimate TNC
        trip propensities, `sigma` varies for cost-based or time-based
        analyses. The `sigma` values are therefor provided in a dictionary.
    imp_axis: String, default="Impedance"
        The name of the axis in `auto_skim` along which to find various
        impedances
    time_label: String, default="TravelTime"
        The impedance axis label where OD travel time estimates are stored.
    dist_label: String, default="Distance"
        The impedance axis label where OD distance estimates are stored.
    logger: Logger

    Returns
    -------
    None - the `tnc_cost_skim` is modified in place.

    """    
    print("Estimating TNC costs")
    if logger is not None:
        logger.info("Estimating TNC costs")
    
    # Place times
    print(" -- Recording estimated duration")
    if logger is not None:
        logger.info(" -- Recording estimated duration")
    crit = {imp_axis: time_label}
    tnc_cost_skim.put(
        auto_skim.take(squeeze=True, **crit).data,
        Components="Duration"
        )
    
    # Place distance
    print(" -- Recording estimated distance")
    if logger is not None:
        logger.info(" -- Recording estimated distance")
    crit = {imp_axis: dist_label}
    tnc_cost_skim.put(
        auto_skim.take(squeeze=True, **crit).data,
        Components="Distance"
        )
    
    # Calculate costs in minutes
    # Convert montetary costs...
    #  numer = (TNC_BASE_FARE + TNC_SERVICE_FEE + (TNC_COST_PER_MILE * distance)
    # ...to minutes
    # denom = (vot * 60))
    # And add resulting "minutes" to duration
    print(" -- Recording estimated total (weighted) time")
    if logger is not None:
        logger.info(" -- Recording estimated (weighted) time")
    duration = tnc_cost_skim.take(Components="Duration", squeeze=True).data[:]
    distance = tnc_cost_skim.take(Components="Distance", squeeze=True).data[:]
    numer = tnc_base_fare + tnc_service_fee + (tnc_cost_per_mile * distance)
    denom = value_of_time * 60
    tnc_cost_skim.put((numer/denom) + duration,
                      Components="EstCostMinutes")
    
    # Calculate costs in dollars
    # Numerator (already expressed in dollars) + duration converted to dollars
    print(" -- Recording estimated total monetary costs")
    if logger is not None:
        logger.info(" -- Recording estimated monetary costs")
    dollars = numer + (duration / 60 * vot)
    tnc_cost_skim.put(dollars, Components="EstCostDollars")
    
    # Setup decay rates
    time_mu = tnc_decay_mu["minutes"]
    time_sigma = tnc_decay_sigma["minutes"]
    time_decay = emma.decay.LogNormalDecay_cdf(time_mu, time_sigma)
    
    money_mu = tnc_decay_mu["dollars"]
    money_sigma = tnc_decay_sigma["dollars"]
    money_decay = emma.decay.LogNormalDecay_cdf(money_mu, money_sigma)
    
    # Apply Decay rates
    print(" -- Recording decay factors")
    if logger is not None:
        logger.info(" -- Recording decay factors")
    tnc_cost_skim.put(
        1 - time_decay.apply(
                tnc_cost_skim.take(
                    Components="EstCostMinutes", squeeze=True).data,
                neg_value=0.0),
        Components="DecayMinutes"
        )
    
    tnc_cost_skim.put(
        1 - money_decay.apply(
                tnc_cost_skim.take(
                    Components="EstCostDollars", squeeze=True).data,
                neg_value=0.0),
        Components="DecayDollars"
        )


def estimateTNCProb(net_config, purpose, tnc_ratio_skim, tnc_cost_skim,
                    decay_refs, mode_dict, mode_impedances,
                    use_units="Dollars",
                    all_purposes=["HBW", "HBO", "HBSch", "NHB"], logger=None):
    """
    Estimate probability ratios for TNC trips. Pulls data for mode- and
    purpose-specific decay, calculates modal cdf, and creates probability
    ratio of modal cdf over tnc cdf.

    Parameters
    ----------
    net_config : String
    purpose : String
    tnc_ratio_skim : Skim
    tnc_cost_skim : Skim
    decay_refs: dict
        A dictionary whose keys corespond to mode names in the emma analysis
        and whose values are strings identifying the name of the hdf file
        containing OD decay factor estimates for that mode.
    mode_dict: dict
        A dictionary whose keys corespond to mode names in the emma analysis
        and whose values are strings identifying the general mode referred to
        in the `decay_specs` input file.
    mode_impedances: dict
        A dictionary whose keys correspond to mode names in the emma analysis
        and whose values are tuples of strings. Each tuple specifies details
        for obtaining mode-specific costs from a skim file, providing the
        hdf node path, axis, label, and units of analysis.
    use_units : String, default="Dollars"
        If "Dollars", the TNC decay in dollars is always referenced; 
        if "Minutes", the TNC decay in minutes is always referenced;
        if "align", the TNC decay referened varies based on the units used
        for each mode's generalized costs.
    all_purposes: [String,...], default=["HBW", "HBO", "HBSch", "NHB"]
    logger: Logger

    Raises
    ------
    ValueError
        If `use_units` is not recognized - use "Dollars", "Minutes", or "Align"

    Returns
    -------
    None - the `tnc_ratio_skim` is modified in place.

    """
    print("Estimating TNC ratios")
    if logger is not None:
        logger.info(f"Estimating TNC ratios (use_units={use_units})")
    _quants_ = np.linspace(0, 1, 10)  
    
    # Iterate over modes
    for mode in tnc_ratio_skim.Mode.labels:
        print(f" -- {mode}")
        if logger is not None:
            logger.info(f" -- {mode}")
        # Calculate modal trip prob
        # -- Get impedance skim file, hdf node, axis, and label
        decay_ref = decay_refs[mode]
        decay_file = r"net\{}\{}.h5".format(net_config, decay_ref)
        node, axis, label, units = mode_impedances[mode]
        if label == all_purposes:
            lbl_idx = all_purposes.index(purpose)
            label = label[lbl_idx]
        decay_skim = emma.od.openSkim_HDF(decay_file, node)
        
        # -- Handle decay specs
        if axis is None:
            decay_crit ={}
        else:
            decay_crit = {axis: label}
            
        # -- Units
        if use_units.lower() == "dollars":
            tnc_decay = "DecayDollars"
        elif use_units.lower() == "minutes":
            tnc_decay = "DecayMinutes"
        elif use_units.lower() == "align":
            if units == "minutes":
                tnc_decay = "DecayMinutes"
            else:
                tnc_decay = "DecayDollars"
        else:
            raise ValueError(f"use_units value ({use_units}) not understood")
        
        # -- Get modal decay specs
        specs_f = "input\DecaySpecs.csv"
        specs = pd.read_csv(specs_f)
        spec_mode = mode_dict[mode]
        fltr = np.logical_and.reduce(
            [
                specs.Mode == spec_mode,
                specs.Purpose == purpose,
                specs.Use == "distribution"
                ]
            )
        mu = specs[fltr].ConstMu
        sigma = specs[fltr].CoefSig
        
        # -- Build the decay object
        print(" -- -- Recording decay factors")
        if logger is not None:
            logger.info(" -- -- Recording decay factors")
        decay = emma.decay.LogNormalDecay_cdf(mu, sigma)
        # -- Apply decay object
        cdf_decay = 1 - decay.apply(
            decay_skim.take(
                squeeze=True, **decay_crit
                ).data,
            neg_value=0
            )
        cdf_quants = np.round(np.quantile(cdf_decay, _quants_), 4)
        if logger is not None:
            logger.info(f" -- -- CDF Decay quantiles: {cdf_quants}")
        # -- Push decay result to skim
        tnc_ratio_skim.put(cdf_decay, Components="Mode_trip_prob", Mode=mode)
        
        # Divide tnc_trip_prob by modal trip prob
        print(" -- -- Recording probability ratios")
        if logger is not None:
            logger.info(" -- -- Recording probability ratios")
        
        tnc_ratio_skim.put(
            np.divide(
                tnc_cost_skim.take(
                    squeeze=True, Components=tnc_decay).data,
                cdf_decay, 
                out=np.zeros_like(cdf_decay, dtype=float),
                where=cdf_decay > 0
                ),
            Components="TNC_prob_ratio", Mode=mode
            )
        
        # Report summaries of results
        pr_quants = np.round(np.quantile(
            tnc_ratio_skim.take(
                Components="TNC_prob_ratio", Mode=mode).data[:],
            _quants_), 4)
        logger.info(f" -- -- TNC prob ratio quantiles: {pr_quants}")

def applyTNCProbRatio(trip_table, tnc_ratio_skim, alpha, hdf_store=None,
                      node_path=None, name=None, overwrite=False,
                      logger=None):
    """
    Estimate which trips in a trip table would switch from the estimated mode
    to TNC, based on the TNC probability ratio (TNC utility relative to the
    estimated mode). Trips between OD pairs with TNC probability ratios above
    a set target (alpha) are assumed to switch to TNC.

    Parameters
    ----------
    trip_table : emma.od.Skim
        A skim with trips by mode distributed from origin to destination
        zones.
    tnc_ratio_skim : emma.od.Skim
        A skim with TNC probability ratios
    alpha : Numeric or [Numeric, ...], default=2.0
        OD pairs with TNC probability ratios above this value are assumed to
        swith to TNC. A single alpha may be provided or a list of alphas
        corresponding to the number of modes in the `Mode` axis of 
        `trip_table` to allows thresholds to vary by mode.
    hdf_store : String, default=None
    node_path : String, default=None
    name : String, default=None
    overwrite : Boolean, default=False
    logger: Logger, default=None

    Returns
    -------
    tnc_table: Skim

    See Also
    --------
    estimateTNCProb
    """
    print(f"Applying TNC probability ratios")
    if logger is not None:
        logger.info(f"Applying TNC probability ratios")
    tnc_table = trip_table.impress(fill_with=0.0, hdf_store=hdf_store,
                                   node_path=node_path, name=name,
                                   overwrite=overwrite)
    trips_by_mode = []
    tnc_by_mode = []
    for i, mode in enumerate(tnc_ratio_skim.Mode.labels):
        print(f" -- {mode}")
        if logger is not None:
            logger.info(f" -- {mode}")
        if isinstance(alpha, emma.od.Iterable):
            a = alpha[i]
        else:
            a = alpha
        print(f" -- -- alpha={a}")
        if logger is not None:
            logger.info(f" -- -- alpha={a}")
        
        # Flag OD pairs
        flags = np.array(
            tnc_ratio_skim.take(Mode=mode, Components="TNC_prob_ratio",
                                squeeze=True).data >= a, dtype=int)
        
        #Push to tnc table
        tnc_table.put(
            np.multiply(
                trip_table.take(Mode=mode, squeeze=True).data,
                flags
                ),
            Mode=mode
            )
        
        # Info
        total_trips = np.sum(trip_table.take(Mode=mode).data)
        tnc_trips = np.sum(tnc_table.take(Mode=mode).data)
        pct = np.round((tnc_trips/total_trips) * 100, 2)
        
        trips_by_mode.append(np.round(total_trips, 2))
        tnc_by_mode.append(np.round(tnc_trips, 2))
        if logger is not None:
            logger.info(f" -- -- count OD pair switches {np.sum(flags)}")
            logger.info(f" -- -- trip switches {tnc_trips} ({pct}%)")
    
    # Reporting all trips
    total_trips = np.sum(trips_by_mode)
    tnc_trips = np.sum(tnc_by_mode)
    pct = np.round((tnc_trips/total_trips) * 100, 2)
    tnc_shares = np.round(np.divide(tnc_by_mode, tnc_trips), 2)
    if logger is not None:
        logger.info(f" -- Total trips switched to tnc: {tnc_trips} ({pct}%)")
        logger.info(f" -- Shares of TNC trips by mode: {tnc_shares}")
    
    return tnc_table
