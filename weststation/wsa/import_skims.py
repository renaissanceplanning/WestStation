import emma
from emma import lba
import numpy as np
import pandas as pd


#Functions
def initImpSkim_wsa(zones_array, index_fields, impedance_attributes,
                    hdf_store, node_path, name, overwrite=False,
                    desc=None, init_val=-1.0):
    """
    Build a basic skim to hold impedance data for a given mode.

    Parameters
    -----------
    zones_array: pd.DataFrame
        The zones array has rows representing zones, with fields to identify
        each zone. Used to set the i and j dimensions of the skim.
    index_fields: [String,...]
        A list of columns in `zones_array` (or a single column name) to use as
        zone indices in the skim.
    impedance_attributes: [String,...]
        A list of impedance attributes that will be stored in the skim.
    hdf_store: String
        A path to an hdf file to store the skim data.
    node_path: String
        The node in `hdf_store` where the skim array will be stored.
    name: String
        The name of skim array at `node_path`.
    overwrite: Boolean, default=False
        If True, the data in the hdf file at `node_path/name` will be
        overwritten.
    desc: String, default=None
        A brief description of the skim's contents.
    init_val: numeric, default=-1.0

    Returns
    --------
    Skim
        A skim object. All values are initialized to `init_val`. These values
        will be updated when skim data are loaded from csv files.
    """
    zones= zones_array[index_fields]
    imp_axis = lba.LbAxis("Impedance", impedance_attributes)
    return emma.od.Skim(zones, init_val, imp_axis, hdf_store=hdf_store,
                          node_path=node_path, name=name,
                          overwrite=True, desc=desc)


def estimateParkingDuration(time_array, cost_array, purpose, max_dur=420):
    """
    Estimated parking costs are a function of 1/2-day pricing pro-rated to 
    hourly assuming 1/2 day is 4 hours.
    
    Hourly estimates are then applied based on the estimated duration of the
    activity (i.e, how long are you parked?)
    
    The parking duration estimate is a function of trip duration, parking cost 
    (1/2 day charge), and trip purpose

    Parameters
    -----------
    time_array: np.ndarray
        An array of OD travel times (in  minutes)
    cost_array: np.ndarray
        An array of destination-end hourly parking charges cast into the full
        OD matrix
    purpose: String ("HBW", "HBO", "HBSch", or "NHB")
        The purpose of travel
    max_dur: Integer, default=420
        Cap the estimated parking duration at the specified value (in minutes)
    
    Returns
    ------------
    np.ndarray
    """
    # Regression parameters
    trip_duration = 0.011149
    parking = -0.007387
    purposes = {
        "HBO": 4.172639,
        "HBSch": 5.593386,
        "HBW": 5.279180,
        "NHB": 3.097651
        }

    est = time_array * trip_duration + cost_array * parking + purposes[purpose]
    est = np.exp(est)
    est[np.where(est > max_dur)] = max_dur
    return est

def addZonalCosts(skim, imped_axis, imped_name, zone_df, column, factor=1.0,
                  zone_id_level="TAZ", origin_cost=False):
    """
    Given a data frame of zonal costs (parking charges, terminal times, e.g.),
    add these cost to the specified axis and label for the input skim.
    
    Origin-end costs are added when origin_cost=True; otherwise costs
    are assumed to apply to the destination end.

    Parameters
    ------------
    skim: Skim
    imped_axis: String
        The name of the axis in `skim` in which values will be recorded
    imped_name: String
        The label in `imped_axis` where values will be recorded.
    zone_df: pd.DataFrame
        A table of zonal costs. It is assumed its index values correspond
        to those in `skim.zones`
    column: String
        The column in `zone_df` with zonal cost values
    factor: numeric, default=1.0
        A factor by which to scale zonal costs upon import
    taz_id_level: String, default="TAZ"
        If `skim` uses a multiindex for its `zones` attribute, provide the name
        of the level against which `zone_df` will be reindexed.
    origin_cost: Boolean, default=False
        If True, costs are applied to OD pairs by origin location. Otherwise,
        costs are applied based on destination.
    """
    # Make a view of the data
    view = lba.LbViewer(skim, **{imped_axis: imped_name})
    # Re-index the taz data to match the skim
    zone_df_re = emma.lba.reindexDf(
        zone_df.set_index("TAZ_ID"), skim.From, level=zone_id_level)
    # Broadcast the taz_data
    dummies = np.array([1.0 for _ in skim.zones])
    zone_values = zone_df_re[column].values
    if origin_cost:
        m = dummies * zone_values[:, None]
    else:
        m = zone_values * dummies[:, None]
    # Apply the factor
    m *= factor
    #add m to the view
    v = view[0]
    v.data += m
    v.push()