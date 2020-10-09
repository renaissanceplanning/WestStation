"""
Clean Skims
============
A collection of functions to facilitate modifications to csv skim tables
to prepare them for importing into emma Skim objects.
"""

import pandas as pd
import numpy as np


# Functions
def previewSkim(in_file, nrows=5, logger=None, **kwargs):
    """
    Return the top rows of a csv file to preview its contents.

    Parameters
    -----------
    in_file: String
        Path to csv file
    nrows: Integer
        The number of rows at the top of the table to load in the preview.
    logger: Logger
        If desired, pass a logger object to record the skim preview. 
        All logging is done at the INFO level.
    kwargs:
        Keyword arguments that can be passed to pandas.read_csv

    Returns
    ----------
    preview: pd.DataFrame
    """
    preview = pd.read_csv(in_file, nrows=nrows, **kwargs)
    if logger is not None:
        logger.info(str(preview))
    return preview

def cleanSkims(in_file, out_file, criteria, rename={}, chunksize=50000,
               logger=None, **kwargs):
    """
    Ingest a raw skim and retain only those rows that meet the provided
    criteria (all criteria must be true).

    Parameters
    -----------
    in_file: String
        Path to the raw csv data
    out_file: String
        Path to the new csv output
    criteria: [(String, String, Var),...]
        A list of tuples. Each tuple contains specifications for filtering
        the raw csv data by a particular criterion. The tuple consists of
        three parts: (reference column, comparator, value). Use column names
        expected after renaming, if `rename` if provided. Comparators may
        by provided as strings corresponding to built-in class comparison
        methods: 
           - __eq__() = equals [==]
           - __ne__() = not equal to [!=]
           - __lt__() = less than [<]
           - __le__() = less than or equal to [<=]
           - __gt__() = greater than [>]
           - __ge__() = greater than or equal to [>=]
    rename: {String: String,...}, default={}
        Optionally rename columns in the raw data based on key: value
        pairs in a dictionary. The key is the existing column name, and
        the value is the new name for that column. Only columns for which
        renaming is desired need to be included in the dictionary.
    chunksize: Int
        The number of rows to read in from `in_file` at one time. All rows
        are evaluated in chunks to manage memory consumption.
    logger: Logger
        If desired, pass a logger object to record information about the
        skim cleaning process. All logging is done at the INFO level.
    kwargs:
        Any keyword arguments passed to pandas.read_csv for loading `in_file`.
    
    Returns
    -------
    None - outfile is written during this process.
    """
    count_full = 0
    count = 0
    header = True
    mode = "w"
    for chunk in pd.read_csv(in_file, chunksize=50000):
        count_full += len(chunk)
        if rename:
            chunk.rename(rename, axis=1, inplace=True)
        # Apply criteria
        for crit in criteria:
            col_name, comp, value = crit
            chunk = chunk[getattr(chunk[col_name], comp)(value)]
        # Write results
        if len(chunk) > 0:
            count += len(chunk)
            chunk.to_csv(out_file, header=header, mode=mode)
            header=False
            mode="a"
    # Report out
    percent = np.round((count/count_full) * 100, 2)
    report_str = f"Complete. {count}/{count_full} rows retained ({percent}%)"
    print(report_str)
    if logger is not None:
        logger.info(report_str)