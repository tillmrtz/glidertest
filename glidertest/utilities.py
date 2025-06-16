import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame
from skyfield import almanac
from skyfield import api
import gsw
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import cmocean.cm as cmo



def _check_necessary_variables(ds: xr.Dataset, vars: list):
    """
    Checks that all of a list of variables are present in a dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that should be checked
    vars: list
        List of variables

    Raises
    ------
    KeyError:
        Raises an error if all vars not present in ds

    Notes
    -----
    Original Author: Callum Rollo
    """
    missing_vars = set(vars).difference(set(ds.variables))
    if missing_vars:
        msg = f"Required variables {list(missing_vars)} do not exist in the supplied dataset."
        raise KeyError(msg)


def _calc_teos10_variables(ds):
    """
    Calculates TEOS 10 variables not present in the dataset
    :param ds:
    :return:
    """
    _check_necessary_variables(ds, ['DEPTH', 'LONGITUDE', 'LATITUDE', 'TEMP', 'PSAL'])
    if 'DENSITY' not in ds.variables:
        SA = gsw.SA_from_SP(ds.PSAL, ds.DEPTH, ds.LONGITUDE, ds.LATITUDE)
        CT = gsw.CT_from_t(SA, ds.TEMP, ds.DEPTH)
        ds['DENSITY'] = ('N_MEASUREMENTS', gsw.rho(SA, CT, ds.DEPTH).values)
    return ds

def _time_axis_formatter(ax, ds, format_x_axis=True):
    """
    Apply time formatter and locator to an axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to format.
    ds : xarray.Dataset
        Dataset containing TIME coordinate.
    format_x_axis : bool, default=True
        Whether to format the x-axis (True) or y-axis (False).
    """
    formatter, locator, xlabel = _select_time_formatter_and_locator(ds)

    axis = ax.xaxis if format_x_axis else ax.yaxis
    axis.set_major_formatter(formatter)
    if locator:
        axis.set_major_locator(locator)

    if format_x_axis:
        ax.xaxis.set_major_formatter(formatter)
        if locator:
            ax.xaxis.set_major_locator(locator)
        ax.set_xlabel(xlabel)
    else:
        ax.yaxis.set_major_formatter(formatter)
        if locator:
            ax.yaxis.set_major_locator(locator)
        ax.set_ylabel(xlabel)
        

def _select_time_formatter_and_locator(ds):
    """
    Select appropriate time formatter and locator based on time range.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing TIME coordinate.

    Returns
    -------
    formatter : matplotlib.dates.DateFormatter
        Formatter for time axis or colorbar.
    locator : matplotlib.dates.Locator or None
        Locator for axis ticks (None if not needed).
    xlabel : str
        Label for the axis (optional for colorbars).
    """
    start_time = ds.TIME.min().values
    end_time = ds.TIME.max().values
    time_range_hours = (end_time - start_time) / np.timedelta64(1, 'h')

    if time_range_hours < 24:
        formatter = DateFormatter('%H:%M')
        # Dynamic interval based on time range
        if time_range_hours <= 6:
            interval = 1
        elif time_range_hours <= 12:
            interval = 2
        else:
            interval = 3
        locator = mdates.HourLocator(byhour=range(0, 24, 2))
        start_date = pd.to_datetime(start_time).strftime('%Y-%b-%d')
        end_date = pd.to_datetime(end_time).strftime('%Y-%b-%d')
        xlabel = f'Time [UTC] ({start_date})' if start_date == end_date else f'Time [UTC] ({start_date} to {end_date})'
    elif time_range_hours < 24 * 7:
        formatter = DateFormatter('%d-%b')
        locator = mdates.DayLocator(interval=1)
        start_date = pd.to_datetime(start_time).strftime('%Y-%b-%d')
        end_date = pd.to_datetime(end_time).strftime('%Y-%b-%d')
        xlabel = f'Time [UTC] ({start_date})' if start_date == end_date else f'Time [UTC] ({start_date} to {end_date})'
    else:
        formatter = DateFormatter('%d-%b')
        locator = None
        xlabel = 'Time [UTC]'
        
    return formatter, locator, xlabel
        
def construct_2dgrid(x, y, v, xi=1, yi=1, x_bin_center: bool = True, y_bin_center: bool = True, agg: str = 'median'):

    """
    Constructs a 2D gridded representation of input data based on specified resolutions. The function takes in x, y, and v data,
    and generates a grid where each cell contains the aggregated value (e.g., mean, median) of v corresponding to the x and y coordinates.
    If the input data is already binned and you want the grid coordinates to align with the original bin edges, set `x_bin_center` and `y_bin_center` to False and the 
    resolution (i.e. xi and yi) to the bin size.

    Parameters
    ----------
    x : array-like  
        Input data representing the x-dimension.  
    y : array-like  
        Input data representing the y-dimension.  
    v : array-like  
        Input data representing the z-dimension (values to be gridded).  
    xi : int or float, optional, default=1  
        Resolution for the x-dimension grid spacing.  
    yi : int or float, optional, default=1  
        Resolution for the y-dimension grid spacing.
    x_bin_center : bool, optional, default=True
        If True, the x-coordinate grid (`XI`) corresponds to the **center** of each x-bin.
        If False, it corresponds to the **left edge** of each bin.
        This is especially useful if the input `x` data is already binned with the same resolution as `xi`,
        and you want the grid coordinates to align with the original bin edges. (e.g. profile numbers).
    y_bin_center : bool, optional, default=True
        Same as `x_bin_center`, but for the y-coordinate grid (`YI`).
        Set to False if your `y` data is already pre-binned with the same resolution as `yi`.
    agg : str, optional, default='median'
        Aggregation method to be used for gridding. Options include 'mean', 'median', etc.

    Returns
    -------
    grid : numpy.ndarray  
        Gridded representation of the z-values over the x and y space.  
    XI : numpy.ndarray  
        Gridded x-coordinates corresponding to the specified resolution.  
    YI : numpy.ndarray  
        Gridded y-coordinates corresponding to the specified resolution. 

    Notes
    -----
    Original Author: Bastien Queste
    [Source Code](https://github.com/bastienqueste/gliderad2cp/blob/de0652f70f4768c228f83480fa7d1d71c00f9449/gliderad2cp/process_adcp.py#L140)
    
    Modified by Till Moritz: added the aggregation parameter and the option to chose either bin center or bin edge as the grid coordinates.
    """
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x) + xi+1, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y) + yi+1, yi)

    raw = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()
    grid = np.full([len(xi)-1, len(yi)-1], np.nan)

    raw['xbins'], xbin_iter = pd.cut(raw.x, xi, retbins=True, labels=False, include_lowest=True, right=False)
    raw['ybins'], ybin_iter = pd.cut(raw.y, yi, retbins=True, labels=False, include_lowest=True, right=False)

    raw = raw.dropna(subset=['xbins', 'ybins'])  # Remove out-of-bound rows
    _tmp = raw.groupby(['xbins', 'ybins'])['v'].agg(agg)
    grid[_tmp.index.get_level_values(0).astype(int), _tmp.index.get_level_values(1).astype(int)] = _tmp.values
    # Match XI and YI shape to grid using bin centers
    if x_bin_center:
        xi = xi[:-1] + np.diff(xi) / 2
    else:
        xi = xi[:-1]
    if y_bin_center:
        yi = yi[:-1] + np.diff(yi) / 2
    else:
        yi = yi[:-1]
    YI, XI = np.meshgrid(yi, xi)
    return grid, XI, YI


def bin_profile(ds_profile, vars, binning, agg: str = 'mean'):
    """
    Bins the data for a single profile using the construct_2dgrid function. The binning determines the depth resolution.

    Parameters
    ----------
    ds_profile : xr.Dataset or pd.DataFrame
        The dataset or dataframe containing the data of one profile containing at least 'DEPTH', 'PROFILE_NUMBER' and the variables to bin.
    vars : list
        The variables to bin.
    binning : float
        The depth resolution for binning.
    agg : str, optional
        The aggregation method ('mean' or 'median'). Default is 'mean'.

    Returns
    -------
    binned_profile: pd.DataFrame
        A dataframe containing the binned data for the selected profile.

    Notes
    -----
    Original author: Till Moritz
    """
    ## check if only one profile is selected
    if isinstance(ds_profile, xr.Dataset):
        profile_number = ds_profile['PROFILE_NUMBER'].values
    elif isinstance(ds_profile, pd.DataFrame):
        profile_number = ds_profile.index.values
    if len(np.unique(profile_number)) > 1:
        raise ValueError("Only one profile can be selected for binning.")
    
    binned_data = {}
    msk = ds_profile['DEPTH'].values > 0
    depth = ds_profile['DEPTH'].values[msk]
    profile_number = profile_number[msk]

    # Check for short or empty input data and return empty DataFrame
    if any(len(ds_profile[var]) <= 1 for var in vars) or len(depth) <= 1:
        return pd.DataFrame(columns=vars + ['DEPTH', 'PROFILE_NUMBER'])

    for var in vars:
        var_grid, prof_num_grid, depth_grid = construct_2dgrid(profile_number, depth, ds_profile[var].values[msk],
                                                                xi=1, yi=binning, x_bin_center=False, y_bin_center=True, agg=agg)
        binned_data[var] = var_grid[0]
    binned_data['DEPTH'] = depth_grid[0]
    binned_data['PROFILE_NUMBER'] = prof_num_grid[0]

    return pd.DataFrame(binned_data)

def compute_sunset_sunrise(time, lat, lon):
    """
    Calculates the local sunrise/sunset of the glider location from GliderTools.

    The function uses the Skyfield package to calculate the sunrise and sunset
    times using the date, latitude and longitude. The times are returned
    rather than day or night indices, as it is more flexible for the quenching
    correction.

    Parameters
    ----------
    time: numpy.ndarray or pandas.Series
        The date & time array in a numpy.datetime64 format.
    lat: numpy.ndarray or pandas.Series
        The latitude of the glider position.
    lon: numpy.ndarray or pandas.Series
        The longitude of the glider position.

    Returns
    -------
    sunrise: numpy.ndarray
        An array of the sunrise times.
    sunset: numpy.ndarray
        An array of the sunset times.

    Notes
    -----
    Original Author: Function from GliderTools 
    [Source Code](https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/optics.py)
    """

    ts = api.load.timescale()
    eph = api.load("de421.bsp")

    df = DataFrame.from_dict(dict([("time", time), ("lat", lat), ("lon", lon)]))

    # set days as index
    df = df.set_index(df.time.values.astype("datetime64[D]"))

    # groupby days and find sunrise/sunset for unique days
    grp_avg = df.groupby(df.index).mean(numeric_only=False)
    date = grp_avg.index

    time_utc = ts.utc(date.year, date.month, date.day, date.hour)
    time_utc_offset = ts.utc(
        date.year, date.month, date.day + 1, date.hour
    )  # add one day for each unique day to compute sunrise and sunset pairs

    bluffton = []
    for i in range(len(grp_avg.lat)):
        bluffton.append(api.wgs84.latlon(grp_avg.lat.iloc[i], grp_avg.lon.iloc[i]))
    bluffton = np.array(bluffton)

    sunrise = []
    sunset = []
    for n in range(len(bluffton)):

        f = almanac.sunrise_sunset(eph, bluffton[n])
        t, y = almanac.find_discrete(time_utc[n], time_utc_offset[n], f)

        if not t:
            if f(time_utc[n]):  # polar day
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 0, 1
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 23, 59
                    ).to_datetime64()
                )
            else:  # polar night
                sunrise.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 11, 59
                    ).to_datetime64()
                )
                sunset.append(
                    pd.Timestamp(
                        date[n].year, date[n].month, date[n].day, 12, 1
                    ).to_datetime64()
                )

        else:
            sr = t[y == 1]  # y=1 sunrise
            sn = t[y == 0]  # y=0 sunset

            sunup = pd.to_datetime(sr.utc_iso()).tz_localize(None)
            sundown = pd.to_datetime(sn.utc_iso()).tz_localize(None)

            # this doesn't look very efficient at the moment, but I was having issues with getting the datetime64
            # to be compatible with the above code to handle polar day and polar night

            su = pd.Timestamp(
                sunup.year[0],
                sunup.month[0],
                sunup.day[0],
                sunup.hour[0],
                sunup.minute[0],
            ).to_datetime64()

            sd = pd.Timestamp(
                sundown.year[0],
                sundown.month[0],
                sundown.day[0],
                sundown.hour[0],
                sundown.minute[0],
            ).to_datetime64()

            sunrise.append(su)
            sunset.append(sd)

    sunrise = np.array(sunrise).squeeze()
    sunset = np.array(sunset).squeeze()

    grp_avg["sunrise"] = sunrise
    grp_avg["sunset"] = sunset

    # reindex days to original dataframe as night
    df_reidx = grp_avg.reindex(df.index)
    sunrise, sunset = df_reidx[["sunrise", "sunset"]].values.T

    return sunrise, sunset

def calc_DEPTH_Z(ds):
    """
    Calculate the depth (Z position) of the glider using the gsw library to convert pressure to depth.
    
    Parameters
    ----------
    ds: xarray.Dataset
        The input dataset containing 'PRES', 'LATITUDE', and 'LONGITUDE' variables.
    
    Returns
    -------
    xarray.Dataset: 
        The dataset with an additional 'DEPTH_Z' variable.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    _check_necessary_variables(ds, ['PRES', 'LONGITUDE', 'LATITUDE'])

    # Initialize the new variable with the same dimensions as dive_num
    ds['DEPTH_Z'] = (['N_MEASUREMENTS'], np.full(ds.dims['N_MEASUREMENTS'], np.nan))

    # Calculate depth using gsw
    depth = gsw.z_from_p(ds['PRES'], ds['LATITUDE'])
    ds['DEPTH_Z'] = depth

    # Assign the calculated depth to a new variable in the dataset
    ds['DEPTH_Z'].attrs = {
        "units": "meters",
        "positive": "up",
        "standard_name": "depth",
        "comment": "Depth calculated from pressure using gsw library, positive up.",
    }
    return ds

label_dict = {
    "PSAL": {
        "label": "Practical salinity",
        "units": "PSU",
        "colormap": cmo.haline
    },
    "SA": {
        "label": "Absolute salinity",
        "units": "g kg⁻¹",
        "colormap": cmo.haline  # Added best-guess
    },
    "TEMP": {
        "label": "Temperature",
        "units": "°C",
        "colormap": cmo.thermal
    },
    "THETA": {
        "label": "Potential temperature",
        "units": "°C",
        "colormap": cmo.thermal
    },
    "DENSITY": {
        "label": "In situ density",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "SIGMA": {
        "label": "Sigma-t",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "SIGMA_T": {
        "label": "Sigma-t",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "POTDENS0": {
        "label": "Potential density (σ₀)",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "SIGTHETA": {
        "label": "Potential density (σθ)",
        "units": "kg m⁻³",
        "colormap": cmo.dense
    },
    "PRES": {
        "label": "Pressure",
        "units": "dbar",
        "colormap": cmo.deep
    },
    "DEPTH_Z": {
        "label": "Depth",
        "units": "m",
        "colormap": cmo.deep
    },
    "DEPTH": {
        "label": "Depth",
        "units": "m",
        "colormap": cmo.deep
    },
    "DOXY": {
        "label": "Dissolved oxygen",
        "units": "mmol m⁻³",
        "colormap": cmo.oxy
    },
    "CHLA": {
        "label": "Chlorophyll",
        "units": "mg m⁻³",
        "colormap": cmo.algae
    },
    "CNDC": {
        "label": "Conductivity",
        "units": "mS cm⁻¹",
        "colormap": cmo.haline
    },
    "DPAR": {
        "label": "Irradiance PAR",
        "units": "μE cm⁻² s⁻¹",
        "colormap": cmo.solar  # Best guess
    },
    "BBP700": {
        "label": "Red backscatter, b${bp}$(700)",
        "units": "m⁻¹",
        "colormap": cmo.matter  # Best guess
    },
    "GLIDER_VERT_VELO_MODEL": {
        "label": "Vertical glider velocity",
        "units": "cm s⁻¹",
        "colormap": cmo.delta
    },
    "GLIDER_HORZ_VELO_MODEL": {
        "label": "Horizontal glider velocity",
        "units": "cm s⁻¹",
        "colormap": cmo.delta
    },
}

def plotting_labels(var: str):
    """
    Retrieves the label associated with a variable from a predefined dictionary.

    This function checks if the given variable `var` exists as a key in the `label_dict` dictionary.
    If found, it returns the associated label from `label_dict`. If not, it returns the variable name itself as the label.

    Parameters
    ----------
    var: str
        The variable (key) whose label is to be retrieved.

    Returns
    -------
    str: 
        The label corresponding to the variable `var`. If the variable is not found in `label_dict`,
        the function returns the variable name as the label.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    if var in label_dict:
        label = f'{label_dict[var]["label"]}'
    else:
        label= f'{var}'
    return label

def plotting_units(ds: xr.Dataset,var: str):
    """
    Retrieves the units associated with a variable from a dataset or a predefined dictionary.

    This function checks if the given variable `var` exists as a key in the `label_dict` dictionary.
    If found, it returns the associated units from `label_dict`. If not, it returns the units of the variable
    from the dataset `ds` using the `var` key.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset containing the variable `var`.
    var: str 
        The variable (key) whose units are to be retrieved.

    Returns
    -------
    str: 
        The units corresponding to the variable `var`. If the variable is found in `label_dict`,
        the associated units will be returned. If not, the function returns the units from `ds[var]`.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    if var in label_dict:
        return f'{label_dict[var]["units"]}'
    elif 'units' in ds[var].attrs:
        return f'{ds[var].units}'
    else:
        return ""
    
def plotting_colormap(var: str):
    """
    Retrieves the colormap associated with a variable from a predefined dictionary.

    This function checks if the given variable `var` exists as a key in the `label_dict` dictionary.
    If found, it returns the associated colormap from `label_dict`. If not, it returns cmocean.cm.delta as a default colormap.

    Parameters
    ----------
    var: str
        The variable (key) whose colormap is to be retrieved.

    Returns
    -------
    colormap: matplotlib colormap or None
        The colormap corresponding to the variable `var`. If the variable is not found in `label_dict`,
        the function returns None.

    Notes
    -----
    Original Author: Till Moritz
    """
    if var in label_dict:
        colormap = label_dict[var]["colormap"]
    else:
        colormap = cmo.delta
    return colormap

def group_by_profiles(ds, variables=None):
    """
    Group glider dataset by the dive profile number.

    This function groups the dataset by the `PROFILE_NUMBER` column, where each group corresponds to a single profile.
    The resulting groups can be evaluated statistically, for example using methods like `pandas.DataFrame.mean` or
    other aggregation functions. To filter a specific profile, use `xarray.Dataset.where` instead.

    Parameters
    ----------
    ds : xarray.Dataset
        A 1-dimensional glider dataset containing profile information.
    variables : list of str, optional
        A list of variable names to group by, if only a subset of the dataset should be included in the grouping.
        Grouping by a subset is more memory-efficient and faster.

    Returns
    -------
    profiles : pandas.core.groupby.DataFrameGroupBy
        A pandas `GroupBy` object, grouped by the `PROFILE_NUMBER` of the glider dataset.
        This can be further aggregated using methods like `mean` or `sum`.

    Notes
    -----
    This function is based on the original GliderTools implementation and was modified by
    Chiara Monforte to ensure compliance with the OG1 standards.
    [Source Code](https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/utils.py)
    """
    ds = ds.reset_coords().to_pandas().reset_index().set_index("PROFILE_NUMBER")
    if variables:
        return ds[variables].groupby("PROFILE_NUMBER")
    else:
        return ds.groupby("PROFILE_NUMBER")