import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from scipy import stats
from scipy.interpolate import interp1d
import cmocean.cm as cmo
import matplotlib.colors as mcolors
import gsw
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from glidertest import utilities, tools
import os
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter

dir = os.path.dirname(os.path.realpath(__file__))
glidertest_style_file = f"{dir}/glidertest.mplstyle"




def plot_updown_bias(ds: xr.Dataset, var='TEMP', v_res=1, ax: plt.Axes = None, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    This function can be used to plot the up and downcast differences computed with the updown_bias function
    
    Parameters
    ----------
    ds: xarray.Dataset 
        Dataset in **OG1 format**, containing at least **TIME, DEPTH, LATITUDE, LONGITUDE,** and the selected variable.
        Data should not be gridded.
    var: str, optional, default='TEMP'
        Selected variable
    v_res: float, default = 1
        Vertical resolution for the gridding
    ax: matplotlib.axes.Axes, default = None
        Axis to plot the data
    **kw : dict, optional  
    Additional keyword arguments passed to `matplotlib.pyplot.plot()`.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        Figure object containing the generated plot.  
    ax : matplotlib.axes.Axes  
        Axes object with the plotted **climb vs. dive differences** over depth. 

    Notes
    -----
    Original Author: Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            third_width = fig.get_size_inches()[0] / 3.11
            fig.set_size_inches(third_width, third_width * 1.1)
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        df = tools.quant_updown_bias(ds, var=var, v_res=v_res)
        if not all(hasattr(df, attr) for attr in ['dc', 'depth']):
            ax.text(0.5, 0.55, ds[var].standard_name, va='center', ha='center', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            ax.text(0.5, 0.45, 'data unavailable', va='center', ha='center', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        else:
            ax.plot(df.dc, df.depth, label='Dive-Climb', **kw)
            ax.plot(df.cd, df.depth, label='Climb-Dive', **kw)
            ax.legend(loc=3)
            lims = np.abs(df.dc)
            ax.set_xlim(-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
            ax.set_ylim(df.depth.max() + 1, -df.depth.max() / 30)
        ax.set_xlabel(f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds,var)})')
        ax.set_ylabel(f'Depth (m)')
        ax.grid()
        if force_plot:
            plt.show()
        return fig, ax


def plot_basic_vars(ds: xr.Dataset, v_res=1, start_prof=0, end_prof=-1, ax=None):
    """
    This function plots the basic oceanographic variables temperature, salinity and density. A second plot is created and filled with oxygen and 
    chlorophyll data if available.

    Parameters
    ----------

    ds: xarray 
        Dataset in **OG1 format**, containing at least **PROFILE_NUMBER, DEPTH, TEMP, PSAL, LATITUDE, LONGITUDE,** and the selected variable.
    v_res: float, default = 1
        Vertical resolution for the gridding. Horizontal resolution (by profile) is assumed to be 1
    start_prof: int
        Start profile used to compute the means that will be plotted. This can vary in case the dataset spread over a large timescale
        or region and subsections want to be plotted-1     
    end_prof: int
        End profile used to compute the means that will be plotted. This can vary in case the dataset spread over a large timescale
        or region and subsections want to be plotted-1          
    **kw : dict, optional  
        Additional keyword arguments passed to `matplotlib.pyplot.plot()`.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        Figure object containing the generated plot.  
    ax : matplotlib.axes.Axes  
        Axes object with the plotted variables **temperature, salinity and density** over depth. 
    
    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['PROFILE_NUMBER', 'DEPTH', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE'])
    ds = utilities._calc_teos10_variables(ds)
    p = 1
    z = v_res
    tempG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.TEMP, p, z,x_bin_center=False)
    salG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.PSAL, p, z,x_bin_center=False)
    denG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.DENSITY, p, z,x_bin_center=False)

    tempG = tempG[start_prof:end_prof, :]
    salG = salG[start_prof:end_prof, :]
    denG = denG[start_prof:end_prof, :]
    depthG = depthG[start_prof:end_prof, :]

    with plt.style.context(glidertest_style_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if ax is None:
                fig, ax = plt.subplots(1, 2)
                force_plot = True
            else:
                fig = plt.gcf()
                force_plot = False

            # Resize to half-width
            half_width = fig.get_size_inches()[0] / 2.07
            fig.set_size_inches(half_width, half_width * 0.85)
            ax1 = ax[0].twiny()
            ax2 = ax[0].twiny()
            ax2_1 = ax[1].twiny()
            ax2.spines["top"].set_position(("axes", 1.2))
            nticks = 5
            for a in [ax[0], ax1, ax2, ax[1], ax2_1]:
                a.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
                a.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                for tick in a.xaxis.get_majorticklabels():
                    tick.set_horizontalalignment("center")
                plt.setp(a.get_xticklabels()[0], visible=False)
                plt.setp(a.get_xticklabels()[-1], visible=False)
            ax[0].plot(np.nanmean(tempG, axis=0), depthG[0, :], c='blue')
            ax1.plot(np.nanmean(salG, axis=0), depthG[0, :], c='red')
            ax2.plot(np.nanmean(denG, axis=0) - 1000, depthG[0, :], c='black')

            ax[0].set(ylabel='Depth (m)',
                      xlabel=f'{utilities.plotting_labels("TEMP")} \n({utilities.plotting_units(ds, "TEMP")})')
            ax[0].tick_params(axis='x', colors='blue')
            ax[0].xaxis.label.set_color('blue')
            ax1.spines['bottom'].set_color('blue')
            new_font = ax[0].get_xticklabels()[0].get_fontsize()
            ax1.spines['bottom'].set_color('blue')
            ax1.set_xlabel(f'{utilities.plotting_labels("PSAL")} ({utilities.plotting_units(ds, "PSAL")})',
                           fontsize=int(new_font))
            ax1.xaxis.label.set_color('red')
            ax1.spines['top'].set_color('red')
            ax1.tick_params(axis='x', colors='red', labelsize=int(new_font))
            ax2.set_xlabel(xlabel=f'{utilities.plotting_labels("SIGMA")} ({utilities.plotting_units(ds, "SIGMA")})',
                           fontsize=int(new_font))
            ax2.xaxis.label.set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.tick_params(axis='x', colors='black', labelsize=int(new_font))

            # Add text annotation to the right, outside of the plot
            ax2.text(1.68, 1.25, f'Averaged profiles \n {start_prof}-{end_prof}', transform=ax2.transAxes,
                     verticalalignment='center', horizontalalignment='center', rotation=0, fontsize=int(new_font),
                     bbox=dict(facecolor='white', alpha=0.5))

            if 'CHLA' in ds.variables:
                chlaG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.CHLA, p, z,x_bin_center=False)
                chlaG = chlaG[start_prof:end_prof, :]
                ax2_1.plot(np.nanmean(chlaG, axis=0), depthG[0, :], c='green')
                ax2_1.set_xlabel(xlabel=f'{utilities.plotting_labels("CHLA")} ({utilities.plotting_units(ds, "CHLA")})',
                                 fontsize=int(new_font))
                ax2_1.xaxis.label.set_color('green')
                ax2_1.spines['top'].set_color('green')
                ax2_1.tick_params(axis='x', colors='green', labelsize=int(new_font))
                plt.setp(ax2_1.get_yticklabels()[0], visible=False)
                plt.setp(ax2_1.get_yticklabels()[-1], visible=False)
            else:
                plt.setp(ax2_1.get_xticklabels(), visible=False)
                ax[1].text(0.15, 0.6, 'Chlorophyll data \nunavailable', va='top', transform=ax[1].transAxes,
                           fontsize=int(new_font))

            if 'DOXY' in ds.variables:
                oxyG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds.DOXY, p, z,x_bin_center=False)
                oxyG = oxyG[start_prof:end_prof, :]
                ax[1].plot(np.nanmean(oxyG, axis=0), depthG[0, :], c='orange')
                ax[1].set(xlabel=f'{utilities.plotting_labels("DOXY")} \n({utilities.plotting_units(ds, "DOXY")})')
                ax[1].xaxis.label.set_color('orange')
                ax[1].spines['top'].set_color('orange')
                ax[1].tick_params(axis='x', colors='orange')
                ax[1].spines['bottom'].set_color('orange')
            else:
                plt.setp(ax[1].get_xticklabels(), visible=False)
                ax[1].text(0.2, 0.4, 'Oxygen data \nunavailable', va='top', transform=ax[1].transAxes,
                           fontsize=int(new_font))

            [a.set_ylim(depthG.max(), 0) for a in ax]
            [a.grid() for a in ax]
            if force_plot:
                plt.show()
    return fig, ax


def process_optics_assess(ds, var='CHLA'):
    """
    Function to assess visually any drift in deep optics data and the presence of any possible negative data. This function returns  both plots and text
    
    Parameters
    ----------
    ds: xarray.Dataset
        Dataset in **OG1 format**, containing at least **TIME, DEPTH,** and the selected variable.
    var: str, optional, default='CHLA'
        Selected variable        
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the generated plot.
    ax : matplotlib.axes.Axes
        Axes object with the plotted data.
    text_output : str  
        Text displaying where and when negative values are present in the dataset.

    Returns
    -------
    fig, ax : matplotlib.axes.Axes
        Axes object with 
    text_output: str
        Text giving info on where and when negative data was observed

    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, [var, 'TIME', 'DEPTH'])
    # Check how much negative data there is
    neg_chl = np.round((len(np.where(ds[var] < 0)[0]) * 100) / len(ds[var]), 1)
    if neg_chl > 0:
        print(f'{neg_chl}% of scaled {var} data is negative, consider recalibrating data')
        # Check where the negative values occur and if we just see them at specific time of the mission or not
        start = ds.TIME[np.where(ds[var] < 0)][0]
        end = ds.TIME[np.where(ds[var] < 0)][-1]
        min_z = ds.DEPTH[np.where(ds[var] < 0)].min().values
        max_z = ds.DEPTH[np.where(ds[var] < 0)].max().values
        print(f'Negative data in present from {str(start.values)[:16]} to {str(end.values)[:16]}')
        print(f'Negative data is present between {"%.1f" % np.round(min_z, 1)} and {"%.1f" % np.round(max_z, 1)} ')
    else:
        print(f'There is no negative scaled {var} data, recalibration and further checks are still recommended ')
    # Check if there is any missing data throughout the mission
    if len(ds.TIME) != len(ds[var].dropna(dim='N_MEASUREMENTS').TIME):
        print(f'{var} data is missing for part of the mission')  # Add to specify where the gaps are
    else:
        print(f'{var} data is present for the entire mission duration')
    # Check bottom dark count and any drift there
    bottom_opt_data = ds[var].where(ds[var].DEPTH > ds.DEPTH.max() - (ds.DEPTH.max() * 0.1)).dropna(
        dim='N_MEASUREMENTS')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, len(bottom_opt_data)), bottom_opt_data)

    # Generate the plot
    with plt.style.context(glidertest_style_file):
        fig, ax = plt.subplots()  # Create figure and axes

        sns.regplot(
            data=ds,
            x=np.arange(0, len(bottom_opt_data)),
            y=bottom_opt_data,
            scatter_kws={"color": "grey"},
            line_kws={"color": "red", "label": f"y={slope:.8f} x+{intercept:.5f}"},
            ax=ax
        )

        ax.legend(loc=2)
        ax.grid()
        ax.set(
            ylim=(np.nanpercentile(bottom_opt_data, 0.5), np.nanpercentile(bottom_opt_data, 99.5)),
            xlabel='Measurements',
            ylabel=f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds, var)})'
        )
        plt.show()
    percentage_change = (((slope * len(bottom_opt_data) + intercept) - intercept) / abs(intercept)) * 100

    if abs(percentage_change) >= 1:
        print(
            'Data from the deepest 10% of data has been analysed and data does not seem perfectly stable. An alternative solution for dark counts has to be considered. \nMoreover, it is recommended to check the sensor has this may suggest issues with the sensor (i.e water inside the sensor, temporal drift etc)')
        print(
            f'Data changed (increased or decreased) by {"%.1f" % np.round(percentage_change, 1)}% from the beginning to the end of the mission')
    else:
        print(
            f'Data from the deepest 10% of data has been analysed and data seems stable. These deep values can be used to re-assess the dark count if the no {var} at depth assumption is valid in this site and this depth')
    return fig, ax


def plot_daynight_avg(ds,var='PSAL', ax: plt.Axes = None, sel_day=None, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    This function can be used to plot the day and night averages computed with the day_night_avg function
    
    Parameters
    ----------
    ds: xarray.Dataset
        Dataset in **OG1 format**, containing at least **TIME, DEPTH,** and the selected variable.
    var: str, optional, default='PSAL'
        Selected variable  
    ax: matplotlib.axes.Axes, default = None
        Axis to plot the data
    sel_day : str or None, optional, default = `None`  
        The specific day to plot. If `None`, the function selects the **median day** in the dataset.
    
    Returns
    -------
    fig : matplotlib.figure.Figure  
        The generated **figure** containing the plot.  
    ax : matplotlib.axes.Axes  
        The **axes** object with the plotted data.

    Notes
    -----
    Original Author: Chiara Monforte

    """
    day, night = tools.compute_daynight_avg(ds, sel_var=var,start_time=sel_day,end_time=sel_day)
    if not sel_day:
        dates = list(day.date.dropna().values) + list(night.date.dropna().values)
        dates.sort()
        sel_day = dates[int(len(dates)/2)]
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax.plot(night.where(night.date == sel_day).dropna().dat, night.where(night.date == sel_day).dropna().depth,
                label='Night time average')
        ax.plot(day.where(day.date == sel_day).dropna().dat, day.where(day.date == sel_day).dropna().depth,
                label='Daytime average')
        ax.legend()
        ax.invert_yaxis()
        ax.grid()
        ax.set(xlabel=f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds,var)})', ylabel='Depth (m)')
        ax.set_title(sel_day)
        if force_plot:
            plt.show()
    return fig, ax


def plot_quench_assess(ds: xr.Dataset, sel_var: str, ax: plt.Axes = None, start_time=None,
                           end_time=None,start_prof=None, end_prof=None, ylim=35, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    Generates a **section plot** of the selected variable over time and depth,  
    with **sunrise and sunset** times marked to assess potential Non-Photochemical Quenching (NPQ) effects.  

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **TIME, DEPTH, LATITUDE, LONGITUDE**, and the selected variable.
        Data **should not** be gridded.  
    sel_var : str  
        The selected variable to plot.  
    ax : matplotlib.axes.Axes, optional, default = `None`  
        Axis on which to plot the data. If `None`, a new figure and axis will be created.  
    start_time : str or None, optional, default = `None`  
        Start date for the data selection (`'YYYY-MM-DD'`).  
        - If `None`, defaults to **2 days before** the dataset’s median date.  
    end_time : str or None, optional, default = `None`  
        End date** for the data selection (`'YYYY-MM-DD'`).  
        - If `None`, defaults to **2 days after** the dataset’s median date.  
    start_prof : int or None, optional, default = `None`  
        Start profile number for data selection
    end_prof : int or None, optional, default = `None`  
        End profile number for data selection 
    ylim : float, optional, default = `35`  
        Maximum depth limit for the y-axis. The **minimum limit** is automatically set as `ylim / 30`.  
    
    Returns
    -------
    fig : matplotlib.figure.Figure  
        The generated **figure** containing the plot.  
    ax : matplotlib.axes.Axes  
        The **axes** object with the plotted data.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['TIME', sel_var, 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            full_width = fig.get_size_inches()[0] 
            fig.set_size_inches(full_width, full_width * 0.5)
        else:
            fig = plt.gcf()

        if "TIME" not in ds.indexes.keys():
            ds = ds.set_xindex('TIME')

        if not start_time:
            start_time = ds.TIME.mean() - np.timedelta64(2, 'D')
        if not end_time:
            end_time = ds.TIME.mean() + np.timedelta64(2, 'D')

        if start_prof and end_prof:
            t1 = ds.TIME.where(ds.PROFILE_NUMBER==start_prof).dropna(dim='N_MEASUREMENTS')[0]
            t2 = ds.TIME.where(ds.PROFILE_NUMBER==end_prof).dropna(dim='N_MEASUREMENTS')[-1]
            ds_sel = ds.sel(TIME=slice(t1,t2))
        else:
            ds_sel = ds.sel(TIME=slice(start_time, end_time))


        if len(ds_sel.TIME) == 0:
            msg = f"supplied limits start_time: {start_time} end_time: {end_time} do not overlap with dataset TIME range {str(ds.TIME.values.min())[:10]} - {str(ds.TIME.values.max())[:10]}"
            raise ValueError(msg)

        sunrise, sunset = utilities.compute_sunset_sunrise(ds_sel.TIME, ds_sel.LATITUDE, ds_sel.LONGITUDE)
        surf_chla= ds_sel[sel_var].where(ds_sel[sel_var].DEPTH < ylim).dropna(dim='N_MEASUREMENTS')
        logchla=np.log10(surf_chla.where((surf_chla>0)).dropna(dim='N_MEASUREMENTS'))
        c = ax.scatter(logchla.TIME, logchla.DEPTH, c=logchla, s=10, vmin=np.nanpercentile(logchla, 0.5),
                   vmax=np.nanpercentile(logchla, 99.5))
        ax.set_ylim(ylim, -ylim / 30)
        for n in np.unique(sunset):
            ax.axvline(np.unique(n), c='blue')
        for m in np.unique(sunrise):
            ax.axvline(np.unique(m), c='orange')
        ax.set_ylabel('Depth (m)')
    
        # Set x-tick labels based on duration of the selection
        # Could pop out as a utility plotting function?
        utilities._time_axis_formatter(ax, ds_sel, format_x_axis=True)
    
        plt.colorbar(c, label=f'log₁₀({utilities.plotting_labels(sel_var)}) ({utilities.plotting_units(ds,sel_var)})')
        plt.show()
    return fig, ax


def check_temporal_drift(ds: xr.Dataset, var: str, ax: plt.Axes = None, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):
    """
    Assesses potential **temporal drift** in a selected variable by generating a figure with two subplots:  
    1. **Time series scatter plot** of the variable.  
    2. **Depth profile scatter plot**, colored by date.
    
    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **TIME, DEPTH,** and the selected variable.  
        Data **should not be gridded**.  
    var : str, optional  
        Selected variable to analyze.
    ax : matplotlib.axes.Axes, optional  
        Axis to plot the data. If None, a new figure and axis will be created.  
    **kw : dict  
        Additional keyword arguments for customization.
    
    Returns
    -------
    fig : matplotlib.figure.Figure  
        The generated figure.  
    ax : matplotlib.axes.Axes  
        Axes containing the plots. 

    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['TIME', var, 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax[0].scatter(mdates.date2num(ds.TIME), ds[var], s=10)
        # Set x-tick labels based on duration of the selection
        utilities._time_axis_formatter(ax[0], ds, format_x_axis=True)

        ax[0].set(ylim=(np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99)), ylabel=f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds,var)})')

        c = ax[1].scatter(ds[var], ds.DEPTH, c=mdates.date2num(ds.TIME), s=10)
        ax[1].set(xlim=(np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99)), ylabel='Depth (m)', xlabel=f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds,var)})')
        ax[1].invert_yaxis()
        [a.grid() for a in ax]

        color_array = c.get_array()
        formatter, locator, _ = utilities._select_time_formatter_and_locator(ds)

        if locator is not None:
            # Set vmin/vmax from ticks
            vmin_dt = mdates.num2date(color_array.min())
            vmax_dt = mdates.num2date(color_array.max())
            ticks = locator.tick_values(vmin_dt, vmax_dt)

            # Filter to ticks in data range
            color_min = color_array.min()
            color_max = color_array.max()
            ticks = [tick for tick in ticks if color_min <= tick <= color_max]

            # ✅ Pass vmin/vmax to scatter!
            c.set_clim(vmin=min(ticks), vmax=max(ticks))

        # Now plot colorbar
        colorbar = plt.colorbar(c, format=formatter)

        # And set ticks
        if locator is not None and ticks:
            colorbar.set_ticks(ticks)
        if force_plot:
            plt.show()
    return fig, ax


def plot_prof_monotony(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict, ) -> tuple({plt.Figure, plt.Axes}):

    """
    Checks for potential issues with the assigned **profile index** by plotting its progression over time.  

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **TIME, DEPTH,** and **PROFILE_NUMBER**.  
        Data **should not be gridded**.  
    ax : matplotlib.axes.Axes, optional  
        Axis to plot the data. If None, a new figure and axis will be created.  
    **kw : dict  
        Additional keyword arguments for customization. 

    Returns 
    -------
    fig : matplotlib.figure.Figure  
        The generated figure.  
    ax : matplotlib.axes.Axes  
        Axes containing the two plots.
        - 1. One line plot with the profile number over time (expected to be always increasing).
        - 2. Scatter plot showing at which depth over time there was a profile index where the
        difference was neither 0 nor 1 (meaning there are possibly issues with how the profile index was assigned).

    Notes
    -----
    Original Author: Chiara Monforte
    """
    utilities._check_necessary_variables(ds, ['TIME', 'PROFILE_NUMBER', 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True)
        else:
            fig = plt.gcf()

        ax[0].plot(ds.TIME, ds.PROFILE_NUMBER)

        ax[0].set(ylabel='Profile number')
        if len(np.where((np.diff(ds.PROFILE_NUMBER) != 0) & (np.diff(ds.PROFILE_NUMBER) != 1))[0]) == 0:
            ax[1].text(0.2, 0.5, 'Data are monotonically increasing - no issues identified',
                       transform=ax[1].transAxes)
        else:
            ax[1].scatter(ds.TIME[np.where((np.diff(ds.PROFILE_NUMBER) != 0) & (np.diff(ds.PROFILE_NUMBER) != 1))],
                          ds.DEPTH[np.where((np.diff(ds.PROFILE_NUMBER) != 0) & (np.diff(ds.PROFILE_NUMBER) != 1))],
                          s=10, c='red', label='Depth at which we have issues \n with the profile number assigned')
            ax[1].legend()
        ax[1].set(ylabel='Depth (m)')
        ax[1].invert_yaxis()
        ax[1].xaxis.set_major_locator(plt.MaxNLocator(8))
        utilities._time_axis_formatter(ax[1], ds, format_x_axis=True)
        [a.grid() for a in ax]
        plt.show()
    return fig, ax


def plot_glider_track(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    Plots the **glider track** on a map, with latitude and longitude colored by time.  

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **TIME, LATITUDE,** and **LONGITUDE**.  
    ax : matplotlib.axes.Axes, optional  
        Axis to plot the data. If None, a new figure and axis will be created.  
    **kw : dict  
        Additional keyword arguments for the scatter plot.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        The generated figure.  
    ax : matplotlib.axes.Axes  
        Axes containing the plot.

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['TIME', 'LONGITUDE', 'LATITUDE'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        else:
            fig = plt.gcf()

        latitudes = ds.LATITUDE.values
        longitudes = ds.LONGITUDE.values
        times = ds.TIME.values

        # Drop NaN values
        valid_indices = ~np.isnan(latitudes) & ~np.isnan(longitudes) & ~np.isnan(times)
        latitudes = latitudes[valid_indices]
        longitudes = longitudes[valid_indices]
        times = times[valid_indices]

        # Reduce the number of latitudes, longitudes, and times to no more than 2000
        if len(latitudes) > 2000:
            indices = np.linspace(0, len(latitudes) - 1, 2000).astype(int)
            latitudes = latitudes[indices]
            longitudes = longitudes[indices]
            times = times[indices]

        # Convert time to matplotlib numeric format
        numeric_times = mdates.date2num(times)

        # Get formatter and locator
        formatter, locator, _ = utilities._select_time_formatter_and_locator(ds)

       # Precompute ticks BEFORE scatter plot
        if locator is not None:
            vmin_dt = mdates.num2date(numeric_times.min())
            vmax_dt = mdates.num2date(numeric_times.max())
            ticks = locator.tick_values(vmin_dt, vmax_dt)

            # Filter ticks within data range
            ticks = [tick for tick in ticks if numeric_times.min() <= tick <= numeric_times.max()]
        else:
            ticks = []

        # Create scatter plot with vmin/vmax from ticks
        vmin = min(ticks) if ticks else numeric_times.min()
        vmax = max(ticks) if ticks else numeric_times.max()

        sc = ax.scatter(
            longitudes,
            latitudes,
            c=numeric_times,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            **kw
        )

        # Create colorbar with formatter
        cbar = plt.colorbar(sc, ax=ax, format=formatter)

        # Apply ticks to colorbar
        if locator is not None and ticks:
            cbar.set_ticks(ticks)

        ax.set_extent([np.min(longitudes)-1, np.max(longitudes)+1, np.min(latitudes)-1, np.max(latitudes)+1], crs=ccrs.PlateCarree())

        # Add features to the map
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)

        ax.set_xlabel(f'Longitude')
        ax.set_ylabel(f'Latitude')
        ax.set_title('Glider Track')
        gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        plt.show()

    return fig, ax

def plot_grid_spacing(ds: xr.Dataset, ax: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    Plots **histograms of the grid spacing**, showing depth and time differences while excluding the outer 1% of values.  

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **TIME** and **DEPTH**.  
    ax : matplotlib.axes.Axes, optional  
        Axis to plot the data. If None, a new figure and axis will be created.  
    **kw : dict  
        Additional keyword arguments for the histograms.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        The generated figure.  
    ax : matplotlib.axes.Axes  
        Axes containing the histograms. 

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['TIME', 'DEPTH'])
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots(1, 2)
            # Set aspect ration of plot to be 2:1
            fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[0] / 2)
        else:
            fig = plt.gcf()
        # Set font sizes for all annotations
        #def_font_size = 14

        # Calculate the depth and time differences
        depth_diff = np.diff(ds.DEPTH)
        orig_time_diff = np.diff(ds.TIME) / np.timedelta64(1, 's')  # Convert to seconds

        # Remove NaN values
        depth_diff = depth_diff[np.isfinite(depth_diff)]
        time_diff = orig_time_diff[np.isfinite(orig_time_diff)]

        # Calculate some statistics (using original data)
        median_neg_depth_diff = np.median(depth_diff[depth_diff < 0])
        median_pos_depth_diff = np.median(depth_diff[depth_diff > 0])
        max_depth_diff = np.max(depth_diff)
        min_depth_diff = np.min(depth_diff)

        median_time_diff = np.median(orig_time_diff)
        mean_time_diff = np.mean(orig_time_diff)
        max_time_diff = np.max(orig_time_diff)
        min_time_diff = np.min(orig_time_diff)
        max_time_diff_hrs = max_time_diff/3600

        # Remove the top and bottom 0.5% of values to get a better histogram
        # This is hiding some data from the user
        depth_diff = depth_diff[(depth_diff >= np.nanpercentile(depth_diff, 0.5)) & (depth_diff <= np.nanpercentile(depth_diff, 99.5))]
        time_diff = time_diff[(time_diff >= np.nanpercentile(time_diff, 0.5)) & (time_diff <= np.nanpercentile(time_diff, 99.5))]
        print('Depth and time differences have been filtered to the middle 99% of values.')
        print('Numeric median/mean/max/min values are based on the original data.')

        # Histogram of depth spacing
        ax[0].hist(depth_diff, bins=50, **kw)
        ax[0].set_xlabel('Depth Spacing (m)')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Histogram of Depth Spacing')

        annotation_text_left = (
            f'Median Negative: {median_neg_depth_diff:.2f} m\n'
            f'Median Positive: {median_pos_depth_diff:.2f} m\n'
            f'Max: {max_depth_diff:.2f} m\n'
            f'Min: {min_depth_diff:.2f} m'
        )
        # Determine the best location for the annotation based on the x-axis limits
        x_upper_limit = ax[0].get_xlim()[1]
        x_lower_limit = ax[0].get_xlim()[0]
        if abs(x_lower_limit) > abs(x_upper_limit):
            annotation_loc = (0.04, 0.96)  # Top left
            ha = 'left'
        else:
            annotation_loc = (0.96, 0.96)  # Top right
            ha = 'right'
        ax[0].annotate(annotation_text_left, xy=annotation_loc, xycoords='axes fraction',
                       ha=ha, va='top',
                       bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=.5))

        # Histogram of time spacing
        ax[1].hist(time_diff, bins=50, **kw)
        ax[1].set_xlabel('Time Spacing (s)')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Histogram of Time Spacing')

        annotation_text = (
            f'Median: {median_time_diff:.2f} s\n'
            f'Mean: {mean_time_diff:.2f} s\n'
            f'Max: {max_time_diff:.2f} s ({max_time_diff_hrs:.2f} hr)\n'
            f'Min: {min_time_diff:.2f} s'
        )
        ax[1].annotate(annotation_text, xy=(0.96, 0.96), xycoords='axes fraction'
                       , ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=.5))

        # Set font sizes for all annotations
        # Font size 14 looks roughly like fontsize 8 when I drop this figure in Word - a bit small
        # Font size 14 looks like fontsize 13 when I drop the top *half* of this figure in powerpoint - acceptable
        for axes in ax:
            axes.tick_params(axis='both', which='major')
            # More subtle grid lines
            axes.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
        plt.show()

    return fig, ax

def plot_sampling_period_all(ds: xr.Dataset) -> tuple({plt.Figure, plt.Axes}):
    """
    Plots **histograms of the sampling period** for multiple variables.  
    By default, it includes **TEMP** and **PSAL**, with **DOXY** and **CHLA** added if present in the dataset.  

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        The generated figure.  
    ax : matplotlib.axes.Axes  
        Axes containing the histograms.

    Notes
    -----
    Original Author: Louis Clement
    """

    count_vars=2
    variables=['TEMP', 'PSAL']
    if 'DOXY' in set(ds.variables): 
        count_vars+=1
        variables.append('DOXY')
    if 'CHLA' in set(ds.variables): 
        count_vars+=1
        variables.append('CHLA')

    fig, ax = plt.subplots(1, count_vars, figsize=(5*count_vars, 6))
    for i in range(len(variables)):
        ax[i] = plot_sampling_period(ds, ax[i], variables[i])
    plt.show()

    return fig, ax

def plot_sampling_period(ds: xr.Dataset, ax: plt.Axes = None, variable='TEMP'):
    """
    Plots a **histogram of the sampling period** for a selected variable after removing NaN values.

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**.  
    ax : matplotlib.axes.Axes, optional  
        Axis to plot the data. If not provided, a new figure and axis are created.  
    variable : str, default='TEMP'  
        Variable for which the sampling period is displayed.  

    Returns
    -------
    ax : matplotlib.axes.Axes  
        Axis containing the histogram. 

    Notes
    -----
    Original Author: Louis Clement
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    nonan = ~np.isnan(ds[variable].values)
    time_diff = np.diff(ds.TIME.values[nonan]) / np.timedelta64(1, 's')  # Convert to seconds

    median_time_diff = np.median(time_diff)
    mean_time_diff = np.mean(time_diff)
    max_time_diff = np.max(time_diff)
    min_time_diff = np.min(time_diff)
    max_time_diff_hrs = max_time_diff/3600

    # Remove the top and bottom 0.5% of values to get a better histogram
    # This is hiding some data from the user
    time_diff = time_diff[(time_diff >= np.nanpercentile(time_diff, 0.5)) & (time_diff <= np.nanpercentile(time_diff, 99.5))]
    if variable=='TEMP': 
        print('Depth and time differences have been filtered to the middle 99% of values.')
        print('Numeric median/mean/max/min values are based on the original data.')

    ax.hist(time_diff, bins=50)
    ax.set_xlabel('Time Spacing (s)')
    if variable=='TEMP': ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Sampling Period' + '\n' + 
                 'for ' + utilities.plotting_labels(variable) + ', \n' +
                 'valid values: {:.1f}'.format(100*(np.sum(nonan)/ds.TIME.values.shape[0]))+'%')

    annotation_text = (
        f'Median: {median_time_diff:.2f} s\n'
        f'Mean: {mean_time_diff:.2f} s\n'
        f'Max: {max_time_diff:.2f} s ({max_time_diff_hrs:.2f} hr)\n'
        f'Min: {min_time_diff:.2f} s'
    )
    ax.annotate(annotation_text, xy=(0.96, 0.96), xycoords='axes fraction'
                    , ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=.5))

    ax.tick_params(axis='both', which='major')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')

    return ax

def plot_ts(ds: xr.Dataset, percentile: list = [0.5,99.5], axs: plt.Axes = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    Plots **temperature and salinity distributions** using histograms and a 2D density plot.

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset in **OG1 format**, containing at least **DEPTH, LONGITUDE, LATITUDE, TEMP and PSAL**. 
    percentile : list, optional
        The percentiles to use for filtering the data. Default is [0.5, 99.5].
    axs : matplotlib.axes.Axes, optional  
        Axes to plot the data. If not provided, a new figure and axes are created.  
    kw : dict, optional  
        Additional keyword arguments for the histograms.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        Figure containing the plots.  
    axs : matplotlib.axes.Axes  
        Axes containing the histograms.  
        - **Three plots are created:**  
            1. **Histogram of Conservative Temperature** (middle 99%)  
            2. **Histogram of Absolute Salinity** (middle 99%)  
            3. **2D Histogram of Salinity vs. Temperature** with density contours 

    Notes
    -----
    Original Author: Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['DEPTH', 'LONGITUDE', 'LATITUDE', 'PSAL', 'TEMP'])
    with plt.style.context(glidertest_style_file):
        if axs is None:
            fig, ax = plt.subplots(2, 3)
            plt.subplots_adjust(wspace=0.03, hspace=0.03)
            force_plot = True
            axs = ax.flatten()
        else:
            fig = plt.gcf()
            plt.subplots_adjust(wspace=0.03, hspace=0.03)
            force_plot = False
        axs[3].set_visible(False)
        axs[5].set_visible(False)
        num_bins = 30

        # Create a mask once for valid data points
        mask = np.isfinite(ds.TEMP.values) & np.isfinite(ds.PSAL.values)
        temp, sal, depth, long, lat = (ds[var].values[mask] for var in ['TEMP', 'PSAL', 'DEPTH', 'LONGITUDE', 'LATITUDE'])

        # Convert to Absolute Salinity (SA) and Conservative Temperature (CT)
        SA = gsw.SA_from_SP(sal, depth, long, lat)
        CT = gsw.CT_from_t(SA, temp, depth)

        # Filter within percentile range
        p_low, p_high = np.nanpercentile(CT, percentile), np.nanpercentile(SA, percentile)
        CT_filtered, SA_filtered = CT[(p_low[0] <= CT) & (CT <= p_low[1])], SA[(p_high[0] <= SA) & (SA <= p_high[1])]
        print(f"Filtered values between {percentile[0]}% and {percentile[1]}% percentiles.")

        # Generate density contours efficiently
        xi, yi = np.meshgrid(np.linspace(SA_filtered.min() - 0.2, SA_filtered.max() + 0.2, 100),
                             np.linspace(CT_filtered.min() - 0.2, CT_filtered.max() + 0.2, 100))
        zi = gsw.sigma0(xi.ravel(), yi.ravel()).reshape(xi.shape)

        # Temperature Histogram
        axs[0].hist(CT_filtered, bins=num_bins, orientation="horizontal", **kw)
        axs[0].set(ylabel='Conservative Temperature (°C)', xlabel='Frequency')
        axs[0].invert_xaxis()

        # Salinity Histogram
        axs[4].hist(SA_filtered, bins=num_bins, **kw)
        axs[4].set(xlabel='Absolute Salinity', ylabel='Frequency')
        axs[4].yaxis.set_label_position("right")
        axs[4].yaxis.tick_right()
        axs[4].invert_yaxis()

        for tick in axs[1].xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in axs[1].yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

        # 2-d T-S histogram
        h = axs[1].hist2d(SA_filtered, CT_filtered, bins=num_bins, cmap='viridis', norm=mcolors.LogNorm(), **kw)
        axs[1].contour(xi, yi, zi, colors='black', alpha=0.5, linewidths=0.5)
        axs[1].clabel(axs[1].contour(xi, yi, zi, colors='black', alpha=0.5, linewidths=0.5), inline=True)
        cbar = fig.colorbar(h[3], orientation='vertical',cax=axs[2])
        cbar.set_label('Log Counts')
        axs[1].set_title('2D Histogram \n (Log Scale)')
        #Resize axs[2] as colorbar
        box2 = axs[2].get_position()
        axs[2].set_position([box2.x0, box2.y0, box2.width / 6, box2.height])
        # Set x-limits based on salinity plot and y-limits based on temperature plot
        axs[1].set_xlim(axs[4].get_xlim())
        axs[1].set_ylim(axs[0].get_ylim())

        # Set font sizes for all annotations
        for axes in [axs[0],axs[1],axs[4]]:
            axes.tick_params(axis='both', which='major')
            axes.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
        if force_plot:
            plt.show()
        all_ax = axs
        return fig, all_ax
    
def plot_vertical_speeds_with_histograms(ds, start_prof=None, end_prof=None):
    """
    Plots vertical glider speeds with histograms for diagnostic purposes.

    This function generates a **diagnostic plot** for vertical velocity calculations, comparing:
    1. **Modeled glider vertical speed** (flight model)
    2. **Computed glider speed** (from dz/dt, pressure sensor)
    3. **Implied seawater velocity** (difference between the two)

    The histograms allow for identifying biases, and the **median of the seawater velocity** should be close to zero if:
    - A large enough sample of dives is used.
    - The glider flight model is well-tuned.

    Parameters
    ----------
    ds : xarray.Dataset  
        Dataset containing vertical speed data with required variables:  
        - **VERT_GLIDER_SPEED**: Modeled glider speed  
        - **VERT_SPEED_DZDT**: Computed glider speed from pressure sensor  
        - **VERT_SW_SPEED**: Implied seawater velocity  
    start_prof : int, optional  
        Starting profile number for subsetting. Defaults to the **first profile number**.  
    end_prof : int, optional  
        Ending profile number for subsetting. Defaults to the **last profile number**.  

    Returns
    -------
    fig : matplotlib.figure.Figure  
        The figure containing the plots.  
    axs : tuple(matplotlib.axes.Axes)  
        The axes objects for the plots.

    Notes
    ------
    Original Author: Eleanor Frajka-Williams
    """
    utilities._check_necessary_variables(ds, ['GLIDER_VERT_VELO_MODEL', 'GLIDER_VERT_VELO_DZDT', 'VERT_CURR_MODEL','PROFILE_NUMBER'])
    with plt.style.context(glidertest_style_file):
        if start_prof is None:
            start_prof = int(ds['PROFILE_NUMBER'].values.mean())-10

        if end_prof is None:
            end_prof = int(ds['PROFILE_NUMBER'].values.mean())+10

        ds = ds.where((ds['PROFILE_NUMBER'] >= start_prof) & (ds['PROFILE_NUMBER'] <= end_prof), drop=True)
        vert_curr = ds.VERT_CURR_MODEL.values * 100  # Convert to cm/s
        vert_dzdt = ds.GLIDER_VERT_VELO_DZDT.values * 100  # Convert to cm/s
        vert_model = ds.GLIDER_VERT_VELO_MODEL.values * 100  # Convert to cm/s

        # Calculate the median line for the lower right histogram
        median_vert_sw_speed = np.nanmedian(vert_curr)

        # Create a dictionary to map the variable names to their labels for legends
        labels_dict = {
            'vert_dzdt': 'w$_{meas}$ (from dz/dt)',
            'vert_model': 'w$_{model}$ (flight model)',
            'vert_curr': 'w$_{sw}$ (calculated)'
        }

        fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 1]})

        # Upper left subplot for vertical velocity and glider speed
        ax1 = axs[0, 0]
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)  # Add zero horizontal line
        ax1.plot(ds['TIME'], vert_dzdt, label=labels_dict['vert_dzdt'])
        ax1.plot(ds['TIME'], vert_model, color='r', label=labels_dict['vert_model'])
        ax1.plot(ds['TIME'], vert_curr, color='g', label=labels_dict['vert_curr'])
        # Annotations
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Vertical Velocity (cm/s)')
        ax1.legend(loc='lower left')
        ax1.legend(loc='lower right')
        utilities._time_axis_formatter(ax1, ds, format_x_axis=True)

        # Upper right subplot for histogram of vertical velocity
        ax1_hist = axs[0, 1]
        ax1_hist.hist(vert_dzdt, bins=50, orientation='horizontal', alpha=0.5, color='blue', label=labels_dict['vert_dzdt'])
        ax1_hist.hist(vert_model, bins=50, orientation='horizontal', alpha=0.5, color='red', label=labels_dict['vert_model'])
        ax1_hist.hist(vert_curr, bins=50, orientation='horizontal', alpha=0.5, color='green', label=labels_dict['vert_curr'])
        ax1_hist.set_xlabel('Frequency')

        # Determine the best location for the legend based on the y-axis limits and zero
        y_upper_limit = ax1_hist.get_ylim()[1]
        y_lower_limit = ax1_hist.get_ylim()[0]
        if abs(y_upper_limit) > abs(y_lower_limit):
            legend_loc = 'upper right'
        else:
            legend_loc = 'lower right'
        plt.rcParams['legend.fontsize'] = 12
        ax1_hist.legend(loc=legend_loc)
        plt.rcParams['legend.fontsize'] = 15
        # Lower left subplot for vertical water speed
        ax2 = axs[1, 0]
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)  # Add zero horizontal line
        ax2.plot(ds['TIME'], vert_curr, 'g', label=labels_dict['vert_curr'])
        # Annotations
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Vertical Water Speed (cm/s)')
        ax2.legend(loc='upper left')
        utilities._time_axis_formatter(ax2, ds, format_x_axis=True)

        # Lower right subplot for histogram of vertical water speed
        ax2_hist = axs[1, 1]
        ax2_hist.hist(vert_curr, bins=50, orientation='horizontal', alpha=0.5, color='green', label=labels_dict['vert_curr'])
        ax2_hist.axhline(median_vert_sw_speed, color='red', linestyle='dashed', linewidth=1, label=f'Median: {median_vert_sw_speed:.2f} cm/s')
        ax2_hist.set_xlabel('Frequency')

        # Determine the best location for the legend based on the y-axis limits and median
        y_upper_limit = ax2_hist.get_ylim()[1]
        y_lower_limit = ax2_hist.get_ylim()[0]
        if abs(y_upper_limit - median_vert_sw_speed) > abs(y_lower_limit - median_vert_sw_speed):
            legend_loc = 'upper right'
        else:
            legend_loc = 'lower right'
        ax2_hist.legend(loc=legend_loc)

        # Set font sizes for all annotations
        # Font size 14 looks roughly like fontsize 8 when I drop this figure in Word - a bit small
        # Font size 14 looks like fontsize 13 when I drop the top *half* of this figure in powerpoint - acceptable

        for ax in [ax1, ax2, ax1_hist, ax2_hist]:
            ax.tick_params(axis='both', which='major')

        # Adjust the axes so that the distance between y-ticks on the top and lower panel is the same
        # Get the y-axis range of the top left plot
        y1_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
        # Get the y-axis limits of the lower left plot
        y2_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        # Get the height in inches of the top left plot
        box1 = ax1.get_position()
        height1 = box1.height
        # Get the height in inches of the lower left plot
        box2 = ax2.get_position()
        height2 = box2.height
        # Set a scaled height for the lower left plot
        new_height = height1 * y2_range / y1_range
        # Determine the change in height
        height_change = height2 - new_height
        # Shift the y-position of the lower left plot by the change in height
        ax2.set_position([box2.x0, box2.y0 + height_change, box2.width, new_height])

        # Get the position of the lower right plot
        box2_hist = ax2_hist.get_position()
        # Adjust the position of the lower right plot to match the height of the lower left plot
        ax2_hist.set_position([box2_hist.x0, box2_hist.y0 + height_change, box2_hist.width, new_height])

        # Find the distance between the right edge of the top left plot and the left edge of the top right plot
        box1_hist = ax1_hist.get_position()
        distance =  box1_hist.x0 - (box1.x0 + box1.width)
        shift_dist = distance/3 # Not sure this will always work; it may depend on the def_fault_size
        # Adjust the width of the top right plot to extend left by half the distance
        ax1_hist.set_position([box1_hist.x0 - shift_dist, box1_hist.y0, box1_hist.width + shift_dist, box1_hist.height])
        # Adjust the width of the bottom right plot to extend left by half the distance
        box2_hist = ax2_hist.get_position()
        ax2_hist.set_position([box2_hist.x0 - shift_dist, box2_hist.y0, box2_hist.width + shift_dist, box2_hist.height])

    plt.show()

    return fig, axs

def plot_combined_velocity_profiles(ds_out_dives: xr.Dataset, ds_out_climbs: xr.Dataset):
    """
    Plots combined vertical velocity profiles for dives and climbs.

    Replicates Fig 3 in Frajka-Williams et al. 2011, but using an updated dataset from Jim Bennett (2013), 
    now in OG1 format as sg014_20040924T182454_delayed.nc.  Note that flight model parameters may differ from those in the paper.

    The function converts vertical velocities from m/s to cm/s, plots the mean vertical velocities and their ranges for both dives and climbs,
    and customizes the plot with labels, legends, and axis settings.

    Parameters
    ----------
    ds_out_dives : xarray.Dataset
        Dataset containing dive profiles with variables:
        - 'zgrid': Depth grid (meters)
        - 'meanw': Mean vertical velocity (m/s)
        - 'w_lower': Lower bound of velocity range (m/s)
        - 'w_upper': Upper bound of velocity range (m/s)
    
    ds_out_climbs : xarray.Dataset
        Dataset containing climb profiles with the same variables as `ds_out_dives`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object for the plot.

    Notes
    -----
    - Assumes that the vertical velocities are in m/s and the depth grid is in meters.
    Original Author: Eleanor Frajka-Williams
    """
    conv_factor = 100  # Convert m/s to cm/s
    depth_negative = ds_out_dives.zgrid.values * -1
    meanw_dives = ds_out_dives.meanw.values * conv_factor
    zgrid_dives = depth_negative
    w_lower_dives = ds_out_dives.w_lower.values * conv_factor
    w_upper_dives = ds_out_dives.w_upper.values * conv_factor

    meanw_climbs = ds_out_climbs.meanw.values * conv_factor
    zgrid_climbs = ds_out_climbs.zgrid.values * -1
    w_lower_climbs = ds_out_climbs.w_lower.values * conv_factor
    w_upper_climbs = ds_out_climbs.w_upper.values * conv_factor
    with plt.style.context(glidertest_style_file):
        fig, ax = plt.subplots(1, 1)
        # Resize to half-width
        half_width = fig.get_size_inches()[0] / 2.07
        fig.set_size_inches(half_width, half_width * 0.9)
        ax.tick_params(axis='both', which='major')

        # Plot dives
        ax.fill_betweenx(zgrid_dives, w_lower_dives, w_upper_dives, color='black', alpha=0.3)
        ax.plot(meanw_dives, zgrid_dives, color='black', label='w$_{dive}$')

        # Plot climbs
        ax.fill_betweenx(zgrid_climbs, w_lower_climbs, w_upper_climbs, color='red', alpha=0.3)
        ax.plot(meanw_climbs, zgrid_climbs, color='red', label='w$_{climb}$')

        ax.invert_yaxis()  # Invert y-axis to show depth increasing downwards
        ax.axvline(x=0, color='darkgray', linestyle='-', linewidth=0.5)  # Add vertical line through 0
        ax.set_xlabel('Vertical Velocity w$_{sw}$ (cm s$^{-1}$)')
        ax.set_ylabel('Depth (m)')
        ax.set_ylim(top=0, bottom=1000)  # Set y-limit maximum to zero
        #ax.set_title('Combined Vertical Velocity Profiles')

        ax.set_xlim(-1, 1.5)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1.0, 1.5])
        plt.tight_layout()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major')
        ax.legend()
        plt.show()
        return fig, ax


def plot_hysteresis(ds, var='DOXY', v_res=1, threshold=2, ax=None):
    """
    This function creates 4 plots which can help the user visualize any possible hysteresis
    present in their dataset for a specific variable

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in **OG1 format** containing at least **DEPTH, PROFILE_NUMBER** and the selected variable. 
        Data should **not** be gridded.
    var : str, optional, default = 'DOXY'
        The variable to analyze for hysteresis effects
    v_res : int, default = 1
        Vertical resolution for gridding the data
    threshold : float, default = 2
        Percentage error threshold for plotting a reference vertical line
    ax : matplotlib.axes.Axes, default = None
        Specific axes for plotting if adding to an existing figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    ax : list of matplotlib.axes.Axes
        The list of axes corresponding to the four generated subplots.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    varG, profG, depthG = utilities.construct_2dgrid(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], 1, v_res,x_bin_center=False)
    df, diff, err_mean, err_range, rms = tools.compute_hyst_stat(ds, var=var, v_res=v_res)
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig = plt.figure()
            ax = [plt.subplot(4, 4, 1), plt.subplot(4, 4, 2), plt.subplot(4, 4, 3), plt.subplot(4, 4, 4), plt.subplot(4, 1, 2)]
            force_plot = True
        else:
            if len(ax) == 1:
                fig = plt.gcf()
                ax = [plt.subplot(4, 4, 1), plt.subplot(4, 4, 2), plt.subplot(4, 4, 3), plt.subplot(4, 4, 4), plt.subplot(4, 1, 2)]
                force_plot = False
            else:
                fig = plt.gcf()
                force_plot = False

        ax[0].plot(df.climb, df.depth, label='Dive')
        ax[0].plot(df.dive, df.depth, label='Climb')
        ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[0].legend()
        ax[1].plot(diff, df.depth)
        ax[2].plot(err_mean, df.depth)
        ax[2].axvline(threshold, c='red')
        ax[3].plot(err_range, df.depth)
        ax[3].axvline(threshold, c='red')
        [a.grid() for a in ax]
        [a.invert_yaxis() for a in ax]
        ax[0].set_ylabel('Depth (m)')
        ax[0].set_xlabel(f'{utilities.plotting_labels(var)} $=mean$ \n({utilities.plotting_units(ds, var)})')
        ax[1].set_xlabel(f'Absolute difference = |$\Delta$| \n({ds[var].units})')
        ax[2].set_xlabel('Error [|Δ| / mean] (%)')
        ax[3].set_xlabel('Scaled error [|Δ| / range] (%)')
        for ax1 in ax[:-1]:
            ax1.xaxis.set_label_position('top')
        c = ax[4].pcolor(profG[:-1, :], depthG[:-1, :], np.diff(varG, axis=0),
                         vmin=np.nanpercentile(np.diff(varG, axis=0), 0.5),
                         vmax=np.nanpercentile(np.diff(varG, axis=0), 99.5), cmap='seismic')
        plt.colorbar(c, ax=ax[4], label=f'Difference dive-climb \n({ds[var].units})', fraction=0.05)
        ax[4].set(ylabel='Depth (m)', xlabel='Profile number')
        fig.suptitle(utilities.plotting_labels(var), y=.98)
        if force_plot:
            plt.show()
    return fig, ax


def plot_outlier_duration(ds: xr.Dataset, rolling_mean: pd.Series, overtime, std=2, ax=None):
    """
    Generates two plots to visualize profile durations and highlight outliers.
    This helps identify profiles with abnormal durations by comparing the actual profile durations
    to a rolling mean, and by visualizing the shape and depth of the selected outlier profiles.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing at least **TIME, DEPTH, and PROFILE_NUMBER**
    rolling_mean : pandas.Series
        A series representing the rolling mean of profile durations, used to 
        define the range of normal durations.
    overtime : list
        A list of profile numbers identified as outliers based on abnormal durations.
    std : float, default = 2
        The number of standard deviations around the rolling mean used to classify 
        outliers.
    ax : matplotlib.axes.Axes, default = None
        Axes object for plotting. If None, a new figure with two subplots is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the generated plots
    ax : list of matplotlib.axes.Axes
        A list of axes corresponding to the two generated subplots.

    Notes
    -----
    Original Author: Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig,ax = plt.subplots(1,2)
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        data = tools.compute_prof_duration(ds)
        ax[0].plot(data['profile_num'], data['profile_duration'], label='Profile duration')
        ax[0].plot(data['profile_num'], rolling_mean, label='Rolling mean')
        ax[0].fill_between(data['profile_num'], rolling_mean - (np.std(rolling_mean) * std),
                           rolling_mean + (np.std(rolling_mean) * std), color='orange',
                           edgecolor="orange", alpha=0.5,
                           label=f'Rolling mean ± {std} std')
        ax[0].legend()
        ax[0].grid()
        ax[0].set(xlabel='Profile number', ylabel='Profile duration (min)')

        ax[1].scatter(ds.TIME, ds.DEPTH, s=0.1)
        for i in range(len(overtime)):
            profile = ds.TIME.where(ds.PROFILE_NUMBER == overtime[i]).dropna(dim='N_MEASUREMENTS')
            ax[1].scatter(profile.TIME, profile.DEPTH, s=0.1, c='red', label='Profiles with odd duration')
        ax[1].invert_yaxis()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[1].legend(by_label.values(), by_label.keys(), markerscale=8., loc='lower right')
        ax[1].set(ylabel='Depth (m)')
        ax[1].grid()
        every_nth = 2
        for n, label in enumerate(ax[1].xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        if force_plot:
            plt.show()
    return fig, ax


def plot_global_range(ds, var='DOXY', min_val=-5, max_val=600, ax=None):
    """
    This function creates a histogram of the selected variable (`var`) from the dataset (`ds`), 
    allowing for visual inspection of its distribution. It overlays vertical lines at the specified 
    minimum (`min_val`) and maximum (`max_val`) values to indicate the global range, helping 
    identify whether values fall within expected limits.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to be plotted.
    var : str, default = 'DOXY'
        The name of the variable to be plotted in the histogram.
    min_val : float, default = -5
        The minimum value of the global range to highlight on the plot.
    max_val : float, default = 600
        The maximum value of the global range to highlight on the plot.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the histogram. If `None`, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the histogram plot.
    ax : matplotlib.axes.Axes
        The axes object containing the histogram plot.
        
    Notes
    -----
    Original Author: Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax.hist(ds[var], bins=50)
        ax.axvline(min_val, c='r')
        ax.axvline(max_val, c='r')
        ax.set(xlabel=f'{utilities.plotting_labels(var)} ({utilities.plotting_units(ds,var)})', ylabel='Frequency')
        ax.set_title('Global range check')
        ax.grid()
        if force_plot:
            plt.show()
    return fig, ax


def plot_ioosqc(data, suspect_threshold=[25], fail_threshold=[50], title='', ax=None):
    """
    This function generates a scatter plot of the results from an IOOS QC test, labeling data points 
    with quality control categories ('GOOD', 'UNKNOWN', 'SUSPECT', 'FAIL', 'MISSING'). It also overlays 
    threshold-based markers to visualize suspect and fail ranges, aiding in the interpretation of 
    data quality.

    Parameters
    ----------
    data : array-like
        The results from the IOOS QC test, representing numerical data points to be plotted.
    suspect_threshold : list, optional, default = [25]
        A list containing one or two numerical values defining the thresholds for suspect values.
        If one value is provided, it applies to both lower and upper bounds. If two values are given,
        they define distinct lower and upper suspect limits.
    fail_threshold : list, optional, default = [50]
        A list containing one or two numerical values defining the thresholds for fail values.
        The structure follows the same logic as `suspect_threshold`.
    title : str, optional, default = ''
        The title to display at the top of the plot.
    ax : matplotlib.axes.Axes, optional
        The axes object on which to plot the results. If `None`, a new figure and axis are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the QC results plot.
    ax : matplotlib.axes.Axes
        The axes object used for plotting.

    Notes
    -----
    - The plot uses two y-axes: one for labeling data points as 'GOOD', 'UNKNOWN', 'SUSPECT', 'FAIL', or 'MISSING' based
      on thresholds, and another for marking specific suspect and fail ranges.
    Original Author: Chiara Monforte
    """
    with plt.style.context(glidertest_style_file):
        if ax is None:
            fig, ax = plt.subplots()
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False

        ax.scatter(np.arange(len(data)), data, s=4)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a = ax.get_yticks().tolist()
        a[:] = ['', 'GOOD', 'UNKNOWN', 'SUSPECT', 'FAIL', '', '', '', '', 'MISSING', '']
        ax.set_yticklabels(a)
        ax2 = ax.twinx()
        ax2.scatter(np.arange(len(data)), data, s=4)
        ax2.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a_2 = ax2.get_yticks().tolist()
        a_2[:] = ['', '', '', '', '', '', '', '', '', '', '']
        if len(suspect_threshold) > 1:
            a_2[1] = f'x>{suspect_threshold[0]} or \nx<{suspect_threshold[1]}'
        else:
            a_2[1] = f'x<{suspect_threshold[0]}'
        if len(fail_threshold) > 1:
            a_2[3] = f'{suspect_threshold[1]}<x<{fail_threshold[1]} &\n {fail_threshold[0]}<x<{suspect_threshold[0].values}'
            a_2[4] = f'x<{fail_threshold[0]} or \nx>{fail_threshold[1]}'
        else:
            a_2[3] = f'x>{suspect_threshold[0]} and \nx<{fail_threshold[0]}'
            a_2[4] = f'x>{fail_threshold[0]}'
        a_2[9] = 'Nan'

        ax2.set_yticklabels(a_2, fontsize=12)

        ax.set_xlabel('Data index')
        ax.grid()
        ax.set_title(title)
        if force_plot:
            plt.show()
    return fig, ax

def plot_max_depth_per_profile(ds: xr.Dataset, bins= 20, ax = None, **kw: dict) -> tuple({plt.Figure, plt.Axes}):
    """
    Plots the maximum depth of each profile in a dataset.

    This function generates two plots: one showing the maximum depth of each profile and another 
    histogram illustrating the distribution of maximum depths. It provides a visual representation 
    of profile depth variations.
        
    Parameters
    ----------
    ds : xarray.Dataset
        An xarray dataset in OG1 format containing profile numbers and corresponding maximum depths.
    bins : int, optional, default = 20
        The number of bins to use for the histogram of maximum depths.
    ax : matplotlib.axes.Axes, optional
        The axes object on which to plot the results. If `None`, a new figure with two subplots is created.
    **kw : dict, optional
        Additional keyword arguments passed to the plotting function for customization.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    ax : matplotlib.axes.Axes
        A 1x2 array of axes used for the two subplots.

    Notes
    -----
    Original Author: Till Moritz
    """
    max_depths = tools.max_depth_per_profile(ds)
    with plt.style.context(glidertest_style_file):
        if ax is None:  
            fig, ax = plt.subplots(1, 2)  
            force_plot = True
        else:
            fig = plt.gcf()
            force_plot = False
            
        ax[0].plot(max_depths.PROFILE_NUMBER, max_depths,**kw)
        ax[0].set_xlabel('Profile number')
        ax[0].set_ylabel(f'Max depth ({max_depths.units})')
        ax[0].set_title('Max depth per profile')
        ax[1].hist(max_depths, bins=bins)
        ax[1].set_xlabel(f'Max depth ({max_depths.units})')
        ax[1].set_ylabel('Number of profiles')
        ax[1].set_title('Histogram of max depth per profile')
        [a.grid() for a in ax]
        if force_plot:
            plt.show()
    return fig, ax

def plot_profile(ds: xr.Dataset, profile_num: int = None, vars: list = ['TEMP','PSAL','DENSITY'], use_bins: bool = False, binning: float = 2) -> tuple:
    """
    Plots the specified variables against depth for a given profile number from an OG1-format dataset.
    If no profile number is provided, it plots the mean profile for the specified variables.

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PROFILE_NUMBER, DEPTH and the specified variables.
    profile_num: int or None
        Profile number to plot. If None, plots mean profile.
    vars: list
        Variables to plot. Default is ['TEMP','PSAL','DENSITY'].
    use_bins: bool
        Whether to bin the data vertically.
    binning: float
        Bin size in meters if use_bins is True.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax1: list of matplotlib.axes.Axes
        The axis objects containing the subplots.

    Notes
    -----
    Original Author: Till Moritz
    """
    vars = [v for v in vars if v]
    if not vars:
        fig, ax1 = plt.subplots(figsize=(12, 9))
        ax1.set_title('No Variables Selected')
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()
        ax1.grid(True)
        return fig, ax1

    if len(vars) > 3:
        raise ValueError("Only three variables can be plotted at once, chose fewer variables")

    with plt.style.context(glidertest_style_file):  
        fig, ax1 = plt.subplots(figsize=(12, 9)) 
        axs = [ax1, ax1.twiny(), ax1.twiny()]
        colors = ['red', 'blue', 'grey']
        s = 10 + binning

        if profile_num is not None:
            profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)
            if use_bins:
                profile = utilities.bin_profile(profile, vars, binning)
            title_str = f'Profile {profile_num}'
        else:
            # Mean profile logic
            data = {}
            for var in vars:
                df = tools.mean_profile(ds, var=var, v_res=binning)
                data[var] = df['mean'].values
            depth = df['depth'].values
            profile = xr.Dataset({var: (['depth'], data[var]) for var in vars})
            profile['DEPTH'] = (['depth'], depth)
            title_str = 'Mean Profile'

        mission = ds.id.split('_')[1][0:8]
        glider = ds.id.split('_')[0]

        for i, var in enumerate(vars):
            ax = axs[i]
            unit = utilities.plotting_units(ds, var)
            label = utilities.plotting_labels(var)
            ax.plot(profile[var], profile['DEPTH'], color=colors[i], label=label)
            ax.scatter(profile[var], profile['DEPTH'], color=colors[i], marker='o', s=s)
            ax.set_xlabel(f'{label} [{unit}]', color=colors[i])
            ax.tick_params(axis='x', colors=colors[i])
            if i > 0:
                ax.xaxis.set_ticks_position('bottom')
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_position(('axes', -0.09*i))
            ax.xaxis.set_label_coords(0.5, -0.05-0.105*i)

        ax1.grid(True)
        ax1.set_ylabel('Depth (m)')
        ax1.invert_yaxis()
        ax1.set_title(f'{title_str} ({glider} on mission: {mission})')

    return fig, ax1


def plot_CR(ds: xr.Dataset, profile_num: int, use_bins: bool = False, binning: float = 2):
    """
    Plots the convective resistance (CR) of a profile against depth based on calculate_CR_for_all_depth function.
    For the calculation, the density anomaly with reference to 1000 kg/m3 is used ('SIGMA_1')

    Parameters
    ----------
    ds: xarray.Dataset
        Xarray dataset in OG1 format with at least PROFILE_NUMBER, DEPTH, SIGMA_1.
    profile_num: int
        The profile number to plot.
    use_bins: bool
        If True, use binned data instead of raw data.
    binning: int
        The depth resolution for binning.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure object containing the plot.
    ax: matplotlib.axes.Axes
        The axis object containing the primary plot.

    Notes
    -----
    Original Author: Till Moritz
    """
    vars = ['SIGMA_1', 'DEPTH']
    utilities._check_necessary_variables(ds, vars + ['PROFILE_NUMBER'])

    profile = ds.where(ds.PROFILE_NUMBER == profile_num, drop=True)

    if use_bins:
        profile = utilities.bin_profile(profile, vars, binning)

    CR_df = tools.calculate_CR_for_all_depth(profile)

    depth = CR_df['DEPTH'].values
    CR = CR_df['CR'].values

    with plt.style.context(glidertest_style_file):
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(CR, depth, label='CR')
        ax.scatter(CR, depth, marker='o', s=10+binning)
        ax.set_xlabel('Convective Resistance (CR)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        ax.set_title(f'Profile {profile_num} (Convective Resistance)')
        ax.grid(True)
        ax.legend()
        plt.show()
    
    return fig, ax

def plot_section(ds, vars=['PSAL', 'TEMP', 'DENSITY'], v_res=1, start=None, end=None, mld_df = None):
    """
    Plots a section of the dataset with PROFILE_NUMBER on the x-axis, DEPTH on the y-axis,
    and mean TIME per profile as secondary x-axis (automatically spaced).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with at least PROFILE_NUMBER, TIME, DEPTH, and target variables.
    vars : str or list of str
        Variables to visualize. If a single variable is provided, it will be converted to a list.
    v_res : float
        Vertical resolution (DEPTH binning).
    start : int or None
        Start PROFILE_NUMBER (inclusive).
    end : int or None
        End PROFILE_NUMBER (inclusive).
    mld_df : pd.DataFrame
        MLD as a pandas Dataframe, which is the result of the MLD calculation compute_mld(). The dataframe should contain the profile number, MLD and the mean time profile.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : list of matplotlib.axes.Axes

    Notes
    -----
    Original Author: Till Moritz
    """
    if not isinstance(vars, list):
        vars = [vars]
    utilities._check_necessary_variables(ds, vars + ['PROFILE_NUMBER', 'TIME', 'DEPTH'])

    if start is not None or end is not None:
        if start is None:
            start = ds.PROFILE_NUMBER.min().values
        if end is None:
            end = ds.PROFILE_NUMBER.max().values
        mask = (ds.PROFILE_NUMBER >= start) & (ds.PROFILE_NUMBER <= end)
        ds = ds.sel(N_MEASUREMENTS=mask)
        if mld_df is not None:
            mld_df = mld_df[(mld_df['PROFILE_NUMBER'] >= start) & (mld_df['PROFILE_NUMBER'] <= end)]

    num_vars = len(vars)
    fig, ax = plt.subplots(num_vars, 1, figsize=(20, 5 * num_vars), sharex=True, gridspec_kw={'height_ratios': [8] * num_vars})
    if num_vars == 1:
        ax = [ax]

    x_plot = ds['PROFILE_NUMBER'].values

    has_density_plot = any(utilities.plotting_colormap(var) == cmo.dense for var in vars)

    # Compute mean time per profile
    df_time = ds[['TIME', 'PROFILE_NUMBER']].to_dataframe().dropna()
    mean_times = df_time.groupby('PROFILE_NUMBER')['TIME'].mean()

    for i, var in enumerate(vars):
        if var not in ds:
            raise ValueError(f'Variable "{var}" not found in dataset.')

        values = ds[var].values
        depth = ds['DEPTH'].values

        p = 1
        z = v_res

        varG, profG, depthG = utilities.construct_2dgrid(x_plot, depth, values, p, z, x_bin_center=False)

        cmap = utilities.plotting_colormap(var)
        if cmap == cmo.delta and np.any(values < 0) and np.any(values > 0):
            norm = mcolors.TwoSlopeNorm(
                vmin=np.nanpercentile(values, 0.5),
                vcenter=0,
                vmax=np.nanpercentile(values, 99.5)
            )
            im = ax[i].pcolormesh(profG, depthG, varG, cmap=cmap, norm=norm)
        else:
            im = ax[i].pcolormesh(profG, depthG, varG, cmap=cmap,
                                  vmin=np.nanpercentile(values, 0.5),
                                  vmax=np.nanpercentile(values, 99.5))
        if mld_df is not None:
            if (has_density_plot and cmap == cmo.dense) or (not has_density_plot and i == 0):
                ax[i].plot(mld_df['PROFILE_NUMBER'], mld_df['MLD'], color='black', marker='o', linewidth=1,
                           label='Mixed Layer Depth', markersize=2)
                ax[i].legend(loc='upper left', fontsize=8)

        unit = utilities.plotting_units(ds,var)
        label = utilities.plotting_labels(var)

        total_profiles = x_plot[-1] - x_plot[0]
        ax[i].invert_yaxis()
        ax[i].set_ylabel('Depth (m)')
        ax[i].grid(True)
        ax[i].set_title(f'Section plot of {label}')
        ax[i].set_xlim(np.min(x_plot)-total_profiles/50, np.max(x_plot)+total_profiles/50)

        cbar = plt.colorbar(im, ax=ax[i], pad=0.03)
        cbar.set_label(f'{label} [{unit}]', labelpad=20, rotation=270)

    # Main x-axis: profile numbers
    ax[-1].set_xlabel('Profile Number')

    # Get mean time per profile (datetime) and profile numbers
    times = pd.to_datetime(mean_times)
    profiles = mean_times.index.values
    time_nums = mdates.date2num(times)  # matplotlib float format for dates

    # Build interpolators
    to_time = interp1d(profiles, time_nums, bounds_error=False, fill_value="extrapolate")
    to_profile = interp1d(time_nums, profiles, bounds_error=False, fill_value="extrapolate")

    # Create a transform that maps profile numbers → time for the secondary x-axis
    def forward(x):
        return to_time(x)

    def inverse(x):
        return to_profile(x)

    # Create the secondary axis (top), linked to the bottom profile axis
    time_ax = ax[-1].secondary_xaxis("bottom", functions=(forward, inverse))
    time_ax.set_xlabel("Mean Time per Profile")
    time_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    time_delta = time_nums[-1] - time_nums[0]
    if time_delta < 5:
        time_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    else:
        time_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    time_ax.spines['bottom'].set_position(('outward', 40))
    time_ax.tick_params(rotation=35)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

