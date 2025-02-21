import numpy as np
import folium
from glidertest.utilities import _check_necessary_variables

def mission_map(ds):
    """
    Makes an interactive folium map of a glider mission. Needs to run in a jupyter notebook

    Parameters
    ----------
    ds: xarray.DataSet
        OG1 dataset of a glider mission

    Returns
    -------
    m: folium.Map
        A folium map with a point for each glider dive and a sub-sampled line of glider track

    Notes
    ------
    Original author: Callum Rollo
    """
    _check_necessary_variables(ds, ["LONGITUDE", "LATITUDE", "DIVE_NUM"])
    df = ds.to_pandas()[["LONGITUDE", "LATITUDE", "DIVE_NUM"]].groupby("DIVE_NUM").median()
    coordinates = [[lat, lon] for lat, lon in zip(ds['LATITUDE'].values, ds['LONGITUDE'].values)]
    if len(coordinates) > 5000:
        coordinates = coordinates[::int(np.ceil(len(coordinates) / 5000))]
    df['DIVE_NUM'] = df.index
    m = folium.Map(location=[df.LATITUDE.mean(), df.LONGITUDE.mean()], zoom_start=8, tiles="cartodb positron")
    folium.WmsTileLayer(
        url="https://ows.emodnet-bathymetry.eu/wms",
        layers= 'mean_atlas_land',
        attr='EMODnet bathymetry'
    ).add_to(m)

    folium.PolyLine(
        locations=coordinates,
        color="red",
        weight=5,
        tooltip="Glider track",
    ).add_to(m)

    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            tooltip=f"Dive {int(row['DIVE_NUM'])}",
            color= 'black',
            fillOpacity= 1,
            fillColor= "red",
            weight = 2,
        ).add_to(m)
    return m