#region IMPORT STATEMENTS
# use Python 3.10.9

import shapely as sp
from shapely import geometry, ops, wkt
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import polygonize, polygonize_full 
#========================
import numpy as np
import pandas as pd
from descartes import PolygonPatch
from colour import Color
import latex
import pickle
import math
from fractions import Fraction
#========================
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter

##preferences
np.set_printoptions(precision=3)

#============================================================

df_us_profile_file_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_profiles\site3_LL8-9_0_profile.csv'
df_us_profile = pd.read_csv(df_us_profile_file_path, encoding='cp1252')

df_ds_profile_file_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_profiles\site3_LL5-6_0_profile.csv'
df_ds_profile = pd.read_csv(df_ds_profile_file_path, encoding='cp1252')

#=============================================================================================================

########   #######  ##      ## ##    ##  ######  ######## ########  ########    ###    ##     ## 
##     ## ##     ## ##  ##  ## ###   ## ##    ##    ##    ##     ## ##         ## ##   ###   ### 
##     ## ##     ## ##  ##  ## ####  ## ##          ##    ##     ## ##        ##   ##  #### #### 
##     ## ##     ## ##  ##  ## ## ## ##  ######     ##    ########  ######   ##     ## ## ### ## 
##     ## ##     ## ##  ##  ## ##  ####       ##    ##    ##   ##   ##       ######### ##     ## 
##     ## ##     ## ##  ##  ## ##   ### ##    ##    ##    ##    ##  ##       ##     ## ##     ## 
########   #######   ###  ###  ##    ##  ######     ##    ##     ## ######## ##     ## ##     ##

#=============================================================================================================


#region CREATE TUPLES OF CHANNEL AND OBJECTS
unique_assets = df_ds_profile['asset'].unique() # Get the unique asset values from the 'asset' column
asset_points = {} # Create an empty dictionary to store the lists

# Loop through each unique asset
for index, row in df_ds_profile.iterrows():
    # Get the values of 'asset', 'station', and 'elevation' columns
    asset = row['asset']
    station = row['station']
    elevation = row['elevation']
    point = (station, elevation) # Create a tuple with 'station' and 'elevation'
    
    # Check if the asset is already in the dictionary, if not, create an empty list
    if asset not in asset_points:
        asset_points[asset] = []
    # Append the tuple to the corresponding list based on the 'asset' value
    asset_points[asset].append(point)

# Print the lists for each unique asset
for asset, points in asset_points.items():
    globals()[f"{asset.lower()}_points"] = points
    
# Print the variables
print(channel_points)

#endregion
#=================================================================================

#region FINDING THE SECOND HIGHEST CHANNEL ELEVATION

# define the channel
numeric_channel = [(float(x), float(y)) for x, y in channel_points] #converts a list of tuples containing string values to a list of numeric pairs
channel = LineString(numeric_channel)

# Necessary because we need to limit water surface elevations to those that can be bound by the channel data; otherwise we'll have geometry issues.
channel_maxy_left = -5000 # the highest elevation to the LEFT of the thalweg (if your xsection looks downstream).
for point in channel.coords: # For each of the points in the 'channel' linestring:
    if point[0] < channel.bounds[1]: # if the coordinate is to the LEFT of the channel invert
        if point[1] > channel_maxy_left: # and if the next coordinate has a higher y-value than the current highest point:
            channel_maxy_left = point[1] # then give channel_maxy_left that value.

lowest_y_point = min(channel_points, key=lambda point: point[1])  # Find the point with the lowest y-coordinate
lowest_y_x = lowest_y_point[0]  # Get the x-coordinate of the point with the lowest y-coordinate

channel_maxy_right = -5000  # the highest elevation to the RIGHT of the thalweg (if your xsection looks downstream).
for point in channel.coords:  # For each of the points in the 'channel' linestring:
    if point[0] < lowest_y_x:  # if the coordinate is to the RIGHT of the channel invert
        if point[1] > channel_maxy_right:  # and if the next coordinate has a higher y-value than the current highest point:
            channel_maxy_right = point[1]  # then give channel_maxy_right that value.

wse_max = min(channel_maxy_right, channel_maxy_left)  # the maximum considered water surface elevation is the lower of these two values.

print(channel_maxy_left)
print(channel_maxy_right)
#endregion

#=================================================================================

#region CHANNEL

channel_start = Point(channel.coords[0]) # Get the start point of the channel
channel_end = Point(channel.coords[-1]) # Get the end point of the channel
channel_invert = channel.bounds[1] # the lowest elevation of the channel
#channel_max = channel.bounds[3] # maximum channel y value
# x-Coordinate of channel invert
channel_invert_x = None # x-coord of channel invert
for coord in channel.coords:
    if coord[1] == channel_invert:
        channel_invert_x = coord[0]
        break 
    elif channel_invert_x is None or coord[1] < channel_invert:
        channel_invert_x = coord[0]
        channel_invert = coord[1] 

# Make a copy so the original points are not modified
channel_points_modified = channel_points.copy()
channel_points2 = channel_points_modified
channel_points2

# Offsets. Purpose: to reduce errors
vertical_offset = 5 # (meters) offset of y-value
start_point_above = (channel_start.x, channel_start.y + vertical_offset) # Create new tuple by adding the vertical offset to the start and end points
end_point_above = (channel_end.x, channel_end.y + vertical_offset) # Create new tuple by adding the vertical offset to the start and end points

# Check if the tuples already exist in channel_points before inserting
if start_point_above not in channel_points2:
    channel_points2.insert(0, start_point_above)
if end_point_above not in channel_points2:
    channel_points2.append(end_point_above)
channel_points2

channelpoly = Polygon(channel_points2) # . intermediate product. CREATING A CHANNEL POLGYON FOR INTERSECTION WITH WSE POLGYON
channelpoly # to view the shape

#endregion
#=================================================================================

#region WATER SURFACE ELEVATION + CREATE PROFILE METRICS DATASET

# Create a list of values for the `wse_jam` column
wse_jam_values = [round(x, 2) for x in pd.np.arange(channel_invert, wse_max + 0.01, 0.01)]

# Create a DataFrame for the whole project
df_ds_profile_metrics = pd.DataFrame({
    'wse_jam': wse_jam_values,
    'discharge': None,  # Set this to the desired initial value or leave it as None
    'flow_area_extent': None,
})

wse_max
wse_end = channel.bounds[2] # where the WSE line ENDS being drawn. Max X
wse_start = channel.bounds[0] # where the WSE line BEGINS being drawn. Min X

#endregion
#=================================================================================

from geometry_setup import get_intersection_coordinates

# Iterate over the rows of the DataFrame
for index, row in df_ds_profile_metrics.iterrows():
    use_closest_x = False # set to False to extent the waterline all the way across the channel; True to first intersection
    wse_jam_value = row['wse_jam']

    # Create the waterpoly for the current wse_jam_value
    start_point = Point(channel_invert_x, wse_jam_value)
    wse_line = LineString([(wse_start, wse_jam_value), (wse_end, wse_jam_value)])
    intersection_coordinates = get_intersection_coordinates(channel, wse_line)
    sorted_coordinates = sorted(intersection_coordinates, key=lambda c: c[0])

    if use_closest_x:
        # Find the closest x-values to the left and right of channel_invert_x
        wse_left_x = None
        wse_right_x = None

        for coordinate in sorted_coordinates:
            x = coordinate[0]
            if x < channel_invert_x:
                wse_left_x = x
            elif x > channel_invert_x:
                wse_right_x = x
                break
                
    else:
        # Use wse_start as the left x and wse_end as the right x
        wse_left_x = wse_start
        wse_right_x = wse_end

    if wse_left_x is not None and wse_right_x is not None:
        waterpoly = Polygon([(wse_left_x, wse_jam_value),
                             (wse_right_x, wse_jam_value),
                             (wse_right_x, channel_invert_x),
                             (wse_left_x, channel_invert_x)])

    # Flow Extents
    flow_extent = waterpoly.intersection(channelpoly)
    flow_area_extent = flow_extent.area
    df_ds_profile_metrics.at[index, 'flow_area_extent'] = flow_area_extent  # Store the flow extent value in the 'flow_area_extent' column

df_ds_profile_metrics

#=============================================================================================================

##     ## ########   ######  ######## ########  ########    ###    ##     ## 
##     ## ##     ## ##    ##    ##    ##     ## ##         ## ##   ###   ### 
##     ## ##     ## ##          ##    ##     ## ##        ##   ##  #### #### 
##     ## ########   ######     ##    ########  ######   ##     ## ## ### ## 
##     ## ##              ##    ##    ##   ##   ##       ######### ##     ## 
##     ## ##        ##    ##    ##    ##    ##  ##       ##     ## ##     ## 
 #######  ##         ######     ##    ##     ## ######## ##     ## ##     ## 
 
#=============================================================================================================


#region CREATE TUPLES OF CHANNEL AND OBJECTS
unique_assets = df_us_profile['asset'].unique() # Get the unique asset values from the 'asset' column
asset_points = {} # Create an empty dictionary to store the lists

# Loop through each unique asset
for index, row in df_us_profile.iterrows():
    # Get the values of 'asset', 'station', and 'elevation' columns
    asset = row['asset']
    station = row['station']
    elevation = row['elevation']
    point = (station, elevation) # Create a tuple with 'station' and 'elevation'
    
    # Check if the asset is already in the dictionary, if not, create an empty list
    if asset not in asset_points:
        asset_points[asset] = []
    # Append the tuple to the corresponding list based on the 'asset' value
    asset_points[asset].append(point)

# Print the lists for each unique asset
for asset, points in asset_points.items():
    globals()[f"{asset.lower()}_points"] = points
    
# Print the variables
print(channel_points)

#endregion
#=================================================================================

#region FINDING THE SECOND HIGHEST CHANNEL ELEVATION

# define the channel
numeric_channel = [(float(x), float(y)) for x, y in channel_points] #converts a list of tuples containing string values to a list of numeric pairs
channel = LineString(numeric_channel)

# Necessary because we need to limit water surface elevations to those that can be bound by the channel data; otherwise we'll have geometry issues.
channel_maxy_left = -5000 # the highest elevation to the LEFT of the thalweg (if your xsection looks downstream).
for point in channel.coords: # For each of the points in the 'channel' linestring:
    if point[0] < channel.bounds[1]: # if the coordinate is to the LEFT of the channel invert
        if point[1] > channel_maxy_left: # and if the next coordinate has a higher y-value than the current highest point:
            channel_maxy_left = point[1] # then give channel_maxy_left that value.

lowest_y_point = min(channel_points, key=lambda point: point[1])  # Find the point with the lowest y-coordinate
lowest_y_x = lowest_y_point[0]  # Get the x-coordinate of the point with the lowest y-coordinate

channel_maxy_right = -5000  # the highest elevation to the RIGHT of the thalweg (if your xsection looks downstream).
for point in channel.coords:  # For each of the points in the 'channel' linestring:
    if point[0] < lowest_y_x:  # if the coordinate is to the RIGHT of the channel invert
        if point[1] > channel_maxy_right:  # and if the next coordinate has a higher y-value than the current highest point:
            channel_maxy_right = point[1]  # then give channel_maxy_right that value.

wse_max = min(channel_maxy_right, channel_maxy_left)  # the maximum considered water surface elevation is the lower of these two values.

print(channel_maxy_left)
print(channel_maxy_right)
#endregion

#=================================================================================

#region CHANNEL

channel_start = Point(channel.coords[0]) # Get the start point of the channel
channel_end = Point(channel.coords[-1]) # Get the end point of the channel
channel_invert = channel.bounds[1] # the lowest elevation of the channel
#channel_max = channel.bounds[3] # maximum channel y value
# x-Coordinate of channel invert
channel_invert_x = None # x-coord of channel invert
for coord in channel.coords:
    if coord[1] == channel_invert:
        channel_invert_x = coord[0]
        break 
    elif channel_invert_x is None or coord[1] < channel_invert:
        channel_invert_x = coord[0]
        channel_invert = coord[1] 

# Make a copy so the original points are not modified
channel_points_modified = channel_points.copy()
channel_points2 = channel_points_modified
channel_points2

# Offsets. Purpose: to reduce errors
vertical_offset = 5 # (meters) offset of y-value
start_point_above = (channel_start.x, channel_start.y + vertical_offset) # Create new tuple by adding the vertical offset to the start and end points
end_point_above = (channel_end.x, channel_end.y + vertical_offset) # Create new tuple by adding the vertical offset to the start and end points

# Check if the tuples already exist in channel_points before inserting
if start_point_above not in channel_points2:
    channel_points2.insert(0, start_point_above)
if end_point_above not in channel_points2:
    channel_points2.append(end_point_above)
channel_points2

channelpoly = Polygon(channel_points2) # . intermediate product. CREATING A CHANNEL POLGYON FOR INTERSECTION WITH WSE POLGYON
channelpoly # to view the shape

#endregion
#=================================================================================

#region WATER SURFACE ELEVATION + CREATE PROFILE METRICS DATASET

# Create a list of values for the `wse_jam` column
wse_jam_values = [round(x, 2) for x in pd.np.arange(channel_invert, wse_max + 0.01, 0.01)]

# Create a DataFrame for the whole project
df_us_profile_metrics = pd.DataFrame({
    'wse_jam': wse_jam_values,
    'discharge': None,  # Set this to the desired initial value or leave it as None
    'flow_area_extent': None,
})

wse_max
wse_end = channel.bounds[2] # where the WSE line ENDS being drawn. Max X
wse_start = channel.bounds[0] # where the WSE line BEGINS being drawn. Min X

#endregion
#=================================================================================

from geometry_setup import get_intersection_coordinates

# Iterate over the rows of the DataFrame
for index, row in df_us_profile_metrics.iterrows():
    use_closest_x = False # set to False to extent the waterline all the way across the channel; True to first intersection
    wse_jam_value = row['wse_jam']

    # Create the waterpoly for the current wse_jam_value
    start_point = Point(channel_invert_x, wse_jam_value)
    wse_line = LineString([(wse_start, wse_jam_value), (wse_end, wse_jam_value)])
    intersection_coordinates = get_intersection_coordinates(channel, wse_line)
    sorted_coordinates = sorted(intersection_coordinates, key=lambda c: c[0])

    if use_closest_x:
        # Find the closest x-values to the left and right of channel_invert_x
        wse_left_x = None
        wse_right_x = None

        for coordinate in sorted_coordinates:
            x = coordinate[0]
            if x < channel_invert_x:
                wse_left_x = x
            elif x > channel_invert_x:
                wse_right_x = x
                break
                
    else:
        # Use wse_start as the left x and wse_end as the right x
        wse_left_x = wse_start
        wse_right_x = wse_end

    if wse_left_x is not None and wse_right_x is not None:
        waterpoly = Polygon([(wse_left_x, wse_jam_value),
                             (wse_right_x, wse_jam_value),
                             (wse_right_x, channel_invert_x),
                             (wse_left_x, channel_invert_x)])

    # Flow Extents
    flow_extent = waterpoly.intersection(channelpoly)
    flow_area_extent = flow_extent.area
    df_us_profile_metrics.at[index, 'flow_area_extent'] = flow_area_extent  # Store the flow extent value in the 'flow_area_extent' column

df_us_profile_metrics
# =====================================================================================================

   #     #####   #####  ###  #####  #     #             #    ######  #######    #    
  # #   #     # #     #  #  #     # ##    #            # #   #     # #         # #   
 #   #  #       #        #  #       # #   #           #   #  #     # #        #   #  
#     #  #####   #####   #  #  #### #  #  #          #     # ######  #####   #     # 
#######       #       #  #  #     # #   # #          ####### #   #   #       ####### 
#     # #     # #     #  #  #     # #    ##          #     # #    #  #       #     # 
#     #  #####   #####  ###  #####  #     #          #     # #     # ####### #     # 

# =====================================================================================================
# DataFrames
df_ds_profile_metrics
df_us_profile_metrics
with open('expanded_df.pk', 'rb') as f5:
    expanded_df = pickle.load(f5)
expanded_df['flow_area_m2'] = None  # New column
expanded_df['velocity_head_m'] = None  # New column
from wse_interpolation import logger_ds, logger_us # Logger names

# Function to calculate the closest flow area extent
def calculate_closest_flow_area(row, profile_df):
    closest_wse = profile_df['wse_jam'].iloc[(profile_df['wse_jam'] - row['wse_estimated_m']).abs().idxmin()]
    closest_flow_area = profile_df.loc[profile_df['wse_jam'] == closest_wse, 'flow_area_extent'].iloc[0]
    return closest_flow_area

# Iterate through rows of expanded_df
for index, row in expanded_df.iterrows():
    if pd.notna(row['wse_estimated_m']):
        if row['logger'] == logger_ds:
            # Calculate the closest flow area extent using df_ds_profile_metrics
            closest_flow_area = calculate_closest_flow_area(row, df_ds_profile_metrics)
            # Populate the 'flow_area_m2' column in expanded_df with the closest flow area
            expanded_df.at[index, 'flow_area_m2'] = closest_flow_area
        
        elif row['logger'] == logger_us:
            # Calculate the closest flow area extent using df_us_profile_metrics
            closest_flow_area = calculate_closest_flow_area(row, df_us_profile_metrics)
            # Populate the 'flow_area_m2' column in expanded_df with the closest flow area
            expanded_df.at[index, 'flow_area_m2'] = closest_flow_area
            
        # Calculate velocity_head_m for rows where flow_area_m2 is populated
        if pd.notna(row['flow_area_m2']):
            discharge_measured_cms = row['discharge_measured_cms']
            flow_area_m2 = row['flow_area_m2']  
            velocity_head_m = ((discharge_measured_cms / flow_area_m2)**2) / (2 * 9.80665)
            expanded_df.at[index, 'velocity_head_m'] = velocity_head_m

expanded_df