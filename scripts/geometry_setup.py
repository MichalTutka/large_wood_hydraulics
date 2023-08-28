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

#========================

##preferences
np.set_printoptions(precision=3)

#endregion
#=================================================================================

#region GEOMETRY IMPORT
profile_file_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_profiles\site3_LL6-7_32_profile.csv'
df_profile = pd.read_csv(profile_file_path, encoding='cp1252')

df_us_profile_file_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_profiles\site3_LL8-9_0_profile.csv'
df_us_profile = pd.read_csv(df_us_profile_file_path, encoding='cp1252')

df_ds_profile_file_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_profiles\site3_LL5-6_0_profile.csv'
df_ds_profile = pd.read_csv(df_ds_profile_file_path, encoding='cp1252')

#endregion
#=================================================================================

#region CREATE TUPLES OF CHANNEL AND OBJECTS
unique_assets = df_profile['asset'].unique() # Get the unique asset values from the 'asset' column
asset_points = {} # Create an empty dictionary to store the lists

# Loop through each unique asset
for index, row in df_profile.iterrows():
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
print(obj1_points)
#print(obj2_points)
#print(obj3_points)

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
            channel_lefty_x = point[0] # Store the x-coordinate corresponding to channel_maxy_left

lowest_y_point = min(channel_points, key=lambda point: point[1])  # Find the point with the lowest y-coordinate
lowest_y_x = lowest_y_point[0]  # Get the x-coordinate of the point with the lowest y-coordinate

channel_maxy_right = -5000  # the highest elevation to the RIGHT of the thalweg (if your xsection looks downstream).
for point in channel.coords:  # For each of the points in the 'channel' linestring:
    if point[0] < lowest_y_x:  # if the coordinate is to the RIGHT of the channel invert
        if point[1] > channel_maxy_right:  # and if the next coordinate has a higher y-value than the current highest point:
            channel_maxy_right = point[1]  # then give channel_maxy_right that value.
            channel_righty_x = point[0]  # Store the x-coordinate corresponding to channel_maxy_right
            
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
df_profile_metrics = pd.DataFrame({
    'wse_jam': wse_jam_values,
    'discharge': None,  # Set this to the desired initial value or leave it as None
    'slope': None,  # Set this to the desired initial value or leave it as None
    'flow_area_interstitial': None,
    'flow_area_extent': None,
    'wood_sub_area': None
})

wse_max
wse_end = channel.bounds[2] # where the WSE line ENDS being drawn. Max X
wse_start = channel.bounds[0] # where the WSE line BEGINS being drawn. Min X

#endregion
#=================================================================================

#region WSE AND CHANNEL INTERCEPTS; TO BE USED FOR FLOW EXTENTS

def get_intersection_coordinates(line1, line2):
    # Function definition remains the same as before
    intersection_points = line1.intersection(line2)
    coordinates = []
    if intersection_points.is_empty:
        return coordinates
    if intersection_points.geom_type == 'Point':
        coordinates.append((intersection_points.x, intersection_points.y))
    elif intersection_points.geom_type == 'MultiPoint':
        for point in intersection_points:
            coordinates.append((point.x, point.y))
    return coordinates


#endregion 
#=================================================================================

#=================================================================================

#region LARGE WOOD OBJECTS
asset_polygons = {} # Create an empty dictionary to store the lists

# Loop through each unique asset
for index, row in df_profile.iterrows():
    # Get the values of 'asset', 'station', and 'elevation' columns
    asset = row['asset']
    station = row['station']
    elevation = row['elevation']
    
    # Create a tuple with 'station' and 'elevation'
    point = (float(station), float(elevation))  # Convert the string values to float
    
    # Check if the asset is already in the dictionary, if not, create an empty list
    if asset not in asset_polygons:
        asset_polygons[asset] = []
    
    # Append the tuple to the corresponding list based on the 'asset' value
    asset_polygons[asset].append(point)

# Create individual polygons for assets with 'obj' in their name
obj_polygons = {}
for asset, points in asset_polygons.items():
    if 'obj' in asset.lower():
        polygon = Polygon(points)
        obj_polygons[asset] = polygon

# Print the polygons for each asset with 'obj' in their name
for asset, polygon in obj_polygons.items():
    print(f"Polygon for asset {asset}: {polygon}")

# Combining all the objects (LW pieces) into one multipolygon
objects_multipoly = []
for polygon in obj_polygons.values():
    objects_multipoly.append(polygon)
objects_multipoly = MultiPolygon(objects_multipoly)
print("Combined MultiPolygon:", objects_multipoly) # Print the MultiPolygon object

df_profile_metrics
#endregion
#=================================================================================




# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&































#region GRAVEYARD

# # WSE-CHANNEL INTERCEPTS; TO BE USED FOR FLOW EXTENTS
# for index, row in df_profile_metrics.iterrows():
#     start_point = Point(channel_invert_x, row['wse_jam'])
#     wse_line = LineString([(wse_start, row['wse_jam']), (wse_end, row['wse_jam'])])

#     # Rest of your code using start_point and wse_line for each row
#     # ...

# def get_intersection_coordinates(line1, line2): # get all the intercetions of the channel and wse_line
#     intersection_points = line1.intersection(line2)
#     coordinates = []

#     if intersection_points.is_empty:
#         return coordinates

#     if intersection_points.geom_type == 'Point':
#         coordinates.append((intersection_points.x, intersection_points.y))
#     elif intersection_points.geom_type == 'MultiPoint':
#         for point in intersection_points:
#             coordinates.append((point.x, point.y))

#     return coordinates

# intersection_coordinates = get_intersection_coordinates(channel, wse_line)
# print(intersection_coordinates)

# # JUST THE CLOSEST WSE/CHANNEL INTERCEPTS TO CHANNEL INVERT

# # Sort the intersection coordinates based on their x-values
# sorted_coordinates = sorted(intersection_coordinates, key=lambda c: c[0])
# # Find the closest x-values to the left and right of channel_invert_x
# wse_left_x = None
# wse_right_x = None

# for coordinate in sorted_coordinates:
#     x = coordinate[0]
#     if x < channel_invert_x:
#         wse_left_x = x
#     elif x > channel_invert_x:
#         wse_right_x = x
#         break

# print("Closest x-value to the left:", wse_left_x)
# print("Closest x-value to the right:", wse_right_x)

# #endregion

# #region WETTED WIDTH
# flow_area_xmin = flow_extent.bounds[0] # get xmin bounds of flow area
# flow_area_xmax = flow_extent.bounds[2] # get xmax bounds of flow area
# wetted_width = flow_area_xmax - flow_area_xmin # wetted width [m]
# flow_extent_surface = LineString([(flow_area_xmin, wse),(flow_area_xmax,wse)]) # a linestring geometry representing the plane of the water surface to its extents

# flow_extent_perimeter = flow_extent.length # total perimeter of water
# wp_channel = flow_extent_perimeter - wetted_width ## WETTED PERIMETER OF CHANNEL
# flow_interstitial_perimeter = flow_interstitial.length ## total perimeter of water (with wood)
# #endregion
# #=================================================================================

# #region WOOD TOPS (intermediate product)
# flow_area_boundary = flow_extent.boundary # extracting the lines of flow_extent
# wood_sub_boundary = wood_sub.boundary # extracting the lines of wood_sub
# wood_tops2 = flow_area_boundary.intersection(wood_sub_boundary) # linestring geometry of where the water surface is interpolated through the wood (erroneous)
# wood_tops = wood_tops2.length # the length of this erroneous surface
# #endregion
# #=================================================================================

# #region WETTED PERIMETER
# wp_wood = wood_sub.length - wood_tops # the actual wetted perimeter of the wood
# wp_total = wp_channel + wp_wood # WP TOTAL: wetted LW + wetted channel
# water_surface_length = wetted_width - wood_tops # the cross-sectional length of the actual water surface
# #endregion
# #=================================================================================

# #region SUBMERGED WOOD

# # # Iterate over the rows of the DataFrame
# # for index, row in df_profile_metrics.iterrows():
# #     wood_sub = flow_area_extent.difference(flow_area_interstitial)  # Calculate the difference between geometries
# #     wood_sub_area = wood_sub.area  # Calculate the area of the wood_sub geometry
# #     df_profile_metrics.at[index, 'wood_sub_area'] = wood_sub_area  # Assign the wood_sub_area value to the corresponding row

# # # Print the updated DataFrame
# # print(df_profile_metrics)

# #endregion

# #region HYDRAULIC RADIUS
# hydraulic_radius_channel = flow_interstitial.area / wp_channel # Rh: The ratio of the cross-sectional flow area to the cross-sectional wetted perimeter, orthogonal to flow.
# hydraulic_radius_total = flow_interstitial.area / wp_total # Rh: The ratio of the cross-sectional flow area to the cross-sectional wetted perimeter, orthogonal to flow.
# #endregion

# #region DIMENSIONLESS METRICS
# porosity = flow_area_interstitial / flow_extent.area # POROSITY (actual flow area divided by flow extent area)
# blockage_ratio = wood_sub_area / flow_extent.area # BLOCKAGE RATIO (Frontal area of wood divided by flow extent area)
# #endregion

df_profile_metrics