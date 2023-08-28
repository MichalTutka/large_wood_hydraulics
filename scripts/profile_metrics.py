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
import statsmodels.api as sm
#========================
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
#========================
from geometry_setup import *
np.set_printoptions(precision=3) ##preferences
#endregion

#region PROFILE METRICS

# Iterate over the rows of the DataFrame
for index, row in df_profile_metrics.iterrows():
    use_closest_x = True # set to False to extent the waterline to the channel limits defined in geometry_setup; True to first intersection
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
        wse_left_x = channel_lefty_x
        wse_right_x = channel_righty_x

    if wse_left_x is not None and wse_right_x is not None:
        waterpoly = Polygon([(wse_left_x, wse_jam_value),
                             (wse_right_x, wse_jam_value),
                             (wse_right_x, channel_invert_x),
                             (wse_left_x, channel_invert_x)])

    # Flow Extents
    flow_extent = waterpoly.intersection(channelpoly)
    flow_area_extent = flow_extent.area
    df_profile_metrics.at[index, 'flow_area_extent'] = flow_area_extent  # Store the flow extent value in the 'flow_area_extent' column

    # Interstitial Flow
    #flow_interstitial = flow_extent.intersection(channelpoly).difference(objects_multipoly)
    flow_interstitial = flow_extent.difference(objects_multipoly)
    flow_area_interstitial = flow_interstitial.area # Calculate the flow area interstitial
    df_profile_metrics.at[index, 'flow_area_interstitial'] = flow_area_interstitial # Store the flow area interstitial value in the 'flow_area_interstitial' column

    # Transverse area
    wood_sub = flow_extent.difference(flow_interstitial)  # Calculate the difference between geometries
    wood_sub_area = wood_sub.area  # Calculate the area of the wood_sub geometry
    df_profile_metrics.at[index, 'wood_sub_area'] = wood_sub_area  # Assign the wood_sub_area value to the corresponding row

    # Wetted width
    flow_area_xmin = flow_extent.bounds[0] # get xmin bounds of flow area
    flow_area_xmax = flow_extent.bounds[2] # get xmax bounds of flow area
    wetted_width = flow_area_xmax - flow_area_xmin # wetted width [m]
    # flow_extent_surface = LineString([(flow_area_xmin, #wse),(flow_area_xmax,#wse)]) # a linestring geometry representing the plane of the water surface to its extents
    flow_extent_perimeter = flow_extent.length # total perimeter of water
    #flow_interstitial_perimeter = flow_interstitial.length ## total perimeter of water (with wood)
    wp_channel = flow_extent_perimeter - wetted_width ## WETTED PERIMETER OF CHANNEL
    df_profile_metrics.at[index, 'wp_channel'] = wp_channel

    #=================================================================================

    # Wood tops (intermediate product)
    flow_area_boundary = flow_extent.boundary # extracting the lines of flow_extent
    wood_sub_boundary = wood_sub.boundary # extracting the lines of wood_sub
    wood_tops2 = flow_area_boundary.intersection(wood_sub_boundary) # linestring geometry of where the water surface is interpolated through the wood (erroneous)
    wood_tops = wood_tops2.length # the length of this erroneous surface

    # Wetted Perimeter
    wp_wood = wood_sub.length - wood_tops # the actual wetted perimeter of the wood
    df_profile_metrics.at[index, 'wp_wood'] = wp_wood
    wp_total = wp_channel + wp_wood # WP TOTAL: wetted LW + wetted channel
    df_profile_metrics.at[index, 'wp_total'] = wp_total
    water_surface_length = wetted_width - wood_tops # the cross-sectional length of the actual water surface
    #=================================================================================
    # Hydraulic Radius
    hydraulic_radius_channel = flow_area_interstitial / wp_channel # Rh: The ratio of the cross-sectional flow area to the cross-sectional wetted perimeter, orthogonal to flow.
    hydraulic_radius_total = flow_area_interstitial / wp_total # Rh: The ratio of the cross-sectional flow area to the cross-sectional wetted perimeter, orthogonal to flow.
    df_profile_metrics.at[index, 'hydraulic_radius_channel'] = hydraulic_radius_channel
    df_profile_metrics.at[index, 'hydraulic_radius_total'] = hydraulic_radius_total
    
    # Hydraulic Mean Depth
    #top_width = wse_right_x - wse_left_x
    #df_profile_metrics.at[index, 'top_width'] = top_width
    hydraulic_mean_depth = flow_area_interstitial / wetted_width
    df_profile_metrics.at[index, 'hydraulic_mean_depth'] = hydraulic_mean_depth
    
    # Dimensionless metrics
    porosity = flow_area_interstitial / flow_extent.area # POROSITY (actual flow area divided by flow extent area)
    blockage_ratio = wood_sub_area / flow_extent.area # BLOCKAGE RATIO (transverse area of wood divided by flow extent area)
    df_profile_metrics.at[index, 'porosity'] = porosity
    df_profile_metrics.at[index, 'blockage_ratio'] = blockage_ratio
    

print(df_profile_metrics) # Print the updated DataFrame

#endregion
#=================================================================================

#region CALCULATING DISCHARGE
from wse_interpolation import calculate_discharge_from_wse_jam

# Iterate over the rows of the DataFrame
for index, row in df_profile_metrics.iterrows():
    wse_jam_value = row['wse_jam']
    discharge_value = calculate_discharge_from_wse_jam(wse_jam_value) # Calculate the discharge using the wse_jam_value
    if discharge_value is None:
        discharge_value = 0
    df_profile_metrics.at[index, 'discharge'] = discharge_value # Store the discharge value in the 'discharge' column

# Print the updated DataFrame
print(df_profile_metrics)
df_profile_metrics.head(50)

def plot_discharge_vs_wse_jam(df):
    discharge = df['discharge']
    wse_jam = df['wse_jam']

    fig, ax = plt.subplots()
    ax.plot(discharge, wse_jam, '.', markersize=5)
    ax.set_ylabel('Stage at jam (m)')
    ax.set_xlabel('Discharge (cms)')
    ax.set_title('Discharge vs Stage at Jam')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format y-axis tick labels to two significant figures
    ax.grid(True)
    plt.show()
plot_discharge_vs_wse_jam(df_profile_metrics) # Call the function with your DataFrame


#endregion

#=================================================================================

#region VELOCITY

for index, row in df_profile_metrics.iterrows():
    flow_area_interstitial = row['flow_area_interstitial']
    flow_area_extent = row['flow_area_extent']
    discharge = row['discharge']

    if pd.notna(flow_area_interstitial) and pd.notna(discharge):
        velocity_interstitial = discharge / flow_area_interstitial
    else:
        velocity_interstitial = 0
    df_profile_metrics.at[index, 'velocity_interstitial'] = velocity_interstitial
    
    if pd.notna(flow_area_extent) and pd.notna(discharge):
        velocity_extent = discharge / flow_area_extent
    else:
        velocity_extent = 0
    df_profile_metrics.at[index, 'velocity_extent'] = velocity_extent

#endregion

#=================================================================================

#region CALCULATING SLOPE
from wse_interpolation import calculate_slope_from_discharge

# Iterate over the rows of the DataFrame
for index, row in df_profile_metrics.iterrows():
    discharge_value = row['discharge']
    slope_value = calculate_slope_from_discharge(discharge_value) # Calculate the discharge using the wse_jam_value
    df_profile_metrics.at[index, 'slope'] = slope_value # Store the discharge value in the 'discharge' column

def plot_discharge_vs_slope(df):
    discharge = df['discharge']
    slope = df['slope']

    fig, ax = plt.subplots()
    ax.plot(discharge, slope, '.', markersize=5)
    ax.set_ylabel('Slope')
    ax.set_xlabel('Discharge (cms)')
    ax.set_title('Discharge vs Slope')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))  # Format y-axis tick labels as percentages    plt.show()
    ax.grid(True)

plot_discharge_vs_slope(df_profile_metrics) # Call the function with your DataFrame

#endregion

#=================================================================================

#region REYNOLDS # FOR VEGETATION (see Cheng and Nguyen 2011)
for index, row in df_profile_metrics.iterrows():
    water_density = 999.65 # kg/m3 at 10 C
    dynamic_viscosity = 0.0013076 # N s/m^2 at 10 C
    kinematic_viscosity =  dynamic_viscosity / water_density #m2/s
    velocity_interstitial = row['velocity_interstitial']
    hydraulic_radius_total = row['hydraulic_radius_total']
    
    if None not in [hydraulic_radius_total, velocity_interstitial]:
        reynolds_veg = (velocity_interstitial*hydraulic_radius_total)/kinematic_viscosity
    df_profile_metrics.at[index, 'reynolds_veg'] = reynolds_veg
    
# COEFFICIENT OF DRAG
for index, row in df_profile_metrics.iterrows():
    reynolds_veg = row['reynolds_veg']
    blockage_ratio = row['blockage_ratio']
    wood_sub_area = row['wood_sub_area']
    hydraulic_radius_total = row['hydraulic_radius_total']
    flow_area_extent = row['flow_area_extent']
    # ============================================================================
    if None not in [reynolds_veg] and reynolds_veg !=0:
        coef_drag_veg = (50 / reynolds_veg**0.43) + 0.7*(1 - math.exp(-reynolds_veg / 15000))
    else:
        coef_drag_veg = np.nan
    df_profile_metrics.at[index, 'coef_drag_veg'] = coef_drag_veg #(see Cheng and Nguyen 2011, Eq 14)
    # ============================================================================
    if None not in [reynolds_veg, wood_sub_area, hydraulic_radius_total, flow_area_extent] and reynolds_veg !=0 and hydraulic_radius_total !=0:
        veg_density = (wood_sub_area / hydraulic_radius_total) / flow_area_extent
        coef_drag_liu = (189 / reynolds_veg) + 0.82 + blockage_ratio**2 + 6.02 * veg_density
    else:
        coef_drag_liu = np.nan
    df_profile_metrics.at[index, 'coef_drag_liu'] = coef_drag_liu #(see Liu et al. 2020, Eq 15)        
    # ============================================================================
  
#endregion
#=================================================================================
#region DRAG FORCE EQUATION (see Sharpe et al. 2022)

for index, row in df_profile_metrics.iterrows():
    water_density = 999.65 # kg/m3 at 10 C
    drag_coefficient = 1 # estimated 0.82-~1.2
    coef_drag_veg = row['coef_drag_veg'] # method from Cheng and Nguyen 2011, Eq 14
    wood_sub_area = row['wood_sub_area']
    velocity_extent = row['velocity_extent']
    velocity_interstitial = row['velocity_interstitial']

    if pd.notna(wood_sub_area) and pd.notna(velocity_interstitial):
        F_drag_adjusted = (water_density * wood_sub_area * drag_coefficient * (velocity_interstitial**2))/2
    else:
        F_drag_adjusted = 0
    df_profile_metrics.at[index, 'F_drag_adjusted'] = F_drag_adjusted
    # ============================================================================
    if pd.notna(wood_sub_area) and pd.notna(velocity_extent):
        F_drag_standard = (water_density * wood_sub_area * drag_coefficient * (velocity_extent**2))/2
    else:
        F_drag_standard = 0
    df_profile_metrics.at[index, 'F_drag_standard'] = F_drag_standard
    # ============================================================================
    # for the method from Cheng and Nguyen 2011
    if pd.notna(wood_sub_area) and pd.notna(velocity_interstitial):
        F_drag_veg = (water_density * wood_sub_area * coef_drag_veg * (velocity_interstitial**2))/2
    else:
        F_drag_veg = 0
    df_profile_metrics.at[index, 'F_drag_veg'] = F_drag_veg

#endregion

#=================================================================================
#region MANNING'S N 

# Adjusted Manning's n
for index, row in df_profile_metrics.iterrows():
    hydraulic_radius_total = row['hydraulic_radius_total']
    slope = row['slope']
    flow_area_interstitial = row['flow_area_interstitial']
    discharge = row['discharge']

    if None not in [hydraulic_radius_total, slope, flow_area_interstitial]:
        mannings_n_adjusted = (hydraulic_radius_total**(2/3) * slope**(0.5) * flow_area_interstitial) / discharge

    # Check if mannings_n_adjusted is larger than 1, if so, store NaN
        if mannings_n_adjusted > 1.5:
            df_profile_metrics.at[index, 'mannings_n_adjusted'] = np.nan
        else:
            df_profile_metrics.at[index, 'mannings_n_adjusted'] = mannings_n_adjusted

# Standard Manning's n
for index, row in df_profile_metrics.iterrows():
    hydraulic_radius_channel = row['hydraulic_radius_channel']
    slope = row['slope']
    flow_area_extent = row['flow_area_extent']
    discharge = row['discharge']

    # Check if slope is not None and not NaN
    if None not in [hydraulic_radius_channel, slope, flow_area_extent] and discharge !=0:  
        #slope is not None or pd.notna(slope) and discharge !=0:
        mannings_n_standard = (hydraulic_radius_channel**(2/3) * slope**(0.5) * flow_area_extent) / discharge

        # Check if mannings_n_standard is larger than 1.5, if so, store NaN
        if mannings_n_standard > 1.5:
            df_profile_metrics.at[index, 'mannings_n_standard'] = np.nan
        else:    
            df_profile_metrics.at[index, 'mannings_n_standard'] = mannings_n_standard
            
            
# Petryk's n (1975)
for index, row in df_profile_metrics.iterrows():
    n_0 = 0.034 # base value based on D50 of 29.7 mm
    n_1 = 0.010 # effect of surface irregularities
    n_2 = 0.012 # cariations in shape and size of channel cross section
    n_4 = 0.040 # vegetation and flow conditions
    meander = 1.15 # correction factor for channel meander. my channel length: 108m, valley length: 87m. ratio: 1.24
    
    #coef_drag = 1
    coef_drag_veg = row['coef_drag_veg']
    coef_drag_liu = row['coef_drag_liu']
    gravity = 9.80665 #m/s2
    wp_total = row['wp_total']
    hydraulic_radius_total = row['hydraulic_radius_total']
    hydraulic_radius_channel = row['hydraulic_radius_channel']
    hydraulic_mean_depth = row['hydraulic_mean_depth']
    flow_area_extent = row['flow_area_extent']
    wood_sub_area = row['wood_sub_area']
    wp_wood = row['wp_wood']

    if None not in [hydraulic_radius_total, flow_area_extent]:
        #n_3 = n_0*np.sqrt(1 + ((coef_drag_veg*(wood_sub_area/hydraulic_mean_depth))/(2*gravity*flow_area_extent))*((1/n_0)**2)*(hydraulic_radius_channel**(4/3)))
        #if wp_wood !=0:
            # n_3 = n_0*np.sqrt(1 + ((coef_drag_veg*(wood_sub_area/hydraulic_mean_depth))/(2*gravity*flow_area_extent))*((1/n_0)**2)*(hydraulic_radius_channel**(4/3)))
            # #n_3 = n_0*np.sqrt(1 + ((coef_drag_liu*(wood_sub_area/wp_wood))/(2*gravity*flow_area_extent))*((1/n_0)**2)*(hydraulic_radius_channel**(4/3)))
            # df_profile_metrics.at[index, 'petryk_n3'] = n_3
        n_3 = n_0*np.sqrt(1 + ((coef_drag_veg*(wood_sub_area/hydraulic_mean_depth))/(2*gravity*flow_area_extent))*((1/n_0)**2)*(hydraulic_radius_channel**(4/3)))
        df_profile_metrics.at[index, 'petryk_n3'] = n_3
        #petryks_n = (n_0 + n_1 + n_2 + n_3 + n_4)*meander
        petryks_n = (n_1 + n_2 + n_3 + n_4)*meander
        petryks_n

    # Check if mannings_n_adjusted is larger than 1, if so, store NaN
        if petryks_n > 5:
            df_profile_metrics.at[index, 'petryks_n'] = np.nan
        else:
            df_profile_metrics.at[index, 'petryks_n'] = petryks_n
            
# Sharpe's n (2022)
for index, row in df_profile_metrics.iterrows():
    n_boundary = 0.031
    #coef_drag = 1
    coef_drag_veg = row['coef_drag_veg']
    gravity = 9.80665 #m/s2
    wp_total = row['wp_total']
    F_drag_veg = row['F_drag_veg']
    #hydraulic_radius_total = row['hydraulic_radius_total']
    hydraulic_mean_depth = row['hydraulic_mean_depth']
    flow_area_extent = row['flow_area_extent']
    wood_sub_area = row['wood_sub_area']
    porosity = row['porosity']

    if None not in [hydraulic_mean_depth, flow_area_extent]:
        n_wood = np.sqrt((hydraulic_mean_depth**(1/3)*F_drag_veg)/(water_density*gravity*flow_area_extent*porosity))
        sharpes_n = np.sqrt(n_boundary**2 + n_wood**2)
    df_profile_metrics.at[index, 'sharpes_n'] = sharpes_n

    # Check if mannings_n_adjusted is larger than 1, if so, store NaN
        # if sharpes_n > 50:
        #     df_profile_metrics.at[index, 'sharpes_n'] = np.nan
        # else:
        #     df_profile_metrics.at[index, 'sharpes_n'] = sharpes_n
#endregion

#=================================================================================
#region CONVEYANCE RATIO

for index, row in df_profile_metrics.iterrows():
    flow_area_interstitial = row['flow_area_interstitial']
    wood_sub_area = row['wood_sub_area'] if wood_sub_area else np.nan
    conv_ratio = np.nan  # Assign a default value to conv_ratio
    
    if pd.notna(wood_sub_area) and wood_sub_area!=0 and pd.notna(flow_area_interstitial):
        conv_ratio = flow_area_interstitial / wood_sub_area
        if conv_ratio > 100:
            conv_ratio = np.nan
    df_profile_metrics.at[index, 'conveyance_ratio'] = conv_ratio

#endregion
#=================================================================================
#region DARCY'S F (eq from Wei-Jie Wang et al. 2019)
# or reference Knight, David. 1984. Fluvial forms and processes.

# Adjusted Darcy friction factor
for index, row in df_profile_metrics.iterrows():
    gravity = 9.80665 # m/s^2
    velocity_extent = row['velocity_extent']
    velocity_interstitial = row['velocity_interstitial']
    hydraulic_radius_total = row['hydraulic_radius_total']
    hydraulic_radius_channel = row['hydraulic_radius_channel']
    friction_slope = row['slope'] # this is not actually the friction slope, its the water slope
    
    # Adjusted
    if pd.notna(friction_slope) and pd.notna(hydraulic_radius_total):
        shear_velocity = (gravity * hydraulic_radius_total * friction_slope)**0.5
    else:
        shear_velocity = np.nan
    
    if pd.notna(velocity_interstitial) and pd.notna(friction_slope) and pd.notna(hydraulic_radius_total) and pd.notna(shear_velocity):
        darcy_f_adjusted = 8*(shear_velocity / velocity_interstitial)**2
        if darcy_f_adjusted > 10:
            darcy_f_adjusted = np.nan
    else:
        darcy_f_adjusted = np.nan
    df_profile_metrics.at[index, 'darcy_f_adjusted'] = darcy_f_adjusted
    
    # Standard
    if pd.notna(friction_slope) and pd.notna(hydraulic_radius_channel):
        shear_velocity = (gravity * hydraulic_radius_channel * friction_slope)**0.5
    else:
        shear_velocity = np.nan
    
    if pd.notna(velocity_extent) and pd.notna(friction_slope) and pd.notna(hydraulic_radius_channel) and pd.notna(shear_velocity):
        darcy_f_standard = 8*(shear_velocity / velocity_extent)**2
        if darcy_f_standard > 10:
            darcy_f_standard = np.nan
    else:
        darcy_f_standard = np.nan
    df_profile_metrics.at[index, 'darcy_f_standard'] = darcy_f_standard
        
    
# DRAG COEFFICIENT (from Ferguson 2013)
for index, row in df_profile_metrics.iterrows():
    darcy_f_standard = row['darcy_f_standard']
    drag_coefficient = darcy_f_standard/4
    df_profile_metrics.at[index, 'drag_coefficient'] = drag_coefficient

#endregion
#=================================================================================
#region MULTIPLE LINEAR REGRESSION


# df_mlr = df_profile_metrics.copy() # Make a copy of the DataFrame

# # Define the independent variables
# exclude_variables = ['discharge', 'wp_wood', 'wp_channel', 'wp_total', 'reynolds_veg']  # Add variable names to exclude here
# independent_vars = [col for col in df_mlr.columns if col not in exclude_variables]

# # Convert columns to numeric data types
# numeric_columns = df_mlr.columns.difference(exclude_variables)
# df_mlr[numeric_columns] = df_mlr[numeric_columns].apply(pd.to_numeric, errors='coerce')

# df_mlr = df_mlr.dropna() # Drop rows with NaN values in any column
# X = df_mlr[independent_vars] # Select the independent variables
# X = sm.add_constant(X) # Add constant for intercept

# df_mlr['discharge'] = pd.to_numeric(df_mlr['discharge'], errors='coerce') # Convert 'discharge' column to numeric data type
# y = df_mlr['discharge'] # Select the dependent variable
# model = sm.OLS(y, X).fit() # Perform multiple linear regression
# print(model.summary()) # Print regression summary
# # ========================


# from sklearn.linear_model import LinearRegression
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from scipy import stats

# # Make a copy of the DataFrame
# df_mlr = df_profile_metrics.copy()

# # Define the independent variables
# exclude_variables = ['discharge', 'wp_wood', 'wp_channel', 'wp_total', 'reynolds_veg']
# independent_vars = [col for col in df_mlr.columns if col not in exclude_variables]

# # Convert columns to numeric data types
# numeric_columns = df_mlr.columns.difference(exclude_variables)
# df_mlr[numeric_columns] = df_mlr[numeric_columns].apply(pd.to_numeric, errors='coerce')

# df_mlr = df_mlr.dropna()  # Drop rows with NaN values in any column
# X = df_mlr[independent_vars]  # Select the independent variables
# y = df_mlr['discharge']  # Select the dependent variable

# # Perform multiple linear regression using scikit-learn
# model = LinearRegression()
# model.fit(X, y)

# # Calculate residuals
# y_pred = model.predict(X)
# residuals = y - y_pred

# # Calculate degrees of freedom
# n = len(y)
# k = X.shape[1]
# df = n - k - 1

# # Calculate standard errors of coefficients
# std_err = np.sqrt(np.sum(residuals**2) / df) * np.sqrt(np.diag(np.linalg.inv(np.dot(X.T, X))))

# # Calculate t-values and p-values
# t_values = model.coef_ / std_err
# p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))

# # Display coefficients, t-values, and p-values for each independent variable
# print("Variable           | Coefficient | t-value   | p-value")
# print("-" * 50)
# for col, coef, t_val, p_val in zip(X.columns, model.coef_, t_values, p_values):
#     print(f"{col:<20} | {coef:>11.4f} | {t_val:>9.4f} | {p_val:>8.4f}")






#endregion
#=================================================================================

# SAVE DATAFRAME

with open('df_profile_metrics.pk', 'wb') as f1:
    pickle.dump(df_profile_metrics, f1)

import os
output_path = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\output_tables'
# Check if the output directory exists, and if not, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)
# Write the DataFrame to a CSV file
df_profile_metrics.to_csv(f"{output_path}\\df_profile_metrics.csv", index=False)

# ============================================================================================

