#region IMPORT

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from matplotlib.ticker import FormatStrFormatter
import pickle
from datetime import datetime
import os 

save_directory = 'D:\\Michal\Documents\\0_Grad_School_OSU\\0_Research\\0_Data\\Python\\plots\\' 

#===============================

# import files
# this is the transects file: names and distances
with open('df_transect.pk', 'rb') as f6:
    df_transect = pickle.load(f6)

# this is the modified field measurements file: 
with open('expanded_df.pk', 'rb') as f5:
    df_field_points = pickle.load(f5)

columns_to_drop = ['BSE_to_WS_(cm)','top_to_ws_(cm)', 'BSE_to_WS_(m)', 'top_to_ws_(m)',
                   'top_elev','added', 'logger_stage_m']  # List of columns you want to drop
df_field_points = df_field_points.drop(columns=columns_to_drop) # drop specified columns
df_field_points.columns
df_field_points

# data from the geometry file
with open('df_profile_metrics.pk', 'rb') as f7:
    df_profile_metrics = pickle.load(f7)

#endregion
# ======================================================================================

#region TRIANGLES FOR INTERPOLATION

## for jam 3C
logger_ds = 'LL5' # downstream level logger (user defined)
logger_us = 'LL8' # upstream level logger (user defined)
transect = 'LL6-7_32' # transect ID (user defined)

## Distance from downstream logger to upstream logger
upstream_logger_dist = df_transect.loc[df_transect['logger'] ==  logger_us, 'dist_upstr'].values[0] 
downstream_logger_dist = df_transect.loc[df_transect['logger'] == logger_ds, 'dist_upstr'].values[0]          
minireach_distance = upstream_logger_dist - downstream_logger_dist ## upstream logger; upstream distance ((MINUS)) downstream logger; upstream distance 
minireach_distance

## Distance from downstream logger to transect
transect_dist = df_transect.loc[df_transect['profile_id'] == transect, 'dist_upstr'].values[0]        
transect_distance = transect_dist - downstream_logger_dist ## upstream distance to transect ((MINUS)) downstream logger; upstream distance
transect_distance

#endregion
# ======================================================================================

#region COMPILING DATA

unique_discharge_values = df_field_points['discharge_measured_cms'].unique() # extracts unique discharge values
corresponding_dates = df_field_points.groupby('discharge_measured_cms')['date'].first().values #groups by date and selects the first value
df_pred_vs_obs = pd.DataFrame({'date': corresponding_dates, 'q_observed': unique_discharge_values}) # new dataframe with date and corresponding Q values

# Filter based on desired loggers
filtered_ds = df_field_points[df_field_points['logger'] == logger_ds] # Filter df_field_points based on 'logger' condition
filtered_us = df_field_points[df_field_points['logger'] == logger_us] # Filter df_field_points based on 'logger' condition

# Create a dictionary of values
wse_ds_dict = filtered_ds.groupby('discharge_measured_cms')['wse_measured_m'].first().to_dict() # Create a dictionary with 'wse_measured_m' values based on 'discharge_measured_cms'
wse_us_dict = filtered_us.groupby('discharge_measured_cms')['wse_measured_m'].first().to_dict() # Create a dictionary with 'wse_measured_m' values based on 'discharge_measured_cms'

df_pred_vs_obs['wse_ds'] = df_pred_vs_obs['q_observed'].map(wse_ds_dict) # Add 'wse_ds' column to df_pred_vs_obs and populate it with 'wse_measured_m' values
df_pred_vs_obs['wse_ds'] = df_pred_vs_obs['wse_ds'].copy() # Add 'wse_ds' column as a copy of 'wse_ds'

df_pred_vs_obs['wse_us'] = df_pred_vs_obs['q_observed'].map(wse_us_dict) # Add 'wse_us' column to df_pred_vs_obs and populate it with 'wse_measured_m' values
df_pred_vs_obs['wse_us'] = df_pred_vs_obs['wse_us'].copy() # Add 'wse_us' column as a copy of 'wse_us'

#===============================
# INTERPOLATING POINTS
df_pred_vs_obs['wse_jam'] = ((df_pred_vs_obs['wse_us'] - df_pred_vs_obs['wse_ds']) / minireach_distance) * transect_distance + df_pred_vs_obs['wse_ds']
df_pred_vs_obs

#endregion

# ======================================================================================

#region AREA, HYD. RADIUS, AND SLOPE FOR STANDARD MANNING'S FORMULA

# Function to find the nearest value in an array
def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Create a new column 'flow_area_approach' and populate it with the 'flow_area_extent' values based on the nearest 'wse_jam' values
df_pred_vs_obs['flow_area_approach'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'flow_area_extent'].iloc[0])
df_pred_vs_obs['flow_area_pore'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'flow_area_interstitial'].iloc[0])
df_pred_vs_obs['hyd_radius_channel'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'hydraulic_radius_channel'].iloc[0])
df_pred_vs_obs['hyd_radius_total'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'hydraulic_radius_total'].iloc[0])
df_pred_vs_obs['slope'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'slope'].iloc[0])
df_pred_vs_obs['frontal_area'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'wood_sub_area'].iloc[0])
df_pred_vs_obs['wp_total'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'wp_total'].iloc[0])
df_pred_vs_obs['coef_drag_veg'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'coef_drag_veg'].iloc[0])
df_pred_vs_obs['F_drag_veg'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'F_drag_veg'].iloc[0])
df_pred_vs_obs['porosity'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'porosity'].iloc[0])
df_pred_vs_obs['hydraulic_mean_depth'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'hydraulic_mean_depth'].iloc[0])
df_pred_vs_obs['coef_drag_liu'] = df_pred_vs_obs['wse_jam'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['wse_jam'].apply(lambda y: find_nearest_value(df_profile_metrics['wse_jam'], x) == y), 'coef_drag_liu'].iloc[0])



# Standard Manning's Formula
for index, row in df_pred_vs_obs.iterrows():
    
    # based on Arcement Jr and Schneider (1989): 
    n_b = 0.034 # base value based on D50 of 29.7 mm
    n_1 = 0.010 # effect of surface irregularities
    n_2 = 0.012 # variations in shape and size of channel cross section
    n_3 = 0.020 # obstructions
    n_4 = 0.040 # vegetation and flow conditions
    meander = 1.15 # correction factor for channel meander. my channel length: 108m, valley length: 87m. ratio: 1.24
    mannings_n_arcement = (n_b + n_1 + n_2 + n_3 + n_4)*meander
    mannings_n_arcement
    
    hyd_radius_channel = row['hyd_radius_channel']
    slope = row['slope']
    flow_area_approach = row['flow_area_approach']
    flow_area_pore = row['flow_area_pore']
    #discharge = row['discharge']

    # Check if slope is not None and not NaN
    if None not in [hyd_radius_channel, slope, flow_area_approach]:  
        q_manning = (flow_area_approach * hyd_radius_channel**(2/3) * slope**(0.5) ) / mannings_n_arcement
    df_pred_vs_obs.at[index, 'q_manning'] = q_manning

df_pred_vs_obs

#endregion
# ======================================================================================
# Petryk's n and Q
for index, row in df_pred_vs_obs.iterrows():    

    n_0 = 0.034 # base value based on D50 of 29.7 mm
    n_1 = 0.010 # effect of surface irregularities
    n_2 = 0.012 # cariations in shape and size of channel cross section
    n_4 = 0.040 # vegetation and flow conditions
    meander = 1.15 # correction factor for channel meander. my channel length: 108m, valley length: 87m. ratio: 1.24
    n_boundary = (n_0 + n_1 + n_2 + 0 + n_4)*meander
    n_boundary
    
    gravity = 9.80665 #m/s2
    #coef_drag = 1
    coef_drag_veg = row['coef_drag_veg']
    coef_drag_liu = row['coef_drag_liu']
    wp_total = row['wp_total']
    hyd_radius_total = row['hyd_radius_total']
    flow_area_approach = row['flow_area_approach']
    hyd_radius_channel = row['hyd_radius_channel']
    flow_area_pore = row['flow_area_pore']
    frontal_area = row['frontal_area']
    slope = row['slope']
    hydraulic_mean_depth = row['hydraulic_mean_depth']
    
    if None not in [hyd_radius_total, flow_area_approach]:
        petryks_n = n_0*np.sqrt(1 + ((coef_drag_veg *(frontal_area/hydraulic_mean_depth))/(2*gravity*flow_area_approach))*((1/n_0)**2)*(hyd_radius_channel**(4/3)))
        petryk_cowan = (n_1 + n_2 + petryks_n + n_4)*meander
        
        q_petryk = (flow_area_approach * hyd_radius_channel**(2/3) * slope**(0.5) ) / petryk_cowan
        petryks_n

    df_pred_vs_obs.at[index, 'q_petryk'] = q_petryk
df_pred_vs_obs


# ======================================================================================
# Sharpes's n and Q
for index, row in df_pred_vs_obs.iterrows():    

    n_boundary = 0.034 # base value based on D50 of 29.7 mm
    gravity = 9.80665 #m/s2
    water_density = 999.65 # kg/m3 at 10 C
    #coef_drag = 1
    coef_drag_veg = row['coef_drag_veg']
    wp_total = row['wp_total']
    hyd_radius_total = row['hyd_radius_total']
    flow_area_approach = row['flow_area_approach']
    flow_area_pore = row['flow_area_pore']
    frontal_area = row['frontal_area']
    slope = row['slope']
    F_drag_veg = row['F_drag_veg']
    hydraulic_mean_depth = row['hydraulic_mean_depth']
    porosity = row['porosity']

    if None not in [hydraulic_mean_depth, flow_area_approach]:
        n_wood = np.sqrt((hydraulic_mean_depth**(1/3)*F_drag_veg)/(water_density*gravity*flow_area_approach*porosity))
        sharpes_n = np.sqrt(n_boundary**2 + n_wood**2)
        q_sharpe = (flow_area_pore * hyd_radius_total**(2/3) * slope**(0.5) ) / sharpes_n
        sharpes_n
       
    df_pred_vs_obs.at[index, 'q_sharpe'] = q_sharpe

df_pred_vs_obs
# ======================================================================================

#region Petryk's n and Q - DISCHARGE

df_pred_vs_obs.dropna(subset=['q_manning', 'q_petryk', 'q_observed'], inplace=True) # Handle NaN values
fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis

# Plot the 1:1 line
min_value = min(df_pred_vs_obs['q_observed'].min(), df_pred_vs_obs[['q_manning', 'q_petryk']].min().min())
max_value = max(df_pred_vs_obs['q_observed'].max(), df_pred_vs_obs[['q_manning', 'q_petryk']].max().max())
ax.plot([0, max_value + (max_value * 2)], [0, max_value + (max_value * 2)], color='black', alpha=1, zorder=1)

# Plot the data
point_size = 50
line_width = 1
man_color = '#2a6dc8'
pet_color = '#b3842b'
ax.scatter(df_pred_vs_obs['q_manning'], df_pred_vs_obs['q_observed'], label='Manning', marker='o', 
           color=man_color, edgecolors='none', alpha=0.6, s=point_size) # face Manning
ax.scatter(df_pred_vs_obs['q_manning'], df_pred_vs_obs['q_observed'], marker='o', facecolors='none',
           edgecolors=(0, 0, 0), s=point_size, linewidths=line_width) # edge Manning
ax.scatter(df_pred_vs_obs['q_petryk'], df_pred_vs_obs['q_observed'], label='Petryk', marker='o', 
           color=pet_color, edgecolors='none', alpha=0.6, s=point_size) # face Petryk
ax.scatter(df_pred_vs_obs['q_petryk'], df_pred_vs_obs['q_observed'], marker='o', facecolors='none',
           edgecolors=(0, 0, 0), s=point_size, linewidths=line_width) # edge Petryk
ax.set_xlim(0, max_value + (max_value * 0.25))
ax.set_ylim(0, max_value + (max_value * 0.25))
ax.grid(True, alpha=0.3)


# Set labels and title
ax.set_xlabel('Q expected (cms)', fontsize=12)
ax.set_ylabel('Q observed (cms)', fontsize=12)
ax.set_title('Observed vs. Expected Discharge', fontsize=16)

# Annotate R2 values
annotation_positions = {'q_manning': (0.2, 2.3), 'q_petryk': (0.2, 2.15)} # Manual annotation positions for each method
for method, label_color, label_name in [('q_manning', man_color, 'Cowan'), ('q_petryk', pet_color, 'Petryk+Cowan')]:
    r2 = r2_score(df_pred_vs_obs['q_observed'], df_pred_vs_obs[method])
    ax.annotate(f'R$^2$ {label_name} = {r2:.2f}', 
                xy=annotation_positions[method],  # Specify manual annotation position
                color=label_color, fontsize=12, ha='left', va='center')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.legend(fontsize=10)  # Add legend
ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
plt.tight_layout()

# Save the plot to a file
current_date = datetime.now().strftime('%Y%m%d')  # Get today's date in YYYYMMDD format
save_filename = f"{current_date}_expected_vs_observed_q_plot.png"  # Combine date and file name
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)

plt.show()  # Show the plot


#endregion 

#==========================================================================================


#region Petryk's n and Q - WSE

# get the WSE values for each based on the discharge from profile metrics
df_pred_vs_obs['wse_manning'] = df_pred_vs_obs['q_manning'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['discharge'].apply(lambda y: find_nearest_value(df_profile_metrics['discharge'], x) == y), 'wse_jam'].iloc[0])
df_pred_vs_obs['wse_petryk'] = df_pred_vs_obs['q_petryk'].apply(lambda x: df_profile_metrics.loc[df_profile_metrics['discharge'].apply(lambda y: find_nearest_value(df_profile_metrics['discharge'], x) == y), 'wse_jam'].iloc[0])

# change q_manning to wse_manning
# chang q_petryk to wse_petryk

df_pred_vs_obs.dropna(subset=['wse_manning', 'wse_petryk', 'wse_jam'], inplace=True) # Handle NaN values
fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis

# Plot the 1:1 line
min_value = min(df_pred_vs_obs['wse_jam'].min(), df_pred_vs_obs[['wse_manning', 'wse_petryk']].min().min())
max_value = max(df_pred_vs_obs['wse_jam'].max(), df_pred_vs_obs[['wse_manning', 'wse_petryk']].max().max())
ax.plot([0, max_value + (max_value * 2)], [0, max_value + (max_value * 2)], color='black', alpha=1, zorder=1)

# Plot the data
point_size = 50
line_width = 1
man_color = '#2a6dc8'
pet_color = '#b3842b'
ax.scatter(df_pred_vs_obs['wse_manning'], df_pred_vs_obs['wse_jam'], label='Manning', marker='o', 
           color=man_color, edgecolors='none', alpha=0.6, s=point_size) # face Manning
ax.scatter(df_pred_vs_obs['wse_manning'], df_pred_vs_obs['wse_jam'], marker='o', facecolors='none',
           edgecolors=(0, 0, 0), s=point_size, linewidths=line_width) # edge Manning
ax.scatter(df_pred_vs_obs['wse_petryk'], df_pred_vs_obs['wse_jam'], label='Petryk', marker='o', 
           color=pet_color, edgecolors='none', alpha=0.6, s=point_size) # face Petryk
ax.scatter(df_pred_vs_obs['wse_petryk'], df_pred_vs_obs['wse_jam'], marker='o', facecolors='none',
           edgecolors=(0, 0, 0), s=point_size, linewidths=line_width) # edge Petryk
ax.set_xlim(999.7, 1000.3)
ax.set_ylim(999.7, 1000.3)
ax.xaxis.set_major_formatter('{:.1f}'.format)
ax.yaxis.set_major_formatter('{:.1f}'.format)
ax.grid(True, alpha=0.3)

# Set labels and title
ax.set_xlabel('WSE estimated (m)', fontsize=12)
ax.set_ylabel('WSE observed (m)', fontsize=12)
ax.set_title('Observed vs. Estimated WSE', fontsize=16)

# Annotate R2 values
annotation_positions = {'wse_manning': (999.74, 1000.18), 'wse_petryk': (999.74, 1000.145)} # Manual annotation positions for each method
for method, label_color, label_name in [('wse_manning', man_color, 'Cowan'), ('wse_petryk', pet_color, 'Petryk+Cowan')]:
    r2 = r2_score(df_pred_vs_obs['wse_jam'], df_pred_vs_obs[method])
    ax.annotate(f'R$^2$ {label_name} = {r2:.2f}', 
                xy=annotation_positions[method],  # Specify manual annotation position
                color=label_color, fontsize=12, ha='left', va='center')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.legend(fontsize=10)  # Add legend
ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
plt.tight_layout()

# Save the plot to a file
current_date = datetime.now().strftime('%Y%m%d')  # Get today's date in YYYYMMDD format
save_filename = f"{current_date}_estimated_vs_observed_wse_plot.png"  # Combine date and file name
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)

plt.show()  # Show the plot





#endregion

#==========================================================================================


# #region TESTING MY SHIT
# 'tutkas_q_1_1', 'tutkas_q_1_2', 'tutkas_q_1_3',
#        'tutkas_q_2_3', 'tutkas_q_4_3', 'tutkas_q_2_1', 'tutkas_q_3_2'

# fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure and axis

# # Plot the 1:1 line
# min_value = min(df_pred_vs_obs['q_observed'].min(), df_pred_vs_obs[['q_manning', 'q_petryk', 'tutkas_q_1_1', 'tutkas_q_1_2', 'tutkas_q_1_3',
#        'tutkas_q_2_3', 'tutkas_q_4_3', 'tutkas_q_2_1', 'tutkas_q_3_2']].min().min())
# max_value = max(df_pred_vs_obs['q_observed'].max(), df_pred_vs_obs[['q_manning', 'q_petryk']].max().max())
# ax.plot([0, max_value + (max_value * 2)], [0, max_value + (max_value * 2)], color='black', alpha=0.9, label='1:1 line')

# # Plot the data
# ax.scatter(df_pred_vs_obs['q_manning'], df_pred_vs_obs['q_observed'], label='q_manning', marker='o', edgecolors=(0, 0, 0))
# ax.scatter(df_pred_vs_obs['q_petryk'], df_pred_vs_obs['q_observed'], label='q_petryk', marker='o', edgecolors=(0, 0, 0))
# ax.set_xlim(0, max_value + (max_value * 0.25))
# ax.set_ylim(0, max_value + (max_value * 0.25))

# # Set labels and title
# ax.set_xlabel('Q predicted (cms)')
# ax.set_ylabel('Q observed (cms)')
# ax.set_title('Observed vs. Predicted Flow')

# # Calculate and annotate R2 values
# for method, label_color, label_name in [('q_manning', 'blue', 'Manning'), ('q_petryk', 'green', 'Petryk')]:
#     observed = df_pred_vs_obs['q_observed'].astype(float)  # Convert to float type
#     predicted = df_pred_vs_obs[method].astype(float)       # Convert to float type
    
#     # Remove NaN values and align data lengths
#     mask = ~np.isnan(observed) & ~np.isnan(predicted)
#     observed_aligned = observed[mask]
#     predicted_aligned = predicted[mask]
    
#     r2 = r2_score(observed_aligned, predicted_aligned)
    
#     ax.annotate(f'R2 {label_name} = {r2:.2f}', (predicted_aligned.max(), observed_aligned.max()), color=label_color, fontsize=10, ha='right', textcoords="offset points", xytext=(0,10))

# ax.legend()  # Add legend
# plt.show()  # Show the plot










#endregion





# ======================================================================================
# Save the modified DataFrame to a new pickle file
with open('modified_df.pk', 'wb') as f6:
    pickle.dump(df_field_points, f6)