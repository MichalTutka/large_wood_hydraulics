#region IMPORT STATEMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.legend_handler import HandlerTuple
import seaborn as sn
import itertools
import math
import pickle
import os
from IPython import display
display.set_matplotlib_formats("svg")
from datetime import datetime
from matplotlib.ticker import FuncFormatter


# IMPORT FILES
#from geometry_setup import df_profile_metrics

with open('df_profile_metrics.pk', 'rb') as f1:
    df_profile_metrics = pickle.load(f1)   
    
df_profile_metrics.columns

save_directory = 'D:\\Michal\Documents\\0_Grad_School_OSU\\0_Research\\0_Data\\Python\\plots\\' 

# Define the column name mappings
column_mappings = {
    'wse_jam': 'WSE at jam (m)',
    'discharge': 'Discharge (cms)',
    'slope': 'Slope (%)',
    'flow_area_interstitial': 'Flow area (interstitial) (m2)',
    'flow_area_extent': 'Flow area (extent) (m2)',
    'wood_sub_area': 'Frontal area (submerged) (m2)',
    'mannings_n_standard': "Manning's n (standard)",
    'mannings_n_adjusted': "Manning's n (adjusted)",
    'wp_channel': 'WP (channel) (m)',
    'wp_wood': 'WP (wood) (m)',
    'wp_total': 'WP (total) (m)',
    'hydraulic_radius_channel': 'R (channel) (m)',
    'hydraulic_radius_total': 'R (total) (m)',
    'porosity': 'Porosity (%)',
    'blockage_ratio': 'Blockage ratio'
}
    
#endregion
# ========================================================================================
#region WSE-WETTED PERIMETER


# Define wetted mappings and y_vars
wetted_mappings = {'wp_channel': 'channel', 'wp_wood': 'wood', 'wp_total': 'total'}
y_vars = ['wp_total', 'wp_channel', 'wp_wood']
filtered_data = df_profile_metrics.dropna(subset=['wse_jam'] + y_vars) # Filter data and create the plot
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')

# Plot each wetted perimeter variable against wse_jam with specific colors and linestyles
line_styles = ['-', '--', '-']
colors = ['black', '#F6511D', '#3F88C5']
line_widths = [2.0, 2.0, 2.0]
line_labels = ['Total', 'Channel', 'Wood']
label_rotation_angles = [41, 32.5, 9.5]
#text_offsets = [0.5, 0.40, 0.18]  # Adjust these values to manually control the text location for each label
text_offsets = [0.5, 0.70, 0.14]

for i, var in enumerate(y_vars):
    ax.plot(filtered_data['wse_jam'], filtered_data[var], label=wetted_mappings[var], color=colors[i],
            linestyle=line_styles[i], linewidth=line_widths[i], alpha=1)
    ax.xaxis.set_major_formatter('{:.1f}'.format)

# Add labels just below the lines with specified rotation angle
    text = wetted_mappings[var]
    bbox_props = dict(boxstyle="square,pad=0.3", facecolor='white', edgecolor='white', alpha=1)
    ax.text(filtered_data['wse_jam'].iloc[-40], filtered_data[var].iloc[-40] + text_offsets[i],
            wetted_mappings[var], fontsize=12, color=colors[i], rotation=label_rotation_angles[i],
            ha='left', va='center', bbox=bbox_props)
    
ax.set_title('Wetted Perimeter vs WSE', fontsize=16)
ax.set_xlabel('Water Surface Elevation (m)', fontsize=14)
ax.set_ylabel('Wetted Perimeter (m)', fontsize=14)

# Set tick parameters for both x and y axes
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))  # Set major tick every 0.5 units
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Set minor tick every 0.1 units
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))  
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=14, direction='in')
ax.grid(False)
plt.tight_layout()

# save_filename = "wp-wse_plot.png"
# plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)
plt.show()

#endregion
#======================================================================================================

#region WSE-AREA

# Define area mappings and y_vars
area_mappings = {'flow_area_interstitial': 'flow', 'wood_sub_area': 'wood', 'flow_area_extent': 'total'}
y_vars = ['flow_area_extent', 'flow_area_interstitial', 'wood_sub_area']
filtered_data = df_profile_metrics.dropna(subset=['wse_jam'] + y_vars) # Filter data and create the plot
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')

# Plot each area variable against wse_jam with specific colors and linestyles
line_styles = ['-', '--', '-']
colors = ['black', '#F6511D', '#3F88C5']
line_widths = [2.0, 2.0, 2.0]
line_labels = ['Total', 'Flow', 'Wood']
label_rotation_angles = [55, 49, 13]
text_offsets = [0.49, 0.32, 0.12]

for i, var in enumerate(y_vars):
    ax.plot(filtered_data['wse_jam'], filtered_data[var], label=area_mappings[var], color=colors[i],
            linestyle=line_styles[i], linewidth=line_widths[i], alpha=1)
    ax.xaxis.set_major_formatter('{:.1f}'.format)

# Add labels just below the lines with specified rotation angle
    text = area_mappings[var]
    bbox_props = dict(boxstyle="square,pad=0.3", facecolor='white', edgecolor='white', alpha=1)
    ax.text(filtered_data['wse_jam'].iloc[-40], filtered_data[var].iloc[-40] + text_offsets[i],
            area_mappings[var], fontsize=12, color=colors[i], rotation=label_rotation_angles[i],
            ha='left', va='center', bbox=bbox_props)
    
ax.set_title('XS Areas vs WSE', fontsize=16)
ax.set_xlabel('Water Surface Elevation [m]', fontsize=14)
ax.set_ylabel('Area [m2]', fontsize=14)

# Set tick parameters for both x and y axes
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))  # Set major tick every 0.5 units
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Set minor tick every 0.1 units
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16, direction='in')
ax.tick_params(axis='both', which='minor', labelsize=14, direction='in')
ax.grid(False)
plt.tight_layout()

# save_filename = "area-wse_plot.png"
# plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)
plt.show()

#endregion

#=======================================================================================

#region WP and AREA SUB PLOTS (COMBINED)

# Define mappings and variables
wetted_mappings = {'wp_channel': 'channel', 'wp_wood': 'wood', 'wp_total': 'total'}
area_mappings = {'flow_area_interstitial': 'flow', 'wood_sub_area': 'wood', 'flow_area_extent': 'total'}
y_vars_wetted = ['wp_total', 'wp_channel', 'wp_wood']
y_vars_area = ['flow_area_extent', 'flow_area_interstitial', 'wood_sub_area']

# Filter data for the subplots
filtered_data_wetted = df_profile_metrics.dropna(subset=['wse_jam'] + y_vars_wetted)
filtered_data_area = df_profile_metrics.dropna(subset=['wse_jam'] + y_vars_area)

# Create subplots with space between them
fig, axs = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'wspace': 0.3})
fig.patch.set_facecolor('white')

# Loop through subplots
for i, (filtered_data, y_vars, mappings) in enumerate(zip([filtered_data_wetted, filtered_data_area],
                                                           [y_vars_wetted, y_vars_area],
                                                           [wetted_mappings, area_mappings])):
    ax = axs[i]
    title = 'Wetted Perimeter vs. WSE' if i == 0 else 'XS Areas vs. WSE'
    y_label = 'Wetted Perimeter (m)' if i == 0 else 'Area (m$^\mathrm{2}$)'
    line_styles = ['-', '--', '-']
    colors = ['black', '#F6511D', '#3F88C5']
    line_widths = [2.0, 2.0, 2.0]
    label_rotation_angles = [41, 32.5, 9.5] if i == 0 else [55, 49, 13]
    text_offsets = [0.5, 0.70, 0.14] if i == 0 else [0.49, 0.32, 0.12]

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Water Surface Elevation (m)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    for j, var in enumerate(y_vars):
        ax.plot(filtered_data['wse_jam'], filtered_data[var], label=mappings[var], color=colors[j],
                linestyle=line_styles[j], linewidth=line_widths[j], alpha=1)
        ax.xaxis.set_major_formatter('{:.1f}'.format)

        text = mappings[var]
        bbox_props = dict(boxstyle="square,pad=0.3", facecolor='white', edgecolor='white', alpha=1)
        ax.text(filtered_data['wse_jam'].iloc[-40], filtered_data[var].iloc[-40] + text_offsets[j],
                mappings[var], fontsize=12, color=colors[j], rotation=label_rotation_angles[j],
                ha='left', va='center', bbox=bbox_props)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) if i == 0 else ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')
    ax.grid(False)
    
    # Add subplot labels "A." and "B." in the upper left corners
    ax.text(0.15, 0.90, f"({chr(97+i)})", transform=ax.transAxes, color='grey', fontsize=26, weight='regular', va='top', ha='left')

plt.tight_layout()

# Save the combined subplots to a file with today's date affixed to the front
current_date = datetime.now().strftime('%Y%m%d')  # Get today's date in YYYYMMDD format
save_filename = f"{current_date}_wp-area-wse_plot.png"  # Combine date and file name
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)

plt.show()

#endregion

#==============================================================================================

#region MANNING-DARCY vs DISCHARGE & BLOCKAGE RATIO (COMBINED)

friction_mappings = {'mannings_n_adjusted': '$n$ adjusted', 'mannings_n_standard': '$n$ standard',
                     'darcy_f_adjusted': '$f$ adjusted', 'darcy_f_standard': '$f$ standard'}
y_vars1 = ['mannings_n_adjusted', 'mannings_n_standard']
y_vars2 = ['darcy_f_adjusted', 'darcy_f_standard']

# Filter data for the subplots
filtered_data_discharge = df_profile_metrics.dropna(subset=['discharge'] + y_vars1)
filtered_data_blockage = df_profile_metrics.dropna(subset=['blockage_ratio'] + y_vars1)

# Create subplots with space between them
fig, axs = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'wspace': 0.3})
fig.patch.set_facecolor('white')

# Define line styles and colors
line_styles = ['--', '-']

# Loop through subplots
for i, filtered_data in enumerate([filtered_data_discharge, filtered_data_blockage]):
    ax = axs[i]
    ax.set_title("Roughness Coefs. vs. Discharge" if i == 0 else "Roughness Coefs. vs. Blockage Ratio", fontsize=16)
    ax.set_xlabel("Discharge (cms)" if i == 0 else "Blockage Ratio", fontsize=14)
    ax.set_ylabel("Manning's $n$", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')

    for j, var in enumerate(y_vars1):
        ax.plot(filtered_data['discharge'] if i == 0 else filtered_data['blockage_ratio'],
                filtered_data[var], 
                label=friction_mappings[var], 
                color='#2e52B5', 
                linestyle=line_styles[j], 
                alpha=1)

    # Create a twin y-axis (ax2) for plotting Darcy's f data
    ax2 = ax.twinx()
    for j, var in enumerate(y_vars2):
        ax2.plot(filtered_data['discharge'] if i == 0 else filtered_data['blockage_ratio'],
                 filtered_data[var], 
                 label=friction_mappings[var],  # Include Darcy's f in the legend
                 color='#B82B25', 
                 linestyle=line_styles[j], 
                 alpha=1)
    ax2.set_ylabel("Darcy's $f$", fontsize=14)  # Set labels for twin axis
    
    if i == 1:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))  # Format x-axis differently for the second subplot
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax2.set_xlim(0, 0.15)
        ax2.set_xlabel("Blockage Ratio (%)", fontsize=12)  # Add x-axis label for the second subplot
        ax2.set_xticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_xticks()])  # Format x-axis ticks as percentages
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    # Add legend for each axis
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    combined_lines = lines1 + lines2
    combined_labels = labels1 + labels2

    # Rearrange the legend labels order (2, 1, 4, 3)
    legend_order = [1, 0, 3, 2]
    rearranged_labels = [combined_labels[i] for i in legend_order]
    rearranged_lines = [combined_lines[i] for i in legend_order]

    ax.legend(rearranged_lines, rearranged_labels, loc='best', fontsize=12, frameon=False)

    # Set tick parameters for both x and y axes
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax2.tick_params(axis='both', which='minor', labelsize=12, direction='in')
    
    # Add subplot labels "A." and "B." in the upper left corners
    ax.text(0.63, 0.65, f"({chr(97+i)})", transform=ax.transAxes, color='grey', fontsize=24, weight='regular', va='top', ha='left')


plt.tight_layout()

# Save the plot to a file
current_date = datetime.now().strftime('%Y%m%d')
save_filename = f"{current_date}_roughness_discharge_plot.png"
save_path = os.path.join(save_directory, save_filename)

plt.savefig(save_path, format='png', dpi=450)
print(f"Plot saved to: {save_path}")
plt.show()




#endregion

#========================================================================================


#region BACKCALCULATED MANNING-PETRYK vs. DISCHARGE-BLOCKAGE (COMBINED)
friction_mappings = {'mannings_n_standard': 'Manning\'s $n$','petryks_n': 'Petryk\' $n$'}
y_vars1 = ['mannings_n_standard']
y_vars2 = ['petryks_n']

# Filter data for the subplots
filtered_data_discharge = df_profile_metrics.dropna(subset=['discharge'] + y_vars1)
filtered_data_blockage = df_profile_metrics.dropna(subset=['blockage_ratio'] + y_vars1)

# Create subplots with space between them
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={'wspace': 0.4})
fig.patch.set_facecolor('white')
line_styles = ['-', '-'] # Define line styles and colors

# Loop through subplots
for i, filtered_data in enumerate([filtered_data_discharge, filtered_data_blockage]):
    ax = axs[i]
    ax.set_title("Roughness Coefs. vs. Discharge" if i == 0 else "Roughness Coefs. vs. Blockage Ratio", fontsize=16)
    ax.set_xlabel("Discharge (cms)" if i == 0 else "Blockage Ratio", fontsize=14)
    ax.set_ylabel("Manning's $n$", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')

    # Plot the first set of y variables (Manning's n)
    for j, var in enumerate(y_vars1):
        ax.plot(filtered_data['discharge'] if i == 0 else filtered_data['blockage_ratio'],
                filtered_data[var], 
                label=friction_mappings[var], 
                color='#2a6dc8', 
                linestyle=line_styles[j], 
                alpha=1)

    # Create a twin y-axis (ax2) for plotting Darcy's f data
    ax2 = ax.twinx()
    
    # Plot the second set of y variables (Petryk's n)
    for j, var in enumerate(y_vars2):
        ax2.plot(filtered_data['discharge'] if i == 0 else filtered_data['blockage_ratio'],
                 filtered_data[var], 
                 label=friction_mappings[var],  # Include Petryk's n in the legend
                 color='#b3842b', 
                 linestyle=line_styles[j], 
                 alpha=1)
    
    ax2.set_ylabel("Petryk's $n$", fontsize=14)  # Set labels for twin axis
    
    # Add combined legend
    combined_labels = [friction_mappings[var] for var in y_vars1 + y_vars2]
    handles, labels = ax.get_legend_handles_labels()  # Get legend handles and labels for the first axis
    handles2, labels2 = ax2.get_legend_handles_labels()  # Get legend handles and labels for the twin axis
    combined_handles = handles + handles2
    combined_labels.reverse()  # Reverse the order of the legend labels
    combined_handles.reverse()  # Reverse the order of the legend handles
    combined_legend = ax.legend(
        combined_handles, combined_labels,
        loc='upper right',  # You can choose other locations as well
        bbox_to_anchor=(0.95, 0.40),  # Adjust these coordinates as needed
        fontsize=12, frameon=False
    )   
    combined_legend.legendHandles[1].set_color('#2a6dc8')  # Set color for Manning's n line
    combined_legend.legendHandles[0].set_color('#b3842b')  # Set color for Petryk's n line

    # Set tick parameters for both x and y axes
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(.005))
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax2.tick_params(axis='both', which='minor', labelsize=12, direction='in')
    
    # Format blockage ratio tick labels as percentages for the second subplot
    if i == 1:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax2.set_xlim(0.015, 0.15)
        #ax2.set_ylim(filtered_data['petryks_n'].min(), filtered_data['petryks_n'].max())
        ax2.set_xlabel("Blockage Ratio (%)", fontsize=12)
        ax2.set_xticklabels(['{:.0f}%'.format(x * 100) for x in ax2.get_xticks()])
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    
    # Add subplot labels "A." and "B." in the upper left corners
    ax.text(0.15, 0.80, f"({chr(97+i)})", transform=ax.transAxes, color='grey', fontsize=24, weight='regular', va='top', ha='left')

plt.tight_layout()
current_date = datetime.now().strftime('%Y%m%d')
save_filename = f"{current_date}_petryk_back_manning_plot.png"
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)
plt.show()

#endregion

#========================================================
#========================================================

# HORIZONTAL LINE

#region ESTIMATED MANNING-PETRYK vs. DISCHARGE-BLOCKAGE (COMBINED)

friction_mappings = {'mannings_n_standard': "Backcalculated Manning", 'petryks_n': "Petryk + Cowan"}
y_vars1 = ['mannings_n_standard']
y_vars2 = ['petryks_n']

# Filter data for the subplots
filtered_data_discharge = df_profile_metrics.dropna(subset=['discharge'] + y_vars1)
filtered_data_blockage = df_profile_metrics.dropna(subset=['blockage_ratio'] + y_vars1)

# Create subplots with space between them
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={'wspace': 0.4})
fig.patch.set_facecolor('white')
line_styles = ['-', '-']  # Define line styles and colors

# Loop through subplots
for i, filtered_data in enumerate([filtered_data_discharge, filtered_data_blockage]):
    ax = axs[i]
    ax.set_title("Roughness Coefs. vs. Discharge" if i == 0 else "Roughness Coefs. vs. Blockage Ratio", fontsize=16)
    ax.set_xlabel("Discharge (cms)" if i == 0 else "Blockage Ratio", fontsize=14)
    ax.set_ylabel("$n$-value", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')

    # Plot both sets of y variables (Petryk's n and Manning's n) on the same y-axis
    for j, var in enumerate(y_vars2 + y_vars1):
        ax.plot(filtered_data['discharge'] if i == 0 else filtered_data['blockage_ratio'],
                 filtered_data[var],
                 label=friction_mappings[var],
                 color='#b3842b' if j < len(y_vars2) else 'black',
                 linestyle='--' if var == 'mannings_n_standard' else line_styles[j % len(line_styles)],
                 alpha=1 if j < len(y_vars1) else 0.25)
        
        # Save the line handles for legend
        if var == 'petryks_n':
            petryk_cowan_line = ax.get_lines()[-1]  # Save Petryk + Cowan line handle
        elif var == 'mannings_n_standard':
            backcalculated_manning_line = ax.get_lines()[-1]  # Save Backcalculated Manning line handle
    
    manning_arcement_n = 0.1334
    if i == 0:
        hline = ax.hlines(y=manning_arcement_n, xmin=filtered_data['discharge'].min(), xmax=filtered_data['discharge'].max()*1.2, color='#2a6dc8', linestyle='-')
    else:
        hline = ax.hlines(y=manning_arcement_n, xmin=filtered_data['blockage_ratio'].min(), xmax=filtered_data['blockage_ratio'].max()*1.05, color='#2a6dc8', linestyle='-')
    
    # Add combined legend
    combined_labels = ["Petryk + Cowan", "Backcalculated Manning", "Cowan"]
    combined_handles = [petryk_cowan_line, backcalculated_manning_line, hline]
    
    combined_legend = ax.legend(
        combined_handles, combined_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper right',
        bbox_to_anchor=(1.05, 0.6),
        fontsize=11, frameon=False
    )
    
    combined_legend.legendHandles[0].set_color('#b3842b')  # Set color for Petryk + Cowan line
    combined_legend.legendHandles[1].set_color('black')    # Set color for Backcalculated Manning line
    combined_legend.legendHandles[2].set_color('#2a6dc8')     # Set color for Cowan line

    # Set tick parameters for both x and y axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    
    if i == 1:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.set_xlim(0.015, 0.15)
        ax.set_xlabel("Blockage Ratio (%)")
        ax.set_xticklabels(['{:.0f}%'.format(x * 100) for x in ax2.get_xticks()])
        
        petryks_n_max = filtered_data['petryks_n'].max() + 0.01
        ax.set_ylim(manning_arcement_n - 0.005, petryks_n_max)
        ax.legend().set_visible(False) # Remove legend from the second plot
        
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.set_xlim(0, 4)
    ax.set_ylim(manning_arcement_n - 0.01, filtered_data['petryks_n'].max() + 0.005)
    
    # Add subplot labels "A." and "B."
    ax.text(0.08, 0.8, f"({chr(97+i)})", transform=ax.transAxes, color='grey', fontsize=24, weight='regular', va='top', ha='left')

plt.tight_layout()
current_date = datetime.now().strftime('%Y%m%d')
save_filename = f"{current_date}_petryk_est_manning_plot.png"
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)
plt.show()

#endregion


#====================================================
#====================================================
#====================================================
#====================================================


# HORIZONTAL LINE but trying to flip the axes

#region ESTIMATED MANNING-PETRYK vs. DISCHARGE-BLOCKAGE (COMBINED)

friction_mappings = {'mannings_n_standard': "Backcalculated Manning", 'petryks_n': "Petryk + Cowan"}
y_vars1 = ['mannings_n_standard']
y_vars2 = ['petryks_n']

# Filter data for the subplots
filtered_data_discharge = df_profile_metrics.dropna(subset=['discharge'] + y_vars1)
filtered_data_blockage = df_profile_metrics.dropna(subset=['blockage_ratio'] + y_vars1)

# Create subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={'wspace': 0.4})
fig.patch.set_facecolor('white')
line_styles = ['-', '-']  # Define line styles and colors

# Loop through subplots
for i, filtered_data in enumerate([filtered_data_discharge, filtered_data_blockage]):
    ax = axs[i]
    ax.set_title("Roughness Coefs. vs. Discharge" if i == 0 else "Roughness Coefs. vs. Blockage Ratio", fontsize=16)
    ax.set_xlabel("$n$-value", fontsize=14)
    ax.set_ylabel("Discharge (cms)" if i == 0 else "Blockage Ratio", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')

    # Plot both sets of y variables (Petryk's n and Manning's n) on the same x-axis
    for j, var in enumerate(y_vars2 + y_vars1):
        ax.plot(filtered_data[var],
                 filtered_data['discharge'] if i == 0 else filtered_data['blockage_ratio'],
                 label=friction_mappings[var],
                 color='#b3842b' if j < len(y_vars2) else 'black',
                 linestyle='--' if var == 'mannings_n_standard' else line_styles[j % len(line_styles)],
                 alpha=1 if j < len(y_vars1) else 0.25)
        
        # Save the line handles for legend
        if var == 'petryks_n':
            petryk_cowan_line = ax.get_lines()[-1]  # Save Petryk + Cowan line handle
        elif var == 'mannings_n_standard':
            backcalculated_manning_line = ax.get_lines()[-1]  # Save Backcalculated Manning line handle
    
    manning_arcement_n = 0.1334
    if i == 0:
        vline = ax.vlines(x=manning_arcement_n, ymin=filtered_data['discharge'].min(), ymax=filtered_data['discharge'].max()*1.2, color='#2a6dc8', linestyle='-')
    else:
        vline = ax.vlines(x=manning_arcement_n, ymin=filtered_data['blockage_ratio'].min(), ymax=filtered_data['blockage_ratio'].max()*1.05, color='#2a6dc8', linestyle='-')
    
    # Add combined legend
    combined_labels = ["Petryk + Cowan", "Backcalculated Manning", "Cowan"]
    combined_handles = [petryk_cowan_line, backcalculated_manning_line, vline]
    
    combined_legend = ax.legend(
        combined_handles, combined_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper right',
        bbox_to_anchor=(0.9, 0.3),
        fontsize=11, frameon=False
    )
    
    combined_legend.legendHandles[0].set_color('#b3842b')  # Set color for Petryk + Cowan line
    combined_legend.legendHandles[1].set_color('black')    # Set color for Backcalculated Manning line
    combined_legend.legendHandles[2].set_color('#2a6dc8')     # Set color for Cowan line

    # Set tick parameters for both x and y axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
    
    if i == 1:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.set_ylim(0.015, 0.15)
        ax.set_ylabel("Blockage Ratio (%)")
        ax.set_yticklabels(['{:.0f}%'.format(y * 100) for y in ax.get_yticks()])
        
        petryks_n_max = filtered_data['petryks_n'].max() + 0.01
        ax.set_xlim(manning_arcement_n - 0.005, petryks_n_max)
        
        
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.set_xlim(manning_arcement_n - 0.01, filtered_data['petryks_n'].max() + 0.005)
        ax.set_ylim(0, 4)
        ax.legend().set_visible(False) # Remove legend from the second plot
    
    # Add subplot labels "A." and "B."
    ax.text(0.45, 0.85, f"({chr(97+i)})", transform=ax.transAxes, color='grey', fontsize=24, weight='regular', va='top', ha='left')

plt.tight_layout()
current_date = datetime.now().strftime('%Y%m%d')
save_filename = f"{current_date}_petryk_est_manning_plot.png"
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)
plt.show()




#endregion


#====================================================
#====================================================
#====================================================
#====================================================

#region DRAG COEF. vs WSE

from geometry_setup import channel_invert
channel_invert

wse_jam = df_profile_metrics['wse_jam'] - channel_invert
reynolds_veg = df_profile_metrics['reynolds_veg']
coef_drag_veg = df_profile_metrics['coef_drag_veg']
F_drag_standard = df_profile_metrics['F_drag_standard']

coef_drag_veg_excluded = coef_drag_veg[coef_drag_veg < 2.0] # exclude the first outlier
wse_jam_excluded = wse_jam[coef_drag_veg < 2.0] # exclude the first outlier

# Create a figure and axis
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)  # Adjust the figure size as needed

# Create a scatter plot on the axis
ax1.plot(wse_jam, reynolds_veg, color='#52528C', linestyle='-') 
ax1.set_ylabel('Reynolds #')
def thousands_format(x, pos):
    return '{:.0f}k'.format(x/1000)
ax1.yaxis.set_major_formatter(FuncFormatter(thousands_format))

# Create a line plot on the second axis
ax2.plot(wse_jam_excluded, coef_drag_veg_excluded, color='#06A77D', linestyle='-', label='Data Line')
ax2.set_ylabel('Drag coefficient')
ax2.annotate('', xy=(1, 1.35), xytext=(1, 1.63), arrowprops=dict(arrowstyle='->'))

# Create a line plot on the second axis
ax3.plot(wse_jam, F_drag_standard, color='#EE6352', linestyle='-', label='Data Line')
ax3.set_xlabel('Depth (m)')
ax3.set_ylabel('Drag force (N)')
ax3.xaxis.set_major_formatter('{:.1f}'.format)
ax3.annotate('', xy=(0.4, 77), xytext=(0.4, 137), arrowprops=dict(arrowstyle='->')) # Add an arrow annotation to the third subplot

x_axis_limit = (0.3, max(wse_jam))
ax1.set_xlim(x_axis_limit)
ax2.set_xlim(x_axis_limit)
ax3.set_xlim(x_axis_limit)

#plt.tight_layout()
current_date = datetime.now().strftime('%Y%m%d')
save_filename = f"{current_date}_reynolds_to_drag_plot.png"
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=450)
plt.show()















#endregion