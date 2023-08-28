#region IMPORT STATEMENTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
import seaborn as sn
import itertools
import math
import pickle
from IPython import display
display.set_matplotlib_formats("svg")

# IMPORT FILES
#from geometry_setup import df_profile_metrics

with open('df_profile_metrics.pk', 'rb') as f1:
    df_profile_metrics = pickle.load(f1)   

#endregion
# ========================================================================================

df_profile_metrics.columns

#region SIMPLE PAIR PLOT
columns = df_profile_metrics.columns
columns = columns.drop('flow_area_extent')  # Exclude 'flow_area_extent' from the columns
columns = columns.drop('hydraulic_radius_channel')  # Exclude 'hydraulic_radius_channel' from the columns
columns = columns.drop('hydraulic_radius_total')  # Exclude 'hydraulic_radius_total' from the columns
#columns = columns.drop('wp_channel')  # Exclude 'wp_channel' from the columns
#columns = columns.drop('wp_wood')  # Exclude 'wp_wood' from the columns
#columns = columns.drop('wp_total')  # Exclude 'wp_total' from the columns
num_cols = len(columns)

fig, axs = plt.subplots(num_cols, num_cols, figsize=(120, 120))

# Define the column name mappings
column_mappings = {
    'wse_jam': 'WSE at jam [m]',
    'discharge': 'Discharge [cms]',
    'slope': 'Slope [%]',
    'flow_area_interstitial': 'Flow area (interstitial) [m2]',
    'flow_area_extent': 'Flow area (extent) [m2]',
    'wood_sub_area': 'Frontal area (submerged) [m2]',
    'mannings_n_standard': "Manning's n (standard)",
    'mannings_n_adjusted': "Manning's n (adjusted)",
    'wp_channel': 'WP (channel) [m]',
    'wp_wood': 'WP (wood) [m]',
    'wp_total': 'WP (total) [m]',
    'hydraulic_radius_channel': 'R (channel) [m]',
    'hydraulic_radius_total': 'R (total) [m]',
    'porosity': 'Porosity [%]',
    'blockage_ratio': 'Blockage ratio'
}

# Iterate over unique pairs of variables
for i, col1 in enumerate(columns):
    for j, col2 in itertools.islice(enumerate(columns), i + 1, None):
        ax = axs[i, j]
        if col1 == col2:  # Check if variable is plotted against itself
            ax.set_facecolor('lightgray')  # Set background color to gray
        elif col1 == 'mannings_n_standard' or col2 == 'mannings_n_standard' or col1 == 'mannings_n_adjusted' or col2 == 'mannings_n_adjusted':  # Check if either axis is 'mannings_n'
            mask = df_profile_metrics['mannings_n_adjusted'] <= 1  # Create a mask to filter values <= 1
            ax.plot(df_profile_metrics.loc[mask, col1], df_profile_metrics.loc[mask, col2])
        else:
            ax.plot(df_profile_metrics[col1], df_profile_metrics[col2])
        ax.set_xlabel(column_mappings.get(col1, col1), fontsize=12)
        ax.set_ylabel(column_mappings.get(col2, col2), fontsize=12)
        ax.set_title(f'{column_mappings.get(col1, col1)} vs {column_mappings.get(col2, col2)}', fontsize=12)

        if col2 == 'slope':
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        if col2 == 'porosity':
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        if col1 == 'wse_jam':
            ax.xaxis.set_major_formatter('{:.2f}'.format)

# Remove empty subplots
if num_cols % 2 != 0:
    fig.delaxes(axs[-1, -1])
for i in range(num_cols):
    for j in range(i + 1, num_cols):
        fig.delaxes(axs[j, i])

fig.patch.set_facecolor('white')
#plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()



#endregion
# ===================================================

#region WETTED PERIMETER PAIR PLOT 

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'hydraulic_radius_channel',
                   'hydraulic_radius_total', 'porosity', 'blockage_ratio',
                   'mannings_n_adjusted', 'darcy_f_adjusted']

# Create a list with the three columns you want to plot together on the x-axis
y_vars = ['wp_channel', 'wp_wood', 'wp_total']

# Dictionary for mapping column names to labels
column_mappings = {
    'wse_jam': 'WSE at jam [m]',
    'discharge': 'Discharge [cms]',
    'slope': 'Slope [%]',
    'flow_area_interstitial': 'Flow area (interstitial) [m2]',
    'flow_area_extent': 'Flow area (extent) [m2]',
    'wood_sub_area': 'Frontal area (submerged) [m2]',
    'mannings_n_standard': "Manning's n (standard)",
    'mannings_n_adjusted': "Manning's n (adjusted)",
    'wp_channel': 'WP (channel) [m]',
    'wp_wood': 'WP (wood) [m]',
    'wp_total': 'WP (total) [m]',
    'hydraulic_radius_channel': 'R (channel) [m]',
    'hydraulic_radius_total': 'R (total) [m]',
    'porosity': 'Porosity [%]',
    'blockage_ratio': 'Blockage ratio [%]',
    'conveyance_ratio': 'Conveyance ratio',
    'drag_coefficient': 'Cd', 
    'velocity_interstitial': 'Velocity (interstitial) [m/s]',
    'velocity_extent': 'Velocity (extent) [m/s]' , 
    'F_drag_adjusted': 'F drag (adjusted)', 
    'F_drag_standard': 'F drag (standard)',
    'darcy_f_adjusted': 'DW friction factor (adjusted)', 
    'darcy_f_standard': 'DW friction factor (standard)'
}

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows)) # Create the figure and axes for subplots
axes = axes.flatten() # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, x_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Filter data for subplots with 'mannings_n_standard' or 'mannings_n_adjusted'
    if x_var in ['mannings_n_adjusted']:
        filtered_data = df_profile_metrics[df_profile_metrics[x_var] < 1]
    else:
        filtered_data = df_profile_metrics

    # Plot wp_channel, wp_wood, and wp_total as separate lines on the x-axis
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[2]], label='wp_total', color='black', alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[0]], label='wp_channel', color='red', linestyle='--', alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[1]], label='wp_wood', color='green', alpha=1)
 
    axes[i].set_title(f'{column_mappings[x_var]} vs Wetted Perimeter') # Set the subplot title  
    axes[i].set_ylabel("Wetted perimeter [m]") # Set labels for x and y axes using column mappings
    axes[i].set_xlabel(column_mappings[x_var]) # Set labels for x and y axes using column mappings
    
    # Format y-axis as a percentage for 'slope' and 'porosity'
    if x_var == 'slope':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if x_var == 'porosity' or x_var == 'blockage_ratio':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if x_var == 'wse_jam':
        axes[i].xaxis.set_major_formatter('{:.2f}'.format)

    axes[i].legend() # Show the legend for the lines
    axes[i].grid(True, alpha=0.4) # add gri

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])

fig.patch.set_facecolor('white')

fig.tight_layout() # Adjust the layout to prevent overlapping titles
plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.show() # Show the plots

#endregion


# ===================================================

#region HYDRAULIC RADIUS PLOTS

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'porosity', 'blockage_ratio',
                   'mannings_n_adjusted', 'mannings_n_standard', 
                   'darcy_f_adjusted','conveyance_ratio', 'F_drag_adjusted']

# Create a list with the three columns you want to plot together on the x-axis
x_vars = ['hydraulic_radius_channel', 'hydraulic_radius_total']

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(12, 6 * n_rows)) # Create the figure and axes for subplots
axes = axes.flatten() # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, y_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Filter data for subplots with 'mannings_n_standard' or 'mannings_n_adjusted'
    if y_var in ['mannings_n_standard', 'mannings_n_adjusted']:
        filtered_data = df_profile_metrics[df_profile_metrics[y_var] < 1]
    else:
        filtered_data = df_profile_metrics

    # Plot wp_channel, wp_wood, and wp_total as separate lines on the x-axis
    line_colors = ['red', 'black'] # Colors for the lines
    axes[i].plot(filtered_data[x_vars[1]], filtered_data[y_var], label='hydraulic_radius_total', color=line_colors[1], alpha=1)
    axes[i].plot(filtered_data[x_vars[0]], filtered_data[y_var], label='hydraulic_radius_channel', color=line_colors[0], linestyle='--', alpha=1)
 
    axes[i].set_title(f'{column_mappings[y_var]} vs R') # Set the subplot title  
    axes[i].set_xlabel("Hydraulic radius [m]") # Set labels for x and y axes using column mappings
    axes[i].set_ylabel(column_mappings[y_var]) # Set labels for x and y axes using column mappings
    
    # Format y-axis as a percentage for 'slope' and 'porosity'
    if y_var == 'slope':
        axes[i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if y_var == 'porosity' or y_var == 'blockage_ratio':
        axes[i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if y_var == 'wse_jam':
        axes[i].yaxis.set_major_formatter('{:.2f}'.format)

    axes[i].legend() # Show the legend for the lines
    axes[i].grid(True, alpha=0.4) # add gri

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])

fig.patch.set_facecolor('white')
fig.tight_layout() # Adjust the layout to prevent overlapping titles
plt.show() # Show the plots

#endregion

# ===================================================

#region MANNING'S N PLOTS

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'porosity', 'blockage_ratio', 
                   'darcy_f_adjusted','conveyance_ratio', 'F_drag_adjusted']

# Create a list with the two columns you want to plot together on the x-axis
y_vars = ['mannings_n_adjusted', 'mannings_n_standard']

# Filter data to exclude values of 'mannings_n_standard' and 'mannings_n_adjusted' >= 1
filtered_data = df_profile_metrics[(df_profile_metrics['mannings_n_standard'] < 1) &
                                   (df_profile_metrics['mannings_n_adjusted'] < 1)]

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(12, 6 * n_rows))  # Create the figure and axes for subplots
axes = axes.flatten()  # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, x_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Plot mannings_n_adjusted and mannings_n_standard as separate lines on the x-axis
    line_colors = ['blue', 'black']  # Colors for the lines
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[1]], label='mannings_n_standard', color=line_colors[1], alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[0]], label='mannings_n_adjusted', color=line_colors[0], alpha=1)

    axes[i].set_title(f"{column_mappings[x_var]} vs Manning's n")  # Set the subplot title
    axes[i].set_ylabel("Manning's n")  # Set labels for x and y axes using column mappings
    axes[i].set_xlabel(column_mappings[x_var])  # Set labels for x and y axes using column mappings

    # Format y-axis as a percentage for 'slope' and 'porosity'
    if x_var == 'slope':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if x_var == 'porosity' or x_var == 'blockage_ratio':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if x_var == 'wse_jam':
        axes[i].xaxis.set_major_formatter('{:.2f}'.format)

    axes[i].legend()  # Show the legend for the lines
    axes[i].grid(True, alpha=0.4)  # Add grid

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])
    
fig.patch.set_facecolor('white')
fig.tight_layout()  # Adjust the layout to prevent overlapping titles
plt.show()  # Show the plots

#endregion

# ===================================================
#region MANNING AND DARCY COMBINED PLOTS

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'porosity', 'blockage_ratio', 
                   'conveyance_ratio', 'F_drag_adjusted']

# Create a list with the two columns you want to plot together on the x-axis
y_vars1 = ['mannings_n_adjusted', 'mannings_n_standard']
y_vars2 = ['darcy_f_adjusted', 'darcy_f_standard']

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(14, 5 * n_rows))  # Create the figure and axes for subplots
axes = axes.flatten()  # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, x_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Plot mannings_n_adjusted and mannings_n_standard as separate lines on the x-axis
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars1[1]], label='n (standard)', color="#2e52B5", alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars1[0]], label='n (adjusted)', color="#2e52B5", linestyle='--', alpha=1)
    
    # Create twin axes for y_vars2 and plot them as separate lines on the x-axis
    ax2 = axes[i].twinx()
    ax2.plot(filtered_data[x_var], filtered_data[y_vars2[1]], label='f (standard)', color="#B82B25", alpha=1)
    ax2.plot(filtered_data[x_var], filtered_data[y_vars2[0]], label='f (adjusted)', color="#B82B25", linestyle='--', alpha=1)  
    
    axes[i].set_title(f"{column_mappings[x_var]} vs Manning's n")  # Set the subplot title
    axes[i].set_ylabel("Manning's n")  # Set labels for x and y axes using column mappings
    ax2.set_ylabel("Darcy's f")  # Set labels for twin axis
    axes[i].set_xlabel(column_mappings[x_var])  # Set labels for x and y axes using column mappings

    # Format y-axis as a percentage for 'slope' and 'porosity'
    if x_var == 'slope':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if x_var == 'porosity' or x_var == 'blockage_ratio':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if x_var == 'wse_jam':
        axes[i].xaxis.set_major_formatter('{:.2f}'.format)

    # Combine the legends of both y-axes
    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    axes[i].grid(True, alpha=0.4)  # Add grid

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])
    
fig.patch.set_facecolor('white')
fig.tight_layout()  # Adjust the layout to prevent overlapping titles
plt.show()  # Show the plots


#endregion

# ===================================================
#region MANNING AND PETRYK COMBINED PLOTS

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'porosity', 'blockage_ratio', 
                   'conveyance_ratio', 'F_drag_adjusted']

# Create a list with the two columns you want to plot together on the x-axis
y_vars1 = ['mannings_n_adjusted', 'mannings_n_standard']
y_vars2 = ['petryks_n']

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(14, 5 * n_rows))  # Create the figure and axes for subplots
axes = axes.flatten()  # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, x_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Plot mannings_n_adjusted and mannings_n_standard as separate lines on the x-axis
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars1[1]], label='n (standard)', color="#2e52B5", alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars1[0]], label='n (adjusted)', color="#2e52B5", linestyle='--', alpha=1)
    
    # Create twin axes for y_vars2 and plot them as separate lines on the x-axis
    ax2 = axes[i].twinx()
    ax2.plot(filtered_data[x_var], filtered_data[y_vars2[0]], label='petryks_n', color="#B82B25", alpha=1)
    
    axes[i].set_title(f"{column_mappings[x_var]} vs Manning's n")  # Set the subplot title
    axes[i].set_ylabel("Manning's n")  # Set labels for x and y axes using column mappings
    ax2.set_ylabel("petryks_n")  # Set labels for twin axis
    axes[i].set_xlabel(column_mappings[x_var])  # Set labels for x and y axes using column mappings

    # Format y-axis as a percentage for 'slope' and 'porosity'
    if x_var == 'slope':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if x_var == 'porosity' or x_var == 'blockage_ratio':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if x_var == 'wse_jam':
        axes[i].xaxis.set_major_formatter('{:.2f}'.format)

    # Combine the legends of both y-axes
    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    axes[i].grid(True, alpha=0.4)  # Add grid

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])
    
fig.patch.set_facecolor('white')
fig.tight_layout()  # Adjust the layout to prevent overlapping titles
plt.show()  # Show the plots

#endregion
# ===================================================

#region SHARPE AND PETRYK COMBINED PLOTS

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'porosity', 'blockage_ratio', 
                   'conveyance_ratio', 'F_drag_adjusted']

# Create a list with the two columns you want to plot together on the x-axis
y_vars1 = ['sharpes_n']
y_vars2 = ['petryks_n']

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(14, 5 * n_rows))  # Create the figure and axes for subplots
axes = axes.flatten()  # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, x_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Plot mannings_n_adjusted and mannings_n_standard as separate lines on the x-axis
    # axes[i].plot(filtered_data[x_var], filtered_data[y_vars[1]], label='n (standard)', color="#2e52B5", alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars1[0]], label='n Sharpe', color="purple", alpha=1)
    
    # Create twin axes for y_vars2 and plot them as separate lines on the x-axis
    ax2 = axes[i].twinx()
    ax2.plot(filtered_data[x_var], filtered_data[y_vars2[0]], label='petryks_n', color="#B82B25", alpha=1)
    
    axes[i].set_title(f"{column_mappings[x_var]} vs sharpes_n")  # Set the subplot title
    axes[i].set_ylabel("sharpes_n")  # Set labels for x and y axes using column mappings
    ax2.set_ylabel("petryks_n")  # Set labels for twin axis
    axes[i].set_xlabel(column_mappings[x_var])  # Set labels for x and y axes using column mappings

    # Format y-axis as a percentage for 'slope' and 'porosity'
    if x_var == 'slope':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if x_var == 'porosity' or x_var == 'blockage_ratio':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if x_var == 'wse_jam':
        axes[i].xaxis.set_major_formatter('{:.2f}'.format)

    # Combine the legends of both y-axes
    lines, labels = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    axes[i].grid(True, alpha=0.4)  # Add grid

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])
    
fig.patch.set_facecolor('white')
fig.tight_layout()  # Adjust the layout to prevent overlapping titles
plt.show()  # Show the plots

#endregion
# ===================================================


#region DRAG FORCE PLOTS

# Select the columns you want to use in the pair plots
columns_to_plot = ['wse_jam', 'discharge', 'slope', 'flow_area_interstitial',
                   'wood_sub_area', 'porosity', 'blockage_ratio', 'wp_total', 
                   'velocity_interstitial', 'conveyance_ratio', 'darcy_f_adjusted']

# Create a list with the two columns you want to plot together on the x-axis
y_vars = ['F_drag_adjusted', 'F_drag_standard']

# Filter data to exclude values of 'mannings_n_standard' and 'mannings_n_adjusted' >= 1
filtered_data = df_profile_metrics[(df_profile_metrics['mannings_n_standard'] < 1)]

# Calculate the number of rows needed for the subplots
n_rows = len(columns_to_plot) // 3
if len(columns_to_plot) % 3 != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, 3, figsize=(12, 6 * n_rows))  # Create the figure and axes for subplots
axes = axes.flatten()  # Flatten the axes array to make it easier to work with

# Loop through the columns to be plotted on the y-axis and create the pair plots
for i, x_var in enumerate(columns_to_plot):
    # Determine the subplot's row and column index
    row = i // 3
    col = i % 3

    # Plot mannings_n_adjusted and mannings_n_standard as separate lines on the x-axis
    line_colors = ['orange', 'black']  # Colors for the lines
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[1]], label='Drag force (extent)', color=line_colors[1], alpha=1)
    axes[i].plot(filtered_data[x_var], filtered_data[y_vars[0]], label='Drag force (interstitial)', color=line_colors[0], alpha=1)

    axes[i].set_title(f"{column_mappings[x_var]} vs Drag Force")  # Set the subplot title
    axes[i].set_ylabel("Drag Force [N]")  # Set labels for x and y axes using column mappings
    axes[i].set_xlabel(column_mappings[x_var])  # Set labels for x and y axes using column mappings

    # Format y-axis as a percentage for 'slope' and 'porosity'
    if x_var == 'slope':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    if x_var == 'porosity' or y_var == 'blockage_ratio':
        axes[i].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    if x_var == 'wse_jam':
        axes[i].xaxis.set_major_formatter('{:.2f}'.format)

    axes[i].legend()  # Show the legend for the lines
    axes[i].grid(True, alpha=0.4)  # Add grid

# Remove any empty subplots if the number of columns to plot is odd
if len(columns_to_plot) % 3 != 0:
    fig.delaxes(axes[-1])
    
fig.patch.set_facecolor('white')
fig.tight_layout()  # Adjust the layout to prevent overlapping titles
plt.show()  # Show the plots

#endregion


# ===================================================



#region OTHER PLOTS
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

# Plot 'discharge' vs 'wse_jam' on the first subplot
axes[0, 0].plot(df_profile_metrics['discharge'], df_profile_metrics['wse_jam'])
axes[0, 0].set_xlabel('Discharge [cms]')
axes[0, 0].set_ylabel('WSE (jam) [m]')
axes[0, 0].set_title('Discharge vs Water Surface Elevation (jam)')
axes[0, 0].grid(True)

# Plot 'discharge' vs 'slope' on the second subplot
axes[0, 1].plot(df_profile_metrics['discharge'], df_profile_metrics['slope'])
axes[0, 1].set_xlabel('Discharge [cms]')
axes[0, 1].set_ylabel('Slope')
axes[0, 1].set_title('Discharge vs Slope')
axes[0, 1].grid(True)

# Plot 'porosity' vs 'conveyance_ratio' on the third subplot
axes[1, 0].plot(df_profile_metrics['porosity'], df_profile_metrics['blockage_ratio'])
axes[1, 0].set_xlabel('Porosity')
axes[1, 0].set_ylabel('Blockage ratio')
axes[1, 0].set_title('Porosity vs Conveyance')
axes[1, 0].grid(True)

# Plot 'hydraulic_radius_channel' vs 'hydraulic_radius_total' on the fourth subplot
axes[1, 1].plot(df_profile_metrics['hydraulic_radius_channel'], df_profile_metrics['hydraulic_radius_total'])
axes[1, 1].set_xlabel('R channel [m]')
axes[1, 1].set_ylabel('R total [m]')
axes[1, 1].set_title('R Channel vs R Total')
axes[1, 1].grid(True)

# Plot 'mannings_n_standard' vs 'mannings_n_adjusted' on the fifth subplot
axes[2, 0].plot(df_profile_metrics['mannings_n_standard'], df_profile_metrics['mannings_n_adjusted'])
axes[2, 0].set_xlabel('n (standard)')
axes[2, 0].set_ylabel('n (adjusted)')
axes[2, 0].set_title("Manning's n")
axes[2, 0].grid(True)

# Plot 'flow_area_interstitial' vs 'wood_sub_area' on the sixth subplot
axes[2, 1].plot(df_profile_metrics['flow_area_interstitial'], df_profile_metrics['wood_sub_area'])
axes[2, 1].set_xlabel('Flow area [m2]')
axes[2, 1].set_ylabel('Frontal area [m2]')
axes[2, 1].set_title('Flow area vs Frontal area')
axes[2, 1].grid(True)

#endregion
# ===================================================


# ===================================================














# GRAVEYARD

#region WETTED PERIMETER PAIR PLOT

# columns = df_profile_metrics.columns
# columns = columns.drop('flow_area_extent')  # Exclude 'flow_area_extent' from the columns
# columns = columns.drop('hydraulic_radius_channel')  # Exclude 'hydraulic_radius_channel' from the columns
# columns = columns.drop('hydraulic_radius_total')  # Exclude 'hydraulic_radius_total' from the columns

# num_cols = len(columns)
# num_rows = math.ceil(num_cols / 2)

# fig, axs = plt.subplots(num_rows, 2, figsize=(16, 8 * num_rows))

# # Define the column name mappings
# column_mappings = {
#     'wse_jam': 'WSE at jam [m]',
#     'discharge': 'Discharge [cms]',
#     'slope': 'Slope [%]',
#     'flow_area_interstitial': 'Flow area (interstitial) [m2]',
#     'flow_area_extent': 'Flow area (extent) [m2]',
#     'wood_sub_area': 'Frontal area (submerged) [m2]',
#     'mannings_n': "Manning's n",
#     'hydraulic_radius_channel': 'R (channel) [m]',
#     'hydraulic_radius_total': 'R (total) [m]',
#     'porosity': 'Porosity [%]',
#     'blockage_ratio': 'Blockage ratio'
# }

# # Iterate over variables for y-axis
# plot_count = 0  # Counter for actual number of plots
# for i, col_y in enumerate(columns):
#     row_idx = plot_count // 2
#     col_idx = plot_count % 2
    
#     ax = axs[row_idx, col_idx]
    
#     # Plot wp_channel, wp_total, and wp_wood on the x-axis against col_y
#     if col_y == 'wp_wood' or col_y == 'wp_channel' or col_y == 'wp_total':
#         continue  # Skip plotting
        
#     # Filter values of mannings_n <= 1
#     mask = df_profile_metrics['mannings_n'] <= 1
    
#     ax.plot(df_profile_metrics.loc[mask, 'wp_channel'], df_profile_metrics.loc[mask, col_y], color='blue', label='WP (channel)')
#     ax.plot(df_profile_metrics.loc[mask, 'wp_wood'], df_profile_metrics.loc[mask, col_y], color='red', label='WP (wood)')
#     ax.plot(df_profile_metrics.loc[mask, 'wp_total'], df_profile_metrics.loc[mask, col_y], color='black', label='WP (total)')
    
#     ax.set_xlabel(column_mappings.get('wp_channel', 'Wetted perimeter [m]'), fontsize=12)  # Set x-axis label as 'WP (channel) [m]'
#     ax.set_ylabel(column_mappings.get(col_y, col_y), fontsize=12)  # Set y-axis label as the current variable
    
#     # Format y-axis as a percentage for 'slope' and 'porosity'
#     if col_y == 'slope':
#         ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
#     if col_y == 'porosity':
#         ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
#     if col_y == 'wse_jam':
#         ax.yaxis.set_major_formatter('{:.2f}'.format)
    
#     ax.legend()  # Show legend for wp_channel, wp_total, and wp_wood
    
#     plot_count += 1

# # Remove empty subplots
# for i in range(plot_count, num_rows * 2):
#     row_idx = i // 2
#     col_idx = i % 2
#     fig.delaxes(axs[row_idx, col_idx])

# fig.patch.set_facecolor('white')
# plt.tight_layout()
# plt.show()

#endregion
# =============================================================================

#region HYDRAULIC RADIUS PAIR PLOTS

# columns = df_profile_metrics.columns
# columns = columns.drop('flow_area_extent')  # Exclude 'flow_area_extent' from the columns
# columns = columns.drop('wp_channel')  # Exclude 'wp_channel' from the columns
# columns = columns.drop('wp_wood')  # Exclude 'wp_wood' from the columns
# columns = columns.drop('wp_total')  # Exclude 'wp_total' from the columns

# num_cols = len(columns)
# num_rows = math.ceil(num_cols / 2)

# fig, axs = plt.subplots(num_rows, 2, figsize=(16, 8 * num_rows))

# # Define the column name mappings
# column_mappings = {
#     'wse_jam': 'WSE at jam [m]',
#     'discharge': 'Discharge [cms]',
#     'slope': 'Slope [%]',
#     'flow_area_interstitial': 'Flow area (interstitial) [m2]',
#     'flow_area_extent': 'Flow area (extent) [m2]',
#     'wood_sub_area': 'Frontal area (submerged) [m2]',
#     'mannings_n': "Manning's n",
#     'porosity': 'Porosity [%]',
#     'blockage_ratio': 'Blockage ratio'
# }

# # Iterate over variables for y-axis
# plot_count = 0  # Counter for actual number of plots
# for i, col_y in enumerate(columns):
#     row_idx = plot_count // 2
#     col_idx = plot_count % 2
    
#     ax = axs[row_idx, col_idx]
    
#     # Plot wp_channel, wp_total, and wp_wood on the x-axis against col_y
#     if col_y == 'hydraulic_radius_channel' or col_y == 'hydraulic_radius_total':
#         continue  # Skip plotting
        
#     # Filter values of mannings_n <= 1
#     mask = df_profile_metrics['mannings_n'] <= 1
    
#     ax.plot(df_profile_metrics.loc[mask, 'hydraulic_radius_channel'], df_profile_metrics.loc[mask, col_y], color='purple', label='WP (wood)')
#     ax.plot(df_profile_metrics.loc[mask, 'hydraulic_radius_total'], df_profile_metrics.loc[mask, col_y], color='black', label='WP (total)')
    
#     ax.set_xlabel(column_mappings.get('hydraulic_radius_total', 'R [m]'), fontsize=12)  # Set x-axis label as 'WP (channel) [m]'
#     ax.set_ylabel(column_mappings.get(col_y, col_y), fontsize=12)  # Set y-axis label as the current variable
    
#     # Format y-axis as a percentage for 'slope' and 'porosity'
#     if col_y == 'slope':
#         ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
#     if col_y == 'porosity':
#         ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
#     if col_y == 'wse_jam':
#         ax.yaxis.set_major_formatter('{:.2f}'.format)
    
#     ax.legend()  # Show legend for wp_channel, wp_total, and wp_wood
    
#     plot_count += 1

# # Remove empty subplots
# for i in range(plot_count, num_rows * 2):
#     row_idx = i // 2
#     col_idx = i % 2
#     fig.delaxes(axs[row_idx, col_idx])

# #plt.tight_layout()
# fig.patch.set_facecolor('white')
# plt.show()

#endregion

# =====================================================


# =====================================================



# =====================================================