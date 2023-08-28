#region IMPORT STATEMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#endregion

#region IMPORT dataset
# from import_files import df_timeseries 
# df_timeseries.columns
# df_timeseries.dtypes
# df_timeseries.head()

# from import_files import df_general
# df_general.columns
# df_general.dtypes
# df_general.head()

from import_files import df_field_mmts
from import_files import df_field_mmts2

## =====================================================================
stage_file_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Segura Lab Data\Stage\Stage\Site 3\site3_stage_combined.csv"
df_stage = pd.read_csv(stage_file_path, encoding='cp1252')
df_stage.columns
df_stage.dtypes
df_stage.head()

discharge_file_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Segura Lab Data\Discharge\Discharge\Site 3\site3_discharge_combined.csv"
df_discharge = pd.read_csv(discharge_file_path, encoding='cp1252')
df_discharge.columns
df_discharge.dtypes
df_stage.head()

#endregion

#region COMPILING STAGE-WSE TABLE
# First, fix the time formats
def update_depth_dataset_time(df):
    # Check if 'Time' column exists in the specified format
    if 'Time' in df.columns:
        try:
            # Convert 'Time' column to datetime format
            df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.strftime('%I:%M %p')
        except ValueError:
            # Ignore if the format is incorrect
            pass
    
    return df
# Call the function on the dataframe
depth_dataset = update_depth_dataset_time(depth_dataset)  
    
# Merge the two datasets based on date and time columns
df_stage_wse_merged = pd.merge(df_stage, depth_dataset, left_on=['date', 'time_rounded'], right_on=['Date', 'Time'], how='inner')
# Filter the columns you need from the merged dataframe
df_stage_wse = df_stage_wse_merged[['logger', 'date', 'time_rounded', 'wse_ds', 'wse_us', 'wse_measured_m', 'discharge_measured_cms']]
# Reset the index if needed
df_stage_wse = df_stage_wse.reset_index(drop=True)
#endregion

#region FIXING WSE COLUMNS. CREATING ONE STAGE COLUMN BASED ON LOGGER
def update_stage_measured(df):
    # Check if 'wse_ds' and 'wse_us' columns are present
    if 'wse_ds' in df.columns and 'wse_us' in df.columns:
        # Create a new column 'stage_measured_m' based on the conditions
        df['stage_measured_m'] = np.where(df['logger'] == 'LL5', df['wse_ds'], df['wse_us'])
        
        # Drop the 'wse_ds' and 'wse_us' columns
        df = df.drop(['wse_ds', 'wse_us'], axis=1)
    
    return df
# Call the function on the dataframe
df_stage_wse = update_stage_measured(df_stage_wse)

df_stage_wse = df_stage_wse.reindex(columns=['logger', 'date', 'time_rounded', 'stage_measured_m', 'wse_measured_m', 'discharge_measured_cms']) # moving columns around
df_stage_wse.columns

#endregion

#region STAGE-WSE RATING CURVES (REGRESSION) + PLOTS

grouped_data = df_stage_wse.groupby('logger') # Group the data by 'logger'
# Create separate plots for each 'logger'
for logger, data in grouped_data:
    # Perform linear regression
    x = data['stage_measured_m']
    y = data['wse_measured_m']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Stage (m)')
    ax.set_ylabel('WSE (m)')
    ax.set_title(f'Plot for {logger}')
    ax.legend() # Show the legend
    ax.yaxis.set_major_formatter('{x:.2f}') # Format the x-axis tick labels as absolute values

    # Add regression line
    ax.plot(x, slope * x + intercept, color='red')
    
    # Add regression equation and R-squared value
    equation = f'$\it{{y}} = \it{{{slope:.2f}x + {intercept:.2f}}}$'
    r_squared = f'$\it{{R^2}} = \it{{{r_value**2:.2f}}}$'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.05, 0.88, r_squared, transform=ax.transAxes, fontsize=12, verticalalignment='top')
#endregion


#region MODELED WSE FROM RATING CURVES
## still need to check this work.........

# def calculate_wse_modeled(dataframe):
#     # Group the regression data by 'logger'
#     grouped_data = df_stage_wse.groupby('logger')
    
#     # Iterate over the dataframe rows
#     for index, row in dataframe.iterrows():
#         logger = row['logger']
#         stage = row['stage']
        
#         # Find the corresponding regression equation based on the logger
#         regression_data = grouped_data.get_group(logger)
#         slope = regression_data['slope'].iloc[0]
#         intercept = regression_data['intercept'].iloc[0]
        
#         # Calculate the modeled WSE using the regression equation
#         wse_modeled = slope * stage + intercept
        
#         # Add the modeled WSE value to the dataframe
#         dataframe.loc[index, 'wse_modeled_m'] = wse_modeled
    
#     return dataframe















#endregion