#region IMPORT STATEMENTS

# import packages
import numpy as np
import pandas as pd
import pickle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

# import data
with open('df_timeseries_site3.pk', 'rb') as f1:
    df_timeseries_site3 = pickle.load(f1)
print(df_timeseries_site3)
print(df_timeseries_site3.dtypes)
    
with open('df_field_mmts.pk', 'rb') as f3:
    df_field_mmts = pickle.load(f3)
df_field_mmts

#endregion
#========================================================================================================

#region NEW ROWS SO THAT EACH LOGGER HAS ONE OF EACH DATE
#========================================================================================================
# Sort the dataframe by 'logger' and 'date'
df_field_mmts.sort_values(by=['logger', 'date'], inplace=True)

# Get the unique loggers and dates in the dataset
unique_loggers = df_field_mmts['logger'].unique()
unique_dates = df_field_mmts['date'].unique()

# Create a new dataframe to store the expanded rows
expanded_df = pd.DataFrame(columns=df_field_mmts.columns)

# Iterate over each unique logger
for logger in unique_loggers:
    logger_dates = df_field_mmts[df_field_mmts['logger'] == logger]['date'].unique()
    
    # Find the missing dates for the logger
    missing_dates = set(unique_dates) - set(logger_dates)
    
    # Add rows with missing dates for the logger
    for date in missing_dates:
        new_row = pd.Series([logger, date] + [None] * (df_field_mmts.shape[1] - 2), index=df_field_mmts.columns)
        new_row['added'] = 'yes'  # Set 'added' to 'yes' for the added rows
        
         # Populate 'top_elev' based on the logger
        same_logger_rows = df_field_mmts[df_field_mmts['logger'] == logger]
        if len(same_logger_rows) > 0:
            new_row['top_elev'] = same_logger_rows['top_elev'].iloc[0]
                
        # Populate 'discharge_measured_cms' based on the values with the same date
        same_date_rows = df_field_mmts[df_field_mmts['date'] == date]
        if len(same_date_rows) > 0:
            new_row['discharge_measured_cms'] = same_date_rows['discharge_measured_cms'].iloc[0]
        
        # Find the closest logger with the same date
        closest_logger = None
        closest_time_rounded = None
        for row in same_date_rows.itertuples(index=False):
            if row.logger < logger:
                closest_logger = row.logger
                closest_time_rounded = row.time_rounded
            else:
                break
        
        if closest_logger is not None:
            new_row['time_rounded'] = closest_time_rounded
        
        
        expanded_df = expanded_df.append(new_row, ignore_index=True)

# Append the existing rows to the expanded dataframe
expanded_df = expanded_df.append(df_field_mmts, ignore_index=True)
# Sort the expanded dataframe by 'logger' and 'date'
expanded_df.sort_values(by=['logger', 'date'], inplace=True)
# Reset the index of the expanded dataframe
expanded_df.reset_index(drop=True, inplace=True)
# Populate 'added' column with 'no' for the existing rows
expanded_df.loc[expanded_df['added'].isnull(), 'added'] = 'no'

expanded_df

#endregion
#========================================================================================================

#region CONVERT 'TIME_ROUNDED' IN EXPANDED_DF TO THE DESIRED FORMAT

# Convert 'time_rounded' in expanded_df to the desired format
expanded_df['time_rounded'] = pd.to_datetime(expanded_df['time_rounded'], format='%I:%M %p', errors='coerce').dt.strftime('%I:%M:%S %p')

# Initialize the 'logger_stage_m' column in expanded_df
expanded_df['logger_stage_m'] = None

# Iterate over each row in expanded_df
for index, row in expanded_df.iterrows():
    logger = row['logger']
    date = row['date']
    time_rounded = row['time_rounded']
    
    # Find the corresponding row in df_timeseries_site3
    match = (df_timeseries_site3['logger'] == logger) & (df_timeseries_site3['date'] == date) & (df_timeseries_site3['time'] == time_rounded)
    matching_rows = df_timeseries_site3.loc[match]
    
    # Update the 'logger_stage_m' value in expanded_df if a match is found
    if not matching_rows.empty:
        logger_stage_m = matching_rows['LEVEL'].values[0]
        expanded_df.at[index, 'logger_stage_m'] = logger_stage_m

# Add the 'wse_estimated_m' column to expanded_df based on 'added' column
expanded_df['wse_estimated_m'] = expanded_df.apply(lambda row: row['wse_measured_m'] if row['added'] == 'no' else row['top_elev'] - row['logger_stage_m'] - 2 if row['logger_stage_m'] is not None else None, axis=1)

# Print the updated expanded_df
print(expanded_df)

#endregion
#========================================================================================================

#region RETRIEVE STAGE VALUES FROM COMBINED LOGGER TIMESERIES FILES BASED ON CRITERIA

# Convert 'time_rounded' in expanded_df to match the format in df_timeseries_site3
expanded_df['time_rounded'] = pd.to_datetime(expanded_df['time_rounded'], format='%I:%M:%S %p').dt.strftime('%I:%M:%S %p')

# Initialize the 'logger_stage_m' column in expanded_df
expanded_df['logger_stage_m'] = None

# Iterate over each row in expanded_df
for index, row in expanded_df.iterrows():
    logger = row['logger']
    date = row['date']
    time_rounded = row['time_rounded']
    
    # Find the corresponding row in df_timeseries_site3
    match = (df_timeseries_site3['logger'] == logger) & (df_timeseries_site3['date'] == date) & (df_timeseries_site3['time'] == time_rounded)
    matching_rows = df_timeseries_site3.loc[match]
    
    # Update the 'logger_stage_m' value in expanded_df if a match is found
    if not matching_rows.empty:
        logger_stage_m = matching_rows['LEVEL'].values[0]
        expanded_df.at[index, 'logger_stage_m'] = logger_stage_m

# Add the 'wse_estimated_m' column to expanded_df based on 'added' column
expanded_df['wse_estimated_m'] = expanded_df.apply(lambda row: row['wse_measured_m'] if row['added'] == 'no' else row['top_elev'] - row['logger_stage_m'] if row['logger_stage_m'] is not None else None, axis=1)

# Filter the rows based on the conditions
expanded_df = expanded_df[~(expanded_df['logger'].str.startswith('RB') & (expanded_df['top_to_ws_(m)'] == 0))]
expanded_df.reset_index(drop=True, inplace=True) # Reset the index of the DataFrame
# expanded_df = expanded_df.drop([169, 180]) # Drop two specific rows

print(expanded_df) # Print the updated expanded_df


#endregion
#========================================================================================================

#region PLOT EVERYTHING


# Assuming your DataFrame is named expanded_df
unique_loggers = expanded_df['logger'].unique()

# Create subplots for each logger
fig, axs = plt.subplots(len(unique_loggers), 1, figsize=(8, 6 * len(unique_loggers)))

for i, logger in enumerate(unique_loggers):
    ax = axs[i]
    data = expanded_df[expanded_df['logger'] == logger]
    
    if not data.empty:
        colors = data['added'].map({'yes': 'green', 'no': 'red'})
        
        scatter = ax.scatter(data['discharge_measured_cms'], data['wse_estimated_m'], c=colors)
        ax.set_xlabel('Discharge (cms)')
        ax.set_ylabel('Water Surface Elevation (m)')
        ax.set_title(f'Scatter Plot for Logger {logger}')
        
        #Format y-axis tick labels to two significant figures
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
plt.tight_layout()
plt.show()




#endregion
#========================================================================================================

#region SAVING PICKLES

# Save to a pickle file for easy loading in other .py files
with open('expanded_df.pk', 'wb') as file:
    pickle.dump(expanded_df, file)


#endregion
#========================================================================================================