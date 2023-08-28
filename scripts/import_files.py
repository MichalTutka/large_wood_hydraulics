#region IMPORT PACKAGES
import numpy as np
import pandas as pd
import pickle
import os
import re
#endregion
#========================================================================================================

#region IMPORT DATA
transect_file_path = r'D:\\Michal\Documents\\0_Grad_School_OSU\\0_Research\\0_Data\\Python\site3_python\site3_transects.csv'
df_transect = pd.read_csv(transect_file_path, encoding='cp1252')
with open ('df_transect.pk', 'wb') as f6:
    pickle.dump(df_transect, f6)
df_transect.columns

df_field_mmts_file_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Segura Lab Data\Stage\Stage\Site 3\site3_field_mmts_combined_test2.csv"
df_field_mmts = pd.read_csv(df_field_mmts_file_path, encoding='cp1252')
with open ('df_field_mmts.pk', 'wb') as f3:
    pickle.dump(df_field_mmts, f3)
    
# df_field_mmts gets processed in field_mmts.py to create expanded_df
with open('expanded_df.pk', 'rb') as f5:
    expanded_df = pickle.load(f5)
expanded_df

stage_file_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Segura Lab Data\Stage\Stage\Site 3\site3_field_mmts_combined.csv"
df_stage = pd.read_csv(stage_file_path, encoding='cp1252')
df_stage.columns
df_stage.dtypes
df_stage.head()

discharge_file_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Segura Lab Data\Discharge\Discharge\Site 3\site3_discharge_combined.csv"
df_discharge = pd.read_csv(discharge_file_path, encoding='cp1252')
df_discharge.columns
df_discharge.dtypes
df_discharge.head(20)

# with open ('df_timeseries_site3.pk', 'wb') as f2:
#     pickle.dump(df_timeseries_site3, f2)

    
#endregion
#========================================================================================================

#region BATCH COMBINE CSVs (use this for sites 1 and 2 later)

# # uncomment when you want to use it again

# # Define the directory path
# directory_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\All compensated files"

# # Define the output directory
# output_directory = os.path.join(directory_path, 'Combined CSVs')

# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)

# # Get all the file names in the directory
# file_names = os.listdir(directory_path)

# # Create a dictionary to store the file contents
# file_contents = {}

# # Iterate over the file names
# for file_name in file_names:
#     file_path = os.path.join(directory_path, file_name)
    
#     # Check if the file is a CSV file
#     if file_name.endswith('.csv'):
#         try:
#             # Read the file into a dataframe, skipping the header rows except the first one
#             df = pd.read_csv(file_path, skiprows=range(1, 12), encoding='latin-1', header=None)
            
#             # Get the first 7 characters of the file name as the key
#             key = file_name[:7]
            
#             # Check if the key already exists in the dictionary
#             if key in file_contents:
#                 # Append the dataframe contents to the existing key
#                 file_contents[key] = pd.concat([file_contents[key], df])
#             else:
#                 # Create a new key in the dictionary and assign the dataframe contents
#                 file_contents[key] = df
#         except pd.errors.EmptyDataError:
#             # Handle empty files if necessary
#             pass

# # Iterate over the dictionary and write the combined dataframes to separate files
# for key, combined_df in file_contents.items():
#     # Define the output file path
#     output_file_name = key + '_MillCreek_Site3_Compensated_Combined2.csv'
#     output_file_path = os.path.join(output_directory, output_file_name)

#     # Replace the header with the specified columns
#     combined_df.columns = ["date", "time", "ms", "LEVEL", "TEMPERATURE"]

#     # Delete the second row of the header
#     combined_df = combined_df.iloc[2:]

#     # Write the combined dataframe to the output file without the index
#     combined_df.to_csv(output_file_path, index=False)

# print("Combined CSV files have been generated and stored in the 'Combined CSVs' folder.")

#endregion
#========================================================================================================

#region ADD LOGGER NAME TO FILES
# def process_files(dataframes):
#     processed_dataframes = []
#     for df, var_name in zip(dataframes, ['LL1_timeseries', 'LL2_timeseries', 'LL3_timeseries', 'LL4_timeseries', 'LL5_timeseries', 'LL7_timeseries', 'LL8_timeseries', 'LL9_timeseries', 'LL10_timeseries']):
#         # Extract the logger name from the variable name
#         logger_name = re.search(r'LL\d+', var_name)[0]
        
#         # Check if the "logger" column already exists and is not empty
#         if 'logger' not in df.columns or df['logger'].isnull().all():
#             # Add the "logger" column and populate it with the logger name
#             df['logger'] = logger_name
        
#         # Append the processed DataFrame to the list
#         processed_dataframes.append(df)
    
#     # Concatenate all the processed DataFrames into a single DataFrame
#     combined_df = pd.concat(processed_dataframes, ignore_index=True)
    
#     return combined_df

def process_files(dataframes):
    processed_dataframes = []
    for df, var_name in zip(dataframes, ['LL1_timeseries', 'LL2_timeseries', 'LL3_timeseries', 'LL4_timeseries', 'LL5_timeseries', 'LL7_timeseries', 'LL8_timeseries', 'LL9_timeseries', 'LL10_timeseries']):
        # Extract the logger name from the variable name
        logger_name = re.search(r'LL\d+', var_name)[0]
        
        # Check if the "logger" column already exists and is not empty
        if 'logger' not in df.columns or df['logger'].isnull().all():
            # Add the "logger" column and populate it with the logger name
            df['logger'] = logger_name
        
        # Convert the 'time' column to the desired data type
        df['time'] = df['time'].astype(str)
        
        # Append the processed DataFrame to the list
        processed_dataframes.append(df)
    
    # Concatenate all the processed DataFrames into a single DataFrame
    combined_df = pd.concat(processed_dataframes, ignore_index=True)
    
    return combined_df


#endregion
#========================================================================================================

#region TIME SERIES DATA FILES
header_size = 0

LL1_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2067050_MillCreek_Site3_LL1_Compensated_Combined2.csv"
LL2_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2117754_MillCreek_Site3_LL2_Compensated_Combined2.csv"
LL3_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2033710_MillCreek_Site3_LL3_Compensated_Combined2.csv"
LL4_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2073011_MillCreek_Site3_LL4_Compensated_Combined2.csv"
LL5_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2038114_MillCreek_Site3_LL5_Compensated_Combined2.csv"
LL7_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2060682_MillCreek_Site3_LL7_Compensated_Combined2.csv"
LL8_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2067559_MillCreek_Site3_LL8_Compensated_Combined2.csv"
LL9_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2075351_MillCreek_Site3_LL9_Compensated_Combined2.csv"
LL10_timeseries_path = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\site3_python\site3_loggers\New folder\combined_logger_timeseries_csvs\2075482_MillCreek_Site3_LL10_Compensated_Combined2.csv"

LL1_timeseries = pd.read_csv(LL1_timeseries_path, header = header_size, encoding='cp1252')
LL2_timeseries = pd.read_csv(LL2_timeseries_path, header = header_size, encoding='cp1252')
LL3_timeseries = pd.read_csv(LL3_timeseries_path, header = header_size, encoding='cp1252')
LL4_timeseries = pd.read_csv(LL4_timeseries_path, header = header_size, encoding='cp1252')
LL5_timeseries = pd.read_csv(LL5_timeseries_path, header = header_size, encoding='cp1252')
LL7_timeseries = pd.read_csv(LL7_timeseries_path, header = header_size, encoding='cp1252')
LL8_timeseries = pd.read_csv(LL8_timeseries_path, header = header_size, encoding='cp1252')
LL9_timeseries = pd.read_csv(LL9_timeseries_path, header = header_size, encoding='cp1252')
LL10_timeseries = pd.read_csv(LL10_timeseries_path, header = header_size, encoding='cp1252')
LL10_timeseries.head(15)

# Store the DataFrames in a list
dataframes = [
    LL1_timeseries,
    LL2_timeseries,
    LL3_timeseries,
    LL4_timeseries,
    LL5_timeseries,
    LL7_timeseries,
    LL8_timeseries,
    LL9_timeseries,
    LL10_timeseries]

# Process the files and update the DataFrames
df_timeseries_site3 = process_files(dataframes)
df_timeseries_site3['time'] = df_timeseries_site3['time'].astype(str)
df_timeseries_site3.dropna(subset=['LEVEL'], inplace=True)  # Drop rows with NaN values in 'time' or 'LEVEL' columns
print(df_timeseries_site3)
print(df_timeseries_site3.dtypes)
directory = r'D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\timeseries_dataframes'
filename = 'df_timeseries_site3.csv'
filepath = os.path.join(directory, filename)
df_timeseries_site3.to_csv(filepath, index=False)

df_timeseries_site3 = r"D:\Michal\Documents\0_Grad_School_OSU\0_Research\0_Data\Python\timeseries_dataframes\df_timeseries_site3.csv"
df_timeseries_site3 = pd.read_csv(df_timeseries_site3)

# Convert 'time' column in df_timeseries_site3 to the desired format
df_timeseries_site3['time'] = pd.to_datetime(df_timeseries_site3['time'], format='%I:%M:%S %p').dt.strftime('%I:%M:%S %p')

# Save to pickle file
with open ('df_timeseries_site3.pk', 'wb') as f1:
    pickle.dump(df_timeseries_site3, f1)
#========================================================================================================




















       