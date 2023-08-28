#region IMPORTS

# import packages
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from matplotlib.ticker import FormatStrFormatter

#import datasets
# this is the transects file: names and distances
with open('df_transect.pk', 'rb') as f6:
    df_transect = pickle.load(f6)

# this is ....
with open('expanded_df.pk', 'rb') as f5:
    expanded_df = pickle.load(f5)

# this is ....
with open('df_wse_discharge_regression.pk', 'rb') as f7:
    df_wse_discharge_regression = pickle.load(f7)   

#endregion
# ===================================================

#region TRANSECT DISTANCE for interpolation  

# Transect IDs

## for jam 3C
logger_ds = 'LL5' # downstream level logger (user defined)
logger_us = 'LL8' # upstream level logger (user defined)
transect = 'LL6-7_32' # transect ID (user defined)

# for jam 3A
# logger_ds = 'LL1' # downstream level logger (user defined)
# logger_us = 'RB4' # upstream level logger (user defined)
# transect = 'LL2-3_9' # transect ID (user defined)

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
# ===================================================

#region CALCULATING WSE @ JAM AND SLOPE USING US & DS WSE-Q REGRESSIONS

# Create a new DataFrame df_field_mmts2 with columns ['discharge', 'wse_us', 'wse_ds', 'wse_jam']
df_field_mmts2 = pd.DataFrame(columns=['discharge', 'wse_us', 'wse_ds', 'wse_jam'])

# Calculate the range for the 'discharge' column
min_discharge = 0  # Minimum discharge value set to 0
max_discharge = expanded_df['discharge_measured_cms'].max()  # Maximum discharge value obtained from expanded_df

# Populate the 'discharge' column with values from min to max, every 0.01 unit
discharge_values = np.arange(min_discharge, max_discharge + 0.01, 0.01)
df_field_mmts2['discharge'] = discharge_values

# Set the 'wse_us', 'wse_ds', and 'wse_jam' columns as NaN
df_field_mmts2['wse_us'] = np.nan  # Initialize the 'wse_us' column with NaN values
df_field_mmts2['wse_ds'] = np.nan  # Initialize the 'wse_ds' column with NaN values
df_field_mmts2['wse_jam'] = np.nan  # Initialize the 'wse_jam' column with NaN values
df_field_mmts2['slope'] = np.nan  # Initialize the 'slope' column with NaN values

# Iterate over rows in df_field_mmts2
for index, row in df_field_mmts2.iterrows():
    discharge = row['discharge']  # Get the discharge value for the current row
    
    # Get the equation row for wse_us based on logger_us
    equation_row_us = df_wse_discharge_regression[df_wse_discharge_regression['logger'].str.strip() == logger_us.strip()]
    
    # Check if equation row exists for wse_us
    if not equation_row_us.empty:
        equation_type_us = equation_row_us['equation_type'].values[0]  # Get the equation type for wse_us
        
        # Calculate wse_us based on the equation type
        if equation_type_us == 'power function':
            # Get the coefficients for power function from equation_row_us
            a = equation_row_us['coef_1'].values[0]
            b = equation_row_us['coef_2'].values[0]
            wse_us = a * discharge ** b  # Calculate wse_us using the power function equation
        elif equation_type_us == 'polynomial':
            # Get the coefficients for polynomial from equation_row_us
            a = equation_row_us['coef_1'].values[0]
            b = equation_row_us['coef_2'].values[0]
            c = equation_row_us['coef_3'].values[0]
            wse_us = a * discharge ** 2 + b * discharge + c  # Calculate wse_us using the polynomial equation
        else:
            wse_us = np.nan  # Invalid equation type, assign NaN to wse_us
        
        df_field_mmts2.at[index, 'wse_us'] = wse_us # Assign wse_us value to df_field_mmts2
    
    # Get the equation row for wse_ds based on logger_ds
    equation_row_ds = df_wse_discharge_regression[df_wse_discharge_regression['logger'].str.strip() == logger_ds.strip()]
    
    # Check if equation row exists for wse_ds
    if not equation_row_ds.empty:
        equation_type_ds = equation_row_ds['equation_type'].values[0]  # Get the equation type for wse_ds
        
        # Calculate wse_ds based on the equation type
        if equation_type_ds == 'power function':
            # Get the coefficients for power function from equation_row_ds
            a = equation_row_ds['coef_1'].values[0]
            b = equation_row_ds['coef_2'].values[0]
            wse_ds = a * discharge ** b  # Calculate wse_ds using the power function equation
        elif equation_type_ds == 'polynomial':
            # Get the coefficients for polynomial from equation_row_ds
            a = equation_row_ds['coef_1'].values[0]
            b = equation_row_ds['coef_2'].values[0]
            c = equation_row_ds['coef_3'].values[0]
            wse_ds = a * discharge ** 2 + b * discharge + c  # Calculate wse_ds using the polynomial equation
        else:
            wse_ds = np.nan  # Invalid equation type, assign NaN to wse_ds
        
        df_field_mmts2.at[index, 'wse_ds'] = wse_ds # Assign wse_ds value to df_field_mmts2
    
    # Calculate wse_jam based on wse_us, wse_ds, minireach_distance, and transect_distance
    wse_jam = ((wse_us - wse_ds) / minireach_distance) * transect_distance + wse_ds
    # Calculate slope based on wse_us, wse_ds, and minireach_distance
    slope = (wse_us - wse_ds) / minireach_distance
    
    # Assign wse_jam and slope value to df_field_mmts2
    df_field_mmts2.at[index, 'wse_jam'] = wse_jam
    df_field_mmts2.at[index, 'slope'] = slope


# Print the updated DataFrame
print(df_field_mmts2)

#endregion
# ===================================================

#region CREATING DISCHARGE VS -SLOPE AND -WSE @ JAM REGRESSION EQUATIONS

# Define the power function to fit
def power_function(x, a, b):
    return a * x**b

# Define the polynomial function to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

#**********************************************************
# WSE V. DISCHARGE

# Perform regression for wse_jam vs discharge
x_discharge = df_field_mmts2['discharge']
y_wse_jam = df_field_mmts2['wse_jam']

# Perform power regression using curve_fit
popt_power_discharge, pcov_power_discharge = curve_fit(power_function, x_discharge, y_wse_jam)
r2_power_jam = r2_score(y_wse_jam, power_function(x_discharge, *popt_power_discharge))

# Perform polynomial regression using curve_fit
popt_poly_discharge, pcov_poly_discharge = curve_fit(polynomial_function, x_discharge, y_wse_jam)
r2_poly_jam = r2_score(y_wse_jam, polynomial_function(x_discharge, *popt_poly_discharge))

# Select regression with higher R2 value for wse_jam
if r2_power_jam >= r2_poly_jam:
    wse_jam_regression = power_function(x_discharge, *popt_power_discharge)
    wse_jam_equation = f"y = {popt_power_discharge[0]:.2f} * x\u00b9\u207b\u02e3 + {popt_power_discharge[1]:.2E}"
    r2_jam = r2_power_jam
else:
    wse_jam_regression = polynomial_function(x_discharge, *popt_poly_discharge)
    wse_jam_equation = f"y = {popt_poly_discharge[0]:.2f} * x\u00b2 + {popt_poly_discharge[1]:.2f} * x + {popt_poly_discharge[2]:.2f}"
    r2_jam = r2_poly_jam

#**********************************************************
# SLOPE V. DISCHARGE

# Perform regression for slope vs discharge
y_slope = df_field_mmts2['slope']

# Perform power regression using curve_fit
popt_power_slope, pcov_power_slope = curve_fit(power_function, x_discharge, y_slope)
r2_power_slope = r2_score(y_slope, power_function(x_discharge, *popt_power_slope))

# Perform polynomial regression using curve_fit
popt_poly_slope, pcov_poly_slope = curve_fit(polynomial_function, x_discharge, y_slope)
r2_poly_slope = r2_score(y_slope, polynomial_function(x_discharge, *popt_poly_slope))

# Select regression with higher R2 value for slope
if r2_power_slope >= r2_poly_slope:
    slope_regression = power_function(x_discharge, *popt_power_slope)
    slope_equation = f"y = {popt_power_slope[0]:.2f} * x\u00b9\u207b\u02e3 + {popt_power_slope[1]:.2E}"
    r2_slope = r2_power_slope
else:
    slope_regression = polynomial_function(x_discharge, *popt_poly_slope)
    slope_equation = f"y = {popt_poly_slope[0]:.2f} * x\u00b2 + {popt_poly_slope[1]:.2f} * x + {popt_poly_slope[2]:.2f}"
    r2_slope = r2_poly_slope

#**********************************************************
# PLOTS

# Plot wse_jam vs discharge
fig, ax = plt.subplots()
ax.scatter(x_discharge, y_wse_jam, color='red', label='Data Points')
ax.plot(x_discharge, wse_jam_regression, color='blue', label='Regression')

# Add equation and R2 value as text annotation
equation_text = f"{wse_jam_equation}"
r2_text = f"R\u00b2 = {r2_jam:.2f}"

ax.text(0.05, 0.9, equation_text, transform=ax.transAxes, ha='left', va='top')
ax.text(0.05, 0.85, r2_text, transform=ax.transAxes, ha='left', va='top')

ax.set_xlabel('Discharge (cms)')
ax.set_ylabel('WSE at jam (m)')
ax.set_title('WSE at jam vs Discharge')
ax.legend(loc='lower right')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Format y-axis tick labels to two significant figures
plt.show()

# Plot slope vs discharge
fig, ax = plt.subplots()
ax.scatter(x_discharge, y_slope, color='red', label='Data Points')
ax.plot(x_discharge, slope_regression, color='blue', label='Regression')

# Add equation and R2 value as text annotation
equation_text = f"{slope_equation}"
r2_text = f"R\u00b2 = {r2_slope:.2f}"

ax.text(0.05, 0.9, equation_text, transform=ax.transAxes, ha='left', va='top')
ax.text(0.05, 0.85, r2_text, transform=ax.transAxes, ha='left', va='top')

ax.set_xlabel('Discharge (cms)')
ax.set_ylabel('Slope')
ax.set_title('Slope vs Discharge')
ax.legend(loc='lower right')
plt.show()

#endregion
# ===================================================

#region STORING THE REGRESSION FUNCTIONS

# Function to calculate discharge based on WSE_jam using the selected regression equation
r2_power_jam
r2_poly_jam

def calculate_discharge_from_wse_jam(wse_input):
    if r2_power_jam >= r2_poly_jam:
        a = popt_power_discharge[0]
        b = popt_power_discharge[1]
        discharge = (wse_input / a) ** (1 / b)
    else:
        a = popt_poly_discharge[0]
        b = popt_poly_discharge[1]
        c = popt_poly_discharge[2]
        discriminant = b**2 - 4*a*(c - wse_input)
        
        if discriminant >= 0:
            x1 = (-b + (discriminant**0.5)) / (2*a)
            x2 = (-b - (discriminant**0.5)) / (2*a)
            # Choose the positive root if available
            if x1 >= 0:
                discharge = x1
            elif x2 >= 0:
                discharge = x2
            else:
                discharge = None
        else:
            discharge = None
    
    return discharge if discharge is not None else None

hi = calculate_discharge_from_wse_jam(1000)
print(hi)



# Function to calculate slope based on discharge using the selected regression equation
def calculate_slope_from_discharge(discharge_output):
    if discharge_output is None:
        return None
    
    if r2_power_slope >= r2_poly_slope:
        a = popt_power_slope[0]
        b = popt_power_slope[1]
        slope_output = a * discharge_output**b
    else:
        a = popt_poly_slope[0]
        b = popt_poly_slope[1]
        c = popt_poly_slope[2]
        if discharge_output is None:
            slope_output = None
        else:
            slope_output = a * discharge_output**2 + b * discharge_output + c
        
    return slope_output

hi2 = calculate_slope_from_discharge(1.5)
print(hi2)

# ======================================



#endregion
# ===================================================

#region CULLING VALUES BASED ON:

# TIME:
## for site 3 these shouldn't matter

#start_time = 'x'# based on interview with Maddie
#end_time = 'x' # based on my photos or notes

# SPACE: 
## figure this out from the second to last elevation value from channel (MAX) 
## and from the channel invert (MIN)
## just pass those here

# ---> from from file1 import my_variable
# Now you can use the imported variable


#min_wse = 
#max_wse = 

#endregion

# ===================================================