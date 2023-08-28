#region IMPORT STATEMENTS

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

# import data

with open('expanded_df.pk', 'rb') as f1:
    expanded_df = pickle.load(f1)
expanded_df

#endregion
# ====================================================================================


#region WSE-DISCHARGE RATING CURVES
# POWER OR POLYNOMIAL REGRESSION. OUTPUT WITH BETTER R2 VALUE; STORE IN NEW DATAFRAME

unique_loggers = expanded_df['logger'].unique() # Assuming your DataFrame is named expanded_df
non_empty_loggers = [logger for logger in unique_loggers if (expanded_df['logger'] == logger).any()] # Filter out empty loggers

# Define the power function to fit
def power_function(x, a, b):
    return a * x**b

# Define the polynomial function to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Create an example DataFrame with regression results
regression_results = []

# Iterate over non-empty loggers and create subplots only for loggers with regression
fig = plt.figure(figsize=(8, 6 * len(non_empty_loggers)))
plot_index = 1

for logger in non_empty_loggers:
    data = expanded_df[(expanded_df['logger'] == logger) & (expanded_df['added'] == 'no')]
    
    x = data['discharge_measured_cms']
    y = data['wse_estimated_m']

    positive_mask = (x > 0) & (y > 0)
    x_filtered = x[positive_mask].astype(float)
    y_filtered = y[positive_mask]

    if len(x_filtered) > 3:
        # Create subplot
        ax = fig.add_subplot(len(non_empty_loggers), 1, plot_index)

        # Perform power regression using curve_fit
        try:
            popt_power, pcov_power = curve_fit(power_function, x_filtered, y_filtered)
            x_reg_power = np.linspace(x_filtered.min(), x_filtered.max(), 100) # Generate x values for the power regression line
            y_reg_power = power_function(x_reg_power, *popt_power)  # Calculate y values using the fitted parameters

            # Perform polynomial regression using curve_fit
            popt_poly, pcov_poly = curve_fit(polynomial_function, x_filtered, y_filtered)
            x_reg_poly = np.linspace(x_filtered.min(), x_filtered.max(), 100) # Generate x values for the polynomial regression line
            y_reg_poly = polynomial_function(x_reg_poly, *popt_poly)  # Calculate y values using the fitted parameters

            # Calculate R2 values
            y_pred_power = power_function(x_filtered, *popt_power)
            r2_power = r2_score(y_filtered, y_pred_power)

            y_pred_poly = polynomial_function(x_filtered, *popt_poly)
            r2_poly = r2_score(y_filtered, y_pred_poly)

            # Select regression with higher R2 value
            if r2_power >= r2_poly:
                x_reg = x_reg_power
                y_reg = y_reg_power
                popt = popt_power
                equation_type = 'power function'
                degree = 1
                equation = f"y = {popt[0]:.2f} * x^({popt[1]:.2E})"
                coefficients = [popt[0], popt[1], np.nan, np.nan, np.nan]
            else:
                x_reg = x_reg_poly
                y_reg = y_reg_poly
                popt = popt_poly
                equation_type = 'polynomial'
                degree = 2
                equation = f"y = {popt[0]:.2f} * x^2 + {popt[1]:.2f} * x + {popt[2]:.2f}"
                coefficients = [popt[0], popt[1], popt[2], np.nan, np.nan]

            # Plot selected regression line
            ax.plot(x_reg, y_reg, color='blue', label='Regression')

            # Scatter plot
            scatter = ax.scatter(x_filtered, y_filtered, color='red', label='Data Points')
            ax.set_xlabel('Discharge (cms)')
            ax.set_ylabel('Water Surface Elevation (m)')
            ax.set_title(f'Scatter Plot for Logger {logger}')
            ax.legend()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Format y-axis tick labels to two significant figures

            # Display R2 value as text annotation
            r2_text = f"$\it{{R^2}} = \it{{{r2_power if r2_power >= r2_poly else r2_poly:.2f}}}$"
            
            ax.text(0.95, 0.12, r2_text, transform=ax.transAxes, ha='right', va='bottom')
            ax.text(0.95, 0.05, equation, transform=ax.transAxes, ha='right', va='bottom')
            plot_index += 1
            
            # Append regression results to the DataFrame
            regression_results.append({
                'logger': logger,
                'equation_type': equation_type,
                'degree': degree,
                'equation': equation,
                'coef_1': coefficients[0],
                'coef_2': coefficients[1],
                'coef_3': coefficients[2],
                'coef_4': coefficients[3]
            })
            
        except RuntimeError:
            pass
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

df_wse_discharge_regression = pd.DataFrame(regression_results) # Create a new DataFrame with the specified columns
print(df_wse_discharge_regression) # Print the new DataFrame

# save the dataframe to a pickle
with open('df_wse_discharge_regression.pk', 'wb') as file:
    pickle.dump(df_wse_discharge_regression, file)

#endregion


# *******************************************************************************************
