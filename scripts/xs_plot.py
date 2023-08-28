#region IMPORT STATEMENTS
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
#========================
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerTuple

import os
from IPython import display
display.set_matplotlib_formats("svg")
from datetime import datetime

# Import data

from geometry_setup import channel,  objects_multipoly, channel_invert, channelpoly
from geometry_setup import wse_max, wse_start, wse_end
from geometry_setup import get_intersection_coordinates

save_directory = 'D:\\Michal\Documents\\0_Grad_School_OSU\\0_Research\\0_Data\\Python\\plots\\' 

#endregion
#=====================================================================================================================================

# **********
wse_input = wse_max # set the water surface elevation here
# **********

waterpoly = Polygon([(wse_start, wse_input),(wse_end, wse_input),(wse_end, wse_input - 50),(wse_start, wse_input - 50)])
flow_approach_poly = channelpoly.intersection(waterpoly)
flow_pore_poly = flow_approach_poly.difference(objects_multipoly)
transverse_area_poly = flow_approach_poly.difference(flow_pore_poly)

# Create a function to generate PolygonPatch with specified properties
def create_polygon_patch(geometry, facecolor, alpha, label=None):
    patch = PolygonPatch(geometry, facecolor=facecolor, alpha=alpha, label=label)
    return patch
#............................................

fig, ax = plt.subplots(figsize=(15, 10)) # Create a plot
ax.plot(*channel.xy, color="black", lw=2) # Plot the channel LineString

# Create the PolygonPatch with the specified properties for objects_multipoly
objects_multipoly_patch = PolygonPatch(objects_multipoly, facecolor='#553c00', edgecolor='#553c00', hatch='////', lw=2, label='large wood')
facecolor_with_alpha = objects_multipoly_patch.get_facecolor()
facecolor_with_alpha = (*facecolor_with_alpha[:3], 0.15)
objects_multipoly_patch.set_facecolor(facecolor_with_alpha)
ax.add_artist(objects_multipoly_patch) # Add objects_multipoly_patch to the plot

ax.add_artist(create_polygon_patch(flow_pore_poly, facecolor='#01A7C2', alpha=0.25, label=r'pore flow area, $A_{\mathrm{p}}$')) # draw flow
ax.add_artist(create_polygon_patch(transverse_area_poly, facecolor='red', alpha=0.5, label=r'transverse area, $A_{\mathrm{tr}}$')) # draw transverse area

# Set the y-axis limits to the maximum y-value of objects_multipoly or channel
max_y_value = max(objects_multipoly.bounds[3], channel.bounds[3])
ax.set_ylim(channel_invert - 0.25, max_y_value + 0.5)
ax.set_xlim(3.5, 17) # x limits for the plot

ax.set_xlabel('Station (m)', fontsize=12)
ax.set_ylabel('Elevation (m)', fontsize=12)
ax.set_title('Reference XS Profile', fontsize=18)

ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal
ax.yaxis.set_major_locator(MultipleLocator(0.5))  # Set the y-axis tick interval to 0.5 units
ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Set the y-axis tick interval to 0.1 units
ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # Set the x-axis tick interval
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='lower left', bbox_to_anchor=(0.01, 0.039), frameon=False
        #, ncol=len(ax.patches)
        )

current_date = datetime.now().strftime('%Y%m%d')
save_filename = f"{current_date}_reference_xs_profile_plot.png"
plt.savefig(os.path.join(save_directory, save_filename), format='png', dpi=500)
plt.show()




