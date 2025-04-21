import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from pyproj import Proj, transform
import base64
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import folium
from folium import LinearColormap
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from folium.plugins import MousePosition
from shapely.geometry import Point
from PIL import Image, ImageDraw, ImageFont
from folium import raster_layers, GeoJson, plugins
from rasterio.plot import show
import os 
import tempfile
import matplotlib.pyplot as plt
from rasterio.warp import transform_bounds
import requests
import json
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances
import re
import requests
import io

# Set the title and favicon that appear in the browser's tab bar.
st.set_page_config(
    page_title='Xwulqwselu',
    page_icon=':herb:',
)

# Sidebar for navigation
st.sidebar.title("Xwulqw'selu Sta'lo'")
selected_option = st.sidebar.radio(
    "Select an option:",
    #("Watershed models", "Water interactions", "Recharge", "View Report")
    ("Watershed models", "Whole watershed", "Water use", "Land use")
)

def process_swatmf_data(file_path):
    data = []
    current_month = None
    current_year = None

    with open(file_path, 'r') as file:
        for line in file:
            if 'month:' in line:
                parts = line.split()
                try:
                    current_month = int(parts[1])
                    current_year = int(parts[3])
                except (ValueError, IndexError):
                    continue  # Skip if there's an issue parsing month/year
            elif 'Layer' in line:
                continue  # Skip header line
            elif line.strip() == '':
                continue  # Skip empty line
            else:
                parts = line.split()
                if len(parts) == 4:
                    try:
                        layer = int(parts[0])
                        row = int(parts[1])
                        column = int(parts[2])
                        rate = float(parts[3])
                        data.append([current_year, current_month, layer, row, column, rate])
                    except ValueError:
                        continue  # Skip if there's an issue parsing the data

    df = pd.DataFrame(data, columns=['Year', 'Month', 'Layer', 'Row', 'Column', 'Rate'])
    return df
    
# Conditional Model Selection Display
if selected_option == "Whole watershed":
    # Display Model selection part only when these options are selected
    st.sidebar.title("Model selection")
    st.sidebar.subheader("Climate")
    selected_decade_climate = st.sidebar.selectbox(
        "Choose a decade for Climate:",
        ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020','logged','F30','F60'],
        index=6  # Default to 2010s (index 6)
    )

    st.sidebar.subheader("Land Use")
    selected_decade_land_use = st.sidebar.selectbox(
        "Choose a decade for Land Use:",
        ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020','logged','F30','F60'],
        index=6  # Default to 2010s (index 6)
    )

    st.sidebar.subheader("Water Use")
    selected_decade_water_use = st.sidebar.selectbox(
        "Choose a decade for Water Use:",
        ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020','half', 'double','NPS'],
        index=6  # Default to 2010s (index 6)
    )

    # Define a function to generate folder path based on selected decade
    def get_folder_path(land_use, climate, water_use):
        # Handle "logged" as a special case for land use
        if land_use.lower() == "logged":
            land_use_code = "Logged"  # Use the exact case as in the folder name
        elif land_use.lower() == "f30":
            land_use_code = "F30"  # Directly use F30 as the folder name
        elif land_use.lower() == "f60":
            land_use_code = "F60"  # Directly use F30 as the folder name
        else:
            land_use_code = f'L{land_use[-2:]}'  # Prepend 'L' to the last two characters
        
        # Format climate code
        climate_code = f'C{climate[-2:]}'
    
        # Handle special cases for water_use
        if water_use.lower() == "half":
            water_use_code = "half"
        elif water_use.lower() == "double":
            water_use_code = "double"
        elif water_use.lower() == "nps":
            water_use_code = "NPS"
        else:
            # Default case for regular decade-based water use (e.g., W1950, W2010)
            water_use_code = f'W{water_use[-2:]}'  
    
        # Construct the folder name
        folder_name = f'{land_use_code}_{climate_code}_{water_use_code}'
        return Path(__file__).parent / 'data' / folder_name
    
    # Get the folder based on selected decades
    data_folder = get_folder_path(selected_decade_land_use, selected_decade_climate, selected_decade_water_use)
    
    # Path to your data file
    data_filename = data_folder / 'swatmf_out_MF_gwsw_monthly.csv'

    # Check if the file exists
    if data_filename.exists():
        df = process_swatmf_data(data_filename)
        # st.write(df.head())  # Show the first few rows of the data
    else:
        st.error(f"Data file not found: {data_filename}")
   
# Month names for mapping
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Function to process SWAT-MF data
@st.cache_data
# Function to read the recharge file
def read_recharge_file(file_path):
    data = {}
    current_year = None
    current_month = None
    reading_data = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if 'month:' in line:
                parts = line.split()
                try:
                    current_month = int(parts[1])
                    current_year = int(parts[3])
                    data[(current_year, current_month)] = []
                    reading_data = True  # Start reading grid data
                except (IndexError, ValueError):
                    reading_data = False
            elif line.startswith("Grid data:") or line.startswith("Monthly Averaged Recharge Values"):
                continue
            elif not line:
                continue
            elif reading_data:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        values = [float(v) for v in parts]
                        data[(current_year, current_month)].append(values)
                    except ValueError:
                        continue
            else:
                reading_data = False

    grid_shape = (68, 94)
    for key in data:
        grid_data = np.array(data[key])
        if grid_data.size:
            data[key] = grid_data.reshape(grid_shape)
        else:
            data[key] = np.full(grid_shape, np.nan)

    return data

def compute_monthly_mean(data):
    mean_data = {}
    for (year, month), grid in data.items():
        if month not in mean_data:
            mean_data[month] = []
        mean_data[month].append(grid)
    for month, grids in mean_data.items():
        stacked_grids = np.stack(grids, axis=0)
        mean_data[month] = np.nanmean(stacked_grids, axis=0)
    return mean_data

def create_map(data, selected_month=None):
    # Create a Folium map centered around a specific latitude and longitude
    m = folium.Map(location=[latitude, longitude], zoom_start=8)

    # Create a heat map layer
    if selected_month:
        # Assuming your data has latitude and longitude for each point
        for index, row in data.iterrows():
            folium.CircleMarker(
                location=(row['Latitude'], row['Longitude']),
                radius=5,
                color='blue',
                fill=True,
                fill_opacity=0.6,
                popup=f"Rate: {row['Rate']}"
            ).add_to(m)

    return m

# Path to recharge data
RECHARGE_FILENAME = Path(__file__).parent / 'data/swatmf_out_MF_recharge_monthly.txt'
recharge_data = read_recharge_file(RECHARGE_FILENAME)
monthly_recharge_means = compute_monthly_mean(recharge_data)

# Custom title function
def custom_title(text, size):
    st.markdown(
        f"<h1 style='font-size:{size}px;'>{text}</h1>",
        unsafe_allow_html=True
    )

# Set the width and height based on the device screen size
def get_iframe_dimensions():
    return "100%", "600"

# Define the EPSG code for the shapefiles
epsg = 32610  # Adjust this if necessary

# Set the paths to your shapefiles
main_path = Path(__file__).parent
subbasins_shapefile_path = main_path / 'data/subs1.shp'
grid_shapefile_path = main_path / 'data/koki_mod_grid.shp'
deltas_file = main_path / 'data/subbasin_deltas.xls'

# Load the subbasins GeoDataFrame from the shapefile
try:
    subbasins_gdf = gpd.read_file(subbasins_shapefile_path)
except Exception as e:
    st.error(f"Error loading subbasins shapefile: {e}")
    st.stop()  # Stop execution if there's an error

# Ensure the GeoDataFrame is in the correct CRS
subbasins_gdf = subbasins_gdf.to_crs(epsg=epsg)

# Load the grid GeoDataFrame from the shapefile
try:
    grid_gdf = gpd.read_file(grid_shapefile_path)
except Exception as e:
    st.error(f"Error loading grid shapefile: {e}")
    st.stop()  # Stop execution if there's an error

# Check if the CRS is set for the grid shapefile, and set it manually if needed
if grid_gdf.crs is None:
    grid_gdf.set_crs(epsg=32610, inplace=True)

# Ensure the grid GeoDataFrame is in the correct CRS
grid_gdf = grid_gdf.to_crs(epsg=epsg)

# Define initial location (latitude and longitude for Duncan, BC)
initial_location = [48.67, -123.79]  # Duncan, BC

if selected_option == "Watershed models":
    custom_title("Xwulqw'selu Sta'lo' Watershed Model", 28)

    st.markdown("""
    
    **Summer flows in Xwulqw'selu Sta'lo' (Koksilah River) have been decreasing through time. Watershed models can be useful tools to better understand where, when and why this is happening.**
    
    """)
    
    # Set the data folder
    data_folder = Path(__file__).parent / 'data'
    image1_path = data_folder / '1_con.jpg'
    image2_path = data_folder / '2_log.jpg'
    
    # Custom styling for cards
    st.markdown("""
    <style>
    .card {
        background-color: #eef6f9;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }
    .card h3 {
        color: #1a3c40;
        margin-bottom: 10px;
    }
    .card p {
        font-size: 16px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Section 1: Graphic 1 ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 2])
    with col1:
        st.image(image1_path, caption="Conceptual model developed of the Xwulqw’selu Watershed highlighting: hydrology fluxes and subsurface aquifer units with names from the BC provincial aquifer mapping", use_container_width=True)
    with col2:
        st.markdown("### Introduction")
        st.markdown("""
        The Xwulqw'selu Connections research team at University of Victoria developed a whole-of-watershed model using the best available data to represent current conditions.  
        The complex computer model includes recent climate data and all the ways water travels through the watershed over time and space.  
        The model is useful to explore the impacts of water and land management choices on the health of the watershed now and in the future.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Section 2: Graphic 2 ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col3, col4 = st.columns([1.5, 2])
    with col3:
        st.image(image2_path, caption="This photo shows a view of the upper Xwulqw'selu watershed, where logging and land use change are clearly visible", use_container_width=True)
    with col4:
        st.markdown("### Content")
        st.markdown("""
        The maps and graphs are interactive, and they offer another way of seeing the whole watershed.  
        Explore how changes in [water use](#) and [forestry practice](#) can impact summer streamflow.  
        This model and website were developed mostly by David Serrano.  
        Much more information is available in [David’s thesis](#) if you would like to explore more.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .definition-box {
        background-color: #f0f8ff;
        border-left: 6px solid #4682B4;
        padding: 16px;
        border-radius: 10px;
        font-family: 'Segoe UI', sans-serif;
    }
    .definition-box h4 {
        margin-top: 0;
        color: #2c3e50;
    }
    .definition-box p {
        margin-bottom: 10px;
    }
    </style>
    
    <div class="definition-box">
    <h4>Here are a few definitions of some important technical words:</h4>
    
    <p><strong>Low flows:</strong> Streamflow (measured in volume per time) in the Xwulqw'selu Sta'lo' is lowest during the summer. We focus especially on the streamflow in August while recognizing that watershed health involves many other aspects.</p>
    
    <p><strong>Baseline model:</strong> The model that is the best of the available recent data (2012 – 2023) about geology, climate, topography, streamflow and ecosystems in the Xwulqw’selu watershed.</p>
    
    <p><strong>Scenarios:</strong> Changing the baseline model with different water use or land use so that we can see how water use and land use changes low flows.</p>
    
    <p><strong>Water use:</strong> The amount of water taken from streams or aquifers for agriculture. This info comes from the Province of British Columbia.</p>
    
    <p><strong>Land use:</strong> How people use the land — like growing food (agriculture), cutting trees (forestry), building towns, or leaving it natural. This info comes from Government of Canada.</p>
    </div>
    """, unsafe_allow_html=True)
           
elif selected_option == "Whole watershed":
    
    st.markdown("""
    ### The Importance of the Whole Watershed
    
    The watershed model results reaffirm the importance of a whole-of-watershed approach to watershed management.  
    
    You can zoom into any part of the maps or change which month you are looking at if you want to see how the hydrologic components of the watershed change by season.  
    
    ---
    
    ### Groundwater-Surface Water Interactions  
    
    Groundwater interacts with streams in different ways. Streams can be either **gaining** (with groundwater flowing to streams, shown as blue on the map) or **losing** (with surface water flowing to the aquifer, shown as brown on the map). Most Xwulqw’selu Sta’lo’ tributaries are gaining throughout the whole year, even in winter. This finding underscores the important contributions of groundwater to overall watershed budgets.  
    
    """)

    # Step 1: Group data by Month, Row, and Column, and calculate the mean for each location across all years
    monthly_stats = df.groupby(['Month', 'Row', 'Column'])['Rate'].mean().reset_index()
    
    # Step 2: Pivot the data to make it easier to compare months
    pivoted = monthly_stats.pivot_table(index=['Row', 'Column'], columns='Month', values='Rate').reset_index()
    
    # Step 3: Allow user to select a specific month for analysis
    unique_months = sorted(monthly_stats['Month'].unique())
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    unique_month_names = [month_names[m - 1] for m in unique_months]
    
    # Dropdown to select the month
    selected_month_name = st.selectbox("Select a Month", unique_month_names, index=0)
    selected_month = unique_months[unique_month_names.index(selected_month_name)]
    
    # Step 4: Create a grid to store color codes and hover text
    rows = int(df['Row'].max())
    cols = int(df['Column'].max())
    grid = np.full((rows, cols), np.nan)
    hover_text = np.empty((rows, cols), dtype=object)
    
    # Step 5: Analyze each location (Row, Column) for changes and classify values
    for _, row in pivoted.iterrows():
        row_vals = row.drop(['Row', 'Column']).values  # Extract monthly values for this location
        row_idx = int(row['Row']) - 1
        col_idx = int(row['Column']) - 1
    
        # Get the value for the selected month
        value = row_vals[selected_month - 1]
        grid[row_idx, col_idx] = row_vals[selected_month - 1]
        # Get the previous month value (December if January)
        prev_month_value = row_vals[selected_month - 2] if selected_month > 1 else row_vals[-1]
    
        # Function to classify values based on the provided ranges (from positive to negative)
        # Function to classify values based on simplified ranges
        def classify_based_on_value_range(value):
            if value > 1:  
                return 0  # Brown
            elif 1 >= value > -1:  
                return 1  # Dark Yellow
            else:  # Losing (extreme negatives)
                return 2  # Dark Blue

        # Classify the current value and assign to the grid
        grid[row_idx, col_idx] = classify_based_on_value_range(value)
        # Add hover text for the grid cell
        hover_text[row_idx, col_idx] = f'Value: {value:.2f} (Prev: {prev_month_value:.2f})'
    
    # Function to create the heatmap
    def create_heatmap(classified_grid, selected_month_name, hover_text):
        # Define a color scale for the classified ranges
        colorscale = [
            [0.0, '#8B4513'],  # Losing - Brown
            [0.66, '#FFFF00'],  # No significant contributions - Yellow
            [1.0, '#00008B'],   # Gaining - Dark Blue
        ]
        
        # Create the heatmap for the selected month
        fig = go.Figure(data=go.Heatmap(
            z=classified_grid,
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            showscale=False,  # Hide scale since categories are defined
            hoverinfo='text',
            text=hover_text
        ))
    
        # Update the layout of the heatmap
        fig.update_layout(
            title=f'Groundwater-Surface Water Interaction for {selected_month_name} [m³/day]',
            xaxis_title='Column',
            yaxis_title='Row',
            xaxis=dict(showticklabels=True, ticks='', showgrid=False),
            yaxis=dict(showticklabels=True, ticks='', autorange='reversed', showgrid=False),
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=8, color='black'),
        )
        
        st.plotly_chart(fig)
    
    def count_cells_per_color(grid):
        # Combine categories into four main classifications
        color_counts = {
            'gaining': np.sum(grid == 2),  # positive
            'no_significant_contributions': np.sum(grid == 1),  # 
            'losing': np.sum(grid == 0),  # negative
        }
        return color_counts

    # Count the colors for the selected month
    color_counts = count_cells_per_color(grid)
    
    # Prepare data for pie chart with updated classification ranges
    color_names = [
        'Gaining < -1',           # Categories 3, 4, 5 combined
        'No Significant Contributions 1 to -1',  # Categories 6, 7 combined
        'Losing > 1',            # Category below -225
    ]
    
    color_values = [
        color_counts['gaining'],          # Gaining
        color_counts['no_significant_contributions'],  # No significant contributions
        color_counts['losing'],           # Losing
    ]
    
    # Prepare the pie chart data
    total_cells = sum(color_values)
    percentages = [count / total_cells * 100 if total_cells > 0 else 0 for count in color_values]
    cell_counts = [str(count) for count in color_values]

    # Combine percentages and cell counts for display in the labels
    labels_with_counts = [
        f"{name}: {count} cells ({percentage:.1f}%)"
        for name, count, percentage in zip(color_names, cell_counts, percentages)
    ]
    
    # Create a pie chart with formatted percentages
    pie_colors = [
        '#00008B',  # Dark Blue (extreme negative)
        '#FFFF00',  # Yellow (slightly positive)
        '#8B4513'   # Brown (strong positive)
    ]
    
    # Create the pie chart
    fig = go.Figure(data=[go.Pie(
        labels=color_names,
        values=color_values,
        hole=0.3,  # Optional donut chart
        marker=dict(colors=pie_colors),
        textinfo='none',  # Display only the percentage
        hoverinfo='label+value+percent',  # Display label, value, and percent on hover
    )])
    
    # Display pie chart
    st.plotly_chart(fig)

    create_heatmap(grid, selected_month_name, hover_text)
    
    # Filter data for the selected month
    selected_month_data = monthly_stats[monthly_stats['Month'] == selected_month]

    st.markdown("""
    ---
    
    ### Groundwater Recharge  
    
    Aquifers are recharged by both precipitation and streams. The rate of average groundwater recharge is shown on the map, with the **darkest blue** indicating the highest rates of groundwater recharge. Groundwater recharge occurs mostly in winter, sourced from both precipitation and rivers. Importantly, groundwater recharge occurs across much of the watershed, reaffirming the importance of a **whole-of-watershed management strategy**.  
    """
    )
    # Define the pixel area in square meters
    pixel_area_m2 = 300 * 300  # Each pixel is 300x300 meters, so 90,000 m²

    # Days per month (for conversion from m³/day to m³/month)
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }

    # Convert recharge from m³/day to mm/month for each pixel
    def convert_recharge_to_mm_per_month(recharge_m3_per_day, month, pixel_area_m2):
        days = days_in_month[month]  # Get number of days in the selected month
        recharge_m3_per_month = recharge_m3_per_day * days  # Convert to m³/month
        recharge_mm_per_month = (recharge_m3_per_month / pixel_area_m2) * 1000  # Convert to mm/month
        return recharge_mm_per_month

    # Select the recharge month and convert recharge to mm/month
    recharge_months = list(monthly_recharge_means.keys())
    recharge_month_names = [month_names[m - 1] for m in recharge_months]

    selected_recharge_month_name = st.selectbox("Select Month", recharge_month_names)
    selected_recharge_month = recharge_months[recharge_month_names.index(selected_recharge_month_name)]

    # Assume monthly_recharge_means[selected_recharge_month] is a grid (e.g., a 2D array) of m³/day values
    recharge_grid_m3_per_day = monthly_recharge_means[selected_recharge_month]

    # Convert the recharge grid to mm/month for each pixel (element-wise conversion)
    recharge_grid_mm_per_month = [[convert_recharge_to_mm_per_month(value, selected_recharge_month, pixel_area_m2)
                                for value in row]
                                for row in recharge_grid_m3_per_day]

    # Create heatmap for recharge in mm/month
    fig_recharge = go.Figure(data=go.Heatmap(
        z=recharge_grid_mm_per_month,  # Using the converted recharge values in mm/month
        colorscale='viridis',
        colorbar=dict(
            title='Recharge [mm/month]',
            orientation='h',
            x=0.5,
            y=-0.1,
            xanchor='center',
            yanchor='top',
        )
    ))

    fig_recharge.update_layout(
        title=f'Monthly Recharge - {selected_recharge_month_name}',
        xaxis_title='Column',
        yaxis_title='Row',
        yaxis=dict(autorange='reversed'),  # Reverse y-axis for heatmap
        width=800,
        height=600,
    )

    # Initialize the map centered on Duncan
    m = folium.Map(location=initial_location, zoom_start=11, control_scale=True)

    # Add the subbasins layer to the map but keep it initially turned off
    subbasins_layer = folium.GeoJson(subbasins_gdf, 
                                    name="Subbasins", 
                                    style_function=lambda x: {'color': 'green', 'weight': 2},
                                    # show=False  # Keep the layer off initially
                                    ).add_to(m)

    # Add the grid layer to the map but keep it initially turned off
    grid_layer = folium.GeoJson(grid_gdf, 
                                name="Grid", 
                                style_function=lambda x: {'color': 'blue', 'weight': 1},
                                show=False  # Keep the layer off initially
                            ).add_to(m)

    # Add MousePosition to display coordinates
    MousePosition().add_to(m)

    # Add a layer control to switch between the subbasins and grid layers
    folium.LayerControl().add_to(m)

    # Display the plotly heatmap in Streamlit
    st.plotly_chart(fig_recharge, use_container_width=True)  
    
    # Render the Folium map in Streamlit
    st.title("Watershed Map")
    st_folium(m, width=700, height=600)  


elif selected_option == "Water use":
    
    st.markdown("""
    ### Water Use Scenarios  
    
    Water use, especially surface water use, strongly and quickly impacts summer low flows in the **Xwulqw'selu Sta'lo'** at Cowichan Station.  
    
    You can zoom into any part of the graphs, and if you want to see detailed differences between scenarios, check out **David’s thesis**. Water use refers to any water extracted from streams or aquifers for agriculture, as calculated by the **Province of British Columbia**. **Streamflow** refers to the flow in the Xwulqw'selu Sta'lo', measured in cubic meters per second.  
    
    ---
    
    ### Three Types of Water Use Scenarios  
    
    We explored three key ways water use could impact summer low flows:  
    
    1. **Total Water Use:** We **doubled or halved** total water use (from both groundwater and surface water) compared to the baseline to assess the overall impact of water use.  
    2. **Decreasing Groundwater or Surface Water Use:** We reduced either **only groundwater use** or **only surface water use** throughout the year to evaluate the distinct effects of each water source.  
    3. **Changing the Timing of Water Use Restrictions:** We adjusted the **start month of water use restrictions** (June, July, or August) to see if timing influenced the impact of restrictions.  
    
    ---
    
    ### Total Water Use  
    
    The upper graph displays different **rates of total water use** in various scenarios. The lower graph illustrates how these changes affect **streamflow**, which varies seasonally. The **red line** represents the **fish protection order** threshold.  
    
    - **Doubling total water use** reduces low flows by over **50%**, often dropping below fish protection thresholds for nearly a month.  
    - **Halving total water use** increases low flow by **50%**, emphasizing the benefits of water conservation.  
    
    ---
    
    ### Decreasing Groundwater or Surface Water Use  
    
    In these graphs, **only groundwater or only surface water use** is reduced.  
    
    - **Surface water extraction has an immediate impact on streamflow**—when water is taken from a stream, flow decreases immediately.  
    - **Groundwater pumping affects streamflow more slowly**, sometimes taking days, weeks, or months for the impact to appear.  
    - **Scenarios where groundwater or surface water use is halved** show that surface water use has a more immediate and significant effect on low flows.  
    - The impact of groundwater pumping on low flows depends on well location relative to streams and aquifers.  
    
    ---
    
    ### Changing the Timing of Water Use Restrictions  
    
    This scenario examines the effect of **starting water use restrictions in June, July, or August**.  
    
    - **Starting restrictions earlier in the season is less critical** since surface water restrictions have immediate impacts.  
    - **Water use restrictions can increase low flows by 20–50%**, with later-starting restrictions still yielding significant benefits.  
    
    ---
    
    These insights highlight the importance of **adaptive water management** and **conservation strategies** to maintain healthy summer streamflows in the Xwulqw'selu Sta'lo'.  
    
    """)

    # Settings
    data_folder = os.path.join("data", "reach_csv")
    
    # Scenario colors and labels
    scenario_colors = {
        "Scenario R3": "black",
        "Scenario R3 S 05": "lightblue",
        "Scenario R3 X2": "navy",
        "Scenario jun": "pink",
        "Scenario jul": "#C71585",
        "Scenario aug": "#800080",
        "Scenario R3 G 05": "darkblue",
        "Scenario R3 SG 05": "skyblue",
        "Scenario mat you": "lightgreen",
        "Scenario mat 60": "darkgreen"
    }
    
    scenario_legend = {
        "Scenario jun": "No Water Use June-August",
        "Scenario jul": "No Water Use July-August",
        "Scenario aug": "No Water Use August",
        "Scenario R3 S 05": "Half Surface Water Use",
        "Scenario R3 G 05": "Half Groundwater Use",
        "Scenario R3": "Base case",
        "Scenario R3 X2": "Double Water Use",
        "Scenario R3 SG 05": "Half Water Use",
        "Scenario mat you": "Mature and Immature Forest",
        "Scenario mat 60": "Mature Forest"
    }
    
    scenario_groups = {
        "Jun-Jul-Aug with Base": ["Scenario jun", "Scenario jul", "Scenario aug", "Scenario R3"],
        "Half, Double, Base": ["Scenario R3 SG 05", "Scenario R3 X2", "Scenario R3"],
        "Surface Half, Ground Half, Base": ["Scenario R3 S 05", "Scenario R3 G 05", "Scenario R3"],
        "Mature, Mature-Immature, Base": ["Scenario mat you", "Scenario mat 60", "Scenario R3"]
    }
    
    tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ticktext = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y_ticks = [0.05, 0.18, 1, 10, 50]
    
    st.title("Streamflow Scenario Explorer (Separate CSVs)")
    
    # Select scenario group
    selected_group = st.selectbox("Select Scenario Group", list(scenario_groups.keys()))
    scenarios = scenario_groups[selected_group]
    
    # Load data from pre-saved CSV files
    all_data = []
    for scenario in scenarios:
        file_path = os.path.join(data_folder, f"scenario_{scenario.replace(' ', '_').lower()}_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = df[df["RCH"] == 3]  # Filter for RCH 3
            df["Scenario"] = scenario
            all_data.append(df)
    
    if not all_data:
        st.error("No matching files found in the folder!")
        st.stop()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Mean daily flow
    mean_daily = combined_df.groupby(['Scenario', 'DAY'])['FLOW_OUTcms'].mean().reset_index()
    
    fig1 = px.line(mean_daily, x="DAY", y="FLOW_OUTcms", color="Scenario",
                   title=f"Mean Daily Flow - {selected_group}",
                   color_discrete_map=scenario_colors)
    
    fig1.add_hline(y=0.18, line_dash="dash", line_color="red", annotation_text="Fish Cutoff (0.18 cms)")
    
    fig1.update_layout(
        xaxis=dict(tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(type="log", tickvals=y_ticks),
        width=800, height=400
    )
    
    fig1.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    st.plotly_chart(fig1)
    
    # Delta Flow
    base_flow = mean_daily[mean_daily["Scenario"] == "Scenario R3"].rename(columns={"FLOW_OUTcms": "Base_Flow"})
    merged = pd.merge(mean_daily, base_flow[["DAY", "Base_Flow"]], on="DAY", how="left")
    merged["Delta Flow"] = (merged["FLOW_OUTcms"] - merged["Base_Flow"]) / merged["Base_Flow"]
    
    fig2 = px.line(merged, x="DAY", y="Delta Flow", color="Scenario",
                   title=f"Relative Change in Flow from Base - {selected_group}",
                   color_discrete_map=scenario_colors)
    
    fig2.update_layout(
        xaxis=dict(tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(range=[-1.1, 1.1]),
        width=800, height=400
    )
    
    fig2.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    st.plotly_chart(fig2)
    
    # FDC
    def compute_fdc(df):
        sorted_flow = df["FLOW_OUTcms"].sort_values(ascending=False)
        prob = (sorted_flow.rank(ascending=False) / len(sorted_flow)) * 100
        return pd.DataFrame({"Flow (cms)": sorted_flow.values, "Exceedance Probability (%)": prob.values})
    
    fdc_all = pd.concat([
        compute_fdc(combined_df[combined_df["Scenario"] == s]).assign(Scenario=s) for s in scenarios
    ], ignore_index=True)
    
    fig3 = px.line(fdc_all, x="Exceedance Probability (%)", y="Flow (cms)", color="Scenario",
                   title=f"Flow Duration Curve - {selected_group}",
                   color_discrete_map=scenario_colors)
    
    fig3.update_layout(
        yaxis_type="log",
        width=800, height=400
    )
    
    fig3.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    st.plotly_chart(fig3)

   

elif selected_option == "Land use":   
    
    st.markdown("""
    ### Land Use Scenarios  
    
    Land use, particularly **forest age distributions**, has a less significant impact on **summer low flows** but influences other aspects of the water cycle.  
    
    You can zoom into any part of the graphs, and if you want to explore **detailed differences between scenarios** or their effects on **evapotranspiration**, check out **David’s thesis** for more insights.  
    
    ---
    
    ### Forestry Impacts on the Water Cycle  
    
    We compared **current conditions** to two alternative land use scenarios by modifying **forest age distribution**, since tree age influences hydrological processes in multiple ways.  
    
    The scenarios include:  
    
    1. **Mature Forest Scenario** – Increased land area of **60-year-old trees**.  
    2. **Mature and Immature Forest Scenario** – A mix of **mature (60-year-old) and immature (30-year-old) forests**.  
    
    The percentage of each forest age class in these scenarios is illustrated in **pie charts**. The watershed diagram highlights how these changes influence **runoff, interflow, and evapotranspiration**, demonstrating that **forest age distributions** play a role in hydrological processes.  
    
    ---
    
    ### Impacts on Streamflow  
    
    - A **mature (60-year-old) forest scenario** reduced **summer streamflow by up to 10%**.  
    - A **mixed mature and immature (30-year-old) forest scenario** showed **no significant changes** in summer low flows.  
    
    These results emphasize the need to **consider forest age distributions** when evaluating **long-term hydrological changes** in the watershed.  
    
    ---
    
    This analysis helps illustrate the **complex relationships** between land use, forest age, and watershed hydrology.  
    """)




# tabs = st.tabs(["Watershed models", "Whole watershed", "Water use", "Land use"])
    
#     with tabs[0]:
#         st.header("What is the Model?")
#         st.write("Summer flows in Xwulqw’selu Sta’lo’ have been decreasing...")
#         st.image("watershed_diagram.png", caption="Watershed processes overview")
    
#     with tabs[1]:
#         st.header("Scenarios")
#         scenario = st.selectbox("Choose a scenario", ["Baseline", "Water Use Change", "Forest Harvesting"])
#         st.write(f"You selected: **{scenario}**")
#         st.plotly_chart(...)  # Insert your graph here
#         st.map(...)  # Or interactive map
    
#     with tabs[2]:
#         st.header("Key Definitions")
#         with st.expander("Low Flows"):
#             st.write("The lowest streamflows during summer, typically August...")
#         with st.expander("Baseline Model"):
#             st.write("A model representing 2012–2023 conditions...")


