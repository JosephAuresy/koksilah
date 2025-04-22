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
import streamlit.components.v1 as components
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

# # Sidebar for navigation
# st.sidebar.title("Xwulqw'selu Sta'lo'")
# selected_option = st.sidebar.radio(
#     "Select an option:",
#     #("Watershed models", "Water interactions", "Recharge", "View Report")
#     ("Watershed models", "Whole watershed", "Water use", "Land use")
# )

import streamlit as st

# --- Page options ---
pages = [
    "Home",
    "The importance of the whole watershed",
    "Water use scenarios",
    "Land use scenarios"
]

# --- Initialize session state ---
if "selected_page" not in st.session_state:
    st.session_state.selected_page = pages[0]

# --- Sidebar navigation ---
with st.sidebar:
    st.markdown("## Xwulqw'selu Sta'lo'")
    selected = st.radio(
        "Select an option:",
        pages,
        index=pages.index(st.session_state.selected_page)
    )
    st.session_state.selected_page = selected

# --- Minimal custom CSS for pill-style top nav ---
st.markdown("""
<style>
.navbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.4em;
    margin-bottom: 1em;
}
.nav-pill {
    background-color: #e0e7ff;
    border: none;
    border-radius: 999px;
    padding: 0.4em 1em;
    font-size: 0.9em;
    color: #1e3a8a;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.3s;
}
.nav-pill:hover {
    background-color: #c7d2fe;
}
.nav-pill-active {
    background-color: #3b82f6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Render collapsible-style top nav ---
with st.expander("📂 Open Navigation", expanded=False):
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    for page in pages:
        btn_class = "nav-pill nav-pill-active" if st.session_state.selected_page == page else "nav-pill"
        if st.button(page, key=f"btn_{page}"):
            st.session_state.selected_page = page
        st.markdown(f'<div class="{btn_class}">{page}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    
def clean_text(text):
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",   # en dash
        "—": "-",   # em dash
        "…": "...", # ellipsis
        " ": " ",   # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

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
if selected_option == "The importance of the whole watershed":
    # Hidden default values (instead of sidebar widgets)
    selected_decade_climate = '2010'
    selected_decade_land_use = '2010'
    selected_decade_water_use = '2010'
    # Display Model selection part only when these options are selected

    # st.sidebar.title("Model selection")
    # st.sidebar.subheader("Climate")
    # selected_decade_climate = st.sidebar.selectbox(
    #     "Choose a decade for Climate:",
    #     ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020','logged','F30','F60'],
    #     index=6  # Default to 2010s (index 6)
    # )

    # st.sidebar.subheader("Land Use")
    # selected_decade_land_use = st.sidebar.selectbox(
    #     "Choose a decade for Land Use:",
    #     ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020','logged','F30','F60'],
    #     index=6  # Default to 2010s (index 6)
    # )

    # st.sidebar.subheader("Water Use")
    # selected_decade_water_use = st.sidebar.selectbox(
    #     "Choose a decade for Water Use:",
    #     ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020','half', 'double','NPS'],
    #     index=6  # Default to 2010s (index 6)
    # )

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

if selected_option == "Home":
    custom_title("🌎 Xwulqw'selu Sta'lo' Watershed Model", 28)

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
        st.markdown("""
        The maps and graphs are interactive, and they offer another way of seeing the whole watershed.  
        Explore how changes in [water use](#) and [forestry practice](#) can impact summer streamflow.  
        This model and website were developed mostly by David Serrano.  
        Much more information is available in [David’s thesis](#) if you would like to explore more.
        """)
    
    # st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .definition-box {
            background-color: #f7f9fa;  /* very light grey-blue */
            border-left: 4px solid #cccccc;
            padding: 16px;
            border-radius: 8px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 0.95em;
            color: #333333;
            margin-bottom: 20px;
        }
        .definition-box h4 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.1em;
        }
        .definition-box p {
            margin-bottom: 10px;
            line-height: 1.5;
        }
        </style>
    
        <div class="definition-box">
        <h4>Here are a few definitions of some important technical words:</h4>
    
        <p><strong>Low flows:</strong> Streamflow (measured in volume per time) in the Xwulqw'selu Sta'lo' is lowest during the summer. We focus especially on the streamflow in August while recognizing that watershed health involves many other aspects.</p>
    
        <p><strong>Baseline model:</strong> The model that is the best of the available recent data (2012–2023) about geology, climate, topography, streamflow, and ecosystems in the Xwulqw'selu watershed.</p>
    
        <p><strong>Scenarios:</strong> Changing the baseline model with different water use or land use so that we can see how water use and land use changes low flows.</p>
    
        <p><strong>Water use:</strong> The amount of water taken from streams or aquifers for agriculture. This info comes from the Province of British Columbia.</p>
    
        <p><strong>Land use:</strong> How people use the land — like growing food (agriculture), cutting trees (forestry), building towns, or leaving it natural. This info comes from the Government of Canada.</p>
    
        <p><strong>Groundwater:</strong> Groundwater interacts with streams in different ways. Streams can be either <strong>gaining</strong> (with groundwater flowing to streams, shown as blue on the map) or <strong>losing</strong> (with surface water flowing to the aquifer, shown as brown on the map). Most Xwulqw'selu Sta'lo' tributaries are gaining throughout the whole year, even in winter. This finding underscores the important contributions of groundwater to overall watershed budgets.</p>
        </div>
    """, unsafe_allow_html=True)

    
elif selected_option == "The importance of the whole watershed":
    
    # Title
    st.markdown("### 🗺️ The Importance of the Whole Watershed")
    
    # Light blue definition-style box
    st.markdown("""
        <style>
        .definition-box-alt {
            background-color: #e6f2ff;  /* Light blue */
            border-left: 6px solid #3399ff;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 1.05em;
            margin-bottom: 20px;
            color: #1a1a1a;
        }
        </style>
    
        <div class="definition-box-alt">
            The watershed model results reaffirm the importance of a whole-of-watershed approach to watershed management.
        </div>
    """, unsafe_allow_html=True)
    
    # Small footer-style note
    st.markdown("""
        <p style="font-size: 11px; font-family: 'Segoe UI', sans-serif;">
            You can zoom into any part of the maps or change which month you are looking at if you want to see how the hydrologic components of the watershed change by season.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Groundwater-Surface Water Interactions  
    
    Groundwater interacts with streams in different ways. Streams can be either **gaining** (with groundwater flowing to streams, shown as blue on the map) or **losing** (with surface water flowing to the aquifer, shown as brown on the map). Most Xwulqw'selu Sta'lo' tributaries are gaining throughout the whole year, even in winter. This finding underscores the important contributions of groundwater to overall watershed budgets.  
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
    
    def create_heatmap_with_pie(classified_grid, selected_month_name, hover_text, color_counts):
        # Define the color scale
        colorscale = [
            [0.0, '#8B4513'],   # Losing - Brown
            [0.66, '#FFFF00'],  # No significant contributions - Yellow
            [1.0, '#00008B'],   # Gaining - Dark Blue
        ]
    
        # Create the heatmap trace
        heatmap_trace = go.Heatmap(
            z=classified_grid,
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            showscale=False,
            hoverinfo='text',
            text=hover_text
        )
    
        # Prepare pie chart data
        color_values = [
            color_counts['gaining'],
            color_counts['no_significant_contributions'],
            color_counts['losing'],
        ]
    
        total_cells = sum(color_values)
        if total_cells == 0: total_cells = 1  # Avoid divide by zero
    
        # Pie labels
        pie_labels = ['Gaining < -1', 'No Significant Contributions 1 to -1', 'Losing > 1']
        pie_colors = ['#00008B', '#FFFF00', '#8B4513']
    
        # Pie chart trace as an inset
        pie_trace = go.Pie(
            labels=pie_labels,
            values=color_values,
            hole=0.3,
            marker=dict(colors=pie_colors),
            domain=dict(x=[0.72, 0.95], y=[0.05, 0.28]),  # Bottom-right corner
            textinfo='none',
            hoverinfo='label+value+percent',
            showlegend=False
        )
    
        # Combine both traces into one figure
        fig = go.Figure(data=[heatmap_trace, pie_trace])
    
        # Layout for the combined plot
        fig.update_layout(
            title=f'Groundwater-Surface Water Interaction for {selected_month_name} [m³/day]',
            xaxis_title='Column',
            yaxis_title='Row',
            xaxis=dict(showticklabels=True, ticks='', showgrid=False),
            yaxis=dict(showticklabels=True, ticks='', autorange='reversed', showgrid=False),
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=8, color='black'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
    
        # Show the final figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

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

    create_heatmap_with_pie(grid, selected_month_name, hover_text, color_counts)

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

    # # Initialize the map centered on Duncan
    # m = folium.Map(location=initial_location, zoom_start=11, control_scale=True)

    # # Add the subbasins layer to the map but keep it initially turned off
    # subbasins_layer = folium.GeoJson(subbasins_gdf, 
    #                                 name="Subbasins", 
    #                                 style_function=lambda x: {'color': 'green', 'weight': 2},
    #                                 # show=False  # Keep the layer off initially
    #                                 ).add_to(m)

    # # Add the grid layer to the map but keep it initially turned off
    # grid_layer = folium.GeoJson(grid_gdf, 
    #                             name="Grid", 
    #                             style_function=lambda x: {'color': 'blue', 'weight': 1},
    #                             show=False  # Keep the layer off initially
    #                         ).add_to(m)

    # # Add MousePosition to display coordinates
    # MousePosition().add_to(m)

    # # Add a layer control to switch between the subbasins and grid layers
    # folium.LayerControl().add_to(m)

    # Display the plotly heatmap in Streamlit
    st.plotly_chart(fig_recharge, use_container_width=True)  
    
    # # Render the Folium map in Streamlit
    # st.title("Watershed Map")
    # st_folium(m, width=700, height=600)  

elif selected_option == "Water use scenarios":

    st.markdown("### 🚰 Water Use Scenarios")
    
    # Light blue box for main message
    st.markdown("""
        <style>
        .definition-box-alt {
            background-color: #e6f2ff;  /* Light blue */
            border-left: 6px solid #3399ff;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 1.1em;
            margin-bottom: 20px;
            color: #1a1a1a;
        }
        </style>

        <div class="definition-box-alt">
            Water use, especially surface water use, significantly and quickly changes summer streamflow in the <strong>Xwulqw'selu Sta'lo'</strong> at Cowichan Station.
        </div>
    """, unsafe_allow_html=True)

    # Grey box for scenario explanation
    st.markdown("""
        <style>
        .definition-box {
            background-color: #f5f5f5;
            border-left: 6px solid #999999;
            padding: 14px;
            border-radius: 8px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
        .definition-box h4 {
            margin-top: 0;
            color: #333333;
            font-size: 1em;
        }
        .definition-box p {
            margin-bottom: 8px;
            line-height: 1.4;
        }
        </style>
        
        <div class="definition-box">
        <h4>We modelled water use scenarios to explore how three behavior changes might impact streamflow in Xwulqw'selu Sta'lo' at Cowichan Station:</h4>
        
        <p><strong>Total water use:</strong> We doubled and halved total water use (from both groundwater and surface water) compared to the baseline model.</p>
        
        <p><strong>Source of water use:</strong> We decreased only groundwater use or surface water use year-round to see the differences between the different water sources.</p>
        
        <p><strong>Timing of water use:</strong> We changed the timing of water use restrictions starting in June, July, or August to see if the timing and duration of summer water use restrictions impact streamflow.</p>
        </div>
    """, unsafe_allow_html=True)

    # Small footer
    st.markdown("""
        <p style="font-size: 11px; font-family: 'Segoe UI', sans-serif;">
        You can zoom into any part of the graphs. If you want to see more details about the scenarios, check out <strong>David’s thesis</strong>. Water use means any water extracted from streams or aquifers for agriculture as calculated by the Province of British Columbia. Streamflow means the flow in the <strong>Xwulqw'selu Sta'lo'</strong> in units of volume per time (cubic meters per second).
    </p>
    """, unsafe_allow_html=True)


    # Define colors for each scenario
    scenario_colors = {
        "Scenario R3": "black",
        "Scenario R3 S 05": "lightblue",
        "Scenario R3 X2": "navy",
        "Scenario jun": "pink",  # magenta
        "Scenario jul": "#C71585",
        "Scenario aug": "#800080",
        "Scenario R3 G 05": "darkblue",
        "Scenario R3 SG 05": "skyblue",
    }
    
    # Scenario legend
    scenario_legend = {
        "Scenario jun": "No Water Use June-August",
        "Scenario jul": "No Water Use July-August",
        "Scenario aug": "No Water Use August",
        "Scenario R3 S 05": "Half Surface Water Use",
        "Scenario R3 G 05": "Half Groundwater Use",
        "Scenario R3": "Baseline",
        "Scenario R3 X2": "Double Water Use",
        "Scenario R3 SG 05": "Half Water Use",
    }
    
    # Scenario groups
    scenario_groups = {
        "Total water use": ["scenario_SG_05_data.csv", "scenario_SG_X2_data.csv", "scenario_R3_data.csv"],
        "Decreasing groundwater or surface water use": ["scenario_S_05_data.csv", "scenario_G_05_data.csv", "scenario_R3_data.csv"],
        "Change the timing of water use restrictions": ["scenario_jun_data.csv", "scenario_jul_data.csv", "scenario_aug_data.csv", "scenario_R3_data.csv"],
    }
    
    # Define tick values (start of each month approx)
    tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ticktext = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y_ticks = [0.05, 0.18, 1, 10, 50]
    
    # Process each scenario group
    for title, files in scenario_groups.items():
        scenario_data = []
    
        for file in files:
            file_path = os.path.join("data/reach_csv", file)
    
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Failed to load file: {file_path}\nError: {e}")
                st.stop()
    
            # Capitalize column names just in case
            df.columns = [col.upper() for col in df.columns]
    
            # Add a Scenario column from file name if not in the file
            if "SCENARIO" not in df.columns:
                scenario_name = os.path.splitext(file)[0].replace("scenario_", "").replace("_data", "")
                df["SCENARIO"] = f"Scenario {scenario_name.replace('_', ' ')}"
    
            scenario_data.append(df)
    
        combined_data = pd.concat(scenario_data, ignore_index=True)
    
        # Filter for Reach 3 and valid days
        rch3_data = combined_data[(combined_data["RCH"] == 3) & (combined_data["DAY"].between(1, 365))]
    
        # Mean daily flow per scenario
        mean_daily_flow = rch3_data.groupby(["SCENARIO", "DAY"])["FLOW_OUTCMS"].mean().reset_index()
    
        max_flow = mean_daily_flow["FLOW_OUTCMS"].max()
    
        # === Plot Mean Daily Flow ===
        fig2 = px.line(
            mean_daily_flow,
            x="DAY",
            y="FLOW_OUTCMS",
            color="SCENARIO",
            title=f"Mean Daily Flow for Reach 3 - {title}",
            labels={"DAY": "Day of the Year", "FLOW_OUTCMS": "Mean Flow (cms)"},
            color_discrete_map=scenario_colors
        )
    
        fig2.add_hline(
            y=0.18, line_dash="dash", line_color="red", line_width=2,
            annotation_text="Fish Protection Cutoff (0.18 cms)",
            annotation_position="right",
            annotation_y=0.18,
            annotation_font_size=12
        )
    
        # Update legend names
        fig2.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    
        fig2.update_layout(
            xaxis=dict(
                title="Day of the Year",
                showgrid=True,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            ),
            yaxis=dict(
                title="Mean Flow (cms)",
                showgrid=True,
                type="log",
                range=[np.log10(min(y_ticks)), np.log10(max(y_ticks))],
                tickvals=y_ticks
            ),
            legend=dict(title="Scenario"),
            width=800,
            height=400
        )
    
        st.plotly_chart(fig2)
    
        # === Plot Delta Flow ===
        base_scenario = mean_daily_flow[mean_daily_flow["SCENARIO"] == "Scenario R3"].rename(columns={"FLOW_OUTCMS": "Base_Flow"})
        merged_data = pd.merge(mean_daily_flow, base_scenario[["DAY", "Base_Flow"]], on="DAY", how="left")
        merged_data["Delta Flow"] = (merged_data["FLOW_OUTCMS"] - merged_data["Base_Flow"]) / merged_data["Base_Flow"]
    
        fig4 = px.line(
            merged_data,
            x="DAY",
            y="Delta Flow",
            color="SCENARIO",
            title=f"Change in Summer Streamflow - {title}",
            color_discrete_map=scenario_colors
        )
    
        fig4.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    
        fig4.update_layout(
            xaxis=dict(
                title="Day of the Year",
                showgrid=True,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            ),
            yaxis=dict(
                title="Delta Flow (Relative Change)",
                showgrid=True,
                range=[-1.1, 1.1]
            ),
            legend=dict(title="Scenario"),
            width=800,
            height=400
        )
    
        st.plotly_chart(fig4)
        
        # Add detailed explanation for the scenario group
        if title == "Total water use":
            st.markdown("""        
            **Total water use:**  
            The upper graph shows the different rates of total water use in different scenarios.  
            The middle graph shows changes in streamflow over the same time period. Notice that the mean flow (in volume per time or cms = cubic meters per second) changes seasonally, and the changes are dramatic; we modified the y-axis scale to emphasize and better show the impact during seasonal low flows.
    
            The red line at 0.18 cms represents the streamflow threshold that triggers the provincial government to issue a temporary order to restrict water use to protect fish populations.
    
            The lower graph is the same changes in streamflow shown as a percentage change compared to the baseline (so 0.5 = 50% more streamflow and -0.5 = 50% less streamflow).  
    
            Doubling overall water use results in August streamflow that is 50% lower than the baseline and less than 0.18 cms for nearly a month.  
            In contrast, halving overall water use results in August streamflow that is 50% higher, emphasizing that conserving water is a crucial water management strategy.
            """)
    
        elif title == "Decreasing groundwater or surface water use":
            st.markdown("""    
            **Decreasing groundwater or surface water use:**  
            In these graphs only groundwater OR surface water is decreased. In these scenarios, water use from the other water source was kept constant (so the overall volume of water use also decreased in these scenarios).
    
            Streamflow is impacted quickly when you take water from a stream, whereas when you pump from a well, it takes days, weeks, or months to impact streamflow.
    
            Scenarios of halving groundwater or surface water use show that surface water use has a much more significant and direct impact on low flows.  
            The impact of decreased groundwater use on low flows is slower and less significant and needs additional research; the impact of individual wells depends on the location of the wells, streams, and aquifers.
            """)
    
        elif title == "Change the timing of water use restrictions":
            st.markdown("""    
            **Change the timing of water use restrictions:**  
            In these scenarios, water use restrictions start at the beginning of June, July, or August.  
    
            Scenarios of starting water use restrictions at different times suggested low flows can be increased by up to 100%.  
            Starting earlier in the season is less important since surface water use restrictions change low flows quickly.  
    
            The results suggest the impacts of water source are greater than the impacts of the timing of water use restrictions.  
            Future research could explore the strategic use of groundwater and surface water at different times of the summer season.
            """)
    
        else:
            st.markdown("*Scenario group summary not available.*")

elif selected_option == "Land use scenarios":   
    
    st.markdown("### 🌲 Land Use Scenarios")
    
    # Add styling for the light blue box
    st.markdown("""
        <style>
        .definition-box-alt {
            background-color: #e6f2ff;  /* Light blue */
            border-left: 6px solid #3399ff;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 1.1em;
            margin-bottom: 20px;
            color: #1a1a1a;
        }
        </style>

        <div class="definition-box-alt">
            The age of trees can impact the water cycle in multiple ways.
            Land use, as represented by different ages of forests,
            changes streamflow in the summer less significantly than water use,
            but changes other parts of the water cycle.
        </div>
    """, unsafe_allow_html=True)

    # Keep the grey box for scenario details
    st.markdown("""
        <style>
        .definition-box {
            background-color: #f5f5f5;
            border-left: 6px solid #999999;
            padding: 14px;
            border-radius: 8px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
        .definition-box h4 {
            margin-top: 0;
            color: #333333;
            font-size: 1em;
        }
        .definition-box p {
            margin-bottom: 8px;
            line-height: 1.4;
        }
        </style>
        
        <div class="definition-box">
        <h4>We modelled two land use scenarios by changing the age of Douglas Fir forests across the watershed:</h4>
        
        <p><strong>Baseline model:</strong> In the baseline model, the watershed area is ~78% mature forest (trees 60 years or older), 3% immature forest (trees ~30 years), and ~19% recently logged.</p>
        
        <p><strong>Mature forest scenario:</strong> The percentage area of mature forest is increased from 78% in the baseline model to 96% to show the impact of mature forests.</p>
        
        <p><strong>Mature and immature forest scenario:</strong> The percentage area of mature forest is 66%, while the immature and recently logged are 17% each to show the impact of immature forests.</p>
        </div>
    """, unsafe_allow_html=True)

    # Small footer text
    st.markdown("""
        <p style="font-size: 11px; font-family: 'Segoe UI', sans-serif;">
        You can zoom into any part of the graphs or if you want to see the detailed differences between scenarios or how these scenarios change evapotranspiration, check out <strong>David’s thesis</strong> for more details.
    </p>
    """, unsafe_allow_html=True)
    
    # Define colors for each scenario
    scenario_colors = {
        "Scenario R3": "black",
        "Scenario mat you": "lightgreen",
        "Scenario mat 60": "darkgreen"
    }
    
    # Scenario legend
    scenario_legend = {
        "Scenario R3": "Baseline",
        "Scenario mat you": "Mature and Immature Forest",
        "Scenario mat 60": "Mature Forest"
    }
    
    # Scenario groups
    scenario_groups = {
        "Mature, Mature-Immature, Base": ["scenario_mat_you_data.csv", "scenario_mat_60_data.csv", "scenario_R3_data.csv"]
    }
    
    # Load and display the image
    # Set the data folder relative to the script's location
    data_folder = Path(__file__).parent / 'data'
    image_path = data_folder / 'mature.jpg'
    st.image(image_path)
    
    # Add the paragraph
    st.markdown("""
    We compared current conditions in the baseline model to two other land use scenarios where we changed the distribution of forest age, since the age of trees can impact the water cycle in multiple ways. This watershed drawing shows how increasing the percentage of mature forest affects different parts of the water cycle. Mature forests increased the rates of evapotranspiration and shallow flow below the ground, suggesting the forest age distributions impact various hydrologic processes.
    """)

    # Define tick values (start of each month approx)
    tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ticktext = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y_ticks = [0.05, 0.18, 1, 10, 50]
    
    # Process each scenario group
    for title, files in scenario_groups.items():
        scenario_data = []
    
        for file in files:
            file_path = os.path.join("data/reach_csv", file)
    
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Failed to load file: {file_path}\nError: {e}")
                st.stop()
    
            # Capitalize column names just in case
            df.columns = [col.upper() for col in df.columns]
    
            # Add a Scenario column from file name if not in the file
            if "SCENARIO" not in df.columns:
                scenario_name = os.path.splitext(file)[0].replace("scenario_", "").replace("_data", "")
                df["SCENARIO"] = f"Scenario {scenario_name.replace('_', ' ')}"
    
            scenario_data.append(df)
    
        combined_data = pd.concat(scenario_data, ignore_index=True)
    
        # Filter for Reach 3 and valid days
        rch3_data = combined_data[(combined_data["RCH"] == 3) & (combined_data["DAY"].between(1, 365))]
    
        # Mean daily flow per scenario
        mean_daily_flow = rch3_data.groupby(["SCENARIO", "DAY"])["FLOW_OUTCMS"].mean().reset_index()
    
        max_flow = mean_daily_flow["FLOW_OUTCMS"].max()
    
        # === Plot Mean Daily Flow ===
        fig2 = px.line(
            mean_daily_flow,
            x="DAY",
            y="FLOW_OUTCMS",
            color="SCENARIO",
            title=f"Mean Daily Flow for Reach 3 - {title}",
            labels={"DAY": "Day of the Year", "FLOW_OUTCMS": "Mean Flow (cms)"},
            color_discrete_map=scenario_colors
        )
    
        fig2.add_hline(
            y=0.18, line_dash="dash", line_color="red", line_width=2,
            annotation_text="Fish Protection Cutoff (0.18 cms)",
            annotation_position="right",
            annotation_y=0.18,
            annotation_font_size=12
        )
    
        # Update legend names
        fig2.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    
        fig2.update_layout(
            xaxis=dict(
                title="Day of the Year",
                showgrid=True,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            ),
            yaxis=dict(
                title="Mean Flow (cms)",
                showgrid=True,
                type="log",
                range=[np.log10(min(y_ticks)), np.log10(max(y_ticks))],
                tickvals=y_ticks
            ),
            legend=dict(title="Scenario"),
            width=800,
            height=400
        )
    
        st.plotly_chart(fig2)
    
        # === Plot Delta Flow ===
        base_scenario = mean_daily_flow[mean_daily_flow["SCENARIO"] == "Scenario R3"].rename(columns={"FLOW_OUTCMS": "Base_Flow"})
        merged_data = pd.merge(mean_daily_flow, base_scenario[["DAY", "Base_Flow"]], on="DAY", how="left")
        merged_data["Delta Flow"] = (merged_data["FLOW_OUTCMS"] - merged_data["Base_Flow"]) / merged_data["Base_Flow"]
    
        fig4 = px.line(
            merged_data,
            x="DAY",
            y="Delta Flow",
            color="SCENARIO",
            title=f"Change in Summer Streamflow - {title}",
            color_discrete_map=scenario_colors
        )
    
        fig4.for_each_trace(lambda t: t.update(name=scenario_legend.get(t.name, t.name)))
    
        fig4.update_layout(
            xaxis=dict(
                title="Day of the Year",
                showgrid=True,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext
            ),
            yaxis=dict(
                title="Delta Flow (Relative Change)",
                showgrid=True,
                range=[-1.1, 1.1]
            ),
            legend=dict(title="Scenario"),
            width=800,
            height=400
        )
    
        st.plotly_chart(fig4)

    st.markdown("""
          A mature (60-year-old) forest scenario reduced streamflow slightly by less than 10% during the summer, whereas a mix of mature and immature (30-year-old) forests did not significantly change low flows. These findings highlight the importance of considering forest age distributions when assessing long-term hydrological changes, while remembering that August streamflow and the age of the trees are an overly simplistic approach to assessing the impact of forestry on watersheds.  
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


