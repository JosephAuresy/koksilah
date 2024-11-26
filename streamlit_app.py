import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from pyproj import Proj, transform
import base64
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from streamlit_folium import folium_static
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
    page_title='Koki Dashboard',
    page_icon=':earth_americas:',
)

# Sidebar for navigation
st.sidebar.title("Xwulqw'selu Sta'lo'")
selected_option = st.sidebar.radio(
    "Select an option:",
    #("Watershed models", "Water interactions", "Recharge", "View Report")
    ("Watershed models", "Field data validation", "Groundwater / Surface water interactions", "Recharge", "Report")
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
if selected_option == "Groundwater / Surface water interactions" or selected_option == "Recharge":
    # Display Model selection part only when these options are selected
    st.sidebar.title("Model selection")
    st.sidebar.subheader("Climate")
    selected_decade_climate = st.sidebar.selectbox(
        "Choose a decade for Climate:",
        ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'],
        index=6  # Default to 2010s (index 6)
    )

    st.sidebar.subheader("Land Use")
    selected_decade_land_use = st.sidebar.selectbox(
        "Choose a decade for Land Use:",
        ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'],
        index=6  # Default to 2010s (index 6)
    )

    st.sidebar.subheader("Water Use")
    selected_decade_water_use = st.sidebar.selectbox(
        "Choose a decade for Water Use:",
        ['1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'],
        index=6  # Default to 2010s (index 6)
    )
    # Define a function to generate folder path based on selected decade
    def get_folder_path(land_use, climate, water_use):
        # Assuming data folder structure is like: 'data/L{land_use}_C{climate}_W{water_use}'
        land_use_code = land_use[-2:]  # Extract the last two digits (e.g., '2010s' -> '10')
        climate_code = climate[-2:]  # Extract the last two digits (e.g., '2010s' -> '10')
        water_use_code = water_use[-2:]         
        folder_name = f'L{land_use_code}_C{climate_code}_W{water_use_code}'
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
    
# #Path to your data file
# DATA_FILENAME = Path(__file__).parent / 'data/swatmf_out_MF_gwsw_monthly.csv'
# df = process_swatmf_data(DATA_FILENAME)

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
    custom_title("Watershed models for Xwulqw'selu Sta'lo'", 28)
    
    st.markdown("""
    [Xwulqw’selu Connections](https://onlineacademiccommunity.uvic.ca/xwulqwselu/) research project brings people together to learn about the conditions affecting stream flows in the Xwulqw’selu Watershed, where many are concerned about summer low flows and winter floods.
    
    We developed watershed models with the best available data that complement the valuable field data collected by monitors and previous reports. These models give us more understanding from places and times that don't have field data.
    
    Watershed models use the **SWAT-MODFLOW** model, an internationally recognized standard for analyzing the interactions between groundwater, surface water, climate, land use, and water use. This model provides valuable insights into the hydrological dynamics of the watershed and is calibrated to the best available data from 2013 to 2022.
    
    You can explore interactive maps showing how groundwater and surface water are connected, or view **groundwater recharge** across the watershed. Soon, we’ll add models from other decades in the past to expand our understanding.
    """)
        
    # Set the data folder using Path
    data_folder = Path(__file__).parent / 'data'
    
    # Define the updated image files and their corresponding captions or messages
    image_files = [
        '1_physical_model.jpg', 
        '2_low_flow_river.jpg', 
        '3_monitoring_data.jpg', 
        '4_meeting_ideas.jpg'
    ]
    
    captions = [
        "How does water emerge from the underground? This web application explains the interactions between groundwater and surface water, similar to the physical model shown in the picture, to help users understand the results from the watershed models.",
        
        "A creek with abundant flow supports a healthy habitat, but pressures from low flows drivers such as climate change, land use, and water use must be analyzed to understand their impact. These factors are examined through watershed models.",
        
        "Accurate data from field monitoring activities plays a pivotal role in capturing real-time watershed conditions. The success of these monitoring efforts enhances our ability to track changes over time, offering valuable insights into watershed dynamics and informing evidence-based management strategies.",
        
        "The exchange of diverse ideas during meetings drives the development of innovative projects that aim to improve the health of the watershed. Collaborative efforts such as these contribute to creating long-term solutions that promote sustainability and resilience in watershed management."
    ]
    
    # Set up the layout for small images (small panel)
    st.title("Significance of the Models")
    st.write("Select the description and scroll down")
    
    # Create columns for displaying small images
    cols = st.columns(4)  # 4 images, each in its own column
    selected_image = None  # To store which image is selected
    selected_caption = ""  # To store the selected caption
    
    # Display the small images
    for i, image_file in enumerate(image_files):
        with cols[i]:
            # Load each image
            image_path = data_folder / image_file
            
            try:
                image = Image.open(image_path)
                # Use a button to trigger the larger view instead of the image click
                if st.button(f"Show Image {i+1}", key=f"button_{i}"):
                    selected_image = image
                    selected_caption = captions[i]
                    # Use a success message when the button is clicked
                    st.success(f"Showing: {image_file}")
                st.image(image, width=100)  # Display the small image without a caption
                
            except FileNotFoundError:
                st.error(f"Image file {image_file} not found in {data_folder}.")
            
    # Display the selected image in a larger panel (if selected)
    if selected_image:
        st.image(selected_image, caption="", use_column_width=True)  # No caption below the large image
        st.write(selected_caption)  # Show the caption only after the image is clicked
    else:
        st.write("Click on an image to see a larger view and explanation.")


elif selected_option == "Field data validation":
    
    # Define paths to the main data file and the points file
    main_path = Path(__file__).parent
    DATA_FILENAME = main_path / 'data/swatmf_out_MF_gwsw_monthly.csv'
    points_file_path = main_path / 'data/points_info.csv'
    
    # Define the sites to color in purple
    purple_sites = [
        "Glenora_DoupeRd", "Glenora_MarshallRd", "Kelvin_MountainRd_5",
        "Mainstem_KoksilahRiverPark", "Neel_ShawRd_2", "Patrolas_HillbankRd",
        "WildDeer_WildDeerMain4.8", "WK_East_RenfrewMain1.0"
    ]
    
    # Streamlit app
    st.title("Flow Rate Analysis for August")
    
    # Check if data files exist before proceeding
    if DATA_FILENAME.exists() and points_file_path.exists():
        # Process the main data file and filter for August
        df = process_swatmf_data(DATA_FILENAME)
        august_data = df[df['Month'] == 8]  # Filter for August only
    
        # Read points CSV file
        points_df = pd.read_csv(points_file_path)
    
        # Merge to get data only for the points in points_info.csv
        filtered_data = august_data.merge(points_df, left_on=['Row', 'Column'], right_on=['ROW', 'COLUMN'], how='inner')
    
        # Separate sites into three groups: all negative, all positive, and mixed values
        all_negative_sites = []
        all_positive_sites = []
        mixed_sites = []
    
        # Determine which sites belong to each group
        for site in filtered_data['name'].unique():
            site_data = filtered_data[filtered_data['name'] == site]
            if (site_data['Rate'] <= 0).all():  # All values are negative
                all_negative_sites.append(site)
            elif (site_data['Rate'] > 0).all():  # All values are positive
                all_positive_sites.append(site)
            else:  # Mixed values
                mixed_sites.append(site)
    
        # Function to create box plots
        def create_box_plot(sites, title, color_map):
            fig = go.Figure()
            for site in sites:
                site_data = filtered_data[filtered_data['name'] == site]
                color = color_map.get(site, 'gray')
                
                fig.add_trace(go.Box(
                    y=site_data['Rate'],
                    name=site,
                    marker_color=color,
                    line=dict(width=2),
                    boxmean='sd',
                    marker=dict(outliercolor='red'),
                    showlegend=False
                ))
            fig.update_layout(
                title=title,
                yaxis_title="Flow Rate (cms)",
                boxmode='group',
                height=600,
                plot_bgcolor='white',
                yaxis=dict(gridcolor='LightGray')
            )
            return fig
    
        # Create color maps for each group
        negative_color_map = {site: 'purple' if site in purple_sites else 'blue' for site in all_negative_sites}
        mixed_color_map = {site: 'purple' if site in purple_sites else 'lightblue' for site in mixed_sites}
        positive_color_map = {site: 'purple' if site in purple_sites else 'brown' for site in all_positive_sites}
    
        # Display the plots in the Streamlit app
        st.subheader("Consistently Gaining: Box Plot of August Flow Rates (All Negative Sites)")
        st.plotly_chart(create_box_plot(all_negative_sites, "Consistently Gaining", negative_color_map))
    
        st.subheader("Transition Places: Box Plot of August Flow Rates (Mixed Value Sites)")
        st.plotly_chart(create_box_plot(mixed_sites, "Transition Places", mixed_color_map))
    
        st.subheader("Consistently Losing: Box Plot of August Flow Rates (All Positive Sites)")
        st.plotly_chart(create_box_plot(all_positive_sites, "Consistently Losing", positive_color_map))
    
    else:
        st.error("Required files not found. Please ensure 'swatmf_out_MF_gwsw_monthly.csv' and 'points_info.csv' are in the working directory.")
        
    # Load your data from the CSV file
    csv_file_path = 'data/Simulated_vs_Observed_Flow_Year10_Months6_9.csv'  # Path to your CSV file
    merged_data = pd.read_csv(csv_file_path)
    
    # Create an interactive scatter plot with log scale and limited axis range, coloring points by 'Site'
    fig = px.scatter(
        merged_data, 
        x='Measured_Flow', 
        y='Simulated_Flow', 
        color='Site',  # Color points by site
        labels={'Measured_Flow': 'Measured Flow (cms)', 'Simulated_Flow': 'Simulated Flow (cms)', 'Site': 'Site'},
        title='Simulated vs. Measured Flow by Site (Log Scale, Limited Range)',
        opacity=0.6
    )
    
    # Add a 45-degree reference line
    fig.add_shape(
        type="line",
        x0=0.00001, y0=0.00001, x1=2, y1=2,  # Start from a small positive value to fit the log scale
        line=dict(color="Red", dash="dash")
    )
    
    # Set log scale and limit the range of the plot to 2
    fig.update_layout(
        xaxis=dict(title="Measured Flow (cms)", type="log", range=[np.log10(0.00001), np.log10(2)]),  # Log scale starting at 0.00001
        yaxis=dict(title="Simulated Flow (cms)", type="log", range=[np.log10(0.00001), np.log10(2)]),  # Log scale starting at 0.00001
        autosize=False,
        width=700,
        height=700
    )
    
    # Streamlit app layout
    st.title("Flow Simulation Analysis")
    st.write("This application displays the simulated vs. measured flow data colored by site.")
    
    # Display the plot in the Streamlit app
    st.plotly_chart(fig)
        
elif selected_option == "Groundwater / Surface water interactions":
    custom_title("How groundwater and surface water interact in the Xwulqw’selu watershed?", 28)

    st.markdown("""
    In the Xwulqw’selu Watershed, groundwater plays a key role in sustaining streamflow during low-flow periods, particularly in summer. As surface water levels drop, groundwater discharge becomes the primary source of flow, helping maintain aquatic habitats and water availability. 
    
    Land use changes, and climate shifts can reduce groundwater recharge, worsening low-flow conditions. Understanding this groundwater-surface water interaction is critical for managing water resources and mitigating the impacts of prolonged droughts.
    
    Below is a map of the average monthly groundwater / surface water interactions across the watershed. You can change which month you want to look at or zoom into different parts of the watershed for a closer examination of recharge patterns.
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
        # Get the previous month value (December if January)
        prev_month_value = row_vals[selected_month - 2] if selected_month > 1 else row_vals[-1]

        # Classify based on the value ranges and changes between months:
        def classify_based_on_value_range(value):
            if value < -50:
                return 0  # Dark Blue (strong negative)
            elif -50 <= value < -20:
                return 1  # Light Blue (moderate negative)
            elif -20 <= value < -10:
                return 2  # Cyan (negative but closer to zero)
            elif -10 <= value < -5:
                return 3  # Light Blue (mild negative)
            elif -5 <= value < -1:  # Adjusted range for mild negative
                return 4  # Yellow (near-zero)
            elif -1 <= value <= 1:  # New range for values between -1 and 1
                return 5  # Light Yellow (near-zero, including positive and negative fluctuations)
            elif 1 < value <= 5:
                return 6  # Brown (positive, to aquifer)
            elif value > 5:
                return 7  # Dark Red (strong positive)
        
        # Classify the current value and assign to the grid
        grid[row_idx, col_idx] = classify_based_on_value_range(value)
        # Add hover text for the grid cell
        hover_text[row_idx, col_idx] = f'Value: {value:.2f} (Prev: {prev_month_value:.2f})'


        # # Classify as positive or negative
        # if value < 0:
        #     grid[row_idx, col_idx] = 0  # Negative
        # elif value > 0:
        #     grid[row_idx, col_idx] = 1  # Positive
    
        # # Classify based on the value ranges and changes between months:
        # if value < -50:
        #     grid[row_idx, col_idx] = 0  # Dark Blue (strong negative)
        # elif -50 <= value < -10:
        #     grid[row_idx, col_idx] = 1  # Light Blue (moderate negative)
        # elif -10 <= value <= 1:
        #     grid[row_idx, col_idx] = 2  # Yellow (near-zero)
        # elif value > 1:
        #     grid[row_idx, col_idx] = 3  # Brown (positive, to aquifer)
    
        # # Check for changes in sign between months and assign green colors
        # if prev_month_value > 0 and value < 0:
        #     grid[row_idx, col_idx] = 4  # Shiny Green (positive to negative)
        # elif prev_month_value < 0 and value > 0:
        #     grid[row_idx, col_idx] = 5  # Green (negative to positive)
    
        # Create hover text for the real values
        # hover_text[row_idx, col_idx] = f"Row: {row['Row']}, Column: {row['Column']}, Value: {value:.2f}"

    # # Set the title of the Streamlit app
    # st.title("Hotspot Data Processing")
    
    # # Path to your data file
    # DATA_FILENAME = Path(__file__).parent / 'data/swatmf_out_MF_gwsw_monthly.csv'
    
    # # Load hotspot data
    # hotspots_df = pd.read_csv(DATA_FILENAME)
    
    # # Define grid origin (top-left corner in meters) and cell size
    # origin_x, origin_y = 428379.32689473, 5401283.09659084  # top-left corner coordinates in meters
    # cell_size = 300  # cell size in meters
    # num_cols = 94  # number of columns
    # num_rows = 68  # number of rows
    
    # # Define the projection systems
    # latlng_proj = Proj(init='EPSG:4326')  # WGS84 for latitude/longitude
    # meters_proj = Proj(init='EPSG:32610')  # Replace with the appropriate projected CRS
    
    # # Create a transformer
    # transformer = Transformer.from_proj(latlng_proj, meters_proj)
    
    # # Function to convert lat/lng to x/y coordinates in meters
    # def convert_latlng_to_meters(lat, lng):
    #     x, y = transformer.transform(lat, lng)  # Note the order: (lat, lng)
    #     return x, y
    
    # # Convert lat/lng coordinates to x/y in meters
    # hotspots_df['x'], hotspots_df['y'] = zip(*hotspots_df.apply(lambda row: convert_latlng_to_meters(row['lat'], row['lng']), axis=1))
    
    # # Calculate grid cell centers
    # grid_centers = []
    # for row in range(num_rows):
    #     for col in range(num_cols):
    #         center_x = origin_x + (col * cell_size) + (cell_size / 2)
    #         center_y = origin_y - (row * cell_size) - (cell_size / 2)  # y decreases as you go down
    #         grid_centers.append((center_x, center_y))
    
    # # Create a DataFrame for grid centers
    # grid_centers_df = pd.DataFrame(grid_centers, columns=['grid_x', 'grid_y'])
    # grid_centers_df['row'] = grid_centers_df.index // num_cols
    # grid_centers_df['col'] = grid_centers_df.index % num_cols
    
    # # Determine the position of hotspots relative to the grid
    # def find_grid_cell(hotspot_x, hotspot_y):
    #     for _, row in grid_centers_df.iterrows():
    #         if (row['grid_x'] - (cell_size / 2) <= hotspot_x <= row['grid_x'] + (cell_size / 2) and
    #             row['grid_y'] - (cell_size / 2) <= hotspot_y <= row['grid_y'] + (cell_size / 2)):
    #             return row['row'], row['col']
    #     return None, None
    
    # # Find grid positions for each hotspot
    # hotspots_df['grid_row'], hotspots_df['grid_col'] = zip(*hotspots_df.apply(lambda row: find_grid_cell(row['x'], row['y']), axis=1))
    
    # # Streamlit display
    # st.title('Hotspot Coordinates Transformation and Grid Positions')
    # st.write('Hotspot Data with Grid Positions:')
    # st.dataframe(hotspots_df[['id', 'name', 'grid_row', 'grid_col']])

    def create_heatmap(classified_grid, selected_month_name, hover_text):
        # Define a color scale for the classified ranges
        colorscale = [
            [0.0, 'darkblue'],   # Less than -50
            [0.14, 'blue'],      # Between -50 and -20
            [0.28, 'cyan'],      # Between -20 and -10
            [0.42, 'lightblue'], # Between -10 and -5
            [0.57, 'yellow'],    # Between -5 and -1
            [0.71, 'orange'], # Between -1 and 1 (new range, light yellow)
            [0.85, 'brown'],     # Between 1 and 5
            [1.0, 'purple']     # Higher positive > 5
        ]
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=classified_grid,
            colorscale=colorscale,
            zmin=0,
            zmax=7,
            showscale=False,  # Hide scale since categories are defined
            hoverinfo='text',
            text=hover_text
        ))
        
        # Update the layout of the heatmap
        fig.update_layout(
            title=f'Groundwater-Surface Water Interaction for {selected_month_name}',
            xaxis_title='Column',
            yaxis_title='Row',
            xaxis=dict(showticklabels=False, ticks='', showgrid=False),
            yaxis=dict(showticklabels=False, ticks='', autorange='reversed', showgrid=False),
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=8, color='black')
        )
        
        # Display the heatmap
        st.plotly_chart(fig)
        
    # Function to count cells per classification
    def count_cells_per_color(grid):
        color_counts = {
            'strong_negative': np.sum(grid == 0),  # Dark Blue (strong negative, < -50)
            'moderate_negative': np.sum(grid == 1),  # Light Blue (moderate negative, -50 to -20)
            'negative_closer_to_zero': np.sum(grid == 2),  # Cyan (negative but closer to zero, -20 to -10)
            'mild_negative': np.sum(grid == 3),  # Light Blue (mild negative, -10 to -5)
            'near_zero': np.sum(grid == 4),  # Yellow (near-zero fluctuation, -5 to 1)
            'light_positive': np.sum(grid == 5),  # Light Yellow (positive, -1 to 1)
            'positive': np.sum(grid == 6),  # Brown (positive, to aquifer, 1 to 5)
            'strong_positive': np.sum(grid == 7),  # Dark Red (strong positive, > 5)
        }
        return color_counts
        
    # Count the colors for the selected month
    color_counts = count_cells_per_color(grid)
    
    # Prepare data for pie chart
    color_names = [
        'Less than -50',               # Dark Blue (strong negative)
        'Between -50 and -20',         # Light Blue (moderate negative)
        'Between -20 and -10',         # Cyan (negative but closer to zero)
        'Between -10 and -5',          # Light Blue (mild negative)
        'Between -5 and -1',           # Yellow (near-zero fluctuation)
        'Between -1 and 1',            # Orange (light positive)
        'Between 1 and 5',             # Brown (positive, to aquifer)
        'Higher positive > 5'          # Dark Red (strong positive)
    ]
    color_values = [
        color_counts['strong_negative'], color_counts['moderate_negative'], color_counts['negative_closer_to_zero'],
        color_counts['mild_negative'], color_counts['near_zero'], color_counts['light_positive'],
        color_counts['positive'], color_counts['strong_positive']
    ]
     
    total_cells = sum(color_values)
    
    # Avoid division by zero
    percentages = [count / total_cells * 100 if total_cells > 0 else 0 for count in color_values]
    
    # Create a pie chart with formatted percentages
    pie_colors = [
        '#00008B',  # Dark Blue (strong negative)
        '#ADD8E6',  # Light Blue (moderate negative)
        '#00FFFF',  # Cyan (negative but closer to zero)
        '#E0FFFF',  # Light Cyan (mild negative)
        '#FFFF00',  # Yellow (near-zero fluctuation)
        '#FFA500',  # Orange (light positive)
        '#8B4513',  # Brown (positive, to aquifer)
        '#800080'   # Purple (strong positive)
    ]

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=color_names, values=percentages, hole=.3, marker=dict(colors=pie_colors), textinfo='percent')])
    
    # Update pie chart layout with formatted percentages
    # fig.update_traces(texttemplate='%{label}: %{percent:.2f}%', textfont_size=14)  # Display both label and percentage
    # fig.update_layout(title_text='Percentage of Each Color by Month', annotations=[dict(text='Percentage', font_size=20, showarrow=False)])
    
    # Display pie chart
    st.plotly_chart(fig)
    
    # Create the heatmap and pass in the grid and hover text
    create_heatmap(grid, selected_month_name, hover_text)

    # Define the main path and image path
    main_path = Path(__file__).parent
    ground = main_path / 'data/riv_groundwater.png'
    
    # Check if the image exists before displaying
    if ground.is_file():
        try:
            # Try to open the image using PIL
            image = Image.open(ground)
            st.image(image, caption='Groundwater and River Interaction', use_column_width=True)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
    else:
        st.warning("Image 'riv_groundwater.png' not found in the data folder.")
    
    # Filter data for the selected month
    selected_month_data = monthly_stats[monthly_stats['Month'] == selected_month]

    # Definitions and Formulas Section
    st.header('Spatial Analysis Definitions and Formulas')
    
    # Hotspot Analysis Definition
    st.subheader('Hotspot Analysis')
    st.write("""
    This analysis identifies significant hotspots in the dataset based on z-scores. 
    A hotspot is defined as a location where the 'Rate' is significantly higher or lower than the mean value of the dataset.
    """)
    st.write("**Formula:**")
    st.latex(r'z = \frac{(X - \mu)}{\sigma}')
    st.write("""
    Where:
    - \(X\) = individual observation
    - \(\mu\) = mean of the dataset
    - \(\sigma\) = standard deviation of the dataset
    """)
    
    # Change Detection Definition
    st.subheader('Change Detection')
    st.write("""
    This analysis identifies changes in the 'Rate' values over time. 
    It calculates the difference in the 'Rate' between consecutive entries to detect any significant changes.
    """)
    st.write("**Formula:**")
    st.write("The change \(C\) is calculated as:")
    st.latex(r'C = R_t - R_{t-1}')
    st.write("""
    Where:
    - \(R_t\) = Rate at time \(t\)
    - \(R_{t-1}\) = Rate at the previous time step
    """)
    
    # Hotspot Analysis Function
    def hotspot_analysis(data):
        """
        This function identifies significant hotspots in the dataset based on z-scores.
        A hotspot is defined as a location where the 'Rate' is significantly higher or lower 
        than the mean value of the dataset, determined by z-scores.
        """
        # Compute the z-scores of the Rate values
        data['z_score'] = zscore(data['Rate'])
        # Identify significant clusters based on z-scores
        significant_hotspots = data[np.abs(data['z_score']) > 1.96]  # 95% confidence interval
        return significant_hotspots

    # Change Detection Function
    def change_detection(data):
        """
        This function identifies changes in the 'Rate' values over time.
        It calculates the difference in the 'Rate' between consecutive entries 
        to detect any significant changes.
        """
        # Calculate the change in the 'Rate' column compared to the previous entry
        data['change'] = data['Rate'].diff().fillna(0)  # Fill NA values with 0
        # Filter the dataset to only include rows where a change is detected
        changes = data[data['change'] != 0]
        return changes
    
    # Hotspot Analysis Button
    if st.button('Run Hotspot Analysis'):
        hotspots = hotspot_analysis(selected_month_data)  # Call the hotspot analysis function
        st.write('Hotspots Identified:', hotspots)  # Display identified hotspots
    
    # Change Detection Button
    if st.button('Run Change Detection'):
        changes = change_detection(selected_month_data)  # Call the change detection function
        st.write('Changes Detected:', changes[['Row', 'Column', 'change']])  # Display detected changes
    
    # Plotting function for histogram
    def plot_histogram(data, selected_month_name):
        """
        Plots a histogram of the 'Rate' values for the selected month.
        """
        mean_rate = data['Rate'].mean()  # Calculate the mean
        std_rate = data['Rate'].std()   # Calculate the standard deviation
        
        # Plot histogram
        plt.figure(figsize=(10, 6))  # Set figure size
        plt.hist(data['Rate'], bins=30, color='blue', alpha=0.7)  # Create histogram
        plt.axvline(mean_rate, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_rate:.2f}')  # Add mean line
        plt.text(mean_rate, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_rate:.2f}', color='red', fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.6))   
        plt.axvline(mean_rate + 1.96 * std_rate, color='green', linestyle='dashed', linewidth=1, label='95% CI Upper')  # Upper CI
        plt.axvline(mean_rate - 1.96 * std_rate, color='green', linestyle='dashed', linewidth=1, label='95% CI Lower')  # Lower CI
        plt.title(f'Histogram of Rate Values for {selected_month_name}')  # Title for the histogram
        plt.xlabel('Rate')  # X-axis label
        plt.ylabel('Frequency')  # Y-axis label
        plt.legend()  # Show legend
        st.pyplot(plt)  # Render the plot within the Streamlit app
        plt.clf()  # Clear the current figure to avoid overlapping plots
    
    # Button to plot histogram
    if st.button('Plot Histogram'):
        plot_histogram(selected_month_data, selected_month_name)  # Call the histogram plotting function
    
    # Step 4: Summarize data into a table to compare histograms across all months
    def summarize_histograms(df):
        """
        Summarizes the histograms data across months.
        """
        summary = df.groupby('Month')['Rate'].agg(['mean', 'std', 'min', 'max', 'count'])
        summary = summary.reset_index()
        summary['Month'] = summary['Month'].apply(lambda x: month_names[x - 1])  # Convert month number to name
        return summary
    
    # Display summary table
    summary_table = summarize_histograms(monthly_stats)
    st.write("Summary of Histograms Data across Months:")
    st.dataframe(summary_table)
   
        
elif selected_option == "Recharge":
    custom_title("How much groundwater recharge is there in the Xwulqw’selu watershed?", 28)

    st.markdown("""
    In the SWAT-MODFLOW model, recharge is how groundwater is replenished from  precipitation, surface runoff, and other sources. Understanding recharge is crucial for effective water resource management, as it helps quantify groundwater availability and assess the impacts of land use changes and climate variability on water sustainability in a watershed.

    Below is a map of the average monthly recharge across the watershed. You can change which month you want to look at or zoom into different parts of the watershed...         
    
    """)
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


    # # Initialize the map centered on Duncan
    # m = folium.Map(location=initial_location, zoom_start=11, control_scale=True)
    
    # # Add the subbasins layer to the map with tooltips for subbasin numbers
    # subbasins_layer = folium.GeoJson(
    #     subbasins_gdf,
    #     name="Subbasins",
    #     style_function=lambda x: {'color': 'green', 'weight': 2},
    #     tooltip=folium.GeoJsonTooltip(fields=['subbasin_number'],  # Change 'subbasin_number' to your actual field name
    #                                    aliases=['Subbasin: '],
    #                                    localize=True)
    # ).add_to(m)
    
    # # Add the grid layer to the map
    # grid_layer = folium.GeoJson(
    #     grid_gdf,
    #     name="Grid",
    #     style_function=lambda x: {'color': 'blue', 'weight': 1},
    #     show=False
    # ).add_to(m)
    
    # # Add MousePosition to display coordinates
    # from folium.plugins import MousePosition
    # MousePosition().add_to(m)
    
    # # Add a layer control to switch between the subbasins and grid layers
    # folium.LayerControl().add_to(m)
    
    # # Render the Folium map in Streamlit
    # st.title("Watershed Map")
    # st_folium(m, width=700, height=600)
    
    # # Optionally, add additional functionality to display data for the selected subbasin
    # selected_subbasin = st.sidebar.text_input("Enter Subbasin Number:")
    # if selected_subbasin:
    #     try:
    #         selected_data = subbasins_gdf[subbasins_gdf['subbasin_number'] == selected_subbasin]  # Adjust field name
    #         st.write(selected_data)
    #     except KeyError:
    #         st.error("Subbasin number not found.")

    # # Parsing data from file
    # def parse_data(file_path):
    #     try:
    #         with open(file_path, 'r') as file:
    #             lines = file.readlines()
    #     except FileNotFoundError:
    #         st.error(f"File not found: {file_path}")
    #         return {}, {}
    
    #     # Initialize dictionaries
    #     watershed_data = {'landuse': {}, 'soil': {}, 'slope': {}}
    #     subbasin_data = {}
    #     current_subbasin = None
    
    #     for line in lines:
    #         line = line.strip()  # Remove leading/trailing spaces
    #         if not line:  # Skip empty lines
    #             continue
    
    #         if "Watershed" in line:
    #             # Process watershed level data
    #             parts = line.split()
    #             if parts[-1].replace('.', '', 1).isdigit():  # Ensure the last part is numeric
    #                 try:
    #                     watershed_data['area'] = float(parts[-1])  # Convert area to float
    #                 except ValueError:
    #                     st.warning(f"Skipping non-numeric watershed area: {line}")
    
    #         elif "Subbasin" in line:
    #             # Initialize a new dictionary for each subbasin's data
    #             current_subbasin = line.strip()
    #             subbasin_data[current_subbasin] = {'landuse': {}, 'soil': {}, 'slope': {}}
    
    #         # Process information for current subbasin
    #         elif current_subbasin:
    #             process_subbasin_data(line, subbasin_data[current_subbasin], watershed_data)
    
    #     return watershed_data, subbasin_data
        
    #     def read_subbasin_data(file_path):
    #     subbasin_data = {}
    #     current_subbasin = None
        
    #     with open(file_path, 'r') as file:
    #         lines = file.readlines()
            
    #     i = 0
    #     while i < len(lines):
    #         line = lines[i].strip()
            
    #         # Detect a new subbasin
    #         if line.startswith("Subbasin"):
    #             current_subbasin = line.split()[1]
    #             subbasin_data[current_subbasin] = {"landuse": {}, "soil": {}, "slope": {}}
    #             i += 1  # Skip the header line
    #             continue
            
    #         # Detect land use section
    #         if line.startswith("Landuse"):
    #             i += 1
    #             while lines[i].strip() and not lines[i].startswith("Soil"):
    #                 parts = lines[i].strip().split()
    #                 landuse_type = parts[0]
    #                 area = float(parts[1])
    #                 percent_subbasin = float(parts[3])
    #                 subbasin_data[current_subbasin]["landuse"][landuse_type] = {
    #                     "area": area, 
    #                     "percent_subbasin": percent_subbasin
    #                 }
    #                 i += 1
            
    #         # Detect soil section
    #         if line.startswith("Soil"):
    #             i += 1
    #             while lines[i].strip() and not lines[i].startswith("Slope"):
    #                 parts = lines[i].strip().split()
    #                 soil_type = parts[0]
    #                 area = float(parts[1])
    #                 percent_subbasin = float(parts[3])
    #                 subbasin_data[current_subbasin]["soil"][soil_type] = {
    #                     "area": area, 
    #                     "percent_subbasin": percent_subbasin
    #                 }
    #                 i += 1
            
    #         # Detect slope section
    #         if line.startswith("Slope"):
    #             i += 1
    #             while i < len(lines) and lines[i].strip() and not lines[i].startswith("Subbasin"):
    #                 parts = lines[i].strip().split()
    #                 slope_range = parts[0]
    #                 area = float(parts[1])
    #                 percent_subbasin = float(parts[3])
    #                 subbasin_data[current_subbasin]["slope"][slope_range] = {
    #                     "area": area, 
    #                     "percent_subbasin": percent_subbasin
    #                 }
    #                 i += 1
            
    #         i += 1  # Move to the next line
    
    #     return subbasin_data

    
    # # Load and parse data
    # file_path = Path(__file__).parent / 'data/LanduseSoilSlopeRepSwat.txt'
    # watershed_data, subbasin_data = parse_data(file_path)
    
    # # Convert data into DataFrames
    # def create_dataframe(data_dict):
    #     df_list = []
    #     for subbasin, details in data_dict.items():
    #         data = {
    #             'Subbasin': subbasin,
    #             'Landuse': details['landuse'],
    #             'Soil': details['soil'],
    #             'Slope': details['slope']
    #         }
    #         df_list.append(data)
    #     return pd.DataFrame(df_list)
    
    # df_subbasins = create_dataframe(subbasin_data)
    
    # # Streamlit App Layout
    # st.title("Watershed and Subbasin Analysis")
    
    # # Watershed Level Overview
    # st.subheader("Watershed Overview")
    # st.write(f"Total Area: {watershed_data.get('area', 'N/A')} ha")
    # st.write("Land Use Distribution:")
    # landuse_df = pd.DataFrame.from_dict(watershed_data['landuse'], orient='index', columns=['Area [ha]'])
    # landuse_df.index.name = 'Land Use Type'
    # st.write(landuse_df)
    
    # # Land Use Visualization
    # st.subheader("Land Use Distribution")
    # fig1 = px.pie(landuse_df, values='Area [ha]', names=landuse_df.index, title="Land Use Distribution")
    # st.plotly_chart(fig1)
    
    # # Slope Distribution
    # st.subheader("Slope Distribution in Watershed")
    # slope_df = pd.DataFrame.from_dict(watershed_data['slope'], orient='index', columns=['Area [ha]'])
    # slope_df.index.name = 'Slope Category'
    # st.write(slope_df)
    # fig2 = px.bar(slope_df, x=slope_df.index, y='Area [ha]', title="Slope Distribution")
    # st.plotly_chart(fig2)
    
    # # Soil Distribution
    # st.subheader("Soil Distribution in Watershed")
    # soil_df = pd.DataFrame.from_dict(watershed_data['soil'], orient='index', columns=['Area [ha]'])
    # soil_df.index.name = 'Soil Type'
    # st.write(soil_df)
    # fig3 = px.bar(soil_df, x=soil_df.index, y='Area [ha]', title="Soil Distribution")
    # st.plotly_chart(fig3)
    
    # # Subbasin Level Analysis
    # st.subheader("Subbasin Analysis")
    # subbasin_selection = st.selectbox('Select Subbasin:', df_subbasins['Subbasin'].unique())
    
    # selected_subbasin = df_subbasins[df_subbasins['Subbasin'] == subbasin_selection]
    
    # # Display selected subbasin details
    # st.write(f"Details for {subbasin_selection}:")
    # st.write(selected_subbasin)
    
    # # Visualize land use, soil, and slope for the selected subbasin
    # if not selected_subbasin.empty:
    #     landuse_df_subbasin = pd.DataFrame.from_dict(selected_subbasin['Landuse'].values[0], orient='index', columns=['Area [ha]'])
    #     soil_df_subbasin = pd.DataFrame.from_dict(selected_subbasin['Soil'].values[0], orient='index', columns=['Area [ha]'])
    #     slope_df_subbasin = pd.DataFrame.from_dict(selected_subbasin['Slope'].values[0], orient='index', columns=['Area [ha]'])
    
    #     # Land Use Visualization for Selected Subbasin
    #     st.subheader(f"Land Use Distribution for {subbasin_selection}")
    #     st.write(landuse_df_subbasin)
    #     fig4 = px.pie(landuse_df_subbasin, values='Area [ha]', names=landuse_df_subbasin.index, title="Land Use Distribution in Selected Subbasin")
    #     st.plotly_chart(fig4)
    
    #     # Slope Visualization for Selected Subbasin
    #     st.subheader(f"Slope Distribution for {subbasin_selection}")
    #     st.write(slope_df_subbasin)
    #     fig5 = px.bar(slope_df_subbasin, x=slope_df_subbasin.index, y='Area [ha]', title="Slope Distribution in Selected Subbasin")
    #     st.plotly_chart(fig5)
    
    #     # Soil Visualization for Selected Subbasin
    #     st.subheader(f"Soil Distribution for {subbasin_selection}")
    #     st.write(soil_df_subbasin)
    #     fig6 = px.bar(soil_df_subbasin, x=soil_df_subbasin.index, y='Area [ha]', title="Soil Distribution in Selected Subbasin")
    #     st.plotly_chart(fig6)

    
# elif selected_option == "Forest hydrology":
    
#     # Title of the app
#     st.title("Forest Hydrology and Management")
    
#     # Introduction
#     st.markdown("""
#     ### Introduction
#     Explore the impact of different forest management practices on hydrology, focusing on **Douglas Fir** and **Red Cedar**. 
#     Analyze how various parameters influence runoff, recharge, discharge, and evapotranspiration in forested areas.
    
#     **Key Equations**:
#     """)
    
#     # Evapotranspiration formula (Penman-Monteith)
#     st.write("**Evapotranspiration (Penman-Monteith):**")
#     st.latex(r'ET = \frac{ \Delta (R_n - G) + \rho_a c_p \frac{(e_s - e_a)}{r_a} }{ \Delta + \gamma (1 + r_s/r_a) }')
#     st.write("""
#     Where:
#     - \(ET\) = Evapotranspiration (mm/day)
#     - \(\Delta\) = Slope of the saturation vapor pressure curve (kPa/°C)
#     - \(R_n\) = Net radiation (MJ/m²/day)
#     - \(G\) = Soil heat flux (MJ/m²/day)
#     - \(\rho_a\) = Density of air (kg/m³)
#     - \(c_p\) = Specific heat of air (MJ/kg/°C)
#     - \(e_s\) = Saturation vapor pressure (kPa)
#     - \(e_a\) = Actual vapor pressure (kPa)
#     - \(r_a\) = Aerodynamic resistance (s/m)
#     - \(\gamma\) = Psychrometric constant (kPa/°C)
#     - \(r_s\) = Surface resistance (s/m)
#     """)
    
#     # Runoff formula
#     st.write("**Runoff (SCS Curve Number method):**")
#     st.latex(r'\text{Runoff} = \frac{(R - 0.2 \times (1000/CN - 10))^2}{R + (0.8 \times (1000/CN - 10)}')
#     st.write("""
#     Where:
#     - \(R\) = Rainfall (mm)
#     - \(CN\) = Curve Number
#     """)
    
#     # Function to calculate LAI accumulation
#     def calculate_lai(frLAImax_i, frLAImax_i_1, LAImax, LAIi_1):
#         if LAIi_1 < LAImax:
#             delta_LAI = (frLAImax_i - frLAImax_i_1) * LAImax * (1 - np.exp(5 * (LAIi_1 - LAImax)))
#             return delta_LAI
#         else:
#             return 0  # No change if max LAI is reached
    
#     # Function to calculate adjusted LAI after maximum is reached
#     def adjusted_lai(LAImax, frPHU, frPHU_sen):
#         if frPHU > frPHU_sen:
#             return LAImax * (1 - frPHU / (1 - frPHU_sen))
#         else:
#             return LAImax
    
#     # Function to calculate actual LAI
#     def calculate_actual_lai(delta_LAI, gamma_reg):
#         return delta_LAI * gamma_reg
    
#     # Input parameters
#     st.header("Dynamic Input for Forest Parameters")
#     st.markdown("""
#     ### Parameter Selection Guidelines
#     Select the **tree species** and their **corresponding age** to dynamically adjust hydrological parameters.
#     """)
    
#     # Tree Species and Age Selection
#     species = st.selectbox("Select Tree Species", ["Douglas Fir", "Red Cedar"])
#     age = st.selectbox("Select Age of Tree (Years)", [5, 10, 20, 30, 60, 100, 200, 500])
    
#     # Define parameter sets
#     parameters = {
#         "Douglas Fir": {
#             5: (2.5, 0.5, 0.5, 3.0, 2.5),
#             10: (3.5, 0.6, 0.4, 4.0, 3.5),
#             20: (4.5, 0.8, 0.2, 5.5, 4.5),
#             30: (5.0, 0.9, 0.1, 6.0, 5.0),
#             60: (6.0, 0.95, 0.05, 7.0, 6.0),
#             100: (7.0, 0.95, 0.05, 8.0, 7.0),
#             200: (8.0, 0.95, 0.05, 9.0, 8.0),
#             500: (9.0, 0.95, 0.05, 10.0, 9.0),
#         },
#         "Red Cedar": {
#             5: (2.0, 0.4, 0.6, 2.5, 2.0),
#             10: (3.0, 0.5, 0.5, 3.5, 3.0),
#             20: (4.0, 0.7, 0.3, 4.5, 4.0),
#             30: (5.0, 0.8, 0.2, 5.5, 5.0),
#             60: (6.0, 0.9, 0.1, 6.5, 6.0),
#             100: (7.0, 0.9, 0.1, 7.5, 7.0),
#             200: (8.0, 0.9, 0.1, 8.5, 8.0),
#             500: (9.0, 0.9, 0.1, 9.5, 9.0),
#         },
#     }
    
#     # Fetch parameters based on user selection
#     BLAI, FRGRW1, FRGRW2, LAIMX1, LAIMX2 = parameters[species][age]
    
#     # Display selected parameters
#     st.write(f"### Selected Parameters for {species} at Age {age} Years:")
#     st.write(f"- **BLAI (Biomass Leaf Area Index)**: {BLAI}")
#     st.write(f"- **Fraction of Growing Season (Stage 1)**: {FRGRW1}")
#     st.write(f"- **Fraction of Growing Season (Stage 2)**: {FRGRW2}")
#     st.write(f"- **Maximum LAI for Stage 1 (LAIMX1)**: {LAIMX1}")
#     st.write(f"- **Maximum LAI for Stage 2 (LAIMX2)**: {LAIMX2}")
    
#     # Input for evapotranspiration
#     et_factor = st.number_input("**Evapotranspiration Factor (mm/year)**:", value=500, step=10)
#     area = st.number_input("**HRU Area (hectares)**:", value=10, step=1)
#     et = et_factor * area  # Total ET for the selected HRU area
    
#     # Display ET result
#     st.subheader("Evapotranspiration Result")
#     st.write(f"- **Evapotranspiration (Total) (mm)**: {et:.2f} mm")
    
#     # Create Plotly visualization for ET
#     et_fig = go.Figure()
#     et_fig.add_trace(go.Indicator(
#         mode="number+gauge+delta",
#         value=et,
#         title={'text': "Evapotranspiration (Total) (mm)", 'font': {'size': 24}},
#         gauge={'axis': {'range': [0, 3000]}}
#     ))
#     st.plotly_chart(et_fig)
    
#     # Input for rainfall and area for runoff calculations
#     rainfall = st.number_input("**Rainfall (mm)**:", value=50, step=1)
    
#     # Calculate Runoff (using SCS Curve Number method)
#     def calculate_runoff(rainfall, CN):
#         if rainfall < 0:
#             return 0
#         else:
#             return (rainfall - (0.2 * (1000 / CN - 10))) ** 2 / (rainfall + (0.8 * (1000 / CN - 10)))
    
#     # SCS Curve Number for forested areas
#     CN = 70 if species == "Douglas Fir" else 75  # Adjust CN for different tree species
    
#     # Perform calculations
#     runoff = calculate_runoff(rainfall, CN) * area  # Runoff in mm for the area
#     recharge = (rainfall - runoff) * area  # Recharge in mm for the area
#     discharge = recharge  # Assume all recharge contributes to discharge
    
#     # Display HRU results
#     st.subheader("HRU Results")
#     st.write(f"- **Runoff (mm)**: {runoff:.2f} mm")
#     st.write(f"- **Recharge (mm)**: {recharge:.2f} mm")
#     st.write(f"- **Discharge (mm)**: {discharge:.2f} mm")
    
#     # Create a bar chart for HRU Results
#     hru_fig = go.Figure(data=[
#         go.Bar(name='Runoff', x=['Runoff', 'Recharge', 'Discharge'], y=[runoff, recharge, discharge])
#     ])
#     hru_fig.update_layout(barmode='group', title='HRU Results (mm)', xaxis_title='Process', yaxis_title='Value (mm)')
#     st.plotly_chart(hru_fig)
    
#     # Input parameters for LAI calculations
#     frLAImax_i = st.number_input("Enter Fraction of LAI (Current):", value=0.5)
#     frLAImax_i_1 = st.number_input("Enter Fraction of LAI (Previous):", value=0.4)
#     LAImax = st.number_input("Enter Maximum LAI:", value=6.0)
#     LAIi_1 = st.number_input("Enter Previous LAI:", value=5.0)
#     frPHU = st.number_input("Enter Fraction of Growing Season (frPHU):", value=0.6)
#     frPHU_sen = st.number_input("Enter Sensitivity for Fraction of Growing Season:", value=0.5)
    
#     # Calculate LAI
#     delta_LAI = calculate_lai(frLAImax_i, frLAImax_i_1, LAImax, LAIi_1)
#     actual_LAI = calculate_actual_lai(delta_LAI, 1.0)  # Assuming gamma_reg = 1 for simplicity
#     adjusted_LAI = adjusted_lai(LAImax, frPHU, frPHU_sen)
    
#     # Display LAI results
#     st.subheader("LAI Results")
#     st.write(f"- **Change in LAI (Delta LAI)**: {delta_LAI:.2f}")
#     st.write(f"- **Actual LAI**: {actual_LAI:.2f}")
#     st.write(f"- **Adjusted LAI**: {adjusted_LAI:.2f}")
    
#         # Display equations in LaTeX
#     st.write("### LAI Calculation Equations")
#     st.write("Before the LAI reaches its maximum value, the new LAI on day \(i\) is calculated as follows:")
#     st.latex(r'\Delta LAI_i = (frLAI_{max,i} - frLAI_{max,i-1}) \times LAI_{max} \times \{1 - e^{5 \times (LAI_{i-1} - LAI_{max})\} } \quad (1)')
    
#     st.write("The LAI does not change after reaching its maximum value. However, after leaf senescence exceeds leaf growth, the LAI is calculated as follows:")
#     st.latex(r'LAI = LAI_{max} \times \frac{1 - frPHU}{1 - frPHU_{sen}} \quad (frPHU > frPHU_{sen}) \quad (2)')
    
#     st.write("The actual LAI is affected by the stress factors. The plant growth factor—defined as the fraction of actual plant growth to potential plant growth—is used to adjust the LAI calculation on each day as follows:")
#     st.latex(r'\gamma_{reg} = 1 - \max(w_{strs}, t_{strs}, n_{strs}, p_{strs}) \quad (3)')
    
#     st.write("If one of the four stress factors exceeds 0, the LAI on day \(i\) is adjusted as follows:")
#     st.latex(r'\Delta LAI_{act,i} = \Delta LAI_i \times \gamma_{reg} \quad (4)')
    
#     st.write("Where:")
#     st.write(r'- \(\Delta LAI_i\) is the new LAI on day \(i\);')
#     st.write(r'- \(frLAI_{max,i}\) and \(frLAI_{max,i-1}\) are the maximum LAI calculated based on heat on days \(i\) and \(i-1\), respectively;')
#     st.write(r'- \(LAI_{max}\) is the maximum LAI for a plant;')
#     st.write(r'- \(LAI_{i-1}\) is the LAI on day \(i-1\);')
#     st.write(r'- \(frPHU\) is the accumulated potential heat unit fraction on a day;')
#     st.write(r'- \(frPHU_{sen}\) is the fraction of days where leaf senescence exceeds leaf growth in the entire plant growth season;')
#     st.write(r'- \(\gamma_{reg}\) is the plant growth factor (range, 0–1);')
#     st.write(r'- \(w_{strs}\) is the water stress on a day;')
#     st.write(r'- \(t_{strs}\) is the temperature stress on a day;')
#     st.write(r'- \(n_{strs}\) is the nitrogen stress on a day;')
#     st.write(r'- \(p_{strs}\) is the phosphorus stress on a day;')
#     st.write(r'- \(\Delta LAI_{act,i}\) is the actual LAI on day \(i\);')
    
#     # LaTeX display for z-score formula
#     st.write("**Formula:**")
#     st.latex(r'z = \frac{(X - \mu)}{\sigma}')
#     st.write("""
#     Where:
#     - \(X\) = individual observation
#     - \(\mu\) = mean of the dataset
#     - \(\sigma\) = standard deviation of the dataset
#     """)

# elif selected_option == "Simulator":

#     # Title and Introduction
#     st.title("Forest Growth and Hydrology Simulator")
#     st.write("Explore how different forest parameters influence growth and hydrological processes.")
    
#     # Function to plot data
#     def plot_data(x, y, title, xlabel, ylabel):
#         plt.figure(figsize=(10, 4))
#         plt.plot(x, y, marker='o')
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         plt.grid(True)
#         st.pyplot(plt)

    
#     # Forest Growth Parameters Section
#     st.subheader("Forest Growth Parameters")
    
#     # Input fields for forest growth parameters
#     LAI = st.slider("Leaf Area Index (LAI)", min_value=0.0, max_value=10.0, value=3.0)
#     biomass = st.number_input("Initial Biomass (metric tons/ha)", min_value=0.0, value=150.0)
#     canopy_height = st.number_input("Canopy Height (meters)", min_value=0.0, value=10.0)
#     light_extinction_coefficient = st.number_input("Light Extinction Coefficient (k)", value=0.5)
#     radiation_use_efficiency = st.number_input("Radiation Use Efficiency (RUE, g/MJ)", value=2.0)
#     total_solar_radiation = st.number_input("Total Solar Radiation (MJ/m²/day)", value=10.0)
    
#     st.write(f"**Leaf Area Index (LAI):** {LAI}")
#     st.write(f"**Initial Biomass:** {biomass} metric tons/ha")
#     st.write(f"**Canopy Height:** {canopy_height} meters")
#     st.write(f"**Light Extinction Coefficient:** {light_extinction_coefficient}")
#     st.write(f"**Radiation Use Efficiency (RUE):** {radiation_use_efficiency} g/MJ")
#     st.write(f"**Total Solar Radiation:** {total_solar_radiation} MJ/m²/day")

#     # Leaf Area Development Calculation
#     LAImax = 3.0  # Maximum LAI for corn
#     f_rLAImax_prev = 0.9  # Fraction of LAImax at previous timestep (arbitrary example)
#     f_rLAImax_current = 1.0  # Current fraction (arbitrary example)
    
#     Kf = LAImax * (f_rLAImax_current - f_rLAImax_prev)
#     delta_LAI = Kf * (1 - np.exp(-5 * (LAI - LAImax)))
#     LAI_new = LAI + delta_LAI
    
#     st.write(f"**New LAI after Growth:** {LAI_new:.2f}")

#     # Light Interception Calculation
#     Hphosyn = total_solar_radiation * (1 - np.exp(-light_extinction_coefficient * LAI_new))
#     st.write(f"**Intercepted Photosynthetically Active Radiation:** {Hphosyn:.2f} MJ/m²")

#     # Biomass Production Calculation
#     delta_bio = radiation_use_efficiency * Hphosyn  # Daily biomass increase
#     total_biomass = biomass + delta_bio  # Update total biomass
#     st.write(f"**New Biomass after Growth:** {total_biomass:.2f} metric tons/ha")

#     # Hydrological Processes Section
#     st.subheader("Hydrological Processes")
    
#     st.subheader("Evapotranspiration Calculator")
    
#     # Input for evapotranspiration calculation
#     net_radiation = st.number_input("Net Radiation (MJ/m²/day)", value=10.0)
#     soil_heat_flux = st.number_input("Soil Heat Flux (MJ/m²/day)", value=0.5)
#     air_temp = st.number_input("Air Temperature (°C)", value=20.0)
#     relative_humidity = st.number_input("Relative Humidity (%)", value=60.0)

#     # Calculate and display ET
#     vapor_pressure_saturation = 0.611 * np.exp((17.27 * air_temp) / (air_temp + 237.3))
#     vapor_pressure_actual = (relative_humidity / 100) * vapor_pressure_saturation
#     delta = 4098 * vapor_pressure_saturation / ((air_temp + 237.3) ** 2)
#     gamma = 0.0665  # Psychrometric constant (kPa/°C)
    
#     ET = (delta * (net_radiation - soil_heat_flux) + (gamma * (vapor_pressure_saturation - vapor_pressure_actual))) / (delta + gamma)
    
#     if st.button("Calculate Evapotranspiration"):
#         st.write(f"**Evapotranspiration (ET):** {ET:.2f} mm/day")

#     # Interactive Calculations Section
#     st.subheader("Calculate Low Flows")
#     rainfall = st.number_input("Rainfall (mm)", value=50.0)
#     curve_number = st.number_input("Curve Number (CN)", value=75)

#     # Calculate and display runoff using the SCS runoff equation
#     if st.button("Calculate Runoff"):
#         S = (25400 / curve_number) - 254  # Calculate S
#         Q = (rainfall - 0.2 * S) ** 2 / (rainfall + 0.8 * S) if rainfall > 0.2 * S else 0
#         st.write(f"**Calculated Runoff:** {Q:.2f} mm")

#     # Soil and Root Systems Analysis Section
#     st.subheader("Soil and Root Systems Analysis")
    
#     root_depth = st.number_input("Root Depth (meters)", value=1.5)
#     water_table = st.number_input("Water Table Depth (meters)", value=3.0)

#     # Example calculation of available water
#     soil_porosity = 0.3  # Example value for soil porosity
#     available_water = soil_porosity * (root_depth - water_table)
    
#     st.write(f"**Available Water (from roots):** {available_water:.2f} m³/ha")

#     # Growth Constraints Section
#     st.subheader("Growth Constraints")
    
#     temp = st.slider("Temperature (°C)", min_value=-10.0, max_value=30.0, value=15.0)
#     water_avail = st.slider("Water Availability (mm)", min_value=0.0, max_value=500.0, value=200.0)

#     # Assessment of growth impact based on constraints
#     growth_potential = (temp / 30) * (water_avail / 500) * 100  # Simplified growth potential assessment
#     st.write(f"**Growth Potential Estimate:** {growth_potential:.2f}% of max potential")

#     # Examples and Case Studies Section
#     st.subheader("Examples and Case Studies")
    
#     # Example data for visualization
#     years = np.arange(1, 21)  # Years from 1 to 20
#     LAI_example = 0.85 * (1 - np.exp(-0.1 * years))  # Example LAI growth curve
    
#     # Plotting LAI growth over years
#     plot_data(years, LAI_example, "LAI Growth Over Time", "Years", "Leaf Area Index (LAI)")

# # If you want to add more sections, include them under the "Other Options" or similar option.
# elif selected_option == "New":

#     class Tree:
#         def __init__(self, species, age, height, dbh, lai):
#             self.species = species
#             self.age = age
#             self.height = height
#             self.dbh = dbh
#             self.lai = lai
    
#         def calculate_growth(self, soil, water, climate):
#             growth_rate = (climate['temperature'] * 0.01) + (water * 0.1) + (soil['nutrients'] * 0.02)
#             self.height += growth_rate * self.height * 0.1
#             self.dbh += growth_rate * self.dbh * 0.05
#             self.lai = self.calculate_lai()
    
#         def calculate_lai(self):
#             return self.height * 0.5  # Simplified calculation
    
#     class HRU:
#         def __init__(self, slope, soil_type, water_table_depth):
#             self.slope = slope
#             self.soil_type = soil_type
#             self.water_table_depth = water_table_depth
    
#         def calculate_runoff(self):
#             # Simplified runoff calculation based on slope and soil type
#             runoff_coefficient = 0.1 if self.soil_type == "Sandy" else 0.2 if self.soil_type == "Loamy" else 0.3
#             runoff = self.slope * runoff_coefficient
#             return runoff
    
#     def hydrological_balance(tree, hru, years, precipitation, evaporation):
#         soil_moisture = 100  # Initial soil moisture in mm
#         results = []
    
#         for year in range(years):
#             et = tree.lai * 2.5  # Evapotranspiration based on LAI
#             runoff = hru.calculate_runoff() * precipitation * 0.01  # Calculate runoff based on precipitation
#             soil_moisture += precipitation - et - runoff
    
#             if soil_moisture > 150:
#                 runoff += soil_moisture - 150
#                 soil_moisture = 150
#             else:
#                 runoff = 0
    
#             results.append({
#                 "Year": year + 1,
#                 "Height (m)": round(tree.height, 2),
#                 "DBH (cm)": round(tree.dbh, 2),
#                 "LAI": round(tree.lai, 2),
#                 "Soil Moisture (mm)": round(soil_moisture, 2),
#                 "Runoff (mm)": round(runoff, 2),
#                 "ET (mm)": round(et, 2)
#             })
    
#         return pd.DataFrame(results)
    
#     def simulate_tree_growth_and_hydrology(tree, hru, years, soil_conditions, water_availability, climate_conditions, precipitation, evaporation):
#         results = []
#         for year in range(years):
#             tree.calculate_growth(soil_conditions, water_availability, climate_conditions)
#             results.append({
#                 "Year": year + 1,
#                 "Height (m)": round(tree.height, 2),
#                 "DBH (cm)": round(tree.dbh, 2),
#                 "LAI": round(tree.lai, 2)
#             })
#         return pd.DataFrame(results)
    
#     # Streamlit App
#     st.title("Tree Growth and Hydrology Simulator")
    
#     # 1. Select a Tree
#     species = st.selectbox("Select Tree Species", options=["Douglas Fir", "Red Cedar"])
    
#     # Change the age selection to specific predefined values
#     age = st.selectbox("Select Tree Age (years)", options=[10, 20, 30, 60, 100, 200, 500])
    
#     # Set default values based on species
#     if species == "Douglas Fir":
#         height = st.number_input("Initial Height (m)", value=10.0)  # Default for Douglas Fir
#         dbh = st.number_input("Initial DBH (cm)", value=25.0)  # Default for Douglas Fir
#         lai = st.number_input("Initial LAI", value=4.0)  # Default for Douglas Fir
#     else:  # Red Cedar
#         height = st.number_input("Initial Height (m)", value=8.0)  # Default for Red Cedar
#         dbh = st.number_input("Initial DBH (cm)", value=20.0)  # Default for Red Cedar
#         lai = st.number_input("Initial LAI", value=3.5)  # Default for Red Cedar
    
#     # 2. Display Parameter Effects
#     st.subheader("Tree Parameters")
#     st.write(f"Species: {species}, Age: {age}, Height: {height} m, DBH: {dbh} cm, LAI: {lai}")
    
#     # 3. Growth Calculation Inputs
#     soil_moisture = st.number_input("Soil Moisture (mm)", value=30)
#     soil_nutrients = st.number_input("Soil Nutrients (1-10)", value=5, min_value=1, max_value=10)
#     water_availability = st.number_input("Water Availability (mm)", value=100)
#     temperature = st.number_input("Temperature (°C)", value=25)
#     precipitation = st.number_input("Annual Precipitation (mm)", value=800)
#     evaporation = st.number_input("Annual Evaporation (mm)", value=300)
#     years = st.number_input("Years to Simulate", value=10, min_value=1)
    
#     # 4. HRU Properties
#     st.subheader("Hydrological Response Unit (HRU) Properties")
#     slope = st.number_input("Slope (%)", value=10.0)
#     soil_type = st.selectbox("Soil Type", options=["Sandy", "Loamy", "Clayey"])
#     water_table_depth = st.number_input("Water Table Depth (cm)", value=150)
    
#     # Create HRU object
#     hru = HRU(slope, soil_type, water_table_depth)
    
#     # Simulate on Button Click
#     if st.button("Simulate Growth and Hydrology"):
#         tree = Tree(species, age, height, dbh, lai)
    
#         soil_conditions = {'moisture': soil_moisture, 'nutrients': soil_nutrients}
#         climate_conditions = {'temperature': temperature}
    
#         # Simulate Growth
#         growth_results = simulate_tree_growth_and_hydrology(tree, hru, years, soil_conditions, water_availability, climate_conditions, precipitation, evaporation)
    
#         # Hydrological Balance Calculation
#         hydrology_results = hydrological_balance(tree, hru, years, precipitation, evaporation)
    
#         # Combine Results
#         combined_results = pd.merge(growth_results, hydrology_results, on="Year")
    
#         # Check columns in combined_results for debugging
#         st.write("Columns in combined_results:", combined_results.columns.tolist())
    
#         st.write("Growth Simulation Results:")
#         st.dataframe(combined_results)
    
#         # 6. Visualization with Plotly
#         st.subheader("Visualizations")
    
#     # Tree Growth Over Time
#     fig_growth = go.Figure()
#     fig_growth.add_trace(go.Scatter(x=combined_results['Year'], y=combined_results['Height (m)_x'],
#                                       mode='lines+markers', name='Height (m)', line=dict(color='green')))
#     fig_growth.add_trace(go.Scatter(x=combined_results['Year'], y=combined_results['DBH (cm)_x'],
#                                       mode='lines+markers', name='DBH (cm)', line=dict(color='blue')))
#     fig_growth.add_trace(go.Scatter(x=combined_results['Year'], y=combined_results['LAI_x'],
#                                       mode='lines+markers', name='LAI', line=dict(color='orange')))
#     fig_growth.update_layout(title='Tree Growth Over Time',
#                              xaxis_title='Year',
#                              yaxis_title='Value',
#                              legend_title='Parameters',
#                              template='plotly_white')
#     st.plotly_chart(fig_growth)
    
#     # Hydrological Balance Components Over Time
#     fig_hydrology = go.Figure()
#     fig_hydrology.add_trace(go.Scatter(x=combined_results['Year'], y=combined_results['ET (mm)'],
#                                         mode='lines+markers', name='ET (mm)', line=dict(color='red')))
#     fig_hydrology.add_trace(go.Scatter(x=combined_results['Year'], y=combined_results['Soil Moisture (mm)'],
#                                         mode='lines+markers', name='Soil Moisture (mm)', line=dict(color='blue')))
#     fig_hydrology.add_trace(go.Scatter(x=combined_results['Year'], y=combined_results['Runoff (mm)'],
#                                         mode='lines+markers', name='Runoff (mm)', line=dict(color='orange')))
#     fig_hydrology.update_layout(title='Hydrological Balance Components Over Time',
#                                 xaxis_title='Year',
#                                 yaxis_title='Value',
#                                 legend_title='Components',
#                                 template='plotly_white')
#     st.plotly_chart(fig_hydrology)

elif selected_option == "Report":   
    st.title("Model Validation Report")
    
    # Add a short description
    st.markdown("""
    This report provides a comprehensive validation of the SWAT-MODFLOW model 
    implemented for groundwater and surface water interactions. It includes 
    detailed analysis of the model's performance, statistical metrics, and 
    visualizations that illustrate the model's predictions against observed data.
    """)

    PDF_FILE = Path(__file__).parent / 'data/koki_swatmf_report.pdf'
    with open(PDF_FILE, "rb") as f:
        pdf_data = f.read()
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

    st.download_button(
        label="Download PDF",
        data=pdf_data,
        file_name="koki_swatmf_report.pdf",
        mime="application/pdf"
    )
    
    iframe_width, iframe_height = get_iframe_dimensions()
    st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="{iframe_width}" height="{iframe_height}" style="border:none;"></iframe>', unsafe_allow_html=True)
    
#Extra comments ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    # # Render the Folium map in Streamlit
    # st.title("Watershed Map")
    # st_folium(m, width=700, height=600)  
        
    # monthly_stats = df.groupby(['Month', 'Row', 'Column'])['Rate'].agg(['mean', 'std']).reset_index()
    # monthly_stats.columns = ['Month', 'Row', 'Column', 'Average Rate', 'Standard Deviation']

    # global_min = monthly_stats[['Average Rate', 'Standard Deviation']].min().min()
    # global_max = monthly_stats[['Average Rate', 'Standard Deviation']].max().max()

    # unique_months = sorted(monthly_stats['Month'].unique())
    # unique_month_names = [month_names[m - 1] for m in unique_months]

    # selected_month_name = st.selectbox("Month", unique_month_names, index=0)
    # selected_month = unique_months[unique_month_names.index(selected_month_name)]
    # stat_type = st.radio("Statistic Type", ['Average Rate [m³/day]', 'Standard Deviation'], index=0)

    # df_filtered = monthly_stats[monthly_stats['Month'] == selected_month]
    
    # grid = np.full((int(df_filtered['Row'].max()), int(df_filtered['Column'].max())), np.nan)

    # for _, row in df_filtered.iterrows():
    #     grid[int(row['Row']) - 1, int(row['Column']) - 1] = row['Average Rate'] if stat_type == 'Average Rate [m³/day]' else row['Standard Deviation']

    # # Define color scale and boundaries for heatmap
    # if stat_type == 'Standard Deviation':
    #     zmin = 0
    #     zmax = global_max
    # else:
    #     zmin = global_min
    #     zmax = global_max

    # colorbar_title = (
    #     "Average Monthly<br> Groundwater / Surface<br> Water Interaction<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - To Stream | + To Aquifer<br> [m³/day]"
    #     if stat_type == 'Average Rate [m³/day]' 
    #     else '&nbsp;&nbsp;&nbsp;&nbsp;Standard Deviation'
    # )

    # # Calculate the midpoint for the color bar (usually zero)
    # zmid = 0

    # # Create the heatmap figure
    # fig = go.Figure(data=go.Heatmap(
    #     z=grid,
    #     colorscale='earth_r',
    #     zmid=zmid,
    #     zmin=zmin,
    #     zmax=zmax,
    #     colorbar=dict(
    #         title=colorbar_title, 
    #         orientation='h', 
    #         x=0.5, 
    #         y=-0.1, 
    #         xanchor='center', 
    #         yanchor='top',
    #         tickvals=[zmin, 0, zmax],  # Specify tick positions
    #         ticktext=[f'{zmin:.2f}', '0', f'{zmax:.2f}'],  # Custom tick labels
    #     ),
    #     hovertemplate='%{z:.2f}<extra></extra>',
    # ))

    # fig.update_layout(
    #     title=f'{stat_type} for Month {selected_month}',
    #     xaxis_title=None,
    #     yaxis_title=None,
    #     xaxis=dict(showticklabels=False, ticks='', showgrid=False),
    #     yaxis=dict(showticklabels=False, ticks='', autorange='reversed', showgrid=False),
    #     plot_bgcolor='rgba(240, 240, 240, 0.8)',
    #     paper_bgcolor='white',
    #     font=dict(family='Arial, sans-serif', size=8, color='black')
    # )

    # # Display the heatmap
    # st.plotly_chart(fig)

