import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
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
    ("Watershed models", "Groundwater / Surface water interactions", "Recharge")
)

# # Decade Selection for each feature
# st.sidebar.title("Model selection")
# st.sidebar.subheader("Climate")
# selected_decade_climate = st.sidebar.selectbox(
#     "Choose a decade for Climate:",
#     ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
# )

# st.sidebar.subheader("Land Use")
# selected_decade_land_use = st.sidebar.selectbox(
#     "Choose a decade for Land Use:",
#     ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
# )

# st.sidebar.subheader("Water Use")
# selected_decade_water_use = st.sidebar.selectbox(
#     "Choose a decade for Water Use:",
#     ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
# )

# Month names for mapping
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Function to process SWAT-MF data
@st.cache_data
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
    
#Path to your data file
DATA_FILENAME = Path(__file__).parent / 'data/swatmf_out_MF_gwsw_monthly.csv'
df = process_swatmf_data(DATA_FILENAME)

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
        if value < -50:
            grid[row_idx, col_idx] = 0  # Dark Blue (strong negative)
        elif -50 <= value < -10:
            grid[row_idx, col_idx] = 1  # Light Blue (moderate negative)
        elif -10 <= value <= 1:
            grid[row_idx, col_idx] = 2  # Yellow (near-zero)
        elif value > 1:
            grid[row_idx, col_idx] = 3  # Brown (positive, to aquifer)
    
        # Check for changes in sign between months and assign green colors
        if prev_month_value > 0 and value < 0:
            grid[row_idx, col_idx] = 4  # Shiny Green (positive to negative)
        elif prev_month_value < 0 and value > 0:
            grid[row_idx, col_idx] = 5  # Green (negative to positive)
    
        # Create hover text for the real values
        hover_text[row_idx, col_idx] = f"Row: {row['Row']}, Column: {row['Column']}, Value: {value:.2f}"
    
    # Function to create heatmap
    def create_heatmap(grid, selected_month_name, hover_text):
        # Step 6: Define a custom color scale
        colorscale = [
            [0.0, 'darkblue'],   # Strong groundwater to river
            [0.2, 'lightblue'],  # Moderate groundwater to river
            [0.4, 'yellow'],     # Near-zero fluctuation
            [0.6, 'brown'],      # Groundwater going into aquifer
            [0.8, 'limegreen'],  # Change from positive to negative interaction
            [1.0, 'lightpink']   # Change from negative to positive interaction
        ]
    
        # Step 7: Create the heatmap for the selected month
        fig = go.Figure(data=go.Heatmap(
            z=grid,
            colorscale=colorscale,
            zmin=0,  # Minimum category (dark blue)
            zmax=5,  # Maximum category (light pink)
            showscale=False,  # Hide scale since colors represent categories
            hoverinfo='text',  # Show real values in hover
            text=hover_text  # Hover text with real values
        ))
    
        # Step 8: Update the layout of the heatmap
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
    
        # Step 9: Display the heatmap
        st.plotly_chart(fig)
    
        # # Step 10: Add a legend to explain the color coding
        # st.markdown("""
        # ### Color Legend:
        # - **Dark Blue**: Strong negative interaction (groundwater going to river, -50 to -1000)
        # - **Light Blue**: Moderate negative interaction (groundwater going to river, -10 to -50)
        # - **Yellow**: Near-zero fluctuation (groundwater level stable, -10 to 1)
        # - **Brown**: Positive interaction (groundwater going into aquifer, >1)
        # - **Shiny Green**: Change from positive to negative interaction
        # - **Light Pink**: Change from negative to positive interaction
        # """)
    
    # Create a function to count cells per color
    def count_cells_per_color(grid):
        color_counts = {
            'dark_blue': np.sum(grid == 0),
            'light_blue': np.sum(grid == 1),
            'yellow': np.sum(grid == 2),
            'brown': np.sum(grid == 3),
            'limegreen': np.sum(grid == 4),
            'lightpink': np.sum(grid == 5),
        }
        return color_counts
    
    # Count the colors for the selected month
    color_counts = count_cells_per_color(grid)
    
    # Prepare data for pie chart
    color_names = ['Strongly gaining', 'Gaining', 'No significants contributions', 'Losing', 'Changing to gaining', 'Changing to losing']
    color_values = [color_counts['dark_blue'], color_counts['light_blue'], color_counts['yellow'],
                    color_counts['brown'], color_counts['limegreen'], color_counts['lightpink']]
    
    total_cells = sum(color_values)
    
    # Avoid division by zero
    percentages = [count / total_cells * 100 if total_cells > 0 else 0 for count in color_values]
    
    # Create a pie chart with formatted percentages
    pie_colors = ['#00008B', '#ADD8E6', '#FFFF00', '#A52A2A', '#00FF00', '#FFB6C1']  # Ensure the colors are correct
    
    fig = go.Figure(data=[go.Pie(labels=color_names, values=percentages, hole=.3, marker=dict(colors=pie_colors), textinfo='percent')])
    # fig = go.Figure(data=[go.Pie(labels=color_names, values=percentages, hole=.3, marker=dict(colors=pie_colors), textinfo='none')])

    
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
    
    # Additional: Plotting Histogram for Selected Month
    def plot_histogram(data):
        """
        This function plots a histogram of the 'Rate' values for the selected month.
        """
        plt.figure(figsize=(10, 6))  # Set figure size
        plt.hist(data['Rate'], bins=30, color='blue', alpha=0.7)  # Create histogram
        plt.axvline(data['Rate'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')  # Add mean line
        plt.axvline(data['Rate'].mean() + 1.96 * data['Rate'].std(), color='green', linestyle='dashed', linewidth=1, label='95% CI Upper')  # Upper CI
        plt.axvline(data['Rate'].mean() - 1.96 * data['Rate'].std(), color='green', linestyle='dashed', linewidth=1, label='95% CI Lower')  # Lower CI
        plt.title(f'Histogram of Rate Values for {selected_month_name}')  # Title for the histogram
        plt.xlabel('Rate')  # X-axis label
        plt.ylabel('Frequency')  # Y-axis label
        plt.legend()  # Show legend
        st.pyplot(plt)  # Render the plot within the Streamlit app
        plt.clf()  # Clear the current figure to avoid overlapping plots
    
    # Button to plot histogram
    if st.button('Plot Histogram'):
        plot_histogram(selected_month_data)  # Call the histogram plotting function
        
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

    
elif selected_option == "View Report":
    st.title("Model Validation Report")

    # Add a short description
    st.markdown("""
    This report provides a comprehensive validation of the SWAT-MODFLOW model 
    implemented for groundwater and surface water interactions. It includes 
    detailed analysis of the model's performance, statistical metrics, and 
    visualizations that illustrate the model's predictions against observed data.
    """)

    # PDF_FILE = Path(__file__).parent / 'data/koki_swatmf_report.pdf'
    # with open(PDF_FILE, "rb") as f:
    #     pdf_data = f.read()
    #     pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

    # st.download_button(
    #     label="Download PDF",
    #     data=pdf_data,
    #     file_name="koki_swatmf_report.pdf",
    #     mime="application/pdf"
    # )
    
    # iframe_width, iframe_height = get_iframe_dimensions()
    # st.markdown(f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="{iframe_width}" height="{iframe_height}" style="border:none;"></iframe>', unsafe_allow_html=True)

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

