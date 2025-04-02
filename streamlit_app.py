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
    # # Define a function to generate folder path based on selected decade
    # def get_folder_path(land_use, climate, water_use):
    #     # Handle "logged" as a special case for land use
    #     if land_use.lower() == "logged":
    #         land_use_code = "Logged"  # Use the exact case as in the folder name
    #     elif land_use.lower() == "f30":
    #         land_use_code = "F30"  # Directly use F30 as the folder name
    #     elif land_use.lower() == "f60":
    #         land_use_code = "F60"  # Directly use F30 as the folder name
    #     else:
    #         land_use_code = f'L{land_use[-2:]}'  # Prepend 'L' to the last two characters
    #     climate_code = f'C{climate[-2:]}'
    #     water_use_code = f'W{water_use[-2:]}'
    #     folder_name = f'{land_use_code}_{climate_code}_{water_use_code}'
    #     return Path(__file__).parent / 'data' / folder_name

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
    custom_title("Watershed models for Xwulqw'selu Sta'lo'", 28)
        
    st.markdown("""
    ### Xwulqw'selu Sta'lo' Watershed Model – Key Learnings
    
    Summer flows in Xwulqw'selu Sta'lo' (Koksilah River) have been decreasing over time. Watershed models can be useful tools to better understand why, how, and where this is happening.
    
    The **Xwulqw'selu Connections** research team at the University of Victoria developed a comprehensive watershed model using the best available data to represent current conditions. This complex computer model integrates multiple processes that govern water movement through the watershed across time and space. The main pathways of water flow—from precipitation to streamflow—are depicted in a watershed diagram. The model also incorporates recent climate data and can be used to compare different scenarios of water and land management.
    
    This interactive website features maps and graphs where you can explore the significance of the entire watershed ([link to first page]), examine how changes in water use ([link to second page]) or forestry practices ([link to third page]) affect summer low flows. These visualizations provide an alternative way of 'seeing' the watershed, similar to interpreting an aerial or conceptual representation of the landscape.
    
    This model and website were primarily developed by **David Serrano**. More detailed information can be found in David’s **thesis** ([link]), for those interested in further exploration.
    
    #### Key Definitions:
    
    - **Low Flows:** The lowest streamflows in the Xwulqw'selu Sta'lo' typically occur during summer and are measured in volume per time. This model focuses particularly on August streamflows, though watershed health encompasses many other factors beyond just flow rates.
    - **Baseline Model:** A model representing the best available recent data (2012–2023).
    - **Scenarios:** Variations of the baseline model where water use or land use is altered while keeping all other variables constant. These scenarios help assess the potential impacts of changes in water and land use.
    
    We invite you to engage with this interactive tool and deepen your understanding of the Xwulqw'selu Sta'lo' watershed.
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
    
       
    # Step 6: Add interactivity to select a specific cell
    st.subheader("Cell-Specific Analysis")
    selected_row = st.slider("Select Row Index", min_value=1, max_value=rows, value=1)
    selected_col = st.slider("Select Column Index", min_value=1, max_value=cols, value=1)
    
    # Extract data for the selected cell
    selected_cell_data = pivoted[(pivoted['Row'] == selected_row) & (pivoted['Column'] == selected_col)]
    if not selected_cell_data.empty:
        monthly_rates = selected_cell_data.iloc[0, 2:].values  # Skip 'Row' and 'Column'
        monthly_changes = np.diff(monthly_rates)  # Calculate differences between months
    
        # Create a plot to visualize rates and changes
        fig = go.Figure()
    
        # Add monthly rates line
        fig.add_trace(go.Scatter(
            x=unique_months,
            y=monthly_rates,
            mode='lines+markers',
            name='Rate',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))
    
        # Add changes line (for months 2 to n, as changes are calculated from differences)
        fig.add_trace(go.Scatter(
            x=unique_months[1:],  # Skip the first month for changes
            y=monthly_changes,
            mode='lines+markers',
            name='Change (ΔRate)',
            line=dict(color='red', dash='dot'),
            marker=dict(size=8)
        ))
    
        # Update layout for better visualization
        fig.update_layout(
            title=f"Rate and Change Over Time for Cell ({selected_row}, {selected_col})",
            xaxis_title="Month",
            yaxis_title="Rate / Change",
            xaxis=dict(tickmode='linear', title='Month'),
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            paper_bgcolor='white',
            font=dict(family='Arial, sans-serif', size=12, color='black'),
            legend=dict(x=0.1, y=0.9),
        )
    
        # Display the plot in Streamlit
        st.plotly_chart(fig)
    
        # Display data values for reference
        st.write(f"**Cell ({selected_row}, {selected_col}) Rate Values:** {monthly_rates}")
        st.write(f"**Cell ({selected_row}, {selected_col}) Changes (ΔRate):** {monthly_changes}")
    else:
        st.write(f"No data available for Cell ({selected_row}, {selected_col}).")

    # Define classification function
    def classify_value(value):
        if value > 1:
            return "Above 1"
        elif -1 <= value <= 1:
            return "Between -1 and 1"
        else:
            return "Below -1"
    
    # Process data for visualization
    ranges = ['Below -1', 'Between -1 and 1', 'Above 1']
    monthly_columns = pivoted.columns[2:]  # Monthly columns (Jan, Feb, ...)
    range_means = {r: [] for r in ranges}
    
    fig = go.Figure()
    
    # Add traces for each cell
    for _, row in pivoted.iterrows():
        cell_id = f"Cell ({int(row['Row'])}, {int(row['Column'])})"
        monthly_values = row[monthly_columns].values  # Extract monthly values
        classification = classify_value(np.mean(monthly_values))
        fig.add_trace(go.Scatter(
            x=monthly_columns,
            y=monthly_values,
            mode='lines',
            name=cell_id,
            line=dict(width=1),
            legendgroup=classification,  # Group by range
            visible=True
        ))
        # Accumulate values for mean calculation
        range_means[classification].append(monthly_values)
    
    # Compute and add mean lines for each range
    for range_name, values in range_means.items():
        if values:  # Check if there are values in this range
            mean_values = np.mean(values, axis=0)  # Mean across cells
            fig.add_trace(go.Scatter(
                x=monthly_columns,
                y=mean_values,
                mode='lines',
                name=f"Mean ({range_name})",
                line=dict(width=3, dash='dash'),
                legendgroup=range_name,  # Group by range
                visible=True
            ))
    
    # Streamlit checkboxes for toggling visibility
    show_cells = st.checkbox("Show All Cells", value=True)
    show_means = st.checkbox("Show Mean Lines", value=True)
    
    # Adjust visibility of traces based on checkboxes
    for trace in fig.data:
        if "Mean" in trace.name:
            trace.visible = show_means
        else:
            trace.visible = show_cells
    
    # Customize layout
    fig.update_layout(
        title="Monthly Values by Cell and Range",
        xaxis_title="Month",
        yaxis_title="Values",
        xaxis=dict(tickmode='linear'),
        yaxis=dict(title="Value Range"),
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        legend_title="Cell and Range",
        font=dict(family="Arial, sans-serif", size=10)
    )
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)    

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

    # Create the data for the scenarios
    data = {
        'Scenario': ['LU_2010', 'F30', 'Logged', 'F60'],
        'Area (ha)': [31131.45, 31131.45, 31131.45, 31131.45],  # in hectares
        'GRAS': [36.81, 36.81, 36.81, 36.81],
        'AGRL': [969.66, 969.66, 969.66, 969.66],
        'DFSP': [5415.21, 5415.21, 22732.2, 0],
        'PAST': [21.6, 21.6, 21.6, 21.6],
        'URHD': [22.59, 22.59, 22.59, 22.59],
        'URMD': [200.34, 200.34, 200.34, 200.34],
        'UTRN': [320.67, 320.67, 320.67, 320.67],
        'WETF': [214.2, 214.2, 214.2, 214.2],
        'DFSF': [22732.2, 7841.79, 5415.21, 28147.41],
        'DFST': [483.93, 15374.34, 483.93, 483.93],
        'DFSS': [23.58, 23.58, 23.58, 23.58],
        'WATR': [113.85, 113.85, 113.85, 113.85],
        'URLD': [576.81, 576.81, 576.81, 576.81],
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Streamlit App
    st.title('Scenario Comparison: Land Use and Water Resources')
    
    # Display the table
    st.subheader('Scenario Table (Area in ha)')
    df_display = df.drop(columns=['Area (ha)'])  # Drop the original 'Area (ha)' column
    st.dataframe(df_display)
    
    # Explanation of the scenarios
    st.subheader('Scenario Explanation')
    
    # Add explanations based on the scenarios
    scenario_explanation = {
        'LU_2010': "This represents the land use scenario in the year 2010.",
        'F30': "This scenario represents an increment in 30-year-old forest (F30), which is changed by 60-year-old forest.",
        'Logged': "In this scenario, all subbasins with previous logged areas are fully logged.",
        'F60': "The F60 scenario shows a 60-year-old forest instead of logged areas.",
    }
    
    # Display description for each scenario
    scenario = st.selectbox("Select a Scenario for More Details", df['Scenario'].unique())
    st.write(scenario_explanation[scenario])
    
    # Visualize the scenario differences with Plotly
    st.subheader('Scenario Differences Visualization')
    
    # Reshape data for Plotly
    df_long = df.melt(id_vars=['Scenario'], value_vars=df.columns[2:], var_name='Category', value_name='Value')
    
    # Create Plotly bar chart
    fig = px.bar(df_long, 
                 x='Scenario', 
                 y='Value', 
                 color='Category', 
                 barmode='group', 
                 title='Scenario Comparison',
                 labels={'Value': 'Value (ha)', 'Scenario': 'Scenario', 'Category': 'Category'},
                 height=500)
    
    # Enable legend interaction to hide or show categories
    fig.update_layout(
        xaxis_title='Scenario',
        yaxis_title='Value (ha)',
        barmode='group',
        legend_title='Categories',
        legend=dict(title="Category", tracegroupgap=1)
    )
    
    # Display the plot
    st.plotly_chart(fig)

    # Input data for monthly basin values
    data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Rainfall (mm)': [240.74, 120.73, 124.14, 85.84, 45.46, 35.79, 15.70, 18.55, 89.53, 152.66, 228.13, 212.18],
        'Snowfall (mm)': [28.98, 16.99, 1.91, 0.43, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 7.98, 34.25],
        'Surface Q (mm)': [43.34, 13.65, 11.05, 4.95, 0.66, 0.53, 0.10, 0.13, 3.30, 7.94, 26.30, 28.33],
        'Lateral Q (mm)': [153.89, 83.01, 83.26, 49.41, 21.27, 13.44, 5.40, 5.64, 38.53, 88.14, 147.77, 137.69],
        'Yield (mm)': [202.87, 101.17, 99.24, 58.36, 25.60, 17.30, 8.76, 8.90, 44.75, 99.25, 178.13, 170.52],
        'ET (mm)': [7.45, 10.68, 20.06, 32.73, 49.52, 43.73, 34.07, 16.55, 20.91, 17.50, 9.95, 6.57],
        'PET (mm)': [10.56, 16.75, 35.28, 62.37, 103.80, 117.40, 140.06, 118.15, 63.65, 33.25, 14.04, 9.02],
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Streamlit app title
    st.title('Water Balance Interactive Visualization')
    
    # Create an interactive bar chart using Plotly
    fig = px.bar(df, 
                 x='Month', 
                 y=['Rainfall (mm)', 'Snowfall (mm)', 'Surface Q (mm)', 'Lateral Q (mm)', 'Yield (mm)', 'ET (mm)', 'PET (mm)'],
                 barmode='group', 
                 title="Monthly Water Balance Components",
                 labels={'value': 'Millimeters (mm)', 'variable': 'Water Balance Components'})
    
    # Add hover data for better interactivity
    fig.update_traces(hovertemplate='%{x}: %{y} mm')
    
    # Show the plot in the Streamlit app
    st.plotly_chart(fig)
    
    # Display the DataFrame as a table
    st.write("### Water Balance Data", df)

    # Filter data for August
    august_data = df[df['Month'] == 'Aug']
    
    # Calculate values for August
    et_august = august_data['ET (mm)'].values[0]
    streamflow_august = august_data['Surface Q (mm)'].values[0] + august_data['Lateral Q (mm)'].values[0]
    baseflow_august = august_data['Lateral Q (mm)'].values[0]

    # Display the results
    st.subheader("August Metrics")
    st.write(f"**Evapotranspiration (ET):** {et_august} mm")
    st.write(f"**Average Streamflow:** {streamflow_august} mm")
    st.write(f"**Baseflow:** {baseflow_august} mm")
    
    # Updated data based on the provided values
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    LU_2010_et = [7.45, 10.68, 20.06, 32.73, 49.52, 43.73, 34.07, 16.55, 20.91, 17.50, 9.95, 6.57]
    thornthwaite_et = [31, 16, 11, 12, 18, 27, 45, 71, 88, 102, 85, 56]  # Example Thornthwaite ET values
    logged_et = [4.51, 6.63, 13.11, 20.43, 27.24, 24.53, 19.58, 12.88, 14.66, 10.47, 6.20, 4.08]  # Example Logged ET values
    f30_et = [6.78, 9.71, 18.24, 29.66, 44.90, 40.19, 32.09, 15.96, 19.29, 15.96, 9.07, 5.98]
    f60_et = [8.70, 12.43, 23.11, 37.77, 57.88, 50.77, 39.35, 17.98, 23.67, 20.44, 11.54, 7.64]
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Add Penman-Monteith ET line
    fig.add_trace(go.Scatter(
        x=months, y=LU_2010_et, mode='lines+markers', name='LU_2010 ET',
        line=dict(color='blue', width=2), marker=dict(symbol='circle', size=8, color='blue')
    ))
    
    # Add Thornthwaite ET line
    fig.add_trace(go.Scatter(
        x=months, y=thornthwaite_et, mode='lines+markers', name='Thornthwaite ET',
        line=dict(color='green', width=2), marker=dict(symbol='square', size=8, color='green')
    ))
    
    # Add Logged ET line
    fig.add_trace(go.Scatter(
        x=months, y=logged_et, mode='lines+markers', name='Logged ET',
        line=dict(color='red', width=2), marker=dict(symbol='diamond', size=8, color='red')
    ))

     # Add F30 ET line
    fig.add_trace(go.Scatter(
        x=months, y=f30_et, mode='lines+markers', name='F30 ET',
        line=dict(color='red', width=2), marker=dict(symbol='diamond', size=8, color='purple')
    ))

    # Add F60 ET line
    fig.add_trace(go.Scatter(
        x=months, y=f60_et, mode='lines+markers', name='F60 ET',
        line=dict(color='red', width=2), marker=dict(symbol='diamond', size=8, color='purple')
    ))
    
    # Update layout for better visualization
    fig.update_layout(
        title='Monthly ET Comparison: Penman-Monteith, Thornthwaite, F30, F60 and Logged ET',
        xaxis_title='Month',
        yaxis_title='ET (mm)',
        template='plotly_dark',
        showlegend=True
    )
    
    # Streamlit app header
    st.title("Evapotranspiration Comparison")
    
    # Streamlit app description
    st.markdown("""
    This app compares monthly evapotranspiration (ET) values using the following methods:
    - **LU 2010 ET** (calculated using the Penman-Monteith method)
    - **Thornthwaite ET** (calculated using the Thornthwaite method)
    - **Logged ET** (ET from logged land cover)
    - **F30 ET** (ET from F30 land cover)
    
    Select the chart below to explore the monthly variations in ET.
    """)
    
    # Show the plot in the Streamlit app
    st.plotly_chart(fig)
    
    # Streamlit app title
    #st.title("Flow Duration Curve (FDC) Analysis")
    
    # --- Load Data ---
    subbasins = gpd.read_file(subbasins_shapefile_path)
    deltas = pd.read_csv(deltas_file)
    
    # Merge deltas with the shapefile
    subbasins = subbasins.merge(deltas, left_on="Subbasin", right_on="Subbasin", how="left")
    
    # --- Clean Data ---
    subbasins[["Delta_Logged", "Delta_F60", "Delta_F30"]] = subbasins[["Delta_Logged", "Delta_F60", "Delta_F30"]].replace([-float('inf')], 0)
    subbasins[["Delta_Logged", "Delta_F60", "Delta_F30"]] = subbasins[["Delta_Logged", "Delta_F60", "Delta_F30"]].replace([float('inf')], pd.NA)
    subbasins = subbasins.dropna(subset=["Delta_Logged", "Delta_F60", "Delta_F30"])
    
    # Calculate global min and max
    vmin = subbasins[["Delta_Logged", "Delta_F60", "Delta_F30"]].min().min()
    vmax = subbasins[["Delta_Logged", "Delta_F60", "Delta_F30"]].max().max()
    
    # Initialize the map centered on your location
    initial_location = [49.0600, -123.0200]  # Modify to your map's center coordinates (e.g., Duncan, BC)
    m = folium.Map(location=initial_location, zoom_start=11, control_scale=True)
    
    # --- Add Subbasin Layer ---
    subbasins_gdf = gpd.read_file(subbasins_shapefile_path)  # Replace with your subbasins shapefile
    subbasins_layer = folium.GeoJson(
        subbasins_gdf,
        name="Subbasins",
        style_function=lambda x: {'color': 'green', 'weight': 2}
    ).add_to(m)
    
    # --- Add Scenarios as Layers ---
    def create_scenario_layer(scenario_column, vmin, vmax, color_map="coolwarm"):
        """Function to create a scenario layer from a GeoDataFrame."""
        return folium.GeoJson(
            subbasins_gdf,
            name=scenario_column,
            style_function=lambda feature: {
                'fillColor': folium.colors.color_brewer[color_map][
                    int((feature['properties'][scenario_column] - vmin) / (vmax - vmin) * (len(folium.colors.color_brewer[color_map]) - 1))
                ],
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6
            }
        )
    
    # Assuming you have deltas dataframe with the appropriate columns
    deltas_df = pd.read_csv('path_to_deltas_data.csv')  # Replace with the path to your deltas CSV file
    vmin = deltas_df[["Delta_Logged", "Delta_F60", "Delta_F30"]].min().min()
    vmax = deltas_df[["Delta_Logged", "Delta_F60", "Delta_F30"]].max().max()
    
    # Add the scenario layers
    delta_logged_layer = create_scenario_layer("Delta_Logged", vmin, vmax)
    delta_f60_layer = create_scenario_layer("Delta_F60", vmin, vmax)
    delta_f30_layer = create_scenario_layer("Delta_F30", vmin, vmax)
    
    # Add layers to map
    delta_logged_layer.add_to(m)
    delta_f60_layer.add_to(m)
    delta_f30_layer.add_to(m)
    
    # --- Add MousePosition ---
    MousePosition().add_to(m)
    
    # --- Add LayerControl for Toggle Options ---
    folium.LayerControl(position='topright').add_to(m)
    
    # --- Render the Map in Streamlit ---
    st.title("Watershed Map")
    st_folium(m, width=700, height=600)


    
    #     # Streamlit widget to choose the year
    #     year = st.selectbox("Select Year", options=combined_data['YEAR'].unique())
    
    #     # Filter the data for the selected year
    #     yearly_data = combined_data[combined_data['YEAR'] == year]
    
    #     # --- FDC Analysis for RCH 3 ---
    #     rch3_data = yearly_data[yearly_data['RCH'] == 3]
    
    #     # Compute the FDC for all scenarios in RCH 3
    #     fdc_data = []
    #     for scenario in rch3_data['Scenario'].unique():
    #         scenario_data = rch3_data[rch3_data['Scenario'] == scenario]
    #         sorted_data = scenario_data.sort_values(by="FLOW_OUTcms", ascending=False).reset_index(drop=True)
    #         sorted_data["Rank"] = sorted_data.index + 1
    #         sorted_data["ExceedanceProbability"] = sorted_data["Rank"] / (len(sorted_data) + 1) * 100
    #         sorted_data["Scenario"] = scenario
    #         fdc_data.append(sorted_data)
    
    #     # Combine the processed data for FDC plotting
    #     fdc_data = pd.concat(fdc_data)
    
    #     # Plot the FDC
    #     fig_fdc = px.line(
    #         fdc_data,
    #         x="ExceedanceProbability",
    #         y="FLOW_OUTcms",
    #         color="Scenario",
    #         color_discrete_map=scenario_colors,
    #         labels={
    #             "ExceedanceProbability": "Exceedance Probability (%)",
    #             "FLOW_OUTcms": "Flow Out (cms)",
    #             "Scenario": "Scenario"
    #         },
    #         title=f"Flow Duration Curve for RCH 3 in Year {year}"
    #     )
    
    #     # Set y-axis to logarithmic scale
    #     fig_fdc.update_yaxes(type="log", title="Flow Out (cms, Log Scale)")
    
    #     # Show the FDC plot in the Streamlit app
    #     st.plotly_chart(fig_fdc)
    
    #     # --- August Analysis for Selected Reaches ---
    #     # Filter for August (Days between 213 and 243)
    #     august_data = combined_data[
    #         (combined_data['YEAR'] == year) & (combined_data['DAY'] >= 213) & (combined_data['DAY'] <= 243)
    #     ]
    
    #     # Add a Month column for August
    #     august_data['Month'] = 8
    
    #     # Select reaches to analyze
    #     selected_reaches = st.multiselect("Select Reaches to Analyze", options=combined_data['RCH'].unique(), default=[3])
    
    #     # Filter the data for the selected reaches
    #     filtered_data = august_data[august_data["RCH"].isin(selected_reaches)]
    
    #     # Create a Plotly figure for flow out comparison by reach
    #     fig_august = px.line(
    #         filtered_data, x="DAY", y="FLOW_OUTcms", color="Scenario", line_dash="Scenario",
    #         facet_col="RCH", facet_col_wrap=4,
    #         color_discrete_map=scenario_colors,
    #         labels={"DAY": "Day of Year", "FLOW_OUTcms": "Flow Out (cms)", "Scenario": "Scenario"},
    #         title=f"Flow Out Comparison for Selected Reaches (Year {year})"
    #     )
    #     st.plotly_chart(fig_august)
    
    #     # Calculate daily volume for total flow (m³)
    #     seconds_in_a_day = 24 * 60 * 60
    #     filtered_data['DailyVolume_m3'] = filtered_data['FLOW_OUTcms'] * seconds_in_a_day
    
    #     # Calculate mean flow (m³/s) and total flow (m³) for August
    #     monthly_mean_flow = filtered_data.groupby(["YEAR", "Scenario", "Month"])["FLOW_OUTcms"].mean().reset_index()
    #     monthly_total_flow = filtered_data.groupby(["YEAR", "Scenario", "Month"])["DailyVolume_m3"].sum().reset_index()
    
    #     # Display the results for both mean and total flow
    #     st.subheader("Mean Flow (m³/s) and Total Flow (m³) for August:")
    
    #     # Plot Mean Flow (m³/s) with bars for all scenarios
    #     fig_mean_flow = px.bar(
    #         monthly_mean_flow,
    #         x="Month", y="FLOW_OUTcms", color="Scenario", barmode="group",
    #         color_discrete_map=scenario_colors,
    #         labels={"Month": "Month", "FLOW_OUTcms": "Mean Flow (m³/s)", "Scenario": "Scenario"},
    #         title=f"Mean Flow per Month for Year {year} (August)"
    #     )
    #     st.plotly_chart(fig_mean_flow)

    #     # Calculate mean flow for August by Subbasin and Scenario
    #     august_mean = august_data.groupby(["Subbasin", "Scenario"])["FLOW_OUTcms"].mean().reset_index()
    
    #     # Pivot to get each scenario as a column for comparison
    #     august_pivot = august_mean.pivot(index="Subbasin", columns="Scenario", values="FLOW_OUTcms").reset_index()
    
    #     # Calculate delta flows for each scenario
    #     for scenario in ["Scenario Logged", "Scenario F60", "Scenario F30"]:
    #         august_pivot[f"Delta_{scenario}"] = (
    #             (august_pivot["Scenario 2010"] - august_pivot[scenario]) / august_pivot["Scenario 2010"]
    #         )
    
    #     # Load Subbasin shapefile
    #     subbasins = gpd.read_file(subbasins_shapefile)
    
    #     # Merge delta values with subbasin shapefile
    #     subbasins = subbasins.merge(august_pivot, left_on='Subbasin', right_on='Subbasin', how='left')
    
    #     # Plot each delta scenario on the map
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    #     scenarios = ["Delta_Scenario Logged", "Delta_Scenario F60", "Delta_Scenario F30"]
    #     titles = ["Logged Scenario", "F60 Scenario", "F30 Scenario"]
    
    #     for i, scenario in enumerate(scenarios):
    #         ax = axes[i]
    #         subbasins.plot(
    #             column=scenario, ax=ax, cmap='RdYlBu', legend=True,
    #             legend_kwds={'label': "Delta August Mean Flow", 'orientation': "vertical"}
    #         )
    #         ax.set_title(f"Delta August Mean Flow - {titles[i]}")
    #         ax.axis("off")
    
    #     plt.tight_layout()
    #     plt.show()
        
    #     # Plot Total Flow (m³) with bars for all scenarios
    #     fig_total_flow = px.bar(
    #         monthly_total_flow,
    #         x="Month", y="DailyVolume_m3", color="Scenario", barmode="group",
    #         color_discrete_map=scenario_colors,
    #         labels={"Month": "Month", "DailyVolume_m3": "Total Flow (m³)", "Scenario": "Scenario"},
    #         title=f"Total Flow per Month for Year {year} (August)"
    #     )
    #     st.plotly_chart(fig_total_flow)
    
    # else:
    #     st.warning("Please upload all four scenario Excel files to proceed.")
 

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
    
