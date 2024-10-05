import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import base64
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from folium import plugins
from folium import GeoJson  
from folium.plugins import MousePosition
from shapely.geometry import Point
from PIL import Image, ImageDraw, ImageFont

# Set the title and favicon that appear in the browser's tab bar.
st.set_page_config(
    page_title='Koki Dashboard',
    page_icon=':earth_americas:',
)

# Sidebar for navigation
st.sidebar.title("Xwulqw'selu Sta'lo'")
selected_option = st.sidebar.radio(
    "Select an option:",
    ("Watershed models", "Water interactions", "Recharge", "View Report")
)

# Month names for mapping
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
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
    
# Path to your data file
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

    # Render the Folium map in Streamlit
    st.title("Watershed Map")
    st_folium(m, width=700, height=600)  

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
        "This physical model provides a straightforward way to illustrate the complex interactions between groundwater and surface water, including the effects of pumping. Through this interactive web application, users can deepen their understanding of these dynamics and better interpret the complex results emerging from the watershed's behavior.",
        
        "Reduced water levels during dry periods are evident in the low flow conditions of the river. These conditions highlight the importance of studying low flows, which are critical for understanding drought impacts, water resource availability, and the ecological health of the watershed.",
        
        "Accurate data from field monitoring activities plays a pivotal role in capturing real-time watershed conditions. The success of these monitoring efforts enhances our ability to track changes over time, offering valuable insights into watershed dynamics and informing evidence-based management strategies.",
        
        "The exchange of diverse ideas during meetings drives the development of innovative projects that aim to improve the health of the watershed. Collaborative efforts such as these contribute to creating long-term solutions that promote sustainability and resilience in watershed management."
    ]
    
    # Set up the layout for small images (small panel)
    st.title("Interactive Image Viewer")
    st.write("Select an image to see a larger view and an explanation.")
    
    # Create columns for displaying small images
    cols = st.columns(4)  # 4 images, each in its own column
    selected_image = None  # To store which image is selected
    
    # Display the small images with buttons
    for i, image_file in enumerate(image_files):
        with cols[i]:
            # Load each image
            image_path = data_folder / image_file
            
            try:
                image = Image.open(image_path)
                # Display the small image with a button
                st.image(image, caption=f"{image_file}", width=100)
                
                # When the button is clicked, the corresponding image and message are selected
                if st.button(f"Show {image_file}"):
                    selected_image = image
                    selected_caption = captions[i]
            except FileNotFoundError:
                st.error(f"Image file {image_file} not found in {data_folder}.")
        
    # Display the selected image in a larger panel (if selected)
    if selected_image:
        st.image(selected_image, caption=selected_caption, use_column_width=True)
        st.write(selected_caption)
    else:
        st.write("Click on an image to see a larger view and explanation.")
    
elif selected_option == "Water interactions":
    custom_title("How groundwater and surface water interact in the Xwulqw’selu watershed?", 28)

    st.markdown("""
    In the Xwulqw’selu Watershed, groundwater plays a key role in sustaining streamflow during low-flow periods, particularly in summer. As surface water levels drop, groundwater discharge becomes the primary source of flow, helping maintain aquatic habitats and water availability. 
    
    Land use changes, and climate shifts can reduce groundwater recharge, worsening low-flow conditions. Understanding this groundwater-surface water interaction is critical for managing water resources and mitigating the impacts of prolonged droughts.
    
    Below is a map of the average monthly groundwater / surface water interactions across the watershed. You can change which month you want to look at or zoom into different parts of the watershed for a closer examination of recharge patterns.
    """)

    monthly_stats = df.groupby(['Month', 'Row', 'Column'])['Rate'].agg(['mean', 'std']).reset_index()
    monthly_stats.columns = ['Month', 'Row', 'Column', 'Average Rate', 'Standard Deviation']

    global_min = monthly_stats[['Average Rate', 'Standard Deviation']].min().min()
    global_max = monthly_stats[['Average Rate', 'Standard Deviation']].max().max()

    unique_months = sorted(monthly_stats['Month'].unique())
    unique_month_names = [month_names[m - 1] for m in unique_months]

    selected_month_name = st.selectbox("Month", unique_month_names, index=0)
    selected_month = unique_months[unique_month_names.index(selected_month_name)]
    stat_type = st.radio("Statistic Type", ['Average Rate [m³/day]', 'Standard Deviation'], index=0)

    df_filtered = monthly_stats[monthly_stats['Month'] == selected_month]

    water_interaction_dict = {}
    
    # Iterate through df_filtered to assign water interaction values
    for _, row in df_filtered.iterrows():
        # Calculate the column index (0-based)
        column_index = int(row['Column']) - 1  # Adjusting for zero-based indexing
        # Calculate the row index (0-based)
        row_number = int(row['Row']) - 1  # Adjusting for zero-based indexing
    
        # Store the corresponding value in the dictionary
        water_interaction_dict[(row_number, column_index)] = row['Average Rate']
    
    # Step 2: Create GeoDataFrame for water interactions
    geometry = []
    
    # Create geometry for each cell in the grid_gdf and assign water interaction values
    for row_index, row in grid_gdf.iterrows():
        x, y = row.geometry.centroid.x, row.geometry.centroid.y
        water_value = water_interaction_dict.get((row_index // grid_gdf.shape[1], row_index % grid_gdf.shape[1]), 0)  # Default to 0 if no value found
        geometry.append(Point(x, y))
    
    # Create a new GeoDataFrame for visualization
    gdf_water_interactions = gpd.GeoDataFrame(
        {
            'Water Interaction Value': [water_interaction_dict.get((i // grid_gdf.shape[1], i % grid_gdf.shape[1]), 0) for i in range(len(geometry))],
        },
        geometry=geometry,
        crs="EPSG:32610"
    )
    
    # Step 3: Create a Folium map centered on Duncan
    duncan_lat = 48.67  # Latitude
    duncan_lon = -123.79  # Longitude
    m = folium.Map(location=[duncan_lat, duncan_lon], zoom_start=11, control_scale=True)
    
    # Add a marker for Duncan
    folium.Marker([duncan_lat, duncan_lon], popup='Duncan, BC').add_to(m)
    
    # Add the grid as a GeoJSON layer to the map
    folium.GeoJson(
        grid_gdf,
        name="Grid",
        style_function=lambda x: {'color': 'blue', 'weight': 1},
    ).add_to(m)
    
    # Prepare heatmap data using water interaction values
    heatmap_data = [
        [row.geometry.y, row.geometry.x, row['Water Interaction Value']]
        for _, row in gdf_water_interactions.iterrows()
    ]
    print(heatmap_data)  # Add this line to debug the contents of heatmap_data
    
    # Add the heatmap layer to the map
    heatmap = plugins.HeatMap(
        heatmap_data,
        radius=15,  # Adjust radius for heatmap intensity
        name='Water Interactions',
        overlay=True,
        control=True,
    )
    m.add_child(heatmap)
    
    # Add Layer Control
    folium.LayerControl().add_to(m)
    
    # Render the Folium map in Streamlit
    st.title("Water Interactions Map")
    st_folium(m, width=700, height=600)
    

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

    # Display the plotly heatmap in Streamlit
    st.plotly_chart(fig_recharge, use_container_width=True)
    
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
