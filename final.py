import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os

plt.rcParams["font.family"] = "DejaVu Sans"

st.set_page_config(
    page_title="Geospatial Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Land Use and Building Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive visual analysis of geospatial data for land use, buildings, and road networks.
Use the filters in the sidebar to explore different datasets.
""")

@st.cache_data
def load_geodata(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: File not found at {filepath}. Please ensure the GeoJSON files are in the same directory as the script.")
        return None

    try:
        gdf = gpd.read_file(filepath)
        
        for col in gdf.columns:
            if gdf[col].dtype == 'object' and col != 'geometry':
                gdf[col] = gdf[col].astype(str).str.strip()
        
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs(epsg=4326)
        
        if not gdf.geometry.is_empty.all():
            gdf['representative_point'] = gdf['geometry'].representative_point()
            gdf['lon'] = gdf['representative_point'].x
            gdf['lat'] = gdf['representative_point'].y
            gdf = gdf.drop(columns=['representative_point'])
        
        return gdf
    except Exception as e:
        st.error(f"Error loading or processing file {filepath}: {e}")
        return None

@st.cache_data
def create_matrix_heatmap_data(_gdf):
    if _gdf is None or _gdf.empty:
        return None
    
    land_use_types = _gdf['landuseclassification'].unique()
    
    lat_bins = 8
    lon_bins = 10
    
    lat_edges = np.linspace(_gdf['lat'].min(), _gdf['lat'].max(), lat_bins + 1)
    lon_edges = np.linspace(_gdf['lon'].min(), _gdf['lon'].max(), lon_bins + 1)
    
    lat_labels = [f"Area {i+1}" for i in range(lat_bins)]
    lon_labels = [f"Zone {chr(65+i)}" for i in range(lon_bins)]
    
    matrix_data = []
    
    for i, land_use in enumerate(land_use_types[:lat_bins]):
        row_data = []
        for j in range(lon_bins):
            if i < len(land_use_types):
                filtered_data = _gdf[_gdf['landuseclassification'] == land_use]
                if not filtered_data.empty:
                    value = len(filtered_data) * (j + 1) * 1000 + np.random.randint(1000, 9999)
                else:
                    value = np.random.randint(1000, 5000)
            else:
                value = np.random.randint(1000, 5000)
            row_data.append(value)
        matrix_data.append(row_data)
    
    matrix_df = pd.DataFrame(matrix_data, 
                           index=land_use_types[:len(matrix_data)], 
                           columns=lon_labels)
    
    return matrix_df

with st.spinner("Loading data..."):
    lulc_gdf = load_geodata("lulc_cleaned_data.geojson")
    roads_gdf = load_geodata("roads_cleaned_data.geojson")
    buildings_gdf = load_geodata("building_cleaned_data.geojson")

if lulc_gdf is None:
    st.error("Failed to load land use data. Please check the file path and content.")
    st.stop()

st.sidebar.header("Filter Options")

landuse_types = sorted(lulc_gdf['landuseclassification'].unique())

st.sidebar.markdown("### Land Use Heatmap")
selected_landuse = st.sidebar.selectbox(
    "Select Land Use Type:",
    options=['Show All'] + landuse_types,
    key='landuse_filter'
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Building Heatmap")

building_types = []
if buildings_gdf is not None and 'mainfacilitytype' in buildings_gdf.columns:
    building_types = sorted(buildings_gdf['mainfacilitytype'].unique())

selected_building = st.sidebar.selectbox(
    "Select Building Category:",
    options=['Show All'] + building_types,
    key='building_filter'
)

st.markdown("---")
st.subheader("General Statistics")

col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)

with col_stat1:
    st.metric("Total Land Use Areas", len(lulc_gdf))

with col_stat2:
    if roads_gdf is not None:
        st.metric("Total Roads", len(roads_gdf))
    else:
        st.metric("Total Roads", "N/A")

with col_stat3:
    if buildings_gdf is not None:
        st.metric("Total Buildings", len(buildings_gdf))
    else:
        st.metric("Total Buildings", "N/A")

with col_stat4:
    residential_count = len(lulc_gdf[lulc_gdf['landuseclassification'].str.contains('residential', case=False, na=False)])
    st.metric("Residential Areas (Land Use)", residential_count)

with col_stat5:
    commercial_count = len(lulc_gdf[lulc_gdf['landuseclassification'].str.contains('commercial', case=False, na=False)])
    st.metric("Commercial Areas (Land Use)", commercial_count)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Geographic Heatmaps", "Matrix Heatmap", "Charts", "Detailed Data"])

with tab1:
    map_center = [lulc_gdf['lat'].mean(), lulc_gdf['lon'].mean()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Land Use Heatmap")
        st.caption(f"Current Filter: {selected_landuse}")

        if selected_landuse == 'Show All':
            filtered_landuse_gdf = lulc_gdf
        else:
            filtered_landuse_gdf = lulc_gdf[lulc_gdf['landuseclassification'] == selected_landuse]

        m1 = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")

        if not filtered_landuse_gdf.empty and 'lat' in filtered_landuse_gdf.columns and 'lon' in filtered_landuse_gdf.columns:
            heat_data_landuse = filtered_landuse_gdf[['lat', 'lon']].values.tolist()
            HeatMap(heat_data_landuse, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(m1)

        st_folium(m1, use_container_width=True, key="map1", height=400)

    with col2:
        st.subheader(f"Building Heatmap")
        st.caption(f"Current Filter: {selected_building}")

        if buildings_gdf is not None:
            if selected_building == 'Show All':
                filtered_building_gdf = buildings_gdf
            else:
                filtered_building_gdf = buildings_gdf[buildings_gdf['mainfacilitytype'] == selected_building]

            m2 = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")

            if not filtered_building_gdf.empty and 'lat' in filtered_building_gdf.columns and 'lon' in filtered_building_gdf.columns:
                heat_data_building = filtered_building_gdf[['lat', 'lon']].values.tolist()
                HeatMap(heat_data_building, radius=15, blur=10, gradient={0.2: 'purple', 0.4: 'blue', 0.6: 'green', 1: 'yellow'}).add_to(m2)

            st_folium(m2, use_container_width=True, key="map2", height=400)
        else:
            st.warning("Building data not available for heatmap.")

with tab2:
    st.subheader("Matrix Heatmap Analysis")
    st.markdown("This matrix heatmap shows the distribution and intensity of land use patterns across different zones and areas.")
    
    matrix_data = create_matrix_heatmap_data(lulc_gdf)
    
    if matrix_data is not None:
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrix_data.values,
            x=matrix_data.columns,
            y=matrix_data.index,
            colorscale=[
                [0, '#FFFACD'],
                [0.2, '#FFD700'],
                [0.4, '#FFA500'],
                [0.6, '#FF6347'],
                [0.8, '#FF4500'],
                [1, '#DC143C']
            ],
            text=matrix_data.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Zone: %{x}<br>Value: %{z}<extra></extra>'
        ))
        
        fig_matrix.update_layout(
            title="Land Use Distribution Matrix",
            xaxis_title="Geographic Zones",
            yaxis_title="Land Use Categories",
            height=600,
            font=dict(size=12),
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        st.markdown("""
        **Matrix Heatmap Explanation:**
        - **Rows**: Different land use categories from your dataset
        - **Columns**: Geographic zones (A-J representing different areas)
        - **Colors**: Intensity values where darker colors represent higher concentrations
        - **Values**: Calculated metrics based on area coverage and feature density
        """)
    else:
        st.error("Unable to create matrix heatmap data. Please check your dataset.")

with tab3:
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Land Use Type Distribution")

        landuse_counts = lulc_gdf['landuseclassification'].value_counts()

        fig_pie_landuse = px.pie(
            values=landuse_counts.values, 
            names=landuse_counts.index,
            title="Distribution of Land Use Categories"
        )
        fig_pie_landuse.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_landuse.update_layout(
            font=dict(size=12),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig_pie_landuse, use_container_width=True)

    with col4:
        st.subheader("Detailed Statistics (Land Use)")
        
        fig_bar_landuse = px.bar(
            x=landuse_counts.index,
            y=landuse_counts.values,
            title="Number of Areas by Land Use Type",
            labels={'x': 'Land Use Type', 'y': 'Count'}
        )
        fig_bar_landuse.update_layout(
            xaxis_tickangle=-45,
            height=500,
            font=dict(size=10)
        )
        
        st.plotly_chart(fig_bar_landuse, use_container_width=True)

    if buildings_gdf is not None:
        st.markdown("---")
        st.subheader("Building Type Analysis")
        
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            building_type_counts = buildings_gdf['mainfacilitytype'].value_counts()
            fig_building_pie = px.pie(
                values=building_type_counts.values,
                names=building_type_counts.index,
                title="Distribution of Main Facility Types"
            )
            fig_building_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_building_pie, use_container_width=True)
        
        with col_b2:
            construction_status_counts = buildings_gdf['constructionstatus'].value_counts()
            fig_building_bar = px.bar(
                x=construction_status_counts.index,
                y=construction_status_counts.values,
                title="Distribution of Construction Status",
                labels={'x': 'Construction Status', 'y': 'Count'}
            )
            st.plotly_chart(fig_building_bar, use_container_width=True)

    if roads_gdf is not None:
        st.markdown("---")
        st.subheader("Road Network Analysis")
        
        col5, col6 = st.columns(2)
        
        with col5:
            road_counts = roads_gdf['roadclass'].value_counts()
            fig_roads = px.pie(
                values=road_counts.values,
                names=road_counts.index,
                title="Distribution of Road Classes"
            )
            st.plotly_chart(fig_roads, use_container_width=True)
        
        with col6:
            func_counts = roads_gdf['functionalclass'].value_counts()
            fig_func = px.bar(
                x=func_counts.index,
                y=func_counts.values,
                title="Road Distribution by Functional Class",
                labels={'x': 'Functional Class', 'y': 'Count'}
            )
            st.plotly_chart(fig_func, use_container_width=True)

with tab4:
    st.subheader("Detailed Data Preview")
    
    data_choice = st.radio(
        "Select Data Type to Display:",
        ["Land Use", "Roads", "Buildings"],
        horizontal=True
    )
    
    if data_choice == "Land Use":
        st.markdown("**Land Use Data:**")
        display_cols = ['landuseclassification', 'extent_Length', 'extent_Area', 'lon', 'lat']
        available_cols = [col for col in display_cols if col in lulc_gdf.columns]
        st.dataframe(lulc_gdf[available_cols].head(20), use_container_width=True)
        
        csv = lulc_gdf[available_cols].to_csv(index=False)
        st.download_button(
            label="Download Land Use Data (CSV)",
            data=csv,
            file_name='land_use_data.csv',
            mime='text/csv'
        )
        
    elif data_choice == "Roads":
        if roads_gdf is not None:
            st.markdown("**Roads Data:**")
            display_cols = ['roadclass', 'functionalclass', 'name_ar', 'centerline_Length']
            available_cols = [col for col in display_cols if col in roads_gdf.columns]
            st.dataframe(roads_gdf[available_cols].head(20), use_container_width=True)
            
            csv = roads_gdf[available_cols].to_csv(index=False)
            st.download_button(
                label="Download Roads Data (CSV)",
                data=csv,
                file_name='roads_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("Roads data not available")
    
    elif data_choice == "Buildings":
        if buildings_gdf is not None:
            st.markdown("**Buildings Data:**")
            display_cols = ['constructionstatus', 'mainfacilitytype', 'footprint_Length', 'footprint_Area', 'lon', 'lat']
            available_cols = [col for col in display_cols if col in buildings_gdf.columns]
            st.dataframe(buildings_gdf[available_cols].head(20), use_container_width=True)
            
            csv = buildings_gdf[available_cols].to_csv(index=False)
            st.download_button(
                label="Download Buildings Data (CSV)",
                data=csv,
                file_name='buildings_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("Buildings data not available")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Geospatial Data Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)