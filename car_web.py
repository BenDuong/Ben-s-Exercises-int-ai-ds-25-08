# FILE: my_dashboard.py
# YOUR EXERCISE SOLUTION

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Car Brands in Australia", page_icon="📊", layout="wide")

# Title
st.title("📊 Car Brands in Australia - Created by Ben")
st.markdown("Interactive analysis of car brands.")

# Load data
@st.cache_data
def load_data():
    np.random.seed(42)
    car_brands = ['Toyota', 'Mazda','Ford', 'BMW','Others']
    n_cars = 1000
    car_prob = [0.3,0.2,0.15,0.1,0.25]

    car_data = pd.DataFrame({
      'car_id': range(1,n_cars+1),
      'car_brand': np.random.choice(car_brands, n_cars, p = car_prob),
      'year': np.random.normal(2018, 10, n_cars).astype(int).clip(2010,2025),
      'kilometers': np.random.exponential(15000, n_cars),
     })
    current_year = 2025

    base_price = { 'Toyota': 30000, 'Mazda':32000, 'Ford': 35000, 'BMW': 70000, 'Others': 50000}

    car_data['price'] = car_data.apply( lambda row: 
                                 base_price[row['car_brand']] -
                                 (current_year - row['year']) * row['kilometers'] * np.random.normal(0, 0.01),
                                  axis = 1
                                  )

    return car_data

df = load_data()
# Sidebar
st.sidebar.header("Filters")
# ADD YOUR FILTERS HERE
# Brand filter
selected_brands = st.sidebar.multiselect(
    "Select brand",
    options=df['car_brand'].unique(),
    default=df['car_brand'].unique()
)

# Year filter
year_range = st.sidebar.slider(
    "Year",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2010, 2021)
)

# Km filter
km_range = st.sidebar.slider(
    "KM",
    min_value=int(df['kilometers'].min()),
    max_value=int(df['kilometers'].max()),
    value=(10000, 15000)
)

# Filter the data
filtered_df = df[
    (df['car_brand'].isin(selected_brands)) &
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1]) &
    (df['kilometers'] >= km_range[0]) &
    (df['kilometers'] <= km_range[1])
]
# Metrics
col1, col2, col3 = st.columns(3)
# ADD YOUR METRICS HERE

with col1:
      avg_price = filtered_df['price'].mean()
      st.metric(
        "Average Price",
        f"${avg_price:.0f}",
        f"{avg_price - df['price'].mean():+.1f} vs all"
    )

with col2:
      max_brand = filtered_df['car_brand'][ filtered_df['price']== filtered_df['price'].max()].tolist()[0]
      max_price = filtered_df['price'].max()
      st.metric(
        "The most expensive car brand",
        f"{max_brand}",
        f"${ max_price- df['price'].mean():+.1f} vs all"
    )

with col3:
      min_brand = filtered_df['car_brand'][ filtered_df['price']== filtered_df['price'].min()].tolist()[0]
      min_price = filtered_df['price'].min()
      st.metric(
        "The cheapest car brand",
        f"{min_brand}",
        f"${ min_price- df['price'].mean():+.1f} vs all",
        delta_color="normal"
    )

# Charts
col1, col2 = st.columns(2)
# ADD YOUR CHARTS HERE
with col1:
    st.subheader("📊 Average Price by Year")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    year_avg = filtered_df.groupby('year')['price'].mean()
    
    plots = plt.plot(year_avg.index, year_avg.values, marker='o', linestyle='-', color='teal')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Price')
    ax.set_title('Average Price over Years')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("📊 Average Price by Car Brands")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    car_avg = filtered_df.groupby('car_brand')['price'].mean()
    
    bars = ax.bar(car_avg.index, car_avg.values, color='steelblue')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Average Price')
    ax.set_title('Average Car Prices')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.1f}M',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
# Data table
if st.checkbox("Show data"):
    st.subheader("Filtered Car Data")
    
    # Show only relevant columns
    display_cols = ['car_brand', 'price', 'kilometers', 'year']
    
    # Format the dataframe for display
    display_df = filtered_df[display_cols].copy()
    display_df['price'] = display_df['price'].apply(lambda x: f'${x:.2f}')
    display_df['kilometers'] = display_df['kilometers'].apply(lambda x: f'{x:.0f} km')
    display_df['year'] = display_df['year'].astype(str)
    
    st.dataframe(display_df.head(100), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Dashboard created with Streamlit • Data is synthetic for demonstration purposes")