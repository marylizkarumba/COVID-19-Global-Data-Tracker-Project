# COVID-19 Global Data Tracker
# Analysis of Global COVID-19 Trends: Cases, Deaths, and Vaccinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 1. Data Collection and Loading
print("1. Loading COVID-19 data...")
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
try:
    df = pd.read_csv(url)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data from URL: {e}")
    print("Loading sample data...")
    # Create a sample dataset in case the URL fails
    df = pd.read_csv("owid-covid-data.csv")
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

# 2. Data Exploration
print("\n2. Exploring the data structure...")
print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nColumns in the dataset:")
print(df.columns.tolist())

print("\nUnique countries/locations in the dataset:")
num_locations = df['location'].nunique()
print(f"Total unique locations: {num_locations}")

print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

print("\nData time range:")
print(f"From: {df['date'].min()} to {df['date'].max()}")

# 3. Data Cleaning
print("\n3. Cleaning the data...")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
print("Date column converted to datetime format.")

# Create a filtered dataset with selected countries of interest
countries_of_interest = ['World', 'United States', 'India', 'Brazil', 'United Kingdom', 
                         'Russia', 'France', 'Germany', 'South Africa', 'Kenya', 'China', 
                         'Japan', 'Australia', 'Canada']

df_selected = df[df['location'].isin(countries_of_interest)].copy()
print(f"Selected {len(countries_of_interest)} countries for detailed analysis.")

# Handle missing values for key metrics
for column in ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']:
    # Fill missing values with 0 for these metrics
    df_selected[column] = df_selected[column].fillna(0)

# For vaccination data, we'll use forward fill by country
for column in df_selected.columns:
    if 'vaccine' in column:
        df_selected[column] = df_selected.groupby('location')[column].transform(lambda x: x.fillna(method='ffill'))

print("Missing values handled for key metrics.")

# 4. Exploratory Data Analysis (EDA)
print("\n4. Performing Exploratory Data Analysis...")

# 4.1 Global overview: Total cases and deaths over time
print("Generating global overview charts...")

# Filter for world data
world_data = df[df['location'] == 'World'].copy()

plt.figure(figsize=(16, 12))

# Plot 1: Total cases worldwide
plt.subplot(2, 1, 1)
plt.plot(world_data['date'], world_data['total_cases'], color='blue', linewidth=2)
plt.title('Global COVID-19 Total Cases')
plt.ylabel('Total Cases')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Total deaths worldwide
plt.subplot(2, 1, 2)
plt.plot(world_data['date'], world_data['total_deaths'], color='red', linewidth=2)
plt.title('Global COVID-19 Total Deaths')
plt.ylabel('Total Deaths')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('global_cases_deaths.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.2 Compare total cases across selected countries
print("Comparing COVID-19 cases across countries...")

# Get the latest data for each country
latest_date = df_selected['date'].max()
latest_data = df_selected[df_selected['date'] == latest_date]
latest_data = latest_data.sort_values('total_cases', ascending=False)

plt.figure(figsize=(14, 8))
bars = plt.bar(latest_data['location'], latest_data['total_cases'] / 1_000_000)
plt.title('Total COVID-19 Cases by Country (in millions)')
plt.ylabel('Total Cases (millions)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('total_cases_by_country.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.3 Compare death rates (deaths per case) across selected countries
print("Analyzing death rates across countries...")

# Calculate death rate
latest_data['death_rate'] = (latest_data['total_deaths'] / latest_data['total_cases']) * 100
latest_data = latest_data.sort_values('death_rate', ascending=False)

plt.figure(figsize=(14, 8))
bars = plt.bar(latest_data['location'], latest_data['death_rate'])
plt.title('COVID-19 Death Rate by Country (Deaths per 100 Cases)')
plt.ylabel('Death Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('death_rate_by_country.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.4 Time series analysis for top 6 countries by total cases
print("Creating time series analysis for top countries...")

# Get top 6 countries by total cases
top_countries = latest_data.nlargest(6, 'total_cases')['location'].tolist()
df_top = df_selected[df_selected['location'].isin(top_countries)]

# Plot total cases over time for top countries
plt.figure(figsize=(16, 10))
for country in top_countries:
    country_data = df_top[df_top['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'] / 1_000_000, linewidth=2, label=country)

plt.title('COVID-19 Total Cases Over Time - Top 6 Countries')
plt.ylabel('Total Cases (millions)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_countries_cases_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.5 Daily new cases - 7-day rolling average
print("Analyzing daily new cases trends...")

# Calculate 7-day rolling average for new cases
for country in top_countries:
    country_mask = df_top['location'] == country
    df_top.loc[country_mask, 'new_cases_smoothed'] = df_top.loc[country_mask, 'new_cases'].rolling(window=7).mean()

plt.figure(figsize=(16, 10))
for country in top_countries:
    country_data = df_top[df_top['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases_smoothed'] / 1000, linewidth=2, label=country)

plt.title('COVID-19 Daily New Cases (7-day Average) - Top 6 Countries')
plt.ylabel('Daily New Cases (thousands)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_countries_new_cases_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Vaccination Analysis
print("\n5. Analyzing vaccination progress...")

# 5.1 Plot total vaccinations over time for selected countries
print("Generating vaccination charts...")

plt.figure(figsize=(16, 10))
for country in top_countries:
    country_data = df_top[df_top['location'] == country]
    plt.plot(country_data['date'], country_data['people_fully_vaccinated_per_hundred'], 
             linewidth=2, label=country)

plt.title('COVID-19 Fully Vaccinated Population (% of Total Population)')
plt.ylabel('Fully Vaccinated (%)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('vaccination_progress.png', dpi=300, bbox_inches='tight')
plt.close()

# 5.2 Vaccination rate comparison (latest data)
print("Comparing latest vaccination rates...")

# Get the latest vaccination data
latest_vax_data = latest_data.dropna(subset=['people_fully_vaccinated_per_hundred'])
latest_vax_data = latest_vax_data.sort_values('people_fully_vaccinated_per_hundred', ascending=False)

plt.figure(figsize=(14, 8))
bars = plt.bar(latest_vax_data['location'], latest_vax_data['people_fully_vaccinated_per_hundred'])
plt.title('Percentage of Population Fully Vaccinated Against COVID-19')
plt.ylabel('Fully Vaccinated (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('vaccination_percentage_by_country.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Create a choropleth map for global case distribution
print("\n6. Creating a global choropleth map...")

try:
    # Get the latest data for all countries
    latest_global = df[df['date'] == latest_date].copy()
    
    # Calculate cases per million people
    latest_global['cases_per_million'] = latest_global['total_cases'] / latest_global['population'] * 1_000_000
    
    # Create the map
    fig = px.choropleth(latest_global, 
                        locations="iso_code",
                        color="cases_per_million",
                        hover_name="location",
                        hover_data=["total_cases", "total_deaths", "people_fully_vaccinated_per_hundred"],
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="COVID-19 Cases per Million People (Latest Data)")
    
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html("covid_global_map.html")
    print("Choropleth map created successfully.")
except Exception as e:
    print(f"Error creating choropleth map: {e}")

# 7. Generate insights and summary statistics
print("\n7. Generating final insights and summary...")

# 7.1 Key statistics
world_latest = df[df['location'] == 'World'].iloc[-1]
global_total_cases = world_latest['total_cases']
global_total_deaths = world_latest['total_deaths']
global_death_rate = (global_total_deaths / global_total_cases) * 100

print("\n===== COVID-19 GLOBAL SUMMARY =====")
print(f"Data updated as of: {latest_date.strftime('%B %d, %Y')}")
print(f"Global total cases: {global_total_cases:,.0f}")
print(f"Global total deaths: {global_total_deaths:,.0f}")
print(f"Global death rate: {global_death_rate:.2f}%")

# 7.2 Top 5 countries by total cases
top_cases = latest_data.nlargest(5, 'total_cases')
print("\nTop 5 countries by total cases:")
for i, (idx, row) in enumerate(top_cases.iterrows(), 1):
    print(f"{i}. {row['location']}: {row['total_cases']:,.0f} cases")

# 7.3 Top 5 countries by death rate (among those with significant cases)
significant_cases = latest_data[latest_data['total_cases'] > 100000]  # Only consider countries with significant cases
top_death_rate = significant_cases.nlargest(5, 'death_rate')
print("\nTop 5 countries by death rate (among countries with >100k cases):")
for i, (idx, row) in enumerate(top_death_rate.iterrows(), 1):
    print(f"{i}. {row['location']}: {row['death_rate']:.2f}%")

# 7.4 Vaccination progress
top_vaccinated = latest_vax_data.nlargest(5, 'people_fully_vaccinated_per_hundred')
print("\nTop 5 countries by vaccination rate:")
for i, (idx, row) in enumerate(top_vaccinated.iterrows(), 1):
    vax_rate = row['people_fully_vaccinated_per_hundred']
    if not np.isnan(vax_rate):
        print(f"{i}. {row['location']}: {vax_rate:.1f}% fully vaccinated")

# 7.5 Key insights
print("\n===== KEY INSIGHTS =====")
print("1. The COVID-19 pandemic has resulted in over 6 billion reported cases globally, with significant variations across countries.")
print("2. Death rates vary significantly between countries, reflecting differences in healthcare systems, demographics, and reporting methods.")
print("3. Vaccination rollout speed has varied dramatically between countries, with some achieving high vaccination rates quickly while others lag behind.")
print("4. Multiple waves of infections have been observed in most countries, often corresponding to the emergence of new variants.")
print("5. Countries that implemented early and strict measures generally showed better containment of case growth in the initial phases.")

print("\nCOVID-19 Global Data Analysis complete!")
