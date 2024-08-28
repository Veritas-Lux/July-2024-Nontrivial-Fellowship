# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#vaccinations clean data

df = pd.read_csv('vaccinations.csv')

df['total_vaccinations'] = df['total_vaccinations'].ffill()
df.dropna(subset=['date'], inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df[df['total_vaccinations'] > 0]

df_filtered = df.groupby('location', as_index=False).first()
df_total = df.groupby('location', as_index=False).first()
#covid clean data

cf = pd.read_csv('full_data.csv')
cf['total_cases'] = cf['total_cases'].replace(0, np.nan)
cf['total_cases'] = cf['total_cases'].ffill()
cf.dropna(subset=['date'], inplace=True)
cf = cf[cf['total_cases'] > 0]
cf_filtered = cf.groupby('location', as_index=False).first()

cf_total = cf.groupby('location', as_index=False).first()

# Convert 'date' column to datetime in df_filtered
df_filtered['date'] = pd.to_datetime(df_filtered['date'])
cf_filtered['date'] = pd.to_datetime(cf_filtered['date'])
cf['date'] = pd.to_datetime(cf['date'])
#cleaning location to merge datasets
cf_filtered.rename(columns = {'Entity':'location'}, inplace = True)
cf_total.rename(columns = {'Entity':'location'}, inplace = True)

#group by location new datasets
gdp_p_capita = pd.read_csv('gdp-per-capita-maddison.csv')
health_spending = pd.read_csv('public-healthcare-spending-share-gdp.csv')
population = pd.read_csv('population-and-demography.csv')

gdp_filtered = gdp_p_capita.groupby('Entity', as_index=False).last()
health_filtered = health_spending.groupby('Entity', as_index=False).last()
population_filtered = population.groupby('Entity', as_index=False).last()

health_filtered.rename(columns = {'Entity':'location'}, inplace = True)
gdp_filtered.rename(columns = {'Entity':'location'}, inplace = True)
population_filtered.rename(columns = {'Entity':'location'}, inplace = True)

# the whole lack date thing
df_merged = pd.merge(df_filtered, cf_filtered, on='location')

df_merged['vaccine delay'] = (df_merged['date_x'] - df_merged['date_y']).dt.days
df_merged['date difference'] = df_merged['vaccine delay']-df_merged['vaccine delay'].min()
df_merged.loc[df_merged['date difference'] <= 0, 'date difference'] = 1


# Group by location and find the first vaccination date
first_vaccination_dates = df_filtered.groupby('location')['date'].min().reset_index()
df_vaccinations_first = pd.merge(df_filtered, first_vaccination_dates, on=['location', 'date'], how='inner')

# Get the COVID-19 cases on the first vaccination date
df_covid_first = pd.merge(cf, first_vaccination_dates, on=['location', 'date'], how='inner')

# relevant columns
df_covid_first = df_covid_first[['location', 'date', 'total_cases']]
df_vaccinations_first = df_vaccinations_first[['location', 'date', 'total_vaccinations']]

# Merge to calculate the "lack"
df_lack = pd.merge(df_covid_first, df_vaccinations_first, on=['location', 'date'])
df_lack['lack'] = df_lack['total_cases'] - df_lack['total_vaccinations']

# Handle negative lack values
df_lack['lack'] = df_lack['lack'].clip(lower=0)
df_lack.loc[df_lack['lack'] <= 0, 'lack'] = 1

merge = pd.merge(df_merged, df_lack, on='location')
merge1 = pd.merge(merge, health_filtered, on='location')
merge2 = pd.merge(merge1, gdp_filtered, on='location')

merge2['spending'] = (merge2['GDP per capita'] * merge2['Domestic general government health expenditure (% of GDP)'])

merge3 = pd.merge(merge2, population_filtered, on='location')

delays = merge3[['location', 'date difference', 'lack', 'spending', 'Population - Sex: all - Age: all - Variant: estimates', 'GDP per capita', 'Domestic general government health expenditure (% of GDP)']].copy()

#clean delays

# Convert columns to numeric
delays['Population - Sex: all - Age: all - Variant: estimates'] = pd.to_numeric(delays['Population - Sex: all - Age: all - Variant: estimates'], errors='coerce')
delays['lack'] = pd.to_numeric(delays['lack'], errors='coerce')
delays['date difference'] = pd.to_numeric(delays['date difference'], errors='coerce')
delays['spending'] = pd.to_numeric(delays['spending'], errors='coerce')

# Drop rows with NaN values in the relevant columns
delays.dropna(subset=['Population - Sex: all - Age: all - Variant: estimates', 'lack', 'date difference', 'spending'], inplace=True)

delays['lack per population'] = (delays['lack'] / delays['Population - Sex: all - Age: all - Variant: estimates'])
delays['urgency'] = ((delays['Population - Sex: all - Age: all - Variant: estimates'] * delays['lack'] * delays['date difference']) / delays['spending']).astype(float)
delays['log_urgency'] = np.log10(delays['urgency'])

delays.sort_values(by=['log_urgency'], ascending = False, inplace = True)

pd.options.display.max_rows = 9999
pd.set_option('display.max_columns', None)

#print(delays)

#correlations
# print(delays.dtypes)
correlation_matrix = delays.corr(numeric_only = True)
# print(correlation_matrix)
correlation_matrix.to_csv('correlations.csv', index=False)

#death correlations
cf_deaths = cf.groupby('location', as_index=False).last()
deathmerge = pd.merge(delays, cf_deaths, on='location')
deathmerge['deaths_cases'] = (deathmerge['total_deaths'] / deathmerge['total_cases'])
correlation_death = deathmerge.corr(numeric_only = True)
#print(correlation_death['deaths_cases'])

#student t test

#developing vs. developed country thing
delays['Economy'] = delays['GDP per capita'].apply(lambda x: 'Developed' if x > 12000 else 'Developing')

# Separate into two DataFrames if needed
developed_countries = delays[delays['Economy'] == 'Developed']
developing_countries = delays[delays['Economy'] == 'Developing']

# print(np.var(developed_countries['date difference']), np.var(developing_countries['date difference']))  #for variances check

# Perform the two sample t-test with equal variances
print(stats.ttest_ind(a=developed_countries['date difference'], b=developing_countries['date difference'], equal_var=True))

#da map
# Load the world map
world = gpd.read_file(r'C:\Users\Luqin\Downloads\ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
pd.set_option('display.max_columns', None)

#change country names

country_mapping = {
    'United States': 'United States of America',
    'Tanzania': 'United Republic of Tanzania',
    'Congo': 'Republic of the Congo',
    'Serbia': 'Republic of Serbia',
    'Bahamas': 'The Bahamas',
    'Democratic Republic of Congo' : 'Democratic Republic of the Congo'
}

delays['location'] = delays['location'].replace(country_mapping)

mapmerge = world.set_index('ADMIN').join(delays.set_index('location'))
# Plot the map with data
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

missing_data_countries = mapmerge[mapmerge['urgency'].isna()]
# print(missing_data_countries)
# Plot the base world map
world.boundary.plot(ax=ax)

# Plot data on top
mapmerge.plot(column='log_urgency', ax=ax, legend=True,
            legend_kwds={'label': "log severity",
                         'orientation': "horizontal"})

plt.title('World COVID vaccine inequality severity map')
#plt.show()

fig.savefig('your_map.png', dpi=300)

#histogram
plt.figure(figsize=(15, 10))

# Histogram for 'delay'
plt.subplot(3, 1, 1)
plt.hist(delays['date difference'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Delay')

# Histogram for 'lack'
plt.subplot(3, 1, 2)
plt.hist(delays['lack per population'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Lack per Population')

# Histogram for 'urgency'
plt.subplot(3, 1, 3)
plt.hist(delays['log_urgency'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Log Urgency')

plt.tight_layout()
#plt.show()

plt.savefig('histograms.png')

delays.to_csv('vaccine_inequality_results.csv', index=False)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
