import pandas as pd

# dataset's null values arent recognised, so we explicitly tell it
missing_vals = ["NA", "NaN", "nan", " ", "", "-"]
df = pd.read_csv("data/raw/dly3904.csv", na_values=missing_vals)

# remove rows with missing values
df = df.dropna()

# remove indicator columns
no_indicators = [col for col in df.columns if not col.startswith("ind")]
print(no_indicators)
df = df[no_indicators]

# convert date column to datetime objects and index
df['date'] = pd.to_datetime(df['date'], format="%d-%b-%Y", errors="coerce")
df = df.sort_values('date')
df = df.set_index('date')

# build full daily range from first to last date
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

# reindex to include all days (this creates new rows with NaN if a whole date was missing)
df = df.reindex(full_range)
df.index.name = "date"

# best choice is to interpolate to fill missing values smoothly
df = df.interpolate(method='time')

# rainfall should not be interpolated, safer to fill missing with 0
if 'rain' in df.columns:
    df['rain'] = df['rain'].fillna(0)

# examine rows
#df.info()

# rename columns to better understand features
rename_map = {
    "maxtp": "max_temp",                   # °C
    "mintp": "min_temp",                   # °C
    "igmin": "grass_min_temp",             # °C
    "gmin": "ground_min_temp",             # °C
    "rain": "rainfall",                    # mm
    "cbl": "mean_sea_level_pressure",      # hPa
    "wdsp": "wind_speed",                  # knots
    "hm": "humidity",                      # %
    "ddhm": "wind_direction",              # degrees
    "hg": "wind_gust",                     # knots
    "sun": "sunshine_duration",            # hours
    "dos": "day_of_season",                # index
    "soil": "soil_temperature",            # °C
    "pe": "potential_evapotranspiration",  # mm
    "evap": "evaporation",                 # mm
    "smd_wd": "soil_moisture_wet",         # deficit (mm)
    "smd_md": "soil_moisture_moderate",    # deficit (mm)
    "smd_pd": "soil_moisture_dry"          # deficit (mm)
}
df = df.rename(columns=rename_map)
#print(df.columns)

df.to_csv("data/cleaned/daily_data.csv")