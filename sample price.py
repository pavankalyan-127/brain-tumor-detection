# import pandas as pd   1...

# # Load the dataset
# df = pd.read_csv("CostCancer.csv")   # adjust filename if different

# # Show first few rows
# print(df.head())

# # Show column names
# print(df.columns)

# # Info about dataset
# print(df.info())

# # Basic stats
# print(df.describe())

# brain_data = df[df['CancerType'].str.contains("Brain", case=False, na=False)]
# print(brain_data.head())
# print(len(brain_data))

# import pandas as pd 2.....

# # Load dataset
# df = pd.read_csv("costcancer.csv")

# # Rename first column for clarity
# df.rename(columns={'Cost of Cancer Care by Phase of Care': 'CancerSite'}, inplace=True)

# # Filter rows where cancer site contains 'Brain'
# brain_df = df[df['CancerSite'].str.contains("Brain", case=False, na=False)]

# print(brain_df.head(10))

# import pandas as pd

# # Load and rename columns
# df = pd.read_csv("costcancer.csv")
# df.rename(columns={'Cost of Cancer Care by Phase of Care': 'CancerSite'}, inplace=True)

# # Filter for Brain cancer
# brain_df = df[df['CancerSite'].str.contains("Brain", case=False, na=False)]

# # Pick relevant columns
# brain_df = brain_df[['CancerSite', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 6']]

# # Rename columns properly
# brain_df.columns = ['CancerSite', 'Year', 'Sex', 'AgeGroup', 'InitialYearCostMillions']

# # Convert cost column to numeric
# brain_df['InitialYearCostMillions'] = pd.to_numeric(brain_df['InitialYearCostMillions'], errors='coerce')

# # Drop any rows where cost could not be converted
# brain_df = brain_df.dropna(subset=['InitialYearCostMillions'])

# # Convert to per-patient cost (assume ~20,000 brain cancer patients/year)
# brain_df['CostPerPatientUSD'] = (brain_df['InitialYearCostMillions'] * 1_000_00) / 200000

# # Convert USD → INR (approx 1 USD = 85 INR)
# brain_df['CostPerPatientINR'] = brain_df['CostPerPatientUSD'] * 85

# print(brain_df.head(10))


# import pandas as pd



# # df = pd.read_csv("costcancer.csv")
# # print(df.columns)   # Show column names
# # print(df.head())    # Show first few rows
# import pandas as pd

# # Load dataset, skipping first 3 junk rows
# df = pd.read_csv("costcancer.csv", skiprows=3)

# # Rename the first column to a clean name
# df.rename(columns={'Cancer Site': 'CancerSite'}, inplace=True)

# # Now this will work
# print(df.head())
# print(df['CancerSite'].unique())


# print(df.columns.tolist())

import pandas as pd

# # Load dataset (skip metadata rows if needed)
# df = pd.read_csv("costcancer.csv", skiprows=3)

# # Clean column names (remove spaces)
# df.columns = df.columns.str.strip()

# # Rename for easier access
# df.rename(columns={
#     'Cancer Site': 'CancerSite',
#     'Initial Year After Diagnosis Cost': 'InitialYearCost',
#     'Continuing Phase Cost': 'ContinuingCost',
#     'Last Year of Life Cost': 'LastYearCost'
# }, inplace=True)

# # Convert to numeric
# for col in ['InitialYearCost', 'ContinuingCost', 'LastYearCost']:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Drop NaNs
# df = df.dropna(subset=['InitialYearCost', 'ContinuingCost', 'LastYearCost'])

# print(df.head())
# print(df['CancerSite'].unique())

# df = pd.read_csv("costcancer.csv")

# # Clean up column names (remove spaces, newlines, etc.)
# df.columns = df.columns.str.strip().str.replace(" ", "")
# print("Cleaned Columns:", df.columns.tolist())


# import pandas as pd

# df = pd.read_csv("costcancer.csv")
# df.columns = df.columns.str.strip().str.replace(" ", "")
# print(df.columns.tolist())


# import pandas as pd

# # Tel# Clean column names: remove spaces, dots, and convert to lowercase
# df.columns = [col.replace(' ', '').replace('.', '').lower() for col in df.columns]

# # Look for a column containing both 'initial' and 'cost'
# possible_cols = [col for col in df.columns if 'initial' in col and 'cost' in col]
# if len(possible_cols) == 0:
#     raise Exception(f"No column containing 'Initial' and 'Cost' found in CSV! Columns found: {df.columns}")
# COST_COLUMN = possible_cols[0]

# # Convert to numeric
# df[COST_COLUMN] = pd.to_numeric(df[COST_COLUMN], errors='coerce')
# print(f"Detected cost column: {COST_COLUMN}")

import pandas as pd

# Load CSV
df = pd.read_csv("costcancer.csv")

# Inspect columns
print("Original Columns:", df.columns.tolist())

# Clean columns: remove spaces, lowercase first letters, remove special chars
df.columns = [col.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
              .replace("/", "").replace(".", "").replace("&", "and") for col in df.columns]

print("Cleaned Columns:", df.columns.tolist())

# Map cleaned column names to standard names for the app
column_mapping = {
    'CostofCancerCarebyPhaseofCare': 'CancerSite',
    'Unnamed1': 'Year',
    'Unnamed2': 'Sex',
    'Unnamed3': 'Age',
    'Unnamed4': 'IncidenceandSurvivalAssumptions',
    'Unnamed5': 'AnnualCostIncreaseAppliedtoinitialandlastphases',
    'Unnamed6': 'TotalCosts',
    'Unnamed7': 'InitialYearCost',
    'Unnamed8': 'ContinuingPhaseCost',
    'Unnamed9': 'LastYearCost'
}

df.rename(columns=column_mapping, inplace=True)

# Check final column names
print("Fixed Columns:", df.columns.tolist())


import pandas as pd

# Load CSV
df = pd.read_csv("costcancer.csv")

# Clean column names
df.columns = [col.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
              .replace("/", "").replace(".", "").replace("&", "and") for col in df.columns]

# Map all columns properly
column_mapping = {
    'CostofCancerCarebyPhaseofCare': 'CancerSite',
    'Unnamed:1': 'Year',
    'Unnamed:2': 'Sex',
    'Unnamed:3': 'Age',
    'Unnamed:4': 'IncidenceandSurvivalAssumptions',
    'Unnamed:5': 'AnnualCostIncreaseAppliedtoinitialandlastphases',
    'Unnamed:6': 'TotalCosts',
    'Unnamed:7': 'InitialYearCost',
    'Unnamed:8': 'ContinuingPhaseCost',
    'Unnamed:9': 'LastYearCost'
}

df.rename(columns=column_mapping, inplace=True)

# Check final column names
print("Final Columns:", df.columns.tolist())
