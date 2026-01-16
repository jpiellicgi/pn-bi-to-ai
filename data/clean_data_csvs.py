import pandas as pd
import csv
import glob
import os
import re

# -----------------------------
# Configurations for each dataset
# -----------------------------
datasets = {
    "vehicle": {
        "pattern": "*ehicle_*.csv",
        "output": "vehicle_2017to2023.csv",
        "base_columns": [
            "BODY_TYP","BODY_TYPNAME","DAY","DEATHS","DEFORMED","DR_ZIP",
            "HARM_EVNAME","HOUR","L_RESTRI","L_STATE","L_STATUS","L_TYPE","LAST_MO",
            "LAST_YR","MAK_MOD","MAK_MODNAME","MAKE","MAKENAME","MOD_YEAR","MODEL",
            "MONTH","PREV_ACC","PREV_DWI","PREV_OTH","PREV_SPD","PREV_SUS1","PREV_SUS2",
            "PREV_SUS3","SPEEDREL","ST_CASE","VEH_NO", "STATE"
        ],
        "filter": lambda df: df[df["STATE"].astype(str).str.strip() == "48"] if "STATE" in df.columns else df
    },
    "person": {
        "pattern": "*erson_*.csv",
        "output": "person_2017to2023.csv",
        "base_columns": [
            "AGE", "DRINKING", "REST_MIS", "REST_USE", "SEX", "ST_CASE", "VEH_NO", "STATE", "PER_TYPE"
        ],
        "filter": lambda df: df[df["PER_TYPE"].astype(str).str.strip() == "1"] if "PER_TYPE" in df.columns else df
     },
    "factor": {
        "pattern": "*actor_*.csv",
        "output": "factor_2017to2023.csv",
        "base_columns": [
            "AOI1","MFACTOR","MFACTORNAME","ST_CASE","VEH_NO","VEHICLECC","STATE"
        ],
        "filter": lambda df: df[df["STATE"].astype(str).str.strip() == "48"] if "STATE" in df.columns else df
    },
    "cevent": {
        "pattern": "*vent_*.csv",
        "output": "cevent_2017to2023.csv",
        "base_columns": [
            "AOI1","AOI1NAME","AOI2","AOI2NAME","EVENTNUM","SOE","SOENAME","ST_CASE", "STATE"
        ],
        "filter": lambda df: df[df["STATE"].astype(str).str.strip() == "48"] if "STATE" in df.columns else df
    },
    "accident": {
        "pattern": "*ccident_*.csv",
        "output": "accident_2017to2023.csv",
        "base_columns": [
            "CITY", "CITYNAME", "COUNTY", "COUNTYNAME", "DAY", "DAYNAME", "HARM_EVNAME", 
            "HOURNAME", "LATITUDE", "LATITUDENAME", "LGT_CONDNAME", "LONGITUD", "LONGITUDNAME", 
            "MONTH", "MONTHNAME", "REL_ROADNAME", "ROUTENAME", "RUR_URBNAME", "ST_CASE", 
            "STATE", "STATENAME", "WEATHERNAME", "PEDS", "PERNOTMVIT", "VE_TOTAL", "PVH_INVL", 
            "PERSONS", "DAY_WEEKNAME", "HOUR", "MINUTE", "TWAY_ID", "FUNC_SYSNAME", 
            "RD_OWNERNAME", "SP_JURNAME", "MAN_COLLNAME", "RELJCT2NAME", "TYP_INTNAME", 
            "WRK_ZONENAME", "RAILNAME", "HOSP_MNNAME", "FATALS"
        ],
        "filter": lambda df: df[df["STATE"].astype(str).str.strip() == "48"] if "STATE" in df.columns else df
    }
}

# Folder containing the files
input_folder = r"C:\Users\jacqueline.pielli\Downloads\crash data"

# -----------------------------
# Function to process one dataset
# -----------------------------
def process_dataset(name, config):
    print(f"\n🔍 Processing dataset: {name}")
    file_pattern = os.path.join(input_folder, config["pattern"])
    files = glob.glob(file_pattern)

    if not files:
        print(f"⚠ No files found for pattern: {file_pattern}")
        return

    combined_df = pd.DataFrame()

    for file_path in files:
        print(f"Processing: {file_path}")
        match = re.search(r'_(\d{4})', os.path.basename(file_path))  # Extract year from filename
        year = match.group(1) if match else None

        # Read header
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [h.strip().lstrip('\ufeff') for h in header]
            expected_cols = len(header)

        # Fix malformed rows
        fixed_rows = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < expected_cols:
                    row += [''] * (expected_cols - len(row))
                elif len(row) > expected_cols:
                    row = row[:expected_cols]
                fixed_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(fixed_rows, columns=header)
        df.replace('', pd.NA, inplace=True)

        # Apply dataset-specific filter
        # df = config"filter"

        # Add YEAR column
        df["YEAR"] = year

        
        # Normalize PER_TYPE name for person files
        if name == "person":
            if "PER_TYPE" not in df.columns and "PER_TYP" in df.columns:
                df.rename(columns={"PER_TYP": "PER_TYPE"}, inplace=True)

        # Determine columns to keep (base + YEAR)
        cols_to_keep = config["base_columns"] + ["YEAR"]
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[existing_cols]

        # Append to combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        
        # Apply export filter if provided
        export_filter = config.get("filter", lambda d: d)
        final_df = export_filter(combined_df)


    # Export combined file
    output_path = os.path.join(input_folder, config["output"])
    final_df.to_csv(output_path, sep='|', index=False, encoding='utf-8', na_rep='', quoting=csv.QUOTE_MINIMAL)

    print(f"✅ Exported {name} dataset to: {output_path}")
    print(f"Total rows: {len(final_df)}")

# -----------------------------
# Run for all datasets
# -----------------------------
for dataset_name, config in datasets.items():
    process_dataset(dataset_name, config)
