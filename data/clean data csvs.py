
import pandas as pd
import csv
import glob
import os
import re

# -----------------------------
# Config
# -----------------------------
# Folder containing the files
input_folder = r"C:\Users\jacqueline.pielli\Downloads\crash data"

# Output CSV (pipe-delimited)
output_path = os.path.join(input_folder, "vehicles_combined_limited.csv")

# Base columns to keep
base_columns = [
    "STATE", "ST_CASE", "VEH_NO", "MONTH", "DAY", "HOUR",
    "MAKE", "MODEL", "BODY_TYP", "MOD_YEAR",
    "GVWR_FROM", "GVWR_TO", "TRAV_SP", "VSPD_LIM",
    "ROLLOVER", "FIRE_EXP", "DR_DRINK", "DR_HGT", "DR_WGT", "DEATHS",
    "DEFORMED", "M_HARM", "ADS_PRES", "ADS_LEV", "ADS_ENG",
    "L_STATE", "DR_ZIP", "L_TYPE", "L_STATUS", "L_RESTRI",
    "PREV_ACC", "PREV_SUS1", "PREV_SUS2", "PREV_SUS3",
    "PREV_DWI", "PREV_SPD", "PREV_OTH", "SPEEDREL",
    "VSURCOND", "VTCONT_F"
]

# Exclude NAME versions for STATE and date/time columns
exclude_name_prefixes = ["STATE", "MONTH", "DAY", "HOUR"]

# -----------------------------
# Discover files
# -----------------------------
file_pattern = os.path.join(input_folder, "vehicle_*.csv")
files = glob.glob(file_pattern)

if not files:
    raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

combined_df = pd.DataFrame()

# -----------------------------
# Process each file
# -----------------------------
for file_path in files:
    print(f"Processing: {file_path}")

    # Extract year from filename (e.g., vehicle_2022.csv)
    match = re.search(r'vehicle_(\d{4})', os.path.basename(file_path))
    year = match.group(1) if match else None

    # Read header to determine expected column count
    with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)

    # Fix malformed rows (pad or truncate to header length)
    malformed_rows = []
    fixed_rows = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader, start=2):
            if len(row) != expected_cols:
                malformed_rows.append((i, len(row)))
                if len(row) < expected_cols:
                    row += [''] * (expected_cols - len(row))
                else:
                    row = row[:expected_cols]
            fixed_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(fixed_rows, columns=header)
    # Normalize empty strings to NA
    df.replace('', pd.NA, inplace=True)

    # Filter STATE = 48 (Texas)
    if "STATE" in df.columns:
        df = df[df["STATE"].astype(str).str.strip() == "48"]

    # Add YEAR column
    df["YEAR"] = year

    # Determine columns to keep, including NAME variants (except excluded prefixes)
    cols_to_keep = []
    for col in base_columns:
        if col in df.columns:
            cols_to_keep.append(col)
        name_col = col + "NAME"
        if name_col in df.columns and not any(name_col.startswith(prefix) for prefix in exclude_name_prefixes):
            cols_to_keep.append(name_col)

    # Always include YEAR
    cols_to_keep.append("YEAR")

    # Subset DataFrame to selected columns (only existing ones)
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]

    # Append to combined DataFrame
    combined_df = pd.concat([combined_df, df], ignore_index=True)

    print(f"Year: {year}, Rows after filtering: {len(df)}, Malformed rows: {len(malformed_rows)}")
    print("-" * 50)

# -----------------------------
# Final summary
# -----------------------------
print("✅ All files processed and combined into one DataFrame!")
print(f"Total combined rows: {len(combined_df)}")

# -----------------------------
# Export to pipe-delimited CSV
# -----------------------------
# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write CSV using pipe as delimiter.
# QUOTE_MINIMAL will quote fields only when needed (e.g., if they contain special characters).
combined_df.to_csv(
    output_path,
    sep='|',
    index=False,
    encoding='utf-8',
    na_rep='',
    quoting=csv.QUOTE_MINIMAL
)

print(f"📁 CSV exported to: {output_path}")

# -----------------------------
# Preview
# -----------------------------
print("\nFirst 50 rows of combined dataset:")
print(combined_df.head(50))
