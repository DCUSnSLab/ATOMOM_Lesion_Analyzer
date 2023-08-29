import pandas as pd
import os
# Load the Excel file
path = '~/ATOMOM_Lesion_Analyzer/tmp/'
path = os.path.expanduser(path)

excel_file_path = 'exp_result.xlsx'  # Replace with the path to your Excel file
excel_file_path = os.path.join(path,excel_file_path)

xls = pd.ExcelFile(excel_file_path)

# Get the sheet names
sheet_names = xls.sheet_names

# Iterate through each sheet and save as CSV
for sheet_name in sheet_names:
	# Read the sheet into a DataFrame
	df = pd.read_excel(xls, sheet_name)

	# Save the DataFrame as a CSV file
	csv_file_name = f'{sheet_name}.csv'
	df.to_csv(os.path.join(path,csv_file_name), index=False)

print("CSV files saved successfully.")
