import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from src.entities import FeatureDefinition, EnvironmentConfiguration


def export_unique_values_to_excel(df, environment_configuration= EnvironmentConfiguration):
    """
    Export unique values from each column of a DataFrame to separate sheets in an Excel file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to process
    output_file : str
        Name of the output Excel file (default: 'unique_values.xlsx')
    """
    # Create a new workbook
    wb = Workbook()
    
    # Remove the default sheet
    wb.remove(wb.active)
    
    # Process each column
    for column in df.columns:
        # Create a new sheet for each column
        # Replace invalid characters in sheet name
        sheet_name = str(column)[:31]  # Excel sheet names limited to 31 chars
        sheet_name = "".join(c if c not in r'\/:*?"<>|' else '_' for c in sheet_name)
        ws = wb.create_sheet(title=sheet_name)
        
        # Get unique values and sort them if possible
        unique_values = df[column].unique()
        try:
            unique_values.sort()
        except:
            pass  # If values can't be sorted (mixed types), keep original order
        
        # Convert unique values to DataFrame for easy export
        unique_df = pd.DataFrame(unique_values, columns=[f'Unique values in {column}'])
        
        # Write the values to the sheet
        for r in dataframe_to_rows(unique_df, index=False, header=True):
            ws.append(r)
            
        # Adjust column width
        for column_cells in ws.columns:
            max_length = 0
            column = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws.column_dimensions[column].width = adjusted_width
    
    # Save the workbook
    wb.save(environment_configuration.unique_values_spreadsheet)
    print(f"Excel file created successfully: {environment_configuration.unique_values_spreadsheet}")
