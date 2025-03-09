import pandas as pd
import numpy as np
import re
import ast

def convert_values(values, value_type):
    """
    Converts values of a certain type from strings to floats.

    Args:
        values: List of values to convert. Can be strings or numbers.
        value_type: String indicating the type of value to convert. Can be "temperature", "time" or "humidity".

    Returns:
        List of converted values. If a value can't be converted, it is replaced with None.
    """
    def convert_temperature(value):
        """
        Converts a temperature value from string to float.

        Args:
            value: The temperature value to convert. Can be a string or a number.

        Returns:
            The converted temperature value as a float, or None if the value can't be converted.
        """
        if isinstance(value, (int, float)):
            return value
        value = str(value).strip().lower()
        
        if '-' in value:
            try:
                parts = value.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return None
        
        if '±' in value:
            try:
                num = float(value.split('±')[0].strip())
                return num
            except ValueError:
                return None
        
        if 'room temperature' in value or 'ambient' in value:
            return 25.0
        
        temp_match = re.match(r"(\d+)(°[cfk]+)?", value)
        if temp_match:
            return float(temp_match.group(1))
        
        return None

    def convert_time(value):
        """
        Converts time values from strings to floats.

        Args:
            value: The time value to convert. Can be a string or a number.

        Returns:
            The converted time value as a float, or None if the value can't be converted.
        """
        if isinstance(value, (int, float)):
            return value
        value = str(value).strip().lower()

        if 'h' in value or 'hour' in value:
            try:
                return float(value.replace('h', '').replace('hour', '').strip())
            except ValueError:
                return None
        
        if 'days' in value:
            try:
                return float(value.replace('days', '').strip()) * 24
            except ValueError:
                return None
        if 'weeks' in value:
            try:
                return float(value.replace('weeks', '').strip()) * 7 * 24
            except ValueError:
                return None
        if 'months' in value:
            try:
                return float(value.replace('months', '').strip()) * 30 * 24
            except ValueError:
                return None
        if 'min' in value or 'minute' in value:
            try:
                return float(value.replace('min', '').replace('minute', '').strip()) / 60
            except ValueError:
                return None
        if 's' in value or 'second' in value:
            try:
                return float(value.replace('s', '').replace('second', '').strip()) / 3600
            except ValueError:
                return None

        return None

    def convert_humidity(value):
        """
        Convert a humidity value to a standard format (float between 0 and 100).

        Args:
            value: The humidity value to convert. Can be a string or a number.

        Returns:
            The converted humidity value as a float, or None if the value can't be converted.
        """
        if isinstance(value, (int, float)):
            return value
        value = str(value).strip().lower()

        # Handle ranges like "30-40%" and "20-30% RH"
        if '-' in value and '%' in value:
            try:
                parts = value.split('-')
                return (float(parts[0].replace('%', '').strip()) + float(parts[1].replace('%', '').strip())) / 2
            except ValueError:
                return None
        
        # Handle values with ± like "50 ± 5%" or "70 ± 5% RH"
        if '±' in value:
            try:
                num = float(value.split('±')[0].replace('%', '').strip())
                return num
            except ValueError:
                return None

        # Handle percentages like "45%" or "80 %"
        if '%' in value:
            try:
                return float(value.replace('%', '').strip())
            except ValueError:
                return None

        # Handle specific strings like "ambient", "dry", "high", "low"
        if 'ambient' in value:
            return 52.0  # Arbitrary average value for ambient humidity
        if 'dry' in value:
            return 0.1  # Dry air, 0% RH
        if 'high' in value:
            return 72.0  # High humidity, arbitrary value like 70%
        if 'low' in value:
            return 22.0  # Low humidity, arbitrary value like 20%
        if 'n2' in value:  # For N2 atmosphere or N2 glove box
            return 0.2  # Typically low humidity in inert atmospheres

        # Handle descriptive strings like "moist air with 70 ± 5% RH"
        if 'rh' in value:
            try:
                num = re.search(r"(\d+(\.\d+)?)", value)
                if num:
                    return float(num.group(1))
            except ValueError:
                return None
        
        return None

    def process_values(values, value_type):
        """
        Process a list of values and convert them to a standard format based on the type of value.

        Args:
            values: List of values to convert. Can be strings or numbers.
            value_type: String indicating the type of value to convert. Can be "temperature", "time" or "humidity".

        Returns:
            List of converted values. If a value can't be converted, it is replaced with None.
        """
        if value_type == "temperature":
            return [convert_temperature(v) for v in values]
        elif value_type == "time":
            return [convert_time(v) for v in values]
        elif value_type == "humidity":
            return [convert_humidity(v) for v in values]
        else:
            return []

    return process_values(values, value_type)


def clean_percentage(value):
    if pd.isnull(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).lower().strip()

    # Handle 'null', 'n/a', etc.
    if value in ['null', 'n/a', 'none', '-', 'na']:
        return np.nan

    # Handle ranges like '80-90%'
    range_match = re.match(r'(\d+)\s*[-–]\s*(\d+)', value)
    if range_match:
        nums = [float(range_match.group(1)), float(range_match.group(2))]
        return np.mean(nums)

    # Handle '70 ± 5%'
    plus_minus_match = re.match(r'(\d+)\s*±\s*(\d+)', value)
    if plus_minus_match:
        return float(plus_minus_match.group(1))

    # Extract single number (with or without %)
    num_match = re.search(r'(\d+\.?\d*)', value)
    if num_match:
        return float(num_match.group(1))

    return np.nan

def expand_rows_by_tests(df):
    # Create a list to store the expanded rows
    expanded_rows = []
    
    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        # Iterate through test columns
        for i in range(1, 11):  # Assuming test_1 to test_10
            test_column = f'test_{i}'
            test_data = row[test_column]
            if pd.notnull(test_data):
                try:
                    test_list = ast.literal_eval(test_data)
                    if isinstance(test_list, dict):
                        test_list = [test_list]
                    for test_dict in test_list:
                        new_row = row.drop([f'test_{j}' for j in range(1, 11)])
                        # Add test details to the new row
                        for key, value in test_dict.items():
                            new_row[key] = value
                        expanded_rows.append(new_row)
                except (ValueError, SyntaxError):
                    print(f"Could not parse test data: {test_data}")

    # Create a new dataframe from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df

def clean_and_merge_db(extraction_path, annotation_expanded_path, output_path):
    """
    Clean and merge the annotation and extraction data into a single csv file.

    Args:
        extraction_path (str): Path to the extraction data csv file.
        annotation_expanded_path (str): Path to the expanded annotation data csv file.
        output_path (str): Path to the output csv file.

    Returns:
        None
    """
    annotated_data = pd.read_csv(annotation_expanded_path)
    extracted_data = pd.read_csv(extraction_path)
    extracted_data = expand_rows_by_tests(extracted_data)
    extracted_data['retained_percentage_cont'] = extracted_data['retained_percentage_cont'].apply(clean_percentage)
    extracted_data['retained_percentage_tret'] = extracted_data['retained_percentage_tret'].apply(clean_percentage)

    for col in ['temperature', 'time', 'humidity']:
        extracted_data[col] = extracted_data[col].apply(lambda x: convert_values([x], col)[0])   

    extracted_data = extracted_data.rename(columns={
    'retained_percentage_cont': 'efficiency_cont',
    'retained_percentage_tret': 'efficiency_tret'
    })

    common_columns = annotated_data.columns.intersection(extracted_data.columns)

    df_subset = extracted_data[common_columns]

    df_combined = pd.concat([annotated_data, df_subset], ignore_index=True)
    df_combined.to_csv(output_path, index=False)
