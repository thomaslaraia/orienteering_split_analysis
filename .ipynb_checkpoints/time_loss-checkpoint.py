import pandas as pd
import openpyxl as op
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def cumulative_sum(arr):
    cum_sum = []
    total = 0
    for num in arr:
        total += num
        cum_sum.append(total)
    return cum_sum

def calculate_percentage(df, names):

    difference_from_expected = []

    
    for name in names:
        
        # Extract rows with the specified name
        split_row = df[df['Name'] == name+'_split']
        total_row = df[df['Name'] == name+'_total']
        
        if len(split_row) == 0 or len(total_row) == 0:
            return None
        
        split_row = split_row.drop(columns=['Name'])
        total_row = total_row.drop(columns=['Name'])
        
        # Extract splits and totals
        splits = df[df['Name'].str.endswith('_split')].drop(columns=['Name'])
        totals = df[df['Name'].str.endswith('_total')].drop(columns=['Name'])
        
        # Remove the rows with the specified name from the original DataFrame
        remaining_splits = splits[splits.index != split_row.index[0]]
        
        # Calculate the quickest times for splits and totals
        quickest_times_splits = remaining_splits.min()
        quickest_times_totals = cumulative_sum(quickest_times_splits)
        
        # Calculate the percentage behind (or possibly ahead) for split row
        split_percentage = (split_row - quickest_times_splits) / quickest_times_splits * 100
        
        # Calculate the percentage behind (or possibly ahead) for total row
        total_percentage = (total_row - quickest_times_totals) / quickest_times_totals * 100
        
        # # Detect outliers in splits column using Z-score
        # z_scores = np.abs(stats.zscore(split_percentage, axis=1))
        # outlier_indices = split_row.columns[np.where(z_scores > 1.5)[1]]
        # print(outlier_indices)
    
        # Calculate median along the rows
        median_values = np.median(split_percentage, axis=1)
        
        # Calculate median absolute deviation (MAD)
        median_absolute_deviations = np.median(np.abs(split_percentage - median_values[:, np.newaxis]), axis=1)
        
        # Calculate modified Z-scores using MAD
        modified_z_scores = 0.6745 * np.abs(split_percentage - median_values[:, np.newaxis]) / median_absolute_deviations[:, np.newaxis]
        
        # Find outlier indices where modified Z-score is greater than threshold (e.g., 1.5)
        outlier_indices = split_row.columns[np.where(modified_z_scores > 2)[1]]
        
        subtract = 0
    #     print(quickest_times_splits)
        for i in outlier_indices:
            subtract += quickest_times_splits[i]
        
        # Calculate the weights of the legs in the race
        max(quickest_times_totals)
        weights = {}
        for col in split_row.columns:
            if col in outlier_indices:
                weights[col] = 0
            else:
                weights[col] = quickest_times_splits[col] / (max(quickest_times_totals) - subtract)
                
        mean_split_percentage_behind = np.dot(list(weights.values()), split_percentage.values[0])
        
        # Compute the expected split time for each leg
        expected_split_time = quickest_times_splits * (1 + 0.01 * mean_split_percentage_behind)
        
        diff_from_expected = split_row.values[0] - expected_split_time
        
        # Create DataFrame to hold the result
        result_df = pd.DataFrame({
            'Name': split_percentage.columns,
            'Weight': list(weights.values()),
            'Split_Percentage_Behind': split_percentage.values[0],
            'Total_Percentage_Behind': total_percentage.values[0],
            'Split_Time': split_row.values[0],
            'Expected_Split_Time': expected_split_time,
            'Difference_from_Expected': diff_from_expected
        })

            # print(outlier_indices)
        # print(name)
        # plt.scatter(split_percentage.values[0],np.zeros(len(split_percentage.values[0])))
        # plt.scatter(split_percentage[outlier_indices].values[0],np.zeros(len(outlier_indices)))
        # plt.show()

        difference_from_expected.append(result_df['Difference_from_Expected'].tolist())

    # Create a DataFrame with the expected split times
    expected_split_times_df = pd.DataFrame(difference_from_expected, index=names, columns=result_df['Name'])
    expected_split_times_df
    
    return expected_split_times_df

def attackpoint_logged_out(df):
    df = df.rename(columns={'Unnamed: 1': 'Name'})

    df.columns = ['A'] + list(df.columns[:-1])

    # Shift entries in the first row one position to the right
    first_row = df.iloc[0].tolist()[-1:] + df.iloc[0].tolist()[:-1]
    df.iloc[0] = first_row
    
    # # Define subset excluding the first row
    # subset = df.iloc[1:]
    
    # Extracting desired column names
    desired_columns = []
    
    integer_columns = []
    
    for col in df.columns:
        str_col = str(col)
        if str_col.isdigit() or str_col in ['Finish','Name']:
            desired_columns.append(col)
            if str_col.isdigit() or str_col == 'Finish':
                integer_columns.append(col)
    
    # Selecting desired columns
    df = df[desired_columns]
    
    # Drop entries that have NaN in any of the control columns
    df = df.dropna(subset=integer_columns)

    df = df.reset_index(drop=True)
    
    names = []
    
    # Shift 'Name' column downwards for odd rows
    for i in range(1, len(df), 2):
        # Remove '\xa0?' from the end of the name if present
        name = df.at[i, 'Name']
        if name.endswith('\xa0?'):
            name = name[:-3]
        names.append(name)
        df.at[i + 1, 'Name'] = name + '_total'
        df.at[i, 'Name'] = name + '_split'
        
    # Function to split data into time and position columns
    def split_time_position(data):
        if '(' in data:
            return data.split(' (')[0]
        else:
            return data
    
    # Apply function to relevant integer columns
    for col in integer_columns:
        str_col = str(col)
        # Skip the first row and double its value
    #     df.loc[0, col] = split_time_position(df.loc[0, col])
        
        # Apply the function to the rest of the rows
        df[col] = df[col].apply(split_time_position)
    
    df = df.drop(0)
    
    # Function to convert time to seconds
    def time_to_seconds(time_str):
        if '(' in time_str:
            time_str = time_str.split(' (')[0]  # Remove position part
        parts = time_str.split(':')
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + int(part)
        return seconds
    
    # Apply function to relevant integer columns
    for col in integer_columns:
        df[col] = df[col].apply(time_to_seconds)

    return df, names

def attackpoint_logged_in(df):

    df = df.rename(columns={'Unnamed: 2': 'Name'})
    
    # Extracting desired column names
    desired_columns = []
    
    integer_columns = []
    
    for col in df.columns:
        str_col = str(col)
        if str_col.isdigit() or str_col in ['Finish','Name']:
            desired_columns.append(col)
            if str_col.isdigit() or str_col == 'Finish':
                integer_columns.append(col)
    
    # Selecting desired columns
    df = df[desired_columns]
    
    # Drop entries that have NaN in any of the control columns
    df = df.dropna(subset=integer_columns)
    
    df = df.reset_index(drop=True)
    
    names = []
    
    # Shift 'Name' column downwards for odd rows
    for i in range(1, len(df), 2):
        # Remove '\xa0?' from the end of the name if present
        name = df.at[i, 'Name']
        if name.endswith('\xa0?'):
            name = name[:-3]
        names.append(name)
        df.at[i + 1, 'Name'] = name + '_total'
        df.at[i, 'Name'] = name + '_split'
        
    # Function to split data into time and position columns
    def split_time_position(data):
        if '(' in data:
            return data.split(' (')[0]
        else:
            return data
    
    # Apply function to relevant integer columns
    for col in integer_columns:
        str_col = str(col)
        # Skip the first row and double its value
    #     df.loc[0, col] = split_time_position(df.loc[0, col])
        
        # Apply the function to the rest of the rows
        df[col] = df[col].apply(split_time_position)
    
    df = df.drop(0)
    
    # Function to convert time to seconds
    def time_to_seconds(time_str):
        if '(' in time_str:
            time_str = time_str.split(' (')[0]  # Remove position part
        parts = time_str.split(':')
        seconds = 0
        for part in parts:
            seconds = seconds * 60 + int(part)
        return seconds
    
    # Apply function to relevant integer columns
    for col in integer_columns:
        df[col] = df[col].apply(time_to_seconds)

    return df, names

def winsplits(df):

    # Extracting desired column names
    desired_columns = []
    
    for col in df.columns:
        str_col = str(col).replace(u'\xa0',u' ')
        if str_col == 'Name' or 'leg tot' in str_col:
            desired_columns.append(col)
    
    #Selecting desired columns
    df = df[desired_columns]
    
    df = df.reset_index(drop=True).drop(0)
    
    df = df.dropna()
    
    for i, col in enumerate(df.columns[1:]):
        df = df.rename(columns={col: str(i+1)})
    df = df.rename(columns={df.columns[-1]: 'Finish'})

    names = []
    
    # Shift 'Name' column downwards for odd rows
    for i in range(1, len(df), 2):
        # Remove '\xa0?' from the end of the name if present
        name = df.at[i, 'Name']
        if name.endswith('\xa0?'):
            name = name[:-3]
        names.append(name)
        df.at[i + 1, 'Name'] = name + '_total'
        df.at[i, 'Name'] = name + '_split'
    
    # Function to convert time to seconds
    def time_to_seconds(time_str):
        time_str = str(time_str)
        seconds = 0
        if ':' not in time_str:
            parts = time_str.split('.')
            for i, part in enumerate(parts):
                if len(parts) == 1:
                    seconds = int(part)*60
                    return seconds
                elif i == 1:
                    if len(part) == 1:
                        part = int(part)*10
                seconds = seconds*60 + int(part)
        else:
            parts = time_str.split('.')
            for i, part in enumerate(parts):
                if i == 0:
                    Parts = part.split(':')
                    for Part in Parts:
                        seconds = seconds*60 + int(Part)
                if i == 1:
                    seconds = seconds*60 + int(part[0:2])
                        
        return seconds
    
    # Apply function to relevant integer columns
    for col in df.columns[1:]:
        df[col] = df[col].apply(time_to_seconds)
    
    return df, names

def sientries(df):

    desired_columns = []
    
    for col in df.columns:
        str_col = str(col)
        if str_col == 'Name' or 'LT' in str_col:
            desired_columns.append(col)
    
    df = df[desired_columns]
    
    df = df.reset_index(drop=True).drop(0)
    
    names = []

    df = df.dropna()
    
    # Shift 'Name' column downwards for odd rows
    for i in range(1, len(df), 2):
        # Remove '\xa0?' from the end of the name if present
        name = df.at[i, 'Name']
        if name.endswith('\xa0?'):
            name = name[:-3]
        names.append(name)
        df.at[i + 1, 'Name'] = name + '_total'
        df.at[i, 'Name'] = name + '_split'
    
    for i, col in enumerate(df.columns[1:]):
        df = df.rename(columns={col: str(i+1)})
    df = df.rename(columns={df.columns[-1]: 'Finish'})
    
    def time_to_seconds(time_str):
        time_str = str(time_str)
        seconds = 0
        if '-' not in time_str:
            parts = time_str.split(':')
            for i, part in enumerate(parts[:-1]):
                seconds = seconds*60 + int(part)
        else:
            parts = time_str.split(' ')
            for i, part in enumerate(parts):
                if i == 0:
                    Part = part.split('-')[-1]
                    seconds_ = int(Part)*24*60
                if i == 1:
                    Parts = part.split(':')[:-1]
                    for Part in Parts:
                        seconds = seconds*60 + int(Part)
            seconds = seconds_ + seconds
    
        return seconds
    
    # Apply function to relevant integer columns
    for col in df.columns[1:]:
        df[col] = df[col].apply(time_to_seconds)

    return df, names

def seconds_to_minutes(time):
    if time < 0:
        minutes = int(np.ceil(time/60))
        seconds = int(60 - np.round((time/60-np.floor(time/60))*60))
        if seconds == 60:
            seconds = 0
            minutes -= 1
    else:
        minutes = int(np.floor(time/60))
        seconds = int(np.round((time/60-np.floor(time/60))*60))
        if seconds == 60:
            seconds = 0
            minutes += 1
    
    if np.abs(seconds) < 10:
        if time > 0 or seconds == 0 or minutes != 0:
            time_str = f"{minutes}:0{seconds}"
        else:
            time_str = f"-{minutes}:0{seconds}"
    else:
        if time > 0 or seconds == 0 or minutes != 0:
            time_str = f"{minutes}:{seconds}"
        else:
            time_str = f"-{minutes}:{seconds}"
    return time_str

def excel_to_diff(excel_path):
    df = pd.read_excel(excel_path)

    if 'Unnamed: 1' in df.columns:
        df, names = attackpoint_logged_out(df)
    elif 'Unnamed: 0' in df.columns:
        df, names = attackpoint_logged_in(df)
    elif 'leg\xa0tot' in df.columns:
        df, names = winsplits(df)
    elif 'No.' in df.columns:
        df, names = sientries(df)
    
    df = calculate_percentage(df, names)

    # Apply function to relevant integer columns
    for col in df.columns[:]:
        df[col] = df[col].apply(seconds_to_minutes)
    
    return df

excel_path = input("File Path: ")
df = excel_to_diff(excel_path)
print(df)
