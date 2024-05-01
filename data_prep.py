import pandas as pd
import numpy as np

# read the csv file and rename it.
def ReadCSV(dataset_path):
    data = pd.read_csv(dataset_path, index_col='date', parse_dates=['date'])
    data.columns = ['P', 'G', 'L']
    return data


# handling missing values by interpolation
def MissingValuesImp(dataframe):

    dataframe.sort_index(inplace=True)
    
    interpolated_df = dataframe.interpolate(method='linear')
    
    return interpolated_df

# create a sequenced dataset
def CreateSequencedDataset(data, n_steps):
    sequenced_data = []
    sequenced_index = []

    for i in range(n_steps, len(data)):
        p_values = data['P'].iloc[i-n_steps:i].values # from P(t-n_steps) to P(t-1)

        g_value = data['G'].iloc[i-1] # for G(t-1) fetaure
        l_value = data['L'].iloc[i-1] # for L(t-1) feature
        p_target = data['P'].iloc[i] # for P(t) feature

        sequenced_data.append(np.hstack((p_values, g_value, l_value, p_target)))
        sequenced_index.append(data.index[i])

    col_names = [f'P(t-{n_steps-i})' for i in range(n_steps)] + ['G(t-1)', 'L(t-1)', 'P(t)']
    sequenced_df = pd.DataFrame(sequenced_data, columns=col_names, index=sequenced_index)

    return sequenced_df


# Maybe some calenderical feature will be considered
def AddCalendarFeatures(data):

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Extract calendar features
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute

    # One-hot encode features
    data = pd.get_dummies(data, columns=['hour', 'minute'], drop_first=False)

    return data


# Pipeline of data_processing
def DataProcessing(market_name, n_steps):
    # Read CSV file
    df = ReadCSV(f'dataset_{market_name}.csv')
    
    # Handle missing values
    df_imp = MissingValuesImp(df)
    
    # Create sequenced dataset
    sequenced_df = CreateSequencedDataset(df_imp, n_steps)
    
    return sequenced_df


# For testing
if __name__ == "__main__":
    n_steps = 12
    market_name = 'ieso'  # or 'pjm', 'ieso'

    final_data = DataProcessing(market_name, n_steps)
    print(final_data)