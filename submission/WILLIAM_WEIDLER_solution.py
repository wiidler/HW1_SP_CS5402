import pandas as pd
import ast
import numpy as np
import wfdb
from typing import Dict

"""PLEASE RENAME your solution TO FIRSTNAME_LASTNAME_solution.py"""


# Step-1, part 1
def parse_ptbxl_data() -> pd.DataFrame:
    """
    Use pandas for this task.
    Implement this function to parse the ptbxl samples. It should
    return a dataframe with the filename_lr and diagnostic_class column.
    The filename_lr is the filename of the low-resolution ECG signal, and
    the diagnostic_class is generated by converting the scp_codes
    in the ptbxl database to diagnostic_class in the scp_statements database.
    There are 5 classes in total: [NORM, MI, STTC, CD, HYP].

    Since we want to simulate using the dataset to train a model, the class needs to have
    at least one class, and any empty entries should be removed.

    The end result should look like this:
    ecg_id filename_lr diagnostic_class
    1 records100/xxxxx/xxxxxx_lr  ['HYP']
    2 records100/xxxxx/xxxxxx_lr  ['MI']
    3 records100/xxxxx/xxxxxx_lr  ['MI, STTC']
    ...
    """
    df = pd.read_csv("ptbxl_sample.csv", usecols=["ecg_id", "filename_lr", "scp_codes"])
    resource = pd.read_csv("scp_statements.csv")
    resource.rename(columns={"Unnamed: 0": "scp_code"}, inplace=True)
    resource = resource.dropna(subset=["diagnostic_class"])
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    diagnostic_class = []
    for code in df["scp_codes"]:
        new_append = []
        for i in code.keys():
            match = resource[resource["scp_code"] == i]
            if not match.empty:
                if match["diagnostic_class"].values[0] not in new_append:
                    new_append.append(match["diagnostic_class"].values[0])
        diagnostic_class.append(list(new_append))

    result_df = pd.DataFrame(
        {
            "ecg_id": df["ecg_id"],
            "filename_lr": df["filename_lr"],
            "diagnostic_class": diagnostic_class,
        }
    )
    result_df = result_df[result_df["diagnostic_class"].apply(lambda x: len(x) > 0)]
    return result_df


# Step-1, part 2
def create_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Use numpy and wfdb for this task.
    The data are located in /records100/, and you can read them like this:
    signal, _ = wfdb.rdsamp(filepath), where signal is a numpy array that
    should have shape [signal_length(1000), num_channels(12)].
    Implement this function to create a dataset from the dataframe df, which should be the output of
    parse_ptbxl_data.

    Convert the textual class labels into one-hot encoding. For example, using the label order [NORM, MI, STTC, CD, HYP],
    an ECG signal with labels [HYP, MI, STTC] would be converted to [0, 1, 1, 0, 1].

    Return two numpy arrays:
    - data_x: array should contain the ECG data with shape [num_samples, signal_length(1000), num_channels(12)].
    - data_y: array should contain the labels with shape   [num_samples, num_classes(5)].
    """

    # you can read the raw ECG signal with this function:
    # ECG_signal, _ = wfdb.rdsamp("filename_lr")

    data_x = []
    data_y = []
    for index, row in df.iterrows():
        ECG_signal, _ = wfdb.rdsamp(row["filename_lr"])
        data_x.append(ECG_signal)
        label = [0, 0, 0, 0, 0]
        for i in row["diagnostic_class"]:
            if i == "NORM":
                label[0] = 1
            elif i == "MI":
                label[1] = 1
            elif i == "STTC":
                label[2] = 1
            elif i == "CD":
                label[3] = 1
            elif i == "HYP":
                label[4] = 1
        data_y.append(label)
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    return data_x, data_y


# Step-2:
def data_preprocessing(
    data_x: np.ndarray, data_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform data preprocessing:
    - Check for missing values (or N/A), anomalies, and outliers.
        - Fill missing values with the average of adjacent points in the same channel.
        - Replace outliers using the 97th percentile (np.percentile(x, 97)).
    - Normalize each channel with the equation: (x - xmin)/(xmax - xmin).
        - xmax: represents the maximum value of a channel
        - xmin: represents the minimum value of the channel.
    After normalization, the values will be scaled to range from 0 to 1.
    """
    data_x_normalized = []
    data_y_normalized = []

    for i in range(len(data_x)):
        for j in range(12):  # Assuming 12 channels
            channel_data = data_x[i][:, j]

            # Handle missing values using linear interpolation (if possible)
            if np.isnan(channel_data).any():
                valid_indices = ~np.isnan(channel_data)
                channel_data = np.interp(
                    np.arange(len(channel_data)),
                    np.where(valid_indices)[0],
                    channel_data[valid_indices],
                )

            # Detect outliers using percentiles (3rd and 97th)
            lower_bound = np.percentile(channel_data, 3)
            upper_bound = np.percentile(channel_data, 97)
            channel_data = np.clip(channel_data, lower_bound, upper_bound)

            # Min-max normalization
            min_val, max_val = np.min(channel_data), np.max(channel_data)
            if max_val != min_val:  # Avoid division by zero
                channel_data = (channel_data - min_val) / (max_val - min_val)

            data_x[i][:, j] = channel_data

        data_x_normalized.append(data_x[i])
        data_y_normalized.append(data_y[i])

    return np.array(data_x_normalized), np.array(data_y_normalized)


# Step-3
def split_data(
    data_x: np.ndarray, data_y: np.ndarray
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Implement this function to split the dataset into train, test, and validation sets at a 7:2:1 ratio.
    """
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    train_dataset = {
        "data_x": data_x[: int(len(data_x) * 0.7)],
        "data_y": data_y[: int(len(data_y) * 0.7)],
    }
    val_dataset = {
        "data_x": data_x[int(len(data_x) * 0.7) : int(len(data_x) * 0.9)],
        "data_y": data_y[int(len(data_y) * 0.7) : int(len(data_y) * 0.9)],
    }
    test_dataset = {
        "data_x": data_x[int(len(data_x) * 0.9) :],
        "data_y": data_y[int(len(data_y) * 0.9) :],
    }

    return train_dataset, val_dataset, test_dataset
