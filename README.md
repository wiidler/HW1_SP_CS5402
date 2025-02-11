Provided:
- ptbxl_database.csv
- scp_statements.csv
- autograder.py
- submission
   - xxx_solution.py
   - ...

Instructions:
The PTB-XL database is a collection of ECG recordings and their corresponding annotations. We only use the metadata from the database in this assignment.

This assignment simulates a real-world scenario where you are given a dataset and a task to train a model with the dataset to predict the diagnostic_class of a given data entry.

In the `submission/xxx_solution.py` file, rename the file to `firstname_lastname_solution.py` and implement the required functions which should follow the requirements below.

The autograder will check if your implementation matches the requirements. You can check your output by running the `autograder.py` file, but please do not change the autograder.
You only need to submit the `firstname_lastname_solution.py` file. Remember to rename the file for submission.

Requirements:

1. Data Parsing (25 points):
- Implement `parse_ptbxl_data()` to process the ptbxl_database.csv and scp_statements.csv files
- Return a dataframe with the following columns:
  - filename_lr: str
  - diagnostic_class: list[str]
- The diagnostic_class should be one of: [NORM, MI, STTC, CD, HYP]
- Each entry must have at least one diagnostic class
- Remove entries with empty diagnostic_class lists

2. Dataset Creation (25 points):
- Implement `create_dataset()` to load ECG data and convert labels
- Read ECG signals from /records100/ using wfdb
- Convert text labels to one-hot encoding
- Return:
  - data_x: numpy array of shape [num_samples, 1000, 12]
  - data_y: numpy array of shape [num_samples, 5]

3. Data Preprocessing (25 points):
- Implement `data_preprocessing()` to clean and normalize the data
- Handle missing values using adjacent point averages
- Replace outliers using 97th percentile(from both ends, so 3% and 97% percentiles), clip the values to the percentile
- Normalize each channel to range [0,1]

4. Data Splitting (25 points):
- Implement `split_data()` to divide the dataset
- Split ratio: 70% training, 20% validation, 10% testing
- Return three dictionaries containing 'data_x' and 'data_y' for each split
