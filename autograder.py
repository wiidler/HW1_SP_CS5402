import pandas as pd
import numpy as np
import glob
import importlib.util
import ast
import os
import wfdb
import pickle as pkl

"""You can run this file to check your solution. 
   The results will be written to results.txt.
"""
SUBMISSION_FOLDER = "submission"
RESULTS_FILE = "results.txt"


def load_solution(solution_file):
    spec = importlib.util.spec_from_file_location("solution", solution_file)
    solution_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution_module)

    return (
        solution_module.parse_ptbxl_data,
        solution_module.create_dataset,
        solution_module.data_preprocessing,
        solution_module.split_data,
    )


def check_parsing(result_df):
    """Check Task 1: Data parsing"""
    messages = []
    task1_score = 25

    if not isinstance(result_df, pd.DataFrame):
        task1_score -= 25
        messages.append("Task 1: Output should be a pandas DataFrame")

    expected_columns = ["filename_lr", "diagnostic_class"]
    if not all(col in result_df.columns for col in expected_columns):
        task1_score -= 5
        messages.append(f"Task 1: DataFrame should contain columns: {expected_columns}")

    if not (result_df["diagnostic_class"].map(len) > 0).all():
        task1_score -= 5
        messages.append(
            "Task 1: All entries should have non-empty diagnostic_class, you should remove the empty entries."
        )

    # Updated expected counts based on the sample data
    expected_counts = {"NORM": 64, "MI": 124, "STTC": 125, "CD": 105, "HYP": 100}
    actual_counts = {"NORM": 0, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0}
    for class_list in result_df["diagnostic_class"]:
        for diag_class in class_list:
            if diag_class in actual_counts:
                actual_counts[diag_class] += 1

    if actual_counts != expected_counts:
        task1_score -= 10
        messages.append(
            f"Task 1: Class counts don't match. Expected: {expected_counts}, Got: {actual_counts}"
        )

    return task1_score, messages


def check_dataset_creation(X, y):
    """Check Task 2: Dataset creation"""
    messages = []
    task2_score = 25

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        task2_score -= 25
        messages.append("Task 2: X and y should be numpy arrays")

    if X.shape != (300, 1000, 12):
        task2_score -= 7.5
        messages.append(f"Task 2: X shape should be (300, 1000, 12), got {X.shape}")

    if y.shape != (300, 5):
        task2_score -= 7.5
        messages.append(f"Task 2: y shape should be (300, 5), got {y.shape}")

    if not np.any(y.sum(axis=0) > 0):
        task2_score -= 5
        messages.append("Task 2: y should have at least one positive label per class")

    return task2_score, messages


def check_preprocessing(X_processed, y):
    """Check Task 3: Data preprocessing"""
    messages = []
    task3_score = 25

    if not isinstance(X_processed, np.ndarray):
        task3_score -= 25
        messages.append("Task 3: Processed X should be a numpy array")
        return task3_score, messages

    if np.min(X_processed) < 0 or np.max(X_processed) > 1:
        task3_score -= 7.5
        messages.append("Task 3: Values should be normalized between 0 and 1")

    if np.any(np.isnan(X_processed)):
        task3_score -= 7.5
        messages.append("Task 3: There should be no NaN values in processed data")

    if X_processed.shape != (300, 1000, 12):
        task3_score -= 5
        messages.append(
            f"Task 3: Shape should be (300, 1000, 12), got {X_processed.shape}"
        )

    # Check mean values for each sample and channel
    MIN_ACCEPTABLE_MEAN = 0.09
    MAX_ACCEPTABLE_MEAN = 1 - MIN_ACCEPTABLE_MEAN
    sample_channel_means = np.mean(X_processed, axis=1)  # Shape: (300, 12)
    problematic_samples = []

    for sample_idx in range(X_processed.shape[0]):
        for channel_idx in range(X_processed.shape[2]):
            mean = sample_channel_means[sample_idx, channel_idx]
            if mean < MIN_ACCEPTABLE_MEAN or mean > MAX_ACCEPTABLE_MEAN:
                problematic_samples.append((sample_idx, channel_idx))

    if problematic_samples:
        task3_score -= 3
        messages.append(
            f"Task 3: Found {len(problematic_samples)} sample-channel combinations with mean values outside "
            f"[{MIN_ACCEPTABLE_MEAN}, {MAX_ACCEPTABLE_MEAN}]. This may imply that the outliers are not removed properly."
        )
        # Optionally print first few problematic samples
        if len(problematic_samples) > 0:
            sample_msg = ", ".join(
                [f"(sample {s}, channel {c})" for s, c in problematic_samples[:3]]
            )
            messages.append(f"Task 3: Examples of problematic means: {sample_msg}")

    return task3_score, messages


def check_data_splitting(train_data, val_data, test_data):
    """Check Task 4: Data splitting"""
    messages = []
    task4_score = 25

    expected_keys = {"data_x", "data_y"}
    for dataset, name in [
        (train_data, "train"),
        (val_data, "val"),
        (test_data, "test"),
    ]:
        if not isinstance(dataset, dict):
            task4_score -= 25
            messages.append(f"Task 4: {name}_data should be a dictionary")
            continue
        if not all(key in dataset for key in expected_keys):
            task4_score -= 25
            messages.append(
                f"Task 4: {name}_data should contain keys 'data_x' and 'data_y'"
            )

    total_samples = (
        len(train_data["data_x"]) + len(test_data["data_x"]) + len(val_data["data_x"])
    )
    expected_train = int(0.7 * total_samples)
    expected_test = int(0.1 * total_samples)
    expected_val = total_samples - expected_train - expected_test

    if (
        len(train_data["data_x"]) != expected_train
        or len(test_data["data_x"]) != expected_test
        or len(val_data["data_x"]) != expected_val
    ):
        task4_score -= 10
        messages.append(
            "Task 4: Incorrect split ratios. Should be 7:2:1 for train:valid:test"
        )

    return task4_score, messages


def check_solution(solution_file):
    """Main function to check all tasks"""
    # Load student solutions
    parse_ptbxl_data, create_dataset, data_preprocessing, split_data = load_solution(
        solution_file
    )
    score = 0
    max_score = 100
    all_messages = []

    # Task 1: Check data parsing
    try:
        result_df = parse_ptbxl_data()
        task1_score, messages = check_parsing(result_df)
        score += task1_score
        all_messages.extend(messages)
    except Exception as e:
        all_messages.append(f"Task 1 Error: {str(e)}")

    # Task 2: Check dataset creation
    try:
        X, y = create_dataset(result_df)
        task2_score, messages = check_dataset_creation(X, y)
        score += task2_score
        all_messages.extend(messages)
    except Exception as e:
        all_messages.append(f"Task 2 Error: {str(e)}")

    # Task 3: Check preprocessing
    try:
        X_processed, y_processed = data_preprocessing(X, y)
        task3_score, messages = check_preprocessing(X_processed, y_processed)
        score += task3_score
        all_messages.extend(messages)
    except Exception as e:
        all_messages.append(f"Task 3 Error: {str(e)}")

    # Task 4: Check data splitting
    try:
        train_data, val_data, test_data = split_data(X_processed, y_processed)
        task4_score, messages = check_data_splitting(train_data, val_data, test_data)
        score += task4_score
        all_messages.extend(messages)
    except Exception as e:
        all_messages.append(f"Task 4 Error: {str(e)}")

    message = "All tests passed!" if score == max_score else "\n".join(all_messages)
    return score == max_score, f"Score: {score}/{max_score}. {message}"


if __name__ == "__main__":
    # Create submission folder if it doesn't exist
    if not os.path.exists(SUBMISSION_FOLDER):
        os.makedirs(SUBMISSION_FOLDER)

    # Find all solution files in submission folder
    solution_files = glob.glob(os.path.join(SUBMISSION_FOLDER, "*_solution.py"))
    if not solution_files:
        print(f"No solution files found in {SUBMISSION_FOLDER}")
        exit(1)

    # Process each solution file and collect results
    results = []
    for solution_file in solution_files:
        try:
            student_name = os.path.basename(solution_file).replace("_solution.py", "")
            success, message = check_solution(solution_file)
            score = message.split("/")[0].split(": ")[1]
            if score == "100":
                results.append(f"{student_name}: {score}")
            else:
                error_msg = message.split(". ", 1)[
                    1
                ]  # Get everything after "Score: X/100. "
                results.append(f"{student_name}: {score} ({error_msg})")
        except Exception as e:
            results.append(f"{student_name}: 0 (Error: {str(e)})")

    # Write results to file
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(results))

    print(f"Results have been written to {RESULTS_FILE}")
    exit(0)
