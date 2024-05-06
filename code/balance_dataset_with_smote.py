"""
@author: Dr. Daniel Hier
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = '../data/neurogenetic_4.csv'
data = pd.read_csv(data_path)

# Check the balance of categories in 'type' column
print(data['type'].value_counts())

# Temporarily exclude 'type' and 'name' columns to get features for SMOTE
X = data.drop(['type', 'name'], axis=1)
y = data['type']  # Target variable

# Split the dataset into training and testing sets, including 'name' for recombination after SMOTE
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, data['name'], test_size=0.2, random_state=42)

# Binarize the training data: set values to 0 if < 1, to 1 if > 0
X_train = X_train.applymap(lambda x: 1 if x > 0 else 0)
X_test = X_test.applymap(lambda x: 1 if x > 0 else 0)  # Optionally, also binarize the test set if needed for consistency

# Initialize SMOTE
smote = SMOTE()

# Apply SMOTE to the binarized training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the balance of categories in the target variable after resampling
print(y_train_smote.value_counts())

# Recombine the resampled features with their corresponding 'type' and original 'name'
balanced_data = pd.concat([X_train_smote, y_train_smote.reset_index(drop=True), names_train.reset_index(drop=True)], axis=1)

# Assuming you want to keep the original column order with 'type' and 'name' at the start
column_order = ['name', 'type'] + [col for col in balanced_data.columns if col not in ['name', 'type']]
balanced_data = balanced_data[column_order]

# Specify your desired output file path
output_file_path = '../data/balanced_neurogenetic_with_names.csv'

# Write the DataFrame to a CSV file
balanced_data.to_csv(output_file_path, index=False)
print("Balanced dataset saved to:", output_file_path)
