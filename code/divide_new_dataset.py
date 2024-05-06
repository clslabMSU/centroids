import pandas as pd

# Load the dataset
folder = '../data/'
dataset = 'neurogenetic_4.csv'
df = pd.read_csv(folder+dataset)

# Specify the classes to separate
selected_classes = ['Parkinson', 'myopathy', 'CMT', 'ataxia']

# Create two datasets: one for the selected classes and one for the rest
class_1_df = df[df['type'].isin(selected_classes)]
class_2_df = df[~df['type'].isin(selected_classes)]

# Save the two datasets into new CSV files
class_1_df.to_csv('../data/neurogenetic_4_4classes.csv', index=False)
class_2_df.to_csv('../data/neurogenetic_4_3classes.csv', index=False)

print("Datasets created and saved.")

# Load the dataset
folder = '../data/'
dataset = 'balanced_neurogenetic_with_names.csv'
df = pd.read_csv(folder+dataset)

# Specify the classes to separate
selected_classes = ['Parkinson', 'myopathy', 'CMT', 'ataxia']

# Create two datasets: one for the selected classes and one for the rest
class_1_df = df[df['type'].isin(selected_classes)]
class_2_df = df[~df['type'].isin(selected_classes)]

# Save the two datasets into new CSV files
class_1_df.to_csv('../data/neurogenetic_4_4classes_balanced.csv', index=False)
class_2_df.to_csv('../data/neurogenetic_4_3classes_balanced.csv', index=False)

print("Datasets created and saved.")
