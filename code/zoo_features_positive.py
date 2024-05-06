import pandas as pd

dataset = '../data/archive_data/zoo_binarized.csv'
df = pd.read_csv(dataset)

filtered_df = df.filter(regex='(type|name|^legs=.*|yes)$')

# Modify column names according to the specified rules
new_columns = []
for col in filtered_df.columns:
    if col.endswith('=yes'):
        new_columns.append(col.replace('=yes', ''))
    elif '=' in col:
        parts = col.split('=')
        new_columns.append(parts[0] + '-' + parts[1])
    else:
        new_columns.append(col)

# Apply the new column names to the DataFrame
filtered_df.columns = new_columns

# Save the modified DataFrame to a new CSV file
new_dataset = '../datasets/zoo_binarized_positives.csv'
filtered_df.to_csv(new_dataset, index=False)

print("New dataset saved successfully with updated column names.")
