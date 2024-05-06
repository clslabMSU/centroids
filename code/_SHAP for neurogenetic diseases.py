##################################################################################################################################
# Created on Wed Dec  6 09:49:21 2023
# @author: danielhier
# This program fits a xgboost tree to the neurogenetic data and then uses SHAP to find the 10 most influential phenotype features
# M#ake sure that these libraries are installed
##################################################################################################################################


###################################
#      LIBRARIES                  #
###################################
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib

###################################
#      IMPORT DATA                #
###################################
# Load your data into a Pandas DataFrame
#directory_path = "/Users/danielhier/Desktop/t_SNE"
matplotlib.use('Agg') # to save png file
neurogenetic = pd.read_csv('../data/_neurogenetic.csv')

###################################
#      PREPROCESS DATA            #
###################################
# Extract labels and features
labels = neurogenetic[['type', 'name']]
features = neurogenetic.iloc[:, 2:]

# Convert feature values to 0 or 1
features = features.applymap(lambda x: 1 if x > 0 else 0)

# Concatenate 'features' and 'labels' along the columns axis
df_neurogenetic = pd.concat([labels, features], axis=1)

# Define the features and labels
X = df_neurogenetic.iloc[:, 2:]
y = df_neurogenetic['type']

# Encode the text labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

##############################################
#  FIT XGTBOOST CLASSIFIER to DATA          #
##############################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
clf = xgb.XGBClassifier()

# Fit the model on the training data
clf.fit(X_train, y_train)
# Save the model in JSON format
clf.save_model('../results/paper_dataset/model.json')

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of xgtboost: {accuracy * 100:.2f}%")
print()
print()

#############################################
# FIT a SHAP EXPLAINER to XGTBOOST MODELS   #
#############################################
# Create a SHAP explainer object with the XGBoost model
explainer = shap.Explainer(clf, X_train)

# Calculate SHAP values for your test data
shap_values = explainer.shap_values(X_test)

#############################################
# CREATD AND SAVE SHP SUMMARY PLOT          #
#############################################

# Create a mapping from encoded class labels to disease names
class_label_to_disease = {label: disease for label, disease in zip(label_encoder.classes_, df_neurogenetic['type'])}
class_label_to_disease[0]='CA'
class_label_to_disease[1]='CMT'
class_label_to_disease[2]='HSP'
# Ensure all class labels in clf.classes_ are in the mapping dictionary
for label in clf.classes_:
    if label not in class_label_to_disease:
        # Handle the case where the label is not in the mapping (you can assign a default disease name)
        class_label_to_disease[label] = "Unknown Disease"

# Get class names for the legend
class_names = [class_label_to_disease[label] for label in clf.classes_]

# Summarize the feature importance with class names in the legend
shap.summary_plot(shap_values, X_test, feature_names=X.columns, class_names=class_names, max_display=10)



plt.savefig('../results/paper_dataset/shap_summary_plot.png', dpi=600)
plt.close()
#############################################
# END PROGRAM                               #
#############################################


