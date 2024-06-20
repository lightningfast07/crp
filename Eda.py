import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Drop all columns where all the values are null
df.dropna(axis=1, how='all', inplace=True)

# Drop the columns where more than 60% of the values are null
threshold = len(df) * 0.6
df.dropna(axis=1, thresh=threshold, inplace=True)

# Draw correlation of all remaining features only with the output column
output_column = 'output'  # Replace 'output' with your actual output column name
correlations = df.corr()[output_column].abs()

# Drop the features which have less than 0.4 correlation with the output variable
selected_features = correlations[correlations >= 0.4].index
df = df[selected_features]

# Save the preprocessed DataFrame
df.to_csv('preprocessed_dataset.csv', index=False


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data before applying PCA
features = df.drop(columns=[output_column])
features_scaled = StandardScaler().fit_transform(features)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
features_pca = pca.fit_transform(features_scaled)

# Create a DataFrame with the PCA components
df_pca = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
df_pca[output_column] = df[output_column].values

# Save the PCA DataFrame
df_pca.to_csv('pca_dataset.csv', index=False)

from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier for classification tasks

# Separate features and output
X = df.drop(columns=[output_column])
y = df[output_column]

# Fit Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
important_features = X.columns[importances > 0.01]  # Adjust the threshold as needed

# Select important features
df_rf = df[important_features]

# Save the DataFrame with important features
df_rf.to_csv('rf_important_features_dataset.csv', index=False)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression  # or another model of your choice

# Fit RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)  # Adjust the number of features to select
fit = rfe.fit(X, y)

# Get selected features
selected_features = X.columns[fit.support_]

# Create a DataFrame with selected features
df_rfe = df[selected_features]

# Save the DataFrame with selected features
df_rfe.to_csv('rfe_selected_features_dataset.csv', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Separate features and output
X = df.drop(columns=['Close'])
y = df['Close']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data (optional, but often recommended for ML algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Resampling Techniques
# Option 1: SMOTE (Oversampling the minority class)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Option 2: Undersampling the majority class
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train_scaled, y_train)

# Option 3: Combining SMOTE and Edited Nearest Neighbors (ENN)
smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train_scaled, y_train)

# Now train your model using X_train_res and y_train_res
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Assuming y_test and y_pred are defined as above
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled), multi_class='ovr')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

from imblearn.ensemble import EasyEnsembleClassifier

model = EasyEnsembleClassifier(random_state=42)
model.fit(X_train, y_train)
