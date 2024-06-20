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
