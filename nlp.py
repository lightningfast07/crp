import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import re
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import shap

# 1. Read the csv file
data = pd.read_csv('data.csv')

# 2. Use a list which will contain the features that are to be included.
features = [
    'Short description', 'Description', 'Type', 'State', 
    'Assignment group', 'Requested by', 'Impact', 'Risk', 'Config item'
]

target_feature = 'Close code'

# Text cleaning and tokenization functions
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# List of text-based features
text_features = ['Short description', 'Description']

# Clean and tokenize text-based features
for feature in text_features:
    data[feature] = data[feature].apply(clean_text)
    data[feature + '_tokens'] = data[feature].apply(lambda x: x.split())

# 3. Use Word2Vec for text and label encoding for categorical data
# Train Word2Vec model
all_tokens = data[text_features[0] + '_tokens'].tolist()
for feature in text_features[1:]:
    all_tokens += data[feature + '_tokens'].tolist()

word2vec_model = Word2Vec(all_tokens, vector_size=100, window=5, min_count=1, workers=4)

def get_word2vec_vectors(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# Create Word2Vec vectors for each text-based feature
text_feature_vectors = []
for feature in text_features:
    vectors = np.array([get_word2vec_vectors(text, word2vec_model) for text in data[feature + '_tokens']])
    text_feature_vectors.append(vectors)

# Combine all text-based feature vectors
text_feature_vectors_combined = np.hstack(text_feature_vectors)

# Encoding categorical features
categorical_features = ['Type', 'State', 'Assignment group', 'Requested by', 'Impact', 'Risk', 'Config item']
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])

# Combining all features
features_data = np.hstack((text_feature_vectors_combined, data[categorical_features].values))

# Target variable
le = LabelEncoder()
data[target_feature] = le.fit_transform(data[target_feature])
target = data[target_feature]

# 4. Create a pie chart to show the target class imbalance
plt.figure(figsize=(8, 8))
target.value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Target Class Imbalance')
plt.ylabel('')
plt.show()

# 5. Create the train test split
X_train, X_test, y_train, y_test = train_test_split(features_data, target, test_size=0.2, random_state=42, stratify=target)

# 6. Use undersampling and create a dataset with that
under = RandomUnderSampler(sampling_strategy='auto')
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

plt.figure(figsize=(8, 8))
pd.Series(y_train_under).value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Target Class After Undersampling')
plt.ylabel('')
plt.show()

# 7. Use oversampling and create a dataset with that
over = SMOTE(sampling_strategy='auto')
X_train_over, y_train_over = over.fit_resample(X_train, y_train)

plt.figure(figsize=(8, 8))
pd.Series(y_train_over).value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Target Class After Oversampling')
plt.ylabel('')
plt.show()

# 8. Train an ANN model
def create_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model = create_ann_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    return model, y_pred

# Train and evaluate on original data
model_orig, y_pred_orig = train_and_evaluate(X_train, y_train, X_test, y_test)

# Train and evaluate on undersampled data
model_under, y_pred_under = train_and_evaluate(X_train_under, y_train_under, X_test, y_test)

# Train and evaluate on oversampled data
model_over, y_pred_over = train_and_evaluate(X_train_over, y_train_over, X_test, y_test)

# 9. Evaluation metrics
def print_evaluation_metrics(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Original Data Evaluation Metrics:")
print_evaluation_metrics(y_test, y_pred_orig)

print("\nUndersampled Data Evaluation Metrics:")
print_evaluation_metrics(y_test, y_pred_under)

print("\nOversampled Data Evaluation Metrics:")
print_evaluation_metrics(y_test, y_pred_over)

# 10. Predict probability for each instance of the test split
for i in range(len(X_test)):
    prob = model_orig.predict(np.expand_dims(X_test[i], axis=0))[0][0]
    print(f"Instance {i+1}: Probability of belonging to class 1: {prob*100:.2f}%")

# 11. Use SHAP to explain the model predictions
explainer = shap.KernelExplainer(model_orig.predict, X_test)
shap_values = explainer.shap_values(X_test)

for i in range(len(X_test)):
    print(f"\nSHAP Beeswarm plot for Instance {i+1}:")
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], X_test[i], matplotlib=True)
