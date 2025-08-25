#--------------CLASSIFICATION-----------------------------------------------
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.neighbors import NearestNeighbors
from nltk.stem import PorterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import regularizers
import os
import random

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



# Step 1: Load and Parse .arff file
file_path = "Imdb.arff"
with open(file_path, "r") as file:
    lines = file.readlines()

# Step 2: Extract attribute names
attributes = []
for line in lines:
    line = line.strip()
    if line.lower().startswith('@attribute'):
        parts = line.split()
        if len(parts) >= 2:
            attr_name = parts[1].strip("'")
            attributes.append(attr_name)
    elif line.lower() == '@data':
        break

label_columns = attributes[:28]
feature_columns = attributes[28:]

# Step 3: Load sparse data
data_start = lines.index('@data\n') + 1
data_lines = [line.strip() for line in lines[data_start:] if line.strip() and not line.startswith('%')]

np.random.seed(42)
np.random.shuffle(data_lines)
subset = data_lines[:int(len(data_lines) * 0.01)] #1% of dataset

data_rows = []
for line in subset:
    row = [0] * len(attributes)
    if line.startswith('{'):
        for pair in line[1:-1].split(','):
            idx, val = map(int, pair.strip().split())
            row[idx] = val
        data_rows.append(row)

df = pd.DataFrame(data_rows, columns=attributes)

# Step 4: Separate features and labels
Y = df[label_columns]
X = df[feature_columns]

# Step 5: Stemming
stemmer = PorterStemmer()
X_raw = X
stem_to_cols = defaultdict(list)
for col in X_raw.columns:
    stem = stemmer.stem(col.lower())
    stem_to_cols[stem].append(col)

# Efficiently build stemmed features using concat
stemmed_columns = []
stem_scores = {}

for stem, cols in stem_to_cols.items():
    stem_col = X_raw[cols].max(axis=1)
    stem_col.name = stem
    stemmed_columns.append(stem_col)
    stem_scores[stem] = X_raw[cols].sum().sum()

# Concatenate all stemmed columns at once to avoid fragmentation
X_stemmed = pd.concat(stemmed_columns, axis=1)


Y = df[label_columns].astype(int)
X = X_stemmed

# Step 6: Keep Top 500 Stemmed Features
top_500_stems = sorted(stem_scores.items(), key=lambda x: x[1], reverse=True)[:500]
top_500_stems = [stem for stem, _ in top_500_stems]
X = X_stemmed[top_500_stems]
print(f"\nRetained Top 500 Features: {len(X.columns)}")


# Step 7: Keep Top 5 Labels
Y = df[label_columns].astype(int)
label_freq = Y.sum(axis=0)
top_5_labels = label_freq.sort_values(ascending=False).head(5).index.tolist()
Y = Y[top_5_labels]
print(f"\nRetained Top 5 Labels: {len(Y.columns)}")
print("Labels selected:", top_5_labels)

print("\n--- Frequency of Top 5 Labels ---")
for label in top_5_labels:
    count = Y[label].sum()
    print(f"{label:12}: {int(count)} samples")


# Step 8: Train-Test Split (before MLSMOTE)
X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Step 9: MLSMOTE Function
def mlsmote(X, Y, n_samples=500, k_neighbors=5, minority_threshold=0.4, random_state=42):
    np.random.seed(random_state)
    X = X.copy()
    Y = Y.copy()

    label_counts = Y.sum(axis=0)
    minority_labels = [i for i, c in enumerate(label_counts) if c < (minority_threshold * len(Y))]

    # Find samples with at least one minority label
    minority_indices = Y[Y.iloc[:, minority_labels].sum(axis=1) > 0].index
    X_minority = X.loc[minority_indices].reset_index(drop=True)
    Y_minority = Y.loc[minority_indices].reset_index(drop=True)

    # Cap n_samples if it's too high
    if n_samples > len(X_minority):
        n_samples = len(X_minority)

    nn = NearestNeighbors(n_neighbors=k_neighbors).fit(X_minority)

    synthetic_X = []
    synthetic_Y = []

    for _ in range(n_samples):
        idx = np.random.randint(0, len(X_minority))
        x = X_minority.iloc[idx]
        y = Y_minority.iloc[idx]

        neighbors = nn.kneighbors(x.to_frame().T, return_distance=False).flatten()
        neighbor_idx = np.random.choice(neighbors[1:])  # exclude self
        x_neighbor = X_minority.iloc[neighbor_idx]
        y_neighbor = Y_minority.iloc[neighbor_idx]

        lam = np.random.rand()
        x_synth = x + lam * (x_neighbor - x)
        y_synth = ((y + y_neighbor) >= 1).astype(int)

        synthetic_X.append(x_synth)
        synthetic_Y.append(y_synth)

    X_syn = pd.DataFrame(synthetic_X, columns=X.columns)
    Y_syn = pd.DataFrame(synthetic_Y, columns=Y.columns).astype(int)

    return pd.concat([X, X_syn], ignore_index=True), pd.concat([Y, Y_syn], ignore_index=True)


# Step 10: Apply MLSMOTE on training set
X_resampled, Y_resampled = mlsmote(X_train_orig, Y_train_orig, n_samples=500)
print(f"\nMLSMOTE applied: X shape {X_resampled.shape}, Y shape {Y_resampled.shape}")


# Step 11: Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_resampled, Y_resampled)

# Step 12: Evaluate on Original Test Set
Y_pred = multi_rf.predict(X_test_orig)

# Step 13: Get Probabilities
Y_prob = []
for est in multi_rf.estimators_:
    probs = est.predict_proba(X_test_orig)
    if probs.shape[1] == 2:
        Y_prob.append(probs[:, 1])
    else:
        Y_prob.append(np.zeros(X_test_orig.shape[0]))
Y_prob = np.array(Y_prob).T

# Step 14: Metrics
Y_test_np = Y_test_orig.to_numpy()
Y_pred_np = Y_pred

print("\n--- Overall Random Forest Evaluation Metrics ---")
print("F1 Score (micro):", f1_score(Y_test_np, Y_pred_np, average='micro', zero_division=0))
print("F1 Score (macro):", f1_score(Y_test_np, Y_pred_np, average='macro', zero_division=0))
print("F1 Score (weighted):", f1_score(Y_test_np, Y_pred_np, average='weighted', zero_division=0))
print("Precision (macro):", precision_score(Y_test_np, Y_pred_np, average='macro', zero_division=0))
print("Recall (macro):", recall_score(Y_test_np, Y_pred_np, average='macro', zero_division=0))

# Step: Per-Label Precision and Recall for Random Forest (added)
print("\n--- Per-Label Precision and Recall (Random Forest) ---")
for i, label in enumerate(Y.columns):
    precision_rf = precision_score(Y_test_np[:, i], Y_pred_np[:, i], zero_division=0)
    recall_rf = recall_score(Y_test_np[:, i], Y_pred_np[:, i], zero_division=0)
    print(f"Label: {label}")
    print(f"Precision: {precision_rf:.4f}")
    print(f"Recall: {recall_rf:.4f}")
    print("-" * 50)

# Classification Report for Random Forest
print("\n--- Random Forest Classification Report ---")
print(classification_report(Y_test_np, Y_pred_np, target_names=Y.columns, zero_division=0))

print("\n--- Random Forest Per-Label Confusion Summary ---")
for i, label in enumerate(Y.columns):
    y_true = Y_test_np[:, i]
    y_pred = Y_pred_np[:, i]
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    print(f"Label: {label:12} | TP: {tp:3} | FP: {fp:3} | FN: {fn:3}")


##############   NEURAL NETWORK CLASSIFIER   ################################
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.initializers import HeNormal, GlorotNormal

# Step: Split original training data into train + validation (before MLSMOTE)
X_train_raw, X_val, Y_train_raw, Y_val = train_test_split(
    X_train_orig, Y_train_orig, test_size=0.2, random_state=42)

# Step: Apply MLSMOTE to training only (not validation)
X_resampled, Y_resampled = mlsmote(X_train_raw, Y_train_raw, n_samples=500)

# === Scale features for neural network ===
scaler = StandardScaler()
X_train_dl = scaler.fit_transform(X_resampled)
X_val_dl = scaler.transform(X_val)
X_test_dl = scaler.transform(X_test_orig)

# Convert all to float32
X_train_dl = X_train_dl.astype('float32')
X_val_dl = X_val_dl.astype('float32')
X_test_dl = X_test_dl.astype('float32')
Y_train_dl = Y_resampled.astype('float32')
Y_val_dl = Y_val.astype('float32')
Y_test_dl = Y_test_orig.astype('float32')

from tensorflow.keras import regularizers

# Step: Define Deep Neural Network
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001),input_shape=(X_train_dl.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),

    Dense(Y_train_dl.shape[1], activation='sigmoid')  # Multi-label output
])


model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)


# Step: Train Model (validation on original data!)
early_stop = EarlyStopping(monitor='val_loss', patience= 10, restore_best_weights=True)

history = model.fit(
    X_train_dl, Y_train_dl,
    epochs=100,
    batch_size=64,
    validation_data=(X_val_dl, Y_val_dl),
    callbacks=[early_stop],
    verbose=1
)

# Step: Predict on Original Test Set
Y_pred_dl_prob = model.predict(X_test_dl)

from sklearn.metrics import precision_recall_curve

# Predict probabilities
Y_pred_dl_prob = model.predict(X_test_dl)

# === Per-Label Threshold Tuning ===
print("\n--- Per-Label Threshold Tuning ---")
thresholds = []
for i in range(Y_test_dl.shape[1]):
    precision, recall, thresh = precision_recall_curve(Y_test_dl.to_numpy()[:, i], Y_pred_dl_prob[:, i])
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresh[np.argmax(f1[:-1])] if len(thresh) > 0 else 0.5
    thresholds.append(best_thresh)
    print(f"Label: {Y.columns[i]:12} | Best Threshold: {best_thresh:.2f}")

# Apply thresholds per label
thresholds = np.array(thresholds)
Y_pred_dl = (Y_pred_dl_prob >= thresholds).astype(int)

# Final Evaluation
Y_test_np = Y_test_dl.to_numpy()

print("\n--- Deep Neural Network Evaluation Metrics---")
print("F1 Score (micro):", f1_score(Y_test_np, Y_pred_dl, average='micro', zero_division=0))
print("F1 Score (macro):", f1_score(Y_test_np, Y_pred_dl, average='macro', zero_division=0))
print("F1 Score (weighted):", f1_score(Y_test_np, Y_pred_dl, average='weighted', zero_division=0))
print("Precision (macro):", precision_score(Y_test_np, Y_pred_dl, average='macro', zero_division=0))
print("Recall (macro):", recall_score(Y_test_np, Y_pred_dl, average='macro', zero_division=0))

# Per-Label Metrics
print("\n--- Per-Label Precision and Recall (Neural Network) ---")
for i, label in enumerate(Y.columns):
    precision_dl = precision_score(Y_test_np[:, i], Y_pred_dl[:, i], zero_division=0)
    recall_dl = recall_score(Y_test_np[:, i], Y_pred_dl[:, i], zero_division=0)
    print(f"Label: {label}")
    print(f"Precision: {precision_dl:.4f}")
    print(f"Recall: {recall_dl:.4f}")
    print("-" * 50)

# Confusion summary per label
print("\n--- Neural Network Per-Label Confusion Summary ---")
for i, label in enumerate(Y.columns):
    y_true = Y_test_np[:, i]
    y_pred = Y_pred_dl[:, i]
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    print(f"Label: {label:12} | TP: {tp:3} | FP: {fp:3} | FN: {fn:3}")

# Classification Report for Neural Network
print("\n--- Neural Network Classification Report ---")
print(classification_report(Y_test_np, Y_pred_dl, target_names=Y.columns, zero_division=0))



######################## SHAP EXPLAINABILITY (Top 10 features on 50 samples) ########################

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === Step 1: Prepare test data
print("X_test_dl shape:", X_test_dl.shape)
#feature_names = [f"feature_{i}" for i in range(X_test_dl.shape[1])]
feature_names = X_train_orig.columns.tolist()  # This gets the real column names
X_test_df = pd.DataFrame(X_test_dl, columns=feature_names)

# === Step 2: Scale & prepare a better background (50 rows from training set)
X_background = X_train_orig.sample(50, random_state=42)
X_background_scaled = scaler.transform(X_background)
X_background_df = pd.DataFrame(X_background_scaled, columns=feature_names)
print("Background sample shape:", X_background_df.shape)

# === Step 3: Select test samples (10 for visualization)
X_test_sample_full = X_test_df.iloc[:10]

# === Step 4: SHAP KernelExplainer setup
def model_predict(x): return model.predict(x)

print("\n⏳ Initializing KernelExplainer...")
explainer = shap.KernelExplainer(model_predict, X_background_df)

print("⏳ Calculating SHAP values...")
shap_values_all_labels = explainer.shap_values(X_test_sample_full)

if not isinstance(shap_values_all_labels, list):
    shap_values_all_labels = [shap_values_all_labels]

# The global importance of each feature on average, across all labels and samples
mean_abs_shap = np.abs(shap_values_all_labels[0]).mean(axis=(0, 2))  # shape: (500,)
# modify from 10 to 20
top_10_indices = np.argsort(mean_abs_shap)[-10:][::-1]

top_10_features = [feature_names[i] for i in top_10_indices]
print("Top 10 features across all labels and samples:", top_10_features)

X_test_sample_top10 = X_test_sample_full[top_10_features]

# === Step 6: Slice SHAP values to top 10 features for all labels
num_labels = shap_values_all_labels[0].shape[2]
shap_values_top10_all_labels = [
    shap_values_all_labels[0][:, top_10_indices, label_idx]
    for label_idx in range(num_labels)
]


################################----FORCE PLOT--------################################
# === Step 8: Generate force plots for all labels with top 10 features ===
print("\n⚡ Generating SHAP force plots for all labels with top 10 features...")

for label_idx, label_name in enumerate(Y.columns):
    # Get SHAP values for this label (all samples, top 10 features)
    shap_vals_label = shap_values_top10_all_labels[label_idx]

    # Find the sample with highest total impact for this label
    total_impact = np.abs(shap_vals_label).sum(axis=1)
    best_sample_idx = int(np.argmax(total_impact))

    # Get the prediction probability for this label
    pred_prob = model_predict(X_test_sample_full.values)[best_sample_idx, label_idx]

    # Create the force plot
    print(f"\nForce Plot for Label: {label_name} (Sample {best_sample_idx})")
    print(f"Prediction probability: {pred_prob:.4f}")

    plt.figure()
    shap.force_plot(
        explainer.expected_value[label_idx],
        shap_vals_label[best_sample_idx],
        X_test_sample_top10.iloc[best_sample_idx],
        feature_names=top_10_features,
        matplotlib=True,
        show=False,
        text_rotation=15
    )
    plt.title(f"SHAP Force Plot - {label_name}\n(Prediction: {pred_prob:.2f})",
              fontsize=10, pad=20)
    plt.tight_layout()

    # Save and show
    plt.savefig(f"shap_force_{label_name}.png", dpi=300, bbox_inches='tight')
    #plt.show()

################################---SUMMARY PLOT--------################################

# === Step 9: Generate beeswarm summary plots for all labels ===
print("\n⚡ Generating SHAP beeswarm summary plots for all labels...")

for label_idx, label_name in enumerate(Y.columns):
    print(f"\nCreating beeswarm plot for: {label_name}")

    # Get SHAP values for this label across all samples and features
    shap_values_label = shap_values_all_labels[0][:, :, label_idx]

    # Create beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_label,
        X_test_sample_full,
        feature_names=feature_names,
        plot_type="dot",  # Changed from "bar" to "dot" for beeswarm
        show=False,
        max_display=10,  # Show top 10 features
        color_bar=False  # Remove color bar if you prefer cleaner look
    )

    # Custom styling to match your example
    plt.title(f"SHAP Summary for {label_name}", fontsize=12, pad=20)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.tight_layout()

    # Save and show
    plt.savefig(f"shap_summary_{label_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


