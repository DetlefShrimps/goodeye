import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from imblearn.over_sampling import SMOTE
import shap

# Load your datasets
pitcher_data = pd.read_csv("pitcher_stats.csv")
batter_data = pd.read_csv("batter_stats.csv")
matchup_data = pd.read_csv("matchups.csv")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
pitcher_data = pd.DataFrame(imputer.fit_transform(pitcher_data), columns=pitcher_data.columns)
batter_data = pd.DataFrame(imputer.fit_transform(batter_data), columns=batter_data.columns)

# Merge datasets based on identifiers
data = pd.merge(matchup_data, pitcher_data, on='pitcher_id')
data = pd.merge(data, batter_data, on='batter_id')

# Calculate derived features
data['Swing_diff'] = data['Swing% (pi)_batter'] - data['Z-Swing% (pi)_batter']
data['Contact_HardHit_diff'] = data['Contact%_batter'] - data['HardHit%_batter']
data['FA_avg'] = (data['FA-X (sc)_pitcher'] + data['FA-Z (sc)_pitcher']) / 2 + data['FT-X (sc)_pitcher']
data['CH_FO_SF_sum'] = data['CH%_pitcher'] + data['FO% (sc)_pitcher'] + data['SF%_pitcher']
data['Complex_avg'] = (data[['CH-X (pi)_pitcher', 'CH-Z (pi)_pitcher']].mean(axis=1) + data['SI-X (pi)_pitcher'] + data['FO-X (sc)_pitcher'] + data['FS-Z (pi)_pitcher']) / 4

# Select relevant features for the model
features = ['Swing_diff', 'Contact_HardHit_diff', 'K%_batter', 'HR/FB_batter', 'FA% (sc)_pitcher', 'FA_avg', 'CH_FO_SF_sum', 'Complex_avg']
target = 'event'

X = data[features]
y = data[target]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Handle class imbalance
smote = SMOTE()
X, y_encoded = smote.fit_resample(X, y_encoded)

class BaseballDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {'features': torch.tensor(self.features[idx], dtype=torch.float), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}
        return item

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
train_dataset = BaseballDataset(X_train, y_train)
test_dataset = BaseballDataset(X_test, y_test)

# Define the model with more layers and regularization
class AdvancedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdvancedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

input_dim = X.shape[1]
output_dim = len(le.classes_)
model = AdvancedModel(input_dim, output_dim)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).astype(np.float32).mean().item(),
        "precision": precision_score(p.label_ids, p.predictions.argmax(-1), average='weighted'),
        "recall": recall_score(p.label_ids, p.predictions.argmax(-1), average='weighted'),
        "f1": f1_score(p.label_ids, p.predictions.argmax(-1), average='weighted')
    }
)

# Train the model
trainer.train()

# Model Explainability
explainer = shap.DeepExplainer(model, torch.tensor(X_train, dtype=torch.float))
shap_values = explainer.shap_values(torch.tensor(X_test, dtype=torch.float))

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=features)

# Example prediction for a hypothetical at-bat
def predict_event(pitcher_id, batter_id):
    pitcher_stats = pitcher_data[pitcher_data['pitcher_id'] == pitcher_id].iloc[0]
    batter_stats = batter_data[batter_data['batter_id'] == batter_id].iloc[0]
    
    new_data = {
        'Swing% (pi)_batter': batter_stats['Swing% (pi)'],
        'Z-Swing% (pi)_batter': batter_stats['Z-Swing% (pi)'],
        'Contact%_batter': batter_stats['Contact%'],
        'HardHit%_batter': batter_stats['HardHit%'],
        'K%_batter': batter_stats['K%'],
        'HR/FB_batter': batter_stats['HR/FB'],
        'FA% (sc)_pitcher': pitcher_stats['FA% (sc)'],
        'FA-X (sc)_pitcher': pitcher_stats['FA-X (sc)'],
        'FA-Z (sc)_pitcher': pitcher_stats['FA-Z (sc)'],
        'FT-X (sc)_pitcher': pitcher_stats['FT-X (sc)'],
        'CH%_pitcher': pitcher_stats['CH%'],
        'FO% (sc)_pitcher': pitcher_stats['FO% (sc)'],
        'SF%_pitcher': pitcher_stats['SF%'],
        'CH-X (pi)_pitcher': pitcher_stats['CH-X (pi)'],
        'CH-Z (pi)_pitcher': pitcher_stats['CH-Z (pi)'],
        'SI-X (pi)_pitcher': pitcher_stats['SI-X (pi)'],
        'FO-X (sc)_pitcher': pitcher_stats['FO-X (sc)'],
        'FS-Z (pi)_pitcher': pitcher_stats['FS-Z (pi)']
    }
    
    new_data_df = pd.DataFrame([new_data])

    # Calculate derived features for the new data
    new_data_df['Swing_diff'] = new_data_df['Swing% (pi)_batter'] - new_data_df['Z-Swing% (pi)_batter']
    new_data_df['Contact_HardHit_diff'] = new_data_df['Contact%_batter'] - new_data_df['HardHit%_batter']
    new_data_df['FA_avg'] = (new_data_df['FA-X (sc)_pitcher'] + new_data_df['FA-Z (sc)_pitcher']) / 2 + new_data_df['FT-X (sc)_pitcher']
    new_data_df['CH_FO_SF_sum'] = new_data_df['CH%_pitcher'] + new_data_df['FO% (sc)_pitcher'] + new_data_df['SF%_pitcher']
    new_data_df['Complex_avg'] = (new_data_df[['CH-X (pi)_pitcher', 'CH-Z (pi)_pitcher']].mean(axis=1) + new_data_df['SI-X (pi)_pitcher'] + new_data_df['FO-X (sc)_pitcher'] + new_data_df['FS-Z (pi)_pitcher']) / 4

    # Normalize the new data
    X_new = scaler.transform(new_data_df[features])

    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(X_new, dtype=torch.float))
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Decode the predicted class
    predicted_event = le.inverse_transform([predicted_class])
    return predicted_event[0]

# Example usage
pitcher_id = 123  # Replace with actual pitcher_id
batter_id = 456   # Replace with actual batter_id
predicted_event = predict_event(pitcher_id, batter_id)
print(f"The predicted outcome of the at-bat is: {predicted_event}")