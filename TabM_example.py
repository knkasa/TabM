# https://medium.com/chat-gpt-now-writes-all-my-articles/more-easy-deep-learning-for-tabular-data-with-tabm-code-example-included-83de8615ad06
# Intended for Tabular data using deep learning.

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytabkit.models.nn_models.tabm import Model as TabM
from pytabkit.utils.wrappers import TorchModelWrapper  # Custom wrapper to mimic sklearn-style interface

# Simulate dummy data (numerical + categorical)
n_samples = 1000
n_num_features = 5
n_cat_features = 3
cat_cardinalities = [4, 6, 3]  # Assume 3 categorical features with these cardinalities

X_num = np.random.rand(n_samples, n_num_features).astype(np.float32)
X_cat = np.random.randint(0, 4, size=(n_samples, n_cat_features)).astype(np.int64)
y = (X_num.sum(axis=1) + np.random.randn(n_samples) * 0.1).astype(np.float32)

# Train/Val split
X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)

# Normalize numerical features
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_val = scaler.transform(X_num_val)

# Convert to PyTorch tensors
X_num_train_tensor = torch.tensor(X_num_train, dtype=torch.float32)
X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_num_val_tensor = torch.tensor(X_num_val, dtype=torch.float32)
X_cat_val_tensor = torch.tensor(X_cat_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Define the TabM model
tabm_model = TabM(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    n_classes=None,  # For regression, output dim = 1
    backbone={
        "type": "mlp",
        "n_blocks": 3,
        "d_block": 128,
        "dropout": 0.1,
        "activation": "ReLU"
    },
    bins=None,  # Assuming no binning
    num_embeddings=None,
    arch_type='tabm',
    k=4,  # Number of ensemble members
    share_training_batches=True
)

# Wrap model to mimic sklearn API
model = TorchModelWrapper(
    model=tabm_model,
    loss_fn=torch.nn.MSELoss(),
    optimizer_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
    epochs=50,
    batch_size=64,
    verbose=True
)

# Fit the model
model.fit((X_num_train_tensor, X_cat_train_tensor), y_train_tensor)

# Predict on validation data
y_pred = model.predict((X_num_val_tensor, X_cat_val_tensor))

# Evaluate
mse = ((y_pred.squeeze() - y_val_tensor.squeeze()) ** 2).mean().item()
print(f"Validation MSE: {mse:.4f}")
