# SHAP and LightGBM Integration for Feature Attribution
Execution of Removal-based explanations by Ian Covert

Overview
This README provides a detailed guide on using SHAP values with the rexplain framework and LightGBM to explain model predictions. The code demonstrates how to prepare data, train a model, generate explanations using different techniques, and includes custom extensions for model interpretability.

Dependencies
Ensure you have the following libraries installed:

shap: For SHAP values
lightgbm: For the LightGBM model
scikit-learn: For data splitting
numpy: For numerical operations
matplotlib: For plotting
rexplain: For model explanations
torch: For neural network operations (if using Torch-related classes)

# Load the 'adult' Census dataset from SHAP.
X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)
num_features = X.shape[1]

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Convert to LightGBM datasets.
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
2. Train Model

# Define parameters for LightGBM model.
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True
}

# Train the LightGBM model.
model = lgb.train(params, d_train, num_boost_round=10000, valid_sets=[d_test], callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=1000)])
3. Generate Explanations

import numpy as np
import matplotlib.pyplot as plt
from rexplain import removal, behavior, summary

# Create a lambda function for model prediction.
model_lam = lambda x: model.predict(x)

# Create a marginal extension of the model using the removal module.
marginal_extension = removal.MarginalExtension(X_test.values[:512], model_lam)

# Create a cooperative game instance for the prediction behavior of the model.
game = behavior.PredictionGame(marginal_extension, X.iloc[0, :].values)

# Apply RemoveIndividual summary technique and plot results.
%time attr = summary.RemoveIndividual(game)
plt.figure(figsize=(9, 6))
plt.bar(np.arange(len(attr)), attr)
plt.xticks(np.arange(len(attr)), X.columns, rotation=45, rotation_mode='anchor', ha='right', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('Remove Individual', fontsize=20)
plt.tight_layout()
plt.show()

# Apply IncludeIndividual summary technique and plot results.
%time attr = summary.IncludeIndividual(game)
plt.figure(figsize=(9, 6))
plt.bar(np.arange(len(attr)), attr)
plt.xticks(np.arange(len(attr)), X.columns, rotation=45, rotation_mode='anchor', ha='right', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('Include Individual', fontsize=20)
plt.tight_layout()
plt.show()

# Apply ShapleyValue summary technique and plot results.
%time attr = summary.ShapleyValue(game)
plt.figure(figsize=(9, 6))
plt.bar(np.arange(len(attr)), attr)
plt.xticks(np.arange(len(attr)), X.columns, rotation=45, rotation_mode='anchor', ha='right', fontsize=14)
plt.tick_params(labelsize=14)
plt.title('Shapley Value', fontsize=20)
plt.tight_layout()
plt.show()

4. Custom Code
Model Extensions
DefaultExtension: Replace removed features with default values.

MarginalExtension: Replace removed features using their marginal distribution.

UniformExtension: Replace removed features using a uniform distribution.

UniformContinuousExtension: Replace removed features with uniform distribution for continuous features.

ProductMarginalExtension: Replace removed features with the product of their marginal distributions.

SeparateModelExtension: Use separate models for each subset of features.

ConditionalExtension: Use a model of the conditional distribution to replace removed features.

ConditionalSupervisedExtension: Use a supervised surrogate model to extend the original model.

Custom Functions
crossentropyloss: Computes cross-entropy loss without averaging across samples.
mseloss: Computes Mean Squared Error loss without averaging across samples.
ModelWrapper: A wrapper for models (e.g., sklearn, xgb, lgbm) to provide a consistent callable interface.
ConstantModel: Represents a model with constant output.
Cooperative Games
PredictionGame: A cooperative game for explaining individual predictions using model extensions.

PredictionLossGame: A cooperative game for explaining individual prediction loss using model extensions.

Conclusion
This README provides a comprehensive guide on integrating SHAP values with the rexplain framework to explain model predictions using LightGBM. It includes data preparation, model training, explanation generation, and custom extensions for interpretability.

