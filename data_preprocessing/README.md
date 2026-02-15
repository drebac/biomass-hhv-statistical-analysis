
## Data preprocessing for machine learning

This folder contains a Python pipeline used for preparing biomass datasets for machine learning models.

The script performs:
- selection of input variables (C, H, N, S, O),
- identification of categorical (biomass category) and numerical variables,
- data splitting into training (70%), validation (15%), and test (15%) sets,
- standardization of numerical features,
- one-hot encoding of categorical variables with a reference category,
- export of raw and preprocessed datasets for reproducible modeling.

The output datasets are intended for use in machine learning models for predicting higher heating value (HHV).
