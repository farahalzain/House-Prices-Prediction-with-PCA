# House Prices Prediction with Principal Component Analysis (PCA)

## Project Overview

This project explores **house price prediction** using machine learning, with a focus on **Principal Component Analysis (PCA)** for dimensionality reduction. The central question is whether a smaller set of principal components can preserve useful information while maintaining competitive regression performance.

## Dataset

The project uses the **House Prices: Advanced Regression Techniques** dataset (Ames Housing).

- **Labeled training data:** 1,460 properties, 81 columns (including `Id` and `SalePrice`)
- **Competition test data:** 1,459 properties without `SalePrice`
- **Target:** `SalePrice`
- **Problem type:** Supervised learning — regression

The dataset includes property characteristics such as overall quality, living area, neighborhood, garage capacity, and construction year.

## Workflow

1. **Exploratory Data Analysis:** Inspect target distribution, missing values, numerical correlations, and categorical relationships with sale prices.
2. **Target Analysis:** Examine `SalePrice` and the effect of a `log1p` transformation.
3. **Data Splitting:** Create training and validation sets before fitting preprocessing transformations.
4. **Preprocessing:** Impute missing values, standardize numerical variables, and one-hot encode categorical variables.
5. **PCA:** Analyze explained variance, scree plots, and cumulative explained variance to assess dimensionality reduction.
6. **Regression Modeling:** Compare predictive performance with and without PCA.
7. **Evaluation:** Assess models with MAE, RMSE, and R² on held-out data; report price-scale metrics after reversing any log-target transformation.

## Initial Data Insights

- `SalePrice` is right-skewed; `log1p` substantially improved its distributional symmetry during EDA.
- `OverallQual` and `GrLivArea` showed strong positive correlations with `SalePrice`.
- Several predictors are correlated with one another, including `GarageCars` and `GarageArea`.
- Neighborhood and property-quality categories showed substantial differences in median sale prices.
- In the initial preprocessing experiment, one-hot encoding expanded **79 original input features to 299 transformed features**.

> **Note:** PCA retains variance in the input data, not necessarily the information most predictive of the target. Its usefulness must be judged by validation performance, not explained variance alone.

## Results

See the Jupyter notebook for the final explained-variance analysis, model comparisons, and evaluation figures. Numerical model results should be added here once verified against the final notebook.

## Technologies

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Getting Started

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset CSV files alongside the notebook (or adjust the paths in the notebook).
4. Open the notebook:

   ```bash
   jupyter notebook
   ```

5. Run the cells in order.

## Data Source

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Follow the dataset's applicable terms before redistributing data files.

## Limitations and Future Work

- The labeled dataset is relatively small.
- Missing-value strategies and categorical encoding choices can affect PCA results.
- Explained variance alone does not establish predictive usefulness.
- Future experiments could compare alternative component thresholds, model families, and cross-validation strategies.

## Project Status

Completed exploratory analysis and initial preprocessing; consult the notebook for the latest modeling experiments.
