# Traffic Demand Prediction 🚦

This repository contains an end-to-end machine learning solution for predicting traffic demand based on spatiotemporal features, weather conditions, and road characteristics.

## Project Overview
The goal of this project is to forecast continuous traffic demand using spatial, temporal, and environmental features. The pipeline implements gradient boosting algorithms, Optuna Bayesian hyperparameter tuning, high-cardinality categorical feature encoding, and rigorous 5-Fold Out-of-Fold (OOF) cross-validation evaluation.

## Approach & Learnings
The final pipeline utilizes an ensemble of three gradient boosting architectures: **LightGBM**, **CatBoost**, and **XGBoost**.

To ensure robust evaluation on unseen data and eliminate overfitting risks, all models were validated using **5-Fold Cross-Validation** to measure performance on held-out out-of-fold (OOF) samples:
- **The Baseline (Final Model):** Treating geographic hashes (`geohash`) as native categorical strings. CatBoost and LightGBM's categorical handling naturally binned spatial regions, achieving a **95.30% 5-Fold Out-of-Fold (OOF) CV R²** (OOF RMSE: `0.0308`, OOF MAE: `0.0204`).
- **The Experiment:** Decoding `geohash` into continuous `latitude` and `longitude` coordinates using `pygeohash` alongside cyclical sine/cosine time transformations.
- **The Learning:** Continuous spatial coordinates degraded generalization, dropping 5-Fold OOF CV R² to **90.31%**. High-cardinality categorical binning preserved sharp micro-regional boundaries far better than axis-aligned splits on continuous coordinates.

Hyperparameters were optimized using **Optuna** (Bayesian Hyperparameter Tuning) across 5 cross-validation folds. Predictions from all three models were blended to construct a stable, highly generalized regression pipeline.

## Tools & Libraries Used
- **Python**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, LightGBM, CatBoost, XGBoost
- **Optimization**: Optuna (Bayesian Hyperparameter Tuning)

## Repository Structure
- `Traffic_Demand_Prediction.ipynb`: The main Jupyter Notebook containing data preprocessing, Optuna tuning, 5-fold cross-validation, model training, and ensembling logic.
- `Approach_Explanation.md`: A detailed breakdown of the machine learning pipeline, feature engineering decisions, cross-validation metrics, and experimental trade-offs.

## Running the Code
1. Clone the repository.
2. Ensure required dependencies are installed: `pip install pandas scikit-learn lightgbm catboost xgboost optuna jupyter`
3. Place `train.csv` and `test.csv` in the `dataset/` directory.
4. Run `Traffic_Demand_Prediction.ipynb` to execute preprocessing, cross-validation, ensemble training, and test inference.

