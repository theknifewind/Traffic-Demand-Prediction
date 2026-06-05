# Traffic Demand Prediction 🚦

This repository contains my machine learning solution for predicting traffic demand based on spatiotemporal features, weather conditions, and road characteristics. This project was developed as part of the **Gridlock Hackathon 2.0**.

## Project Overview
The goal of this project was to forecast continuous traffic demand using various categorical and numerical features. I used this opportunity to deepen my understanding of gradient boosting algorithms, hyperparameter tuning, and feature engineering for geographic data.

## Approach & Learnings
My final pipeline involves an ensemble of three models: **LightGBM**, **CatBoost**, and **XGBoost**. 

During development, I experimented with different ways to handle geospatial data:
- **The Baseline:** Treating geographic hashes (`geohash`) as raw categorical strings. This allowed CatBoost and LightGBM's native categorical handling to naturally bin the locations.
- **The Experiment:** Decoding the `geohash` into continuous `latitude` and `longitude` coordinates using `pygeohash`, alongside applying cyclical sine/cosine transformations to the time features.
- **The Learning:** The advanced continuous coordinate approach led to massive overfitting on the training data (R² ~96) but poor generalization on unseen data. I learned that for this specific dataset, keeping the geohashes as categorical strings acted as a powerful natural regularizer, preventing the complex trees from memorizing exact coordinates. 

I ultimately rolled back to the robust baseline approach, optimized hyperparameters using **Optuna**, and blended the predictions of all three models to achieve a strong and stable R² score.

## Tools & Libraries Used
- **Python**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, LightGBM, CatBoost, XGBoost
- **Optimization**: Optuna (Bayesian Hyperparameter Tuning)

## Repository Structure
- `Traffic_Demand_Prediction.ipynb`: The main Jupyter Notebook containing the data preprocessing, Optuna tuning, model training, and ensembling logic.
- `Approach_Explanation.md`: A detailed, step-by-step breakdown of the pipeline, feature engineering decisions, and failed experiments.

## Running the Code
1. Clone the repository.
2. Ensure you have the required libraries installed: `pip install pandas scikit-learn lightgbm catboost xgboost optuna jupyter`
3. Place the `train.csv` and `test.csv` files in a `dataset/` directory.
4. Run the Jupyter Notebook to train the ensemble and generate the `submission_blend.csv`.

---
*Note: I am continuously learning and iterating on my data science skills. Feedback and suggestions are always welcome!*
