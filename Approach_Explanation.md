# Traffic Demand Prediction: Comprehensive Project Walkthrough

This document serves as a complete record of the machine learning pipeline developed for the Traffic Demand Prediction hackathon. It includes our feature engineering strategies, model selection, the errors we encountered, our advanced experiments, and the rationale behind our final, best-performing approach. This is designed to serve as a reference guide for explaining the project end-to-end.

---

## 1. The Baseline Approach (Our Final & Best Model)
Our goal was to predict continuous traffic `demand` based on a mix of spatiotemporal and categorical features. We utilized three powerful gradient boosting frameworks: **LightGBM**, **CatBoost**, and **XGBoost**.

### Feature Engineering
- **Temporal Features**: The `timestamp` column (e.g., `"0:0"`) was split into raw `hour` and `minute` integers. Tree-based models can easily split on these integers to isolate peak vs. off-peak hours.
- **Categorical Features**: Features like `geohash`, `RoadType`, and `Weather` were treated as standard Pandas categorical types.
- **Missing Values**: Missing categoricals were explicitly filled with `'Unknown'` (providing a specific split branch for missingness), while numericals like `Temperature` were imputed with the training median.

### Model Selection & Tuning
- **Why these models?**: CatBoost is unparalleled at natively handling high-cardinality categoricals (like `geohash`) without heavy pre-processing. LightGBM is incredibly fast and memory-efficient (leaf-wise growth), and XGBoost is a robust powerhouse.
- **Optuna Tuning**: We utilized Optuna (Bayesian optimization) combined with K-Fold cross-validation to search for optimal hyperparameters (like `learning_rate` and `num_leaves`) using the hackathon's exact metric: `max(0, 100 * r2_score)`.
- **Final Training**: We trained the final models on **100% of the training dataset** to maximize the data they learned from.

### Ensembling (Blending)
We averaged the predictions of the three models: `(LightGBM_preds + CatBoost_preds + XGBoost_preds) / 3.0`.
- **Advantage**: LightGBM (leaf-wise), XGBoost (depth-wise), and CatBoost (symmetric) grow trees differently, making slightly different errors. Averaging them smooths out individual variances, preventing overfitting and yielding a highly robust score (91-92 range).

---

## 2. Obstacles Encountered: Submission Dimension Error
During early development, we encountered a critical dimensional error:
> `ValueError: Length of values (41778) does not match length of index (5)`

- **The Cause**: We initially tried to inject our 41,778 test predictions into the `sample_submission.csv` template file. However, that file was just a dummy reference containing only 5 rows.
- **The Fix**: We discarded the template and updated the code to dynamically generate a new Pandas DataFrame matching the exact `Index` array from `test.csv` to our predictions. This ensured a flawless 41778 x 2 submission structure.

---

## 3. The "Advanced" Experiment (And Why It Failed)
In an attempt to push our score from 91 up to the 95-98 range, we fundamentally altered the pipeline to extract deeper signal. 

### What We Changed:
1. **Geohash Decoding**: We used the `pygeohash` library to mathematically decode the categorical `geohash` strings into continuous `latitude` and `longitude` coordinates. (Idea: Let the models understand true spatial proximity).
2. **Cyclical Time Encoding**: We applied Sine and Cosine transformations to `hour` and `minute` to map time onto a circle. (Idea: Teach the model that 23:59 and 00:00 are adjacent).
3. **K-Fold OOF Blending**: Instead of training on 100% of the data, we used 5-Fold Cross Validation. We trained 5 models per algorithm, each predicting on the test set, and averaged them to prevent overfitting.

### The Result:
- **Training R² Score**: Dropped from **~96% down to ~90%**.
- **Leaderboard Score**: Dropped drastically from **92 down to 86**.

### Disadvantages & Analysis (Why it failed):
The advanced approach failed to capture the true underlying patterns of the data.
- When we decoded `geohash` into continuous latitude and longitude, the gradient boosting trees struggled to create effective splits. The continuous coordinates created too much noise, preventing the models from learning clear spatial boundaries, which caused performance to drop on both the training and test sets.
- In contrast, our baseline approach (treating `geohash` as a raw categorical string) is extremely effective because CatBoost and LightGBM naturally group and bin categorical strings. This allows the models to cleanly separate distinct geographic regions, leading to a much higher Training R² (96%) and a strong generalization on the test set (92).

---

## 4. Final Conclusion & Rollback
Recognizing the overfitting caused by the advanced spatial decoding, we executed a full rollback to the original baseline pipeline. 

**Summary of the Final State:**
The final code completely relies on treating `geohash` as a native categorical feature, leveraging the inherent regularization of CatBoost and LightGBM's categorical handling. By combining basic temporal integer extraction, median imputation, Optuna hyperparameter tuning, and a simple 3-model ensemble blend, we struck the perfect balance between learning complex traffic patterns and generalizing to unseen data, securing a strong >91 score.
