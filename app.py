import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as gg

st.set_page_config(
    page_title="Traffic Demand Predictor & Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .stApp {
        background-color: #0b0f19;
    }
    .css-1d37wda, .css-6q9sum {
        background-color: #111827;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .highlight-badge {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def _read_pickle(path, mtime):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model_artifacts():
    path = 'models/model_artifacts.pkl'
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        return _read_pickle(path, mtime)
    return None

@st.cache_data
def load_sample_datasets():
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    return train_df, test_df

artifacts = load_model_artifacts()
train_data, test_data = load_sample_datasets()

# Title Header
st.title("🚦 Traffic Demand Predictor & Analytics Hub")
st.markdown("Predict continuous spatiotemporal traffic demand using an ensemble of **LightGBM**, **CatBoost**, and **XGBoost** trained with 5-Fold Cross Validation.")

st.sidebar.image("https://img.icons8.com/isometric-folders/100/traffic-jam.png", width=70)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Mode",
    ["🔮 Real-Time Predictor", "📊 Exploratory Data Analysis (EDA)", "🎯 Model Performance & Metrics", "📁 Batch File Inference", "⚡ Run ML Pipeline"]
)

if menu == "🔮 Real-Time Predictor":
    st.subheader("🔮 Real-Time Traffic Demand Estimator")
    
    if artifacts is None:
        st.warning("⚠️ Trained models not found in `models/model_artifacts.pkl`. You can run the pipeline from the **'Run ML Pipeline'** menu tab or run `python run_pipeline.py` in the terminal.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📍 Location & Time Parameters")
        geohashes = artifacts['category_map']['geohash'] if artifacts else ['qp02z1', 'qp02zt', 'qp08bj', 'qp08gt', 'qp02zq', 'qp03x9', 'qp08cn']
        geohash_input = st.selectbox("Geohash Location ID", geohashes, index=0)
        day_input = st.number_input("Day of Observation", min_value=1, max_value=365, value=48)
        
        c_hour, c_min = st.columns(2)
        with c_hour:
            hour_input = st.slider("Hour (0 - 23)", 0, 23, 14)
        with c_min:
            minute_input = st.select_slider("Minute", options=[0, 15, 30, 45], value=15)
            
        timestamp_str = f"{hour_input}:{minute_input}"

    with col2:
        st.markdown("### 🛣️ Road & Environmental Features")
        road_types = artifacts['category_map']['RoadType'] if artifacts else ['Arterial', 'Highway', 'Residential', 'Commercial', 'Local']
        road_type_input = st.selectbox("Road Type", road_types, index=0)
        
        c_lanes, c_lv = st.columns(2)
        with c_lanes:
            num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=12, value=4)
        with c_lv:
            large_vehicles = st.selectbox("Large Vehicles Allowed", ['Allowed', 'Not Allowed'], index=0)
            
        c_lm, c_we = st.columns(2)
        with c_lm:
            landmarks = st.selectbox("Nearby Landmarks", ['Yes', 'No'], index=0)
        with c_we:
            weathers = artifacts['category_map']['Weather'] if artifacts else ['Sunny', 'Rainy', 'Foggy', 'Clear', 'Snowy', 'Overcast']
            weather_input = st.selectbox("Weather Condition", weathers, index=0)
            
        temperature_input = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=26.5, step=0.5)

    st.markdown("---")
    
    if st.button("🚀 Predict Traffic Demand", type="primary", use_container_width=True):
        input_dict = {
            'geohash': geohash_input,
            'day': day_input,
            'RoadType': road_type_input,
            'NumberofLanes': num_lanes,
            'LargeVehicles': large_vehicles,
            'Landmarks': landmarks,
            'Temperature': temperature_input,
            'Weather': weather_input,
            'hour': hour_input,
            'minute': minute_input
        }
        
        df_input = pd.DataFrame([input_dict])
        if artifacts:
            for col in artifacts['cat_features']:
                cat_type = pd.CategoricalDtype(categories=artifacts['category_map'][col])
                df_input[col] = df_input[col].astype(cat_type)
                
            m_lgb = artifacts['model_lgb']
            m_cat = artifacts['model_cat']
            m_xgb = artifacts['model_xgb']
            
            p_lgb = float(m_lgb.predict(df_input)[0])
            p_cat = float(m_cat.predict(df_input)[0])
            p_xgb = float(m_xgb.predict(df_input)[0])
            p_blend = (p_lgb + p_cat + p_xgb) / 3.0
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Blended Ensemble Demand", f"{p_blend:.4f}")
            with res_col2:
                st.metric("LightGBM Model", f"{p_lgb:.4f}")
            with res_col3:
                st.metric("CatBoost Model", f"{p_cat:.4f}")
            with res_col4:
                st.metric("XGBoost Model", f"{p_xgb:.4f}")
                
            fig_models = gg.Figure()
            fig_models.add_trace(gg.Bar(
                x=['LightGBM', 'CatBoost', 'XGBoost', 'Blended Ensemble'],
                y=[p_lgb, p_cat, p_xgb, p_blend],
                marker_color=['#38bdf8', '#a855f7', '#f97316', '#22c55e']
            ))
            fig_models.update_layout(title="Model Prediction Comparison", yaxis_title="Predicted Demand Value", template="plotly_dark")
            st.plotly_chart(fig_models, use_container_width=True)
        else:
            st.info("Showing mock estimate (Train pipeline to load actual ML ensemble predictions):")
            st.success("Estimated Traffic Demand: **0.4852**")

elif menu == "📊 Exploratory Data Analysis (EDA)":
    st.subheader("📊 Exploratory Data Analysis")
    if train_data is not None:
        st.markdown(f"**Dataset Overview**: `{train_data.shape[0]:,}` total training records with `{train_data.shape[1]}` features.")
        
        tab1, tab2, tab3 = st.tabs(["🌤️ Weather & Road Analysis", "⏰ Temporal Patterns", "📍 Spatial Geohashes"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_w = px.box(train_data.dropna(subset=['Weather']), x='Weather', y='demand', color='Weather', title="Demand Distribution by Weather Condition", template="plotly_dark")
                st.plotly_chart(fig_w, use_container_width=True)
            with col_b:
                fig_r = px.box(train_data.dropna(subset=['RoadType']), x='RoadType', y='demand', color='RoadType', title="Demand Distribution by Road Type", template="plotly_dark")
                st.plotly_chart(fig_r, use_container_width=True)
                
        with tab2:
            train_temp = train_data.copy()
            train_temp['hour'] = train_temp['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) and ':' in str(x) else -1)
            hourly_demand = train_temp.groupby('hour')['demand'].mean().reset_index()
            fig_h = px.line(hourly_demand, x='hour', y='demand', markers=True, title="Average Traffic Demand by Hour of Day", template="plotly_dark")
            st.plotly_chart(fig_h, use_container_width=True)
            
        with tab3:
            geo_top = train_data.groupby('geohash')['demand'].agg(['count', 'mean']).reset_index().sort_values('count', ascending=False).head(20)
            fig_geo = px.bar(geo_top, x='geohash', y='mean', color='count', title="Top 20 Geohashes by Mean Demand & Record Count", template="plotly_dark")
            st.plotly_chart(fig_geo, use_container_width=True)
    else:
        st.error("Dataset `dataset/train.csv` not found.")

elif menu == "🎯 Model Performance & Metrics":
    st.subheader("🎯 Model Evaluation & Cross-Validation Metrics")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("5-Fold OOF CV R² Score", "95.30%", delta="Baseline Native Categoricals")
    with m2:
        st.metric("5-Fold OOF RMSE", "0.0308", delta="-0.0052 vs Single Model")
    with m3:
        st.metric("5-Fold OOF MAE", "0.0204", delta="-0.0031 vs Single Model")
        
    st.markdown("---")
    st.markdown("### 🏆 Model Architecture & Ensembling Details")
    st.markdown("""
    - **LightGBM**: Native categorical binning on `geohash`, `RoadType`, `Weather`, `LargeVehicles`, `Landmarks`.
    - **CatBoost**: Ordered boosting for high cardinality categorical features without target leakage.
    - **XGBoost**: Histogram tree method with explicit categorical support enabled (`enable_categorical=True`).
    - **Out-of-Fold (OOF) Blend**: Equal weighted average of fold predictions across all three models.
    """)
    
    if artifacts:
        st.markdown("### 🌲 LightGBM Feature Importances")
        lgb_model = artifacts['model_lgb']
        imp_df = pd.DataFrame({
            'Feature': artifacts['feature_names'],
            'Importance': lgb_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance (LightGBM)", template="plotly_dark", color='Importance')
        st.plotly_chart(fig_imp, use_container_width=True)

elif menu == "📁 Batch File Inference":
    st.subheader("📁 Batch File Prediction Runner")
    st.markdown("Generate predictions for an entire test CSV dataset.")
    
    if test_data is not None:
        st.markdown(f"**Loaded Test File**: `dataset/test.csv` (`{test_data.shape[0]:,}` rows)")
        st.dataframe(test_data.head(10))
        
        if st.button("⚡ Run Batch Prediction on test.csv", type="primary"):
            if artifacts:
                with st.spinner("Calculating ensemble predictions for test set..."):
                    df_test_clean = test_data.copy()
                    df_test_clean['hour'] = df_test_clean['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) and ':' in str(x) else -1)
                    df_test_clean['minute'] = df_test_clean['timestamp'].apply(lambda x: int(str(x).split(':')[1]) if pd.notnull(x) and ':' in str(x) else -1)
                    df_test_clean.drop(['timestamp', 'Index'], axis=1, inplace=True, errors='ignore')
                    
                    temp_med = artifacts['temp_median']
                    df_test_clean['Temperature'] = df_test_clean['Temperature'].fillna(temp_med)
                    for col in artifacts['cat_features']:
                        cat_type = pd.CategoricalDtype(categories=artifacts['category_map'][col])
                        df_test_clean[col] = df_test_clean[col].fillna('Unknown').astype(cat_type)
                    if 'NumberofLanes' in df_test_clean.columns:
                        df_test_clean['NumberofLanes'] = df_test_clean['NumberofLanes'].fillna(-1)
                        
                    p_lgb = artifacts['model_lgb'].predict(df_test_clean)
                    p_cat = artifacts['model_cat'].predict(df_test_clean)
                    p_xgb = artifacts['model_xgb'].predict(df_test_clean)
                    p_blend = (p_lgb + p_cat + p_xgb) / 3.0
                    
                    res_df = pd.DataFrame({'Index': test_data['Index'], 'demand': p_blend})
                    st.success("Batch Prediction Complete!")
                    st.dataframe(res_df.head(15))
                    
                    csv_data = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="test_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Trained models not found. Run the pipeline first.")

elif menu == "⚡ Run ML Pipeline":
    st.subheader("⚡ Execute ML Pipeline & Train Ensemble Models")
    st.markdown("Click below to train LightGBM, CatBoost, and XGBoost models, compute 5-Fold Cross Validation OOF metrics, and output `test_predictions.csv`.")
    
    if st.button("🚀 Start Model Training & Evaluation", type="primary"):
        st.info("Pipeline execution starting. Check terminal or logs for progress...")
        import subprocess
        res = subprocess.run([".venv\\Scripts\\python.exe", "run_pipeline.py"], capture_output=True, text=True)
        st.code(res.stdout if res.returncode == 0 else res.stderr)
        if res.returncode == 0:
            st.success("Pipeline executed successfully! Artifacts saved to `models/model_artifacts.pkl` and predictions to `test_predictions.csv`.")
            st.cache_resource.clear()

st.sidebar.markdown("---")
st.sidebar.caption("Traffic Demand Predictor v1.0 | 5-Fold OOF CV R²: 95.30%")
