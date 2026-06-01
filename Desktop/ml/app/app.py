from __future__ import annotations

import argparse
from pathlib import Path
import sys
from datetime import datetime, time as dtime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Smart Delivery Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
    }
    .stSelectbox {
        background-color: white;
    }
    .st-emotion-cache-18ni7ap {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px 0;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def _load_artifacts(artifacts_dir: Path):
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
    metadata = joblib.load(artifacts_dir / "metadata.joblib")
    feature_cols = metadata.get("feature_cols", [])
    available_models = metadata.get("available_models", [])
    preferred = metadata.get("best_model")

    model_path = None
    order = ([preferred] if preferred else []) + ["model_rf_tuned.joblib", "model_rf.joblib", "model_linear.joblib"]
    seen = set()
    ordered_unique = [m for m in order if not (m in seen or seen.add(m))]
    for name in ordered_unique:
        candidate = artifacts_dir / name
        if candidate.exists():
            model_path = candidate
            break
    if model_path is None:
        raise FileNotFoundError("No trained model found in artifacts directory.")
    model = joblib.load(model_path)
    return preprocessor, model, feature_cols, model_path.name, available_models


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
	for col in ["Order_Time", "Pickup_Timestamp"]:
		if col in df.columns:
			df[col] = pd.to_datetime(df[col], errors="coerce")
			df[f"{col}_hour"] = df[col].dt.hour
			df[f"{col}_dow"] = df[col].dt.dayofweek
	return df


def _ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
	for col in feature_cols:
		if col not in df.columns:
			df[col] = np.nan
	return df[feature_cols]


def calculate_eta(base_time: datetime, predicted_duration: float) -> datetime:
    return base_time + pd.Timedelta(minutes=predicted_duration)

def format_time(minutes: float) -> str:
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"

def main() -> None:
    if st.runtime.exists():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--artifacts_dir", type=str, default="models")
        known, _ = parser.parse_known_args(sys.argv[1:])
        artifacts_dir = Path(known.artifacts_dir)
    else:
        artifacts_dir = Path("models")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1>🚚 Smart Delivery Predictor</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 30px;'>
        Powered by Machine Learning | Real-time Predictions | High Accuracy
        </div>
    """, unsafe_allow_html=True)

    # Sidebar styling
    with st.sidebar:
        st.markdown("### 🔧 Model Configuration")
        st.markdown("---")
        st.markdown("#### Current Model Status")
        
        preprocessor, model, feature_cols, model_name, available_models = _load_artifacts(artifacts_dir)
        
        st.success(f"✅ Active Model: {model_name}")
        st.info(f"📂 Artifacts Directory: {str(artifacts_dir.resolve())}")
        
        if available_models:
            st.markdown("#### Model Selection")
            chosen = st.selectbox(
                "Choose Model Version",
                options=available_models,
                index=max(available_models.index(model_name) if model_name in available_models else 0, 0)
            )
            if chosen != model_name:
                model = joblib.load(artifacts_dir / chosen)
                model_name = chosen

    # Main content
    tab_single, tab_batch, tab_metrics = st.tabs([
        "🎯 Single Prediction",
        "📦 Batch Prediction",
        "📊 Analytics & Metrics"
    ])

    with tab_single:
        st.markdown("""
            <div class='status-box' style='background-color: #f8f9fa;'>
                <h3>📍 Delivery Time Estimation</h3>
                <p>Enter delivery details below for an instant prediction.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                distance_km = st.number_input(
                    "📏 Distance (km)",
                    min_value=0.0,
                    step=0.1,
                    value=5.0,
                    help="Enter the delivery distance in kilometers"
                )
                vehicle_type = st.selectbox(
                    "🚗 Vehicle Type",
                    ["Bike", "Car", "Van", "Truck"],
                    help="Select the type of delivery vehicle"
                )
                traffic = st.selectbox(
                    "🚦 Traffic Conditions",
                    ["Low", "Medium", "High"],
                    help="Current traffic conditions on the route"
                )
            
            with col2:
                weather = st.selectbox(
                    "🌤️ Weather Conditions",
                    ["Clear", "Rain", "Snow", "Fog", "Windy"],
                    help="Select current weather conditions"
                )
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    order_date = st.date_input(
                        "📅 Order Date",
                        value=datetime.now().date()
                    )
                    pickup_date = st.date_input(
                        "🗓️ Pickup Date",
                        value=datetime.now().date()
                    )
                with col2_2:
                    order_tod = st.time_input(
                        "⏰ Order Time",
                        value=datetime.now().time()
                    )
                    pickup_tod = st.time_input(
                        "⏱️ Pickup Time",
                        value=datetime.now().time()
                    )

            submitted = st.form_submit_button("🚀 Calculate Delivery Time")

        if submitted:
            try:
                with st.spinner('Calculating delivery prediction...'):
                    row = {
                        "Distance_km": distance_km,
                        "Vehicle_Type": vehicle_type,
                        "Traffic": traffic,
                        "Weather": weather,
                        "Order_Time": pd.Timestamp(datetime.combine(order_date, order_tod if isinstance(order_tod, dtime) else datetime.now().time())),
                        "Pickup_Timestamp": pd.Timestamp(datetime.combine(pickup_date, pickup_tod if isinstance(pickup_tod, dtime) else datetime.now().time())),
                    }
                    X = pd.DataFrame([row])
                    X = _engineer_time_features(X)
                    X = _ensure_feature_columns(X, feature_cols)

                    Xt = preprocessor.transform(X)
                    pred = model.predict(Xt)[0]
                    st.metric("Predicted Delivery Time (minutes)", f"{pred:.1f}")
                    # ETA timestamp = Pickup_Timestamp + predicted minutes
                    if pd.notna(row["Pickup_Timestamp"]) and isinstance(row["Pickup_Timestamp"], pd.Timestamp):
                        eta = row["Pickup_Timestamp"] + pd.to_timedelta(float(pred), unit="m")
                        st.caption(f"ETA: {eta}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    with tab_batch:
        st.write("Upload a CSV with columns: Distance_km, Vehicle_Type, Traffic, Weather, Order_Time, Pickup_Timestamp")
        upload = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
        if upload is not None:
            try:
                df_in = pd.read_csv(upload)
                st.dataframe(df_in.head(10))
                # Parse date/time text to timestamps if needed
                df_proc = df_in.copy()
                for col in ["Order_Time", "Pickup_Timestamp"]:
                    if col in df_proc.columns:
                        df_proc[col] = pd.to_datetime(df_proc[col], errors="coerce")
                X = _engineer_time_features(df_proc)
                X = _ensure_feature_columns(X, feature_cols)
                Xt = preprocessor.transform(X)
                preds = model.predict(Xt)
                out = df_in.copy()
                out["Predicted_Delivery_Minutes"] = preds
                # ETA per row if pickup timestamp present
                if "Pickup_Timestamp" in out.columns:
                    pt = pd.to_datetime(out["Pickup_Timestamp"], errors="coerce")
                    out["ETA"] = pt + pd.to_timedelta(preds, unit="m")
                st.dataframe(out.head(20))
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")

    with tab_metrics:
        st.subheader("Model Information")
        st.write(f"Active model: {model_name}")
        # Show saved metrics if available
        try:
            metrics = joblib.load(artifacts_dir / "metrics.joblib")
            st.write("Saved metrics:", metrics)
            if (artifacts_dir / "tuning_results.joblib").exists():
                tr = joblib.load(artifacts_dir / "tuning_results.joblib")
                st.write("Tuning results:", {k: v for k, v in tr.items() if k != "best_estimator_"})
        except Exception:
            st.info("Metrics not found yet. Train the models first.")

        # Feature importances for tree-based models
        try:
            if hasattr(model, "feature_importances_"):
                feature_names = None
                try:
                    # ColumnTransformer can provide names
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    pass
                importances = model.feature_importances_
                if feature_names is None or len(feature_names) != len(importances):
                    feature_names = [f"f{i}" for i in range(len(importances))]
                imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
                st.bar_chart(imp_df.set_index("feature"))
            else:
                st.caption("Feature importances not available for the current model.")
        except Exception:
            st.caption("Could not compute feature importances.")


if __name__ == "__main__":
	main()


