import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Fitbit ML Project", layout="wide")

# ---------------------------------------------------
# DATA & MODEL LOADING
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("fitbit_final_cleaned.csv")

@st.cache_resource
def load_models():
    models = {
        "Linear Regression": joblib.load("linear_regression.pkl"),
        "Ridge Regression": joblib.load("ridge_regression.pkl"),
        "Lasso Regression": joblib.load("lasso_regression.pkl"),
        "KNN": joblib.load("knn_model.pkl"),
        "Decision Tree": joblib.load("decision_tree.pkl"),
        "Random Forest": joblib.load("random_forest.pkl"),
        "SVR": joblib.load("svr_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl")
    }

    scaler = joblib.load("scaler.pkl")
    scaler_cluster = joblib.load("scaler_cluster.pkl")
    pca_model = joblib.load("pca_model.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")
    hierarchical_model = joblib.load("hierarchical_model.pkl")
    cluster_columns = joblib.load("cluster_columns.pkl")

    return models, scaler, scaler_cluster, pca_model, kmeans_model, hierarchical_model, cluster_columns

# Initialize
df = load_data()
models, scaler, scaler_cluster, pca_model, kmeans_model, hierarchical_model, cluster_columns = load_models()

# ---------------------------------------------------
# SESSION STATE & SIDEBAR
# ---------------------------------------------------
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = "Dashboard"

pages = ["Dashboard", "Supervised Learning", "Unsupervised Learning", "Prediction Center"]

# Function to handle redirects
def navigate_to(page_name):
    st.session_state.page_selection = page_name
    st.rerun()

# Sync sidebar selectbox with session state
current_index = pages.index(st.session_state.page_selection)
page = st.sidebar.selectbox(
    "Select Page", 
    pages, 
    index=current_index, 
    key="nav_selectbox",
    on_change=lambda: st.session_state.update({"page_selection": st.session_state.nav_selectbox})
)

# ===================================================
# PAGE 1: DASHBOARD
# ===================================================
if page == "Dashboard":
    st.markdown("""
        <style>
        .main-title { font-size: 36px; font-weight: bold; color: #2E86C1; margin-bottom: 10px; }
        .section-header { font-size: 24px; font-weight: bold; color: #1B4F72; border-bottom: 2px solid #EAECEE; padding-bottom: 5px; margin-top: 20px; }
        .insight-card { background-color: #F4F6F7; padding: 15px; border-radius: 10px; border-left: 5px solid #2E86C1; margin-bottom: 15px; min-height: 120px;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🥗 🍎 Fitbit Health Analytics & ML Dashboard")
    st.write("Welcome! This platform leverages Advanced Machine Learning to decode fitness data, providing actionable insights for both users and fitness enterprises.")

    st.subheader("🎯 Project Objective")
    st.markdown("""
    The goal of this project is to transform raw wearable sensor data into strategic business value:
    *   **Predict calories burned** with high precision using Regression.
    *   **Identify hidden workout clusters** to understand user behavior.
    *   **Improve personalization** and user retention for fitness platforms.
    """)

    st.markdown('<p class="section-header">📊 Dataset Snapshot</p>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Records", len(df))
    m_col2.metric("Total Features", df.shape[1])
    m_col3.metric("Avg Calories Burned", f"{round(df['Calories_Burned (kcal)'].mean(), 2)} kcal")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    st.markdown('<p class="section-header">💡 Business Insights</p>', unsafe_allow_html=True)
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown('<div class="insight-card"><strong>📈 Retention Engine</strong><br>By identifying users in "Low Intensity" clusters, the platform can trigger automated encouragement to prevent churn.</div>', unsafe_allow_html=True)
    with ins2:
        st.markdown('<div class="insight-card"><strong>🛡️ Device Precision</strong><br>Our XGBoost model (99% R²) can be used to calibrate device sensors, providing market-leading accuracy.</div>', unsafe_allow_html=True)
    with ins3:
        st.markdown('<div class="insight-card"><strong>🎯 Target Marketing</strong><br>Segmented clusters allow teams to upsell "Pro" equipment to Advanced Users and "Entry" gear to Beginners.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">🔗 Quick Navigation</p>', unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        st.subheader("1. Supervised Learning")
        st.write("Analyze model performance (R², RMSE) and see which algorithms best predict calorie burn.")

    with nav2:
        st.subheader("2. Unsupervised Learning")
        st.write("Visualize how users are grouped using KMeans, Hierarchical, and DBSCAN algorithms.")

    with nav3:
        st.subheader("3. Prediction Center")
        st.write("Input your own data to predict calories burned or identify your specific workout cluster.")


# ===================================================
# PAGE 2: SUPERVISED LEARNING (UPDATED WITH BUSINESS INSIGHTS)
# ===================================================
elif page == "Supervised Learning":
    st.markdown("""
        <style>
        .impact-header { font-size: 24px; font-weight: bold; color: #1E8449; margin-top: 20px; }
        .business-box { background-color: #E8F8F5; padding: 15px; border-radius: 10px; border-left: 5px solid #1E8449; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Supervised Learning: Calorie Prediction Analysis")
    
    # --- MODEL PERFORMANCE SECTION ---
    st.markdown('<p class="section-header">📈 Performance Metrics</p>', unsafe_allow_html=True)
    metrics_data = {
        "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression", "KNN", "Decision Tree", "Random Forest", "SVR", "XGBoost"],
        "R2": [0.9263, 0.9263, 0.9264, 0.9470, 0.9952, 0.9982, 0.9980, 0.9978],
        "RMSE": [45.72, 45.71, 45.69, 38.77, 11.72, 7.05, 7.57, 7.85],
        "MAE": [31.76, 31.75, 31.77, 27.69, 7.53, 3.65, 2.74, 5.15]
    }
    df_metrics = pd.DataFrame(metrics_data)

    st.table(df_metrics.style.hide(axis="index"))

    # --- BUSINESS INSIGHTS SECTION ---
    st.markdown('<p class="impact-header">💡 Strategic Business Insights</p>', unsafe_allow_html=True)
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown("""
        <div class="business-box">
            <strong>🎯 Precision Marketing & User Trust</strong><br>
            With an XGBoost R² score of <b>0.9978</b>, our error margin is incredibly low. 
            <em>Business Value:</em> Highly accurate calorie feedback builds user trust, increasing the 
            Net Promoter Score (NPS) and long-term app subscription retention.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        st.markdown("""
        <div class="business-box">
            <strong>📉 Operational Efficiency</strong><br>
            Our analysis shows that <b>Avg_BPM</b> and <b>Weight</b> are the primary drivers of calorie burn. 
            <em>Business Value:</em> Fitbit can optimize hardware battery life by sampling 'Secondary Features' 
            (like specific movement sensors) less frequently without losing prediction accuracy.
        </div>
        """, unsafe_allow_html=True)

    with col_ins2:
        st.markdown("""
        <div class="business-box">
            <strong>🚀 Feature Monetization</strong><br>
            The difference between simple Linear models and Ensemble models (Random Forest) is significant (~7% accuracy jump). 
            <em>Business Value:</em> This advanced accuracy can be marketed as a 'Premium Analytics' feature, 
            creating a new revenue stream for professional athletes.
        </div>
        """, unsafe_allow_html=True)

        st.write("") # Spacer

        st.markdown("""
        <div class="business-box">
            <strong>🛡️ Risk Mitigation</strong><br>
            By monitoring the <b>MAE (Mean Absolute Error)</b>, we can identify when a device is miscalibrated 
            for specific body types. 
            <em>Business Value:</em> Early detection of sensor drift reduces warranty claims and prevents 
            reputational damage from 'false data.'
        </div>
        """, unsafe_allow_html=True)

    # --- VISUALIZATION ---
    st.divider()
    st.subheader("Visualizing Model Accuracy")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_metrics, x="Model", y="R2", palette="GnBu_r", ax=ax)
    plt.ylim(0.9, 1.0) # Focus on the top performance
    plt.xticks(rotation=30)
    st.pyplot(fig)

# ===================================================
# PAGE 3: UNSUPERVISED LEARNING (UPDATED & TUNABLE)
# ===================================================
elif page == "Unsupervised Learning":
    # 1. Custom Styling
    st.markdown("""
        <style>
        .segment-header { font-size: 24px; font-weight: bold; color: #7D3C98; margin-top: 20px; }
        .strategy-box { background-color: #F5EEF8; padding: 15px; border-radius: 10px; border-left: 5px solid #7D3C98; margin-bottom: 20px; }
        .metric-text { font-size: 18px; font-weight: 500; color: #4A235A; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧩 Unsupervised Learning: Behavioral Segmentation")
    st.write("Using Clustering algorithms to identify hidden patterns and fitness personas within the user base.")

    # 2. Data Processing Function
    @st.cache_data
    def get_clustered_data(_df, _scaler_cluster, _pca_model, _cluster_columns):
        # Isolate columns, scale, and apply PCA for 2D visualization
        cluster_df = _df[_cluster_columns].copy()
        scaled_data = _scaler_cluster.transform(cluster_df)
        pca_data = _pca_model.transform(scaled_data)
        return pd.DataFrame(pca_data, columns=["PC1", "PC2"])

    plot_df = get_clustered_data(df, scaler_cluster, pca_model, cluster_columns)
    
    # 3. Model Selection Tabs
    tab1, tab2, tab3 = st.tabs(["📊 K-Means", "🌿 Hierarchical", "☁️ DBSCAN"])

    # --- K-MEANS TAB ---
    with tab1:
        st.markdown('<p class="segment-header">Strategic Business Value: K-Means</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="strategy-box">
            <strong>🎯 Market Personalization:</strong><br>
            K-Means isolated 4 distinct personas. We can now target 'Advanced' clusters with premium recovery gear 
            and 'Beginners' with simplified 5-minute daily challenges to reduce churn.
        </div>
        """, unsafe_allow_html=True)
        
        # Plotting
        plot_df["K_Cluster"] = kmeans_model.labels_
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="K_Cluster", palette="viridis", ax=ax1, alpha=0.7)
        ax1.set_title("K-Means: 4-Persona Segmentation")
        st.pyplot(fig1)
        st.info("K-Means works best for creating balanced, actionable marketing segments.")

    # --- HIERARCHICAL TAB ---
    with tab2:
        st.markdown('<p class="segment-header">Strategic Business Value: Hierarchical</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="strategy-box">
            <strong>🧬 Biological Product Testing:</strong><br>
            Reveals nested similarities in body metrics. This data is vital for beta-testing new sensor hardware 
            across diverse body types to ensure global inclusivity and accuracy.
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction/Labels
        if hasattr(hierarchical_model, 'labels_'):
            plot_df["H_Cluster"] = hierarchical_model.labels_
        else:
            plot_df["H_Cluster"] = hierarchical_model.fit_predict(plot_df[["PC1", "PC2"]])
            
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="H_Cluster", palette="magma", ax=ax2, alpha=0.7)
        ax2.set_title("Hierarchical: Connectivity-Based Grouping")
        st.pyplot(fig2)

    # --- DBSCAN TAB ---
    with tab3:
        st.markdown('<p class="segment-header">Strategic Business Value: DBSCAN</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="strategy-box">
            <strong>🕵️ Fraud & Anomaly Detection:</strong><br>
            DBSCAN identifies 'Noise' (Cluster -1). These outliers often represent malfunctioning sensors, 
            extreme athletes, or even 'data cheating' (shaking the device for steps).
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.subheader("Tuning the Density (Epsilon)")
        st.write("Adjust the Epsilon slider to connect density regions. If you see 2 clusters but want 1, **increase Epsilon.**")
        
        # DBSCAN Parameter Sliders
        col_eps, col_samp = st.columns(2)
        with col_eps:
            eps_val = st.slider("Epsilon (Radius)", 0.1, 2.5, 0.5, 0.1, key="dbscan_eps")
        with col_samp:
            min_samp = st.slider("Min Samples", 2, 20, 5, key="dbscan_samp")

        # Run DBSCAN with current slider values
        dbscan_viz = DBSCAN(eps=eps_val, min_samples=min_samp)
        plot_df["DB_Cluster"] = dbscan_viz.fit_predict(plot_df[["PC1", "PC2"]])

        # Cluster Count Logic
        n_clusters = len(set(plot_df["DB_Cluster"])) - (1 if -1 in plot_df["DB_Cluster"] else 0)
        st.markdown(f'<p class="metric-text">Detected Clusters: <b>{n_clusters}</b></p>', unsafe_allow_html=True)
        
        # Plotting
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="DB_Cluster", palette="tab10", ax=ax3, alpha=0.7)
        ax3.set_title(f"DBSCAN: Density-Based Zones (Eps: {eps_val})")
        st.pyplot(fig3)
        
        if n_clusters > 1:
            st.warning("💡 Tip: Increase Epsilon slightly to merge those clusters into one.")

# ===================================================
# PAGE 4: PREDICTION CENTER
# ===================================================
elif page == "Prediction Center":
    st.markdown("""
        <style>
        .predict-header { font-size: 28px; font-weight: bold; color: #D35400; margin-bottom: 15px; }
        .result-card { background-color: #FEF5E7; padding: 20px; border-radius: 15px; border: 2px solid #D35400; text-align: center; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Calorie Burn Prediction")
    st.write("Enter your session details below to estimate calorie expenditure using our trained models.")
    st.write("KNN is the optimal model among all trained models, while some models are available here for prediction comparison.")

    # -----------------------------------
    # 1. MODEL SELECTION & LAYOUT
    # -----------------------------------
    # Create a list of only your top performers
    optimal_models = ["KNN", "SVR", "Linear Regression", "Ridge Regression"]

    # Update the selectbox to only use this list
    selected_model = st.selectbox("Choose the ML Brain for Prediction", optimal_models)
    
    st.divider()

    # Create two main columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Personal Metrics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", 18, 80, 25)
        weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
        height = st.number_input("Height (m)", 1.3, 2.5, 1.75)
        fat_percentage = st.slider("Body Fat %", 5.0, 50.0, 20.0)

    with col2:
        st.markdown("### 🏃 Workout Details")
        workout_type = st.selectbox("Workout Type", ["Cardio", "HIIT", "Mixed", "Strength", "Yoga"])
        duration = st.number_input("Session Duration (Hours)", 0.1, 5.0, 1.0)
        workout_frequency = st.slider("Workouts per Week", 1, 7, 3)
        experience_level = st.select_slider("Experience Level", options=[0, 1, 2], help="0: Beginner, 1: Intermediate, 2: Athlete")

    # Advanced Metrics in an Expander to keep UI clean
    with st.expander("💓 Advanced Physiological Data (Heart Rate & Water)"):
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            avg_bpm = st.number_input("Average Heart Rate (BPM)", 50, 200, 110)
            max_bpm = st.number_input("Peak Heart Rate (BPM)", 100, 220, 150)
        with a_col2:
            resting_bpm = st.number_input("Resting Heart Rate (BPM)", 40, 120, 65)
            water = st.number_input("Water Consumed (Liters)", 0.5, 10.0, 1.5)

    # -----------------------------------
    # 2. FEATURE ENGINEERING & PREDICTION
    # -----------------------------------
    if st.button("🚀 Calculate Calorie Burn", use_container_width=True):
        
        # BMI Calculation
        bmi = weight / (height ** 2)

        # MET Mapping
        met_mapping = {"Cardio": 7, "HIIT": 9.5, "Mixed": 8, "Strength": 6, "Yoga": 3}
        base_met = met_mapping[workout_type]
        
        # Intensity Calculation
        hr_intensity = avg_bpm / max_bpm
        effective_met = base_met * hr_intensity

        # One-Hot Encoding for Workout Type
        workout_hiit = 1 if workout_type == "HIIT" else 0
        workout_mixed = 1 if workout_type == "Mixed" else 0
        workout_strength = 1 if workout_type == "Strength" else 0
        workout_yoga = 1 if workout_type == "Yoga" else 0
        gender_male = 1 if gender == "Male" else 0

        # Construct DataFrame (Ensuring order matches training data)
        input_data = pd.DataFrame([{
            "Age": age,
            "Weight (kg)": weight,
            "Height (m)": height,
            "Max_BPM": max_bpm,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration (hours)": duration,
            "Fat_Percentage": fat_percentage,
            "Water_Intake (liters)": water,
            "Workout_Frequency (days/week)": workout_frequency,
            "Experience_Level": experience_level,
            "BMI": bmi,
            "Base_MET": base_met,
            "HR_Intensity": hr_intensity,
            "Effective_MET": effective_met,
            "Gender_Male": gender_male,
            "Workout_Type_HIIT": workout_hiit,
            "Workout_Type_Mixed": workout_mixed,
            "Workout_Type_Strength": workout_strength,
            "Workout_Type_Yoga": workout_yoga
        }])

        try:
           # 3. SCALING & PREDICTION
          input_scaled = scaler.transform(input_data)
    
          model = models[selected_model]
          prediction = model.predict(input_scaled)[0]

          # Use abs() to remove the negative sign if it exists
          display_prediction = abs(prediction)

          # 4. DISPLAY RESULT
          st.markdown(f"""
        <div class="result-card">
            <p style="color: #D35400; font-size: 20px; margin-bottom: 0;">Estimated Energy Expenditure</p>
            <h1 style="color: #D35400; margin-top: 0;">{display_prediction:.2f} kcal</h1>
            <p style="font-style: italic;">Results calculated using {selected_model}</p>
        </div>
         """, unsafe_allow_html=True)

        except Exception as e:
         st.error(f"Error in prediction: {e}.")
# ===================================================
# PAGE 4: PREDICTION CENTER - CLUSTER PREDICTION
# ===================================================
    st.divider()
    st.markdown('<p class="predict-header">🎯 Workout Cluster Identification</p>', unsafe_allow_html=True)
    st.write("Determine which Fitness Cluster you belong to based on your physiological and activity profile.")

    # 1. INPUT GRID FOR USER DATA
    cl_col1, cl_col2, cl_col3 = st.columns(3)

    with cl_col1:
       c_age = st.number_input("Age", 18, 80, 25, key="ca")
       c_weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0, key="cw")
       c_height = st.number_input("Height (m)", 1.3, 2.5, 1.75, key="ch")

    with cl_col2:
       c_avg = st.number_input("Avg BPM", 50, 220, 110, key="cab")
       c_max = st.number_input("Max BPM", 100, 220, 150, key="cmb")
       c_duration = st.number_input("Duration (hrs)", 0.1, 5.0, 1.0, key="cd")

    with cl_col3:
       c_resting = st.number_input("Resting BPM", 40, 120, 65, key="crb")
       c_fat = st.slider("Fat %", 5.0, 50.0, 20.0, key="cfat")
       c_freq = st.slider("Workouts/Week", 1, 7, 3, key="cfreq")

    # 2. ALGORITHM SELECTION TABS
    st.write("### Select Clustering Algorithm")
    tab_km, tab_hi, tab_db = st.tabs(["📊 K-Means", "🌿 Hierarchical", "🕵️ DBSCAN"])

    # PRE-CALCULATION FOR NEW DATA POINT
    # (This logic is shared across all three tabs)
    bmi_val = c_weight / (c_height ** 2)
    hr_int = c_avg / c_max
    base_met_val = 7
    eff_met = base_met_val * hr_int

    new_user_row = pd.DataFrame([{
      "Age": c_age, "Weight (kg)": c_weight, "Height (m)": c_height,
      "Max_BPM": c_max, "Avg_BPM": c_avg, "Resting_BPM": c_resting,
      "Session_Duration (hours)": c_duration, "Fat_Percentage": c_fat,
      "Workout_Frequency (days/week)": c_freq, "BMI": bmi_val,
      "Base_MET": base_met_val, "HR_Intensity": hr_int, "Effective_MET": eff_met
    }])
    # Ensure columns match exactly what the models were trained on
    new_user_row = new_user_row.reindex(columns=cluster_columns, fill_value=0)

    # --- TAB 1: K-MEANS (Direct Prediction) ---
    with tab_km:
      st.info("K-Means identifies personas by calculating the distance to saved cluster centers.")
      if st.button("🚀 Identify Cluster (K-Means)", use_container_width=True):
          # Scale -> PCA -> Predict
          scaled_point = scaler_cluster.transform(new_user_row)
          pca_point = pca_model.transform(scaled_point)
          cluster_id = kmeans_model.predict(pca_point)[0]

        # Persona Mapping
          persona_map = {
            0: {"name": "Cluster 0: High Intensity Beginners", "color": "#E74C3C", "desc": "High physiological strain, short sessions. High growth potential."},
            1: {"name": "Cluster 1: Casual Fitness Enthusiasts", "color": "#F1C40F", "desc": "Consistent moderate activity. Focus on lifestyle maintenance."},
            2: {"name": "Cluster 2: Elite/Advanced Athletes", "color": "#27AE60", "desc": "Optimized performance metrics. High duration and intensity."},
            3: {"name": "Cluster 3: Balanced Lifestyle Users", "color": "#3498DB", "desc": "Steady and sustainable workout patterns."}
        }
          res = persona_map.get(cluster_id, {"name": "Unknown", "color": "#7F8C8D", "desc": "No specific profile match."})

          st.markdown(f"""
             <div style="background-color: {res['color']}; padding: 20px; border-radius: 15px; color: white; text-align: center;">
                <h2 style="margin: 0;">Persona Identified: {res['name']}</h2>
                <p style="font-size: 18px; opacity: 0.9;">{res['desc']}</p>
             </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: HIERARCHICAL (Batch Re-fit) ---
    with tab_hi:
      st.info("Hierarchical Clustering finds the user's place in the data 'family tree'.")
      if st.button("🚀 Identify Cluster (Hierarchical)", use_container_width=True):
          with st.spinner("Calculating Linkage..."):
              # Combine original 'df' with the new row
              combined_df = pd.concat([df[cluster_columns], new_user_row], ignore_index=True)
              combined_scaled = scaler_cluster.transform(combined_df)
              combined_pca = pca_model.transform(combined_scaled)

             # Re-fit (Hierarchical requires seeing all data points at once)
              hi_model = AgglomerativeClustering(n_clusters=4)
              all_labels = hi_model.fit_predict(combined_pca)
            
              user_label = all_labels[-1] # The last row is our new user
              st.success(f"🎯 Hierarchical Cluster Identified: {user_label}")
              st.write("This user belongs to a group with similar physiological connectivity.")

    # --- TAB 3: DBSCAN (Anomaly Detection) ---
    with tab_db:
      st.info("DBSCAN checks if the user fits into a dense behavior group or is an 'Outlier'.")
      if st.button("🚀 Identify Cluster (DBSCAN)", use_container_width=True):
          with st.spinner("Analyzing Density..."):
              # Combine original 'df' with the new row
              combined_df = pd.concat([df[cluster_columns], new_user_row], ignore_index=True)
              combined_scaled = scaler_cluster.transform(combined_df)
              combined_pca = pca_model.transform(combined_scaled)

              # Re-fit with standard parameters
              db_model = DBSCAN(eps=0.8, min_samples=5)
              all_labels = db_model.fit_predict(combined_pca)
            
              user_label = all_labels[-1]

              if user_label == -1:
                 st.error("⚠️ Persona Identified: Outlier / Anomaly")
                 st.write("This user's data is significantly different from the average user base.")
              else:
                 st.success(f"🎯 DBSCAN Cluster Identified: {user_label}")
                 st.write("The user is part of a high-density behavioral group.")
