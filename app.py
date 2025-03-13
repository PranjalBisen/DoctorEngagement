import streamlit as st
st.set_page_config(page_title="Doctor Engagement Predictor", layout="wide")

import pandas as pd
import joblib
import altair as alt
from datetime import time, timedelta


model = joblib.load('survey_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
doctor_profiles = pd.read_csv('doctor_profiles.csv')

def prepare_features(input_time, input_day):
    """
    Prepare the feature DataFrame for prediction.
    Updates time features based on the user's input, computes deviations,
    and one-hot encodes categorical variables to align with training data.
    """
    current_hour = input_time.hour + input_time.minute / 60
    features = doctor_profiles.copy()
    features['Login_Hour'] = current_hour           
    features['Login_Day'] = input_day               
    features['Hour_Deviation'] = abs(current_hour - features['Doctor_Avg_Login'])
    features['Day_Deviation'] = abs(input_day - features['Login_Day'])
    features['Duration'] = features['Doctor_Avg_Duration'] 

    
    X = pd.get_dummies(features, columns=['Doctor_Speciality', 'Doctor_Region'])
    
    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[feature_columns]
    return X


if "show_predictor" not in st.session_state:
    st.session_state.show_predictor = False


if not st.session_state.show_predictor:
    st.title("Doctor Engagement Predictor")
    st.markdown(
        """
        Welcome to the Doctor Engagement Predictor!
        
        This tool uses historical survey and login data to suggest optimal survey timing for outreach.
        Our model uses a simple Logistic Regression approach with L2 regularization 
        to rank doctors by engagement probability.
        """
    )
    if st.button("Get Started"):
        st.session_state.show_predictor = True
    st.stop()


st.title("Doctor Engagement Predictor")
st.markdown("Select appointment details from the sidebar and click **Predict Engagement**.")

with st.sidebar:
    st.header("Settings")
    selected_time = st.slider(
        "Preferred survey time",
        value=time(12, 0),
        min_value=time(0, 0),
        max_value=time(23, 59),
        step=timedelta(minutes=15),
        format="HH:mm",
        help="Choose the time you plan to send the survey invitation"
    )
    selected_day = st.number_input(
        "Day of month",
        min_value=1,
        max_value=31,
        value=15,
        help="Choose the day of the month for survey distribution"
    )
    st.subheader("Filters")
    specialties = doctor_profiles['Doctor_Speciality'].unique()
    selected_specialties = st.multiselect(
        "Specialties",
        options=list(specialties),
        default=list(specialties),
        help="Select which specialties to include in predictions"
    )
    regions = doctor_profiles['Doctor_Region'].unique()
    selected_regions = st.multiselect(
        "Regions",
        options=list(regions),
        default=list(regions),
        help="Select which regions to include in predictions"
    )
    predict_button = st.button("Predict Engagement")

if predict_button:
    with st.spinner("Crunching numbers..."):
        X_pred = prepare_features(selected_time, selected_day)
        probabilities = model.predict_proba(X_pred)[:, 1]
        results = pd.DataFrame({
            'NPI': doctor_profiles['NPI'],
            'Speciality': doctor_profiles['Doctor_Speciality'],
            'Region': doctor_profiles['Doctor_Region'],
            'Probability': probabilities
        })
        
        filtered_results = results[
            (results['Speciality'].isin(selected_specialties)) &
            (results['Region'].isin(selected_regions))
        ].copy()
        
        filtered_results['Recommendation'] = pd.cut(
            filtered_results['Probability'],
            bins=[0, 0.4, 0.7, 1],
            labels=['Low', 'Medium', 'High']
        )
        st.session_state.results = filtered_results.sort_values('Probability', ascending=False)

if "results" in st.session_state:
    st.header("Prediction Results")
    st.subheader(f"Time: {selected_time.strftime('%I:%M %p')} on Day {selected_day}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Probability", f"{st.session_state.results['Probability'].mean():.1%}")
    with col2:
        st.metric("High Candidates", (st.session_state.results['Probability'] > 0.7).sum())
    with col3:
        st.metric("Total Doctors", len(st.session_state.results))
    
    st.subheader("Top Recommendations")
    top_10 = st.session_state.results.head(10)
    st.dataframe(
        top_10.sort_values('Probability', ascending=False)
        .style.format({'Probability': '{:.2%}'})
        .background_gradient(subset=['Probability'], cmap='Blues'),
        use_container_width=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Probability Distribution")
        hist_chart = alt.Chart(st.session_state.results).mark_bar().encode(
            x=alt.X('Probability:Q', bin=alt.Bin(maxbins=20)),
            y='count()',
            tooltip=['count()']
        )
        st.altair_chart(hist_chart, use_container_width=True)
    with col2:
        st.subheader("By Speciality")
        specialty_dist = st.session_state.results.groupby('Speciality')['Probability'].mean().reset_index()
        bar_chart = alt.Chart(specialty_dist).mark_bar().encode(
            x='Speciality',
            y='Probability',
            tooltip=['Speciality', 'Probability']
        ).properties(height=300)
        st.altair_chart(bar_chart, use_container_width=True)
    
    st.subheader("Full Results")
    st.dataframe(
        st.session_state.results.sort_values('Probability', ascending=False)
        .style.format({'Probability': '{:.2%}'})
        .background_gradient(subset=['Probability'], cmap='Blues'),
        height=400,
        use_container_width=True
    )
    
    st.download_button(
        "Download Results",
        data=st.session_state.results.to_csv(index=False),
        file_name="doctor_predictions.csv",
        mime="text/csv"
    )
    
    with st.expander("About This Model"):
        st.markdown(
            """
            **Performance Information**
            - Model: Logistic Regression with L2 regularization
            - The model ranks doctors based on relative engagement probability.
            
            """
        )
    
    st.markdown("---")
    st.markdown("Predictions are based on historical login patterns and survey attempt data.")
