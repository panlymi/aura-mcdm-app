import streamlit as st
import pandas as pd
from aura_calculator import calculate_aura

st.set_page_config(page_title="AURA MCDM Calculator", layout="wide")

st.title("Adaptive Utility Ranking Algorithm (AURA) Calculator")
st.markdown("""
This application implements the Adaptive Utility Ranking Algorithm (AURA) for Multi-Criteria Decision Making (MCDM).
Upload your decision matrix as an Excel or CSV file. The file should have alternatives as rows and criteria as columns.
The first column should contain the names of the alternatives.
""")

st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Decision Matrix", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, index_col=0)
    else:
        df = pd.read_excel(uploaded_file, index_col=0)
    
    st.subheader("Input Decision Matrix")
    st.dataframe(df)
    
    # Ensure numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        st.error("The uploaded file does not contain numeric data suitable for MCDM.")
        st.stop()
        
    criteria = numeric_df.columns.tolist()
    
    st.sidebar.subheader("AURA Parameters")
    alpha = st.sidebar.slider(
        "Balance Parameter (α)", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Balances relative closeness (TOPSIS) vs. distance to Average Solution. 1.0 means pure relative closeness."
    )
    
    p_metric = st.sidebar.selectbox(
        "Distance Metric (p)",
        options=[1, 2],
        index=1,
        help="1 = Manhattan Distance, 2 = Euclidean Distance"
    )
    
    st.sidebar.subheader("Criteria Weights & Directions")
    
    # Initialize dataframe for data editor
    weights_df_init = pd.DataFrame({
        "Criterion": criteria,
        "Weight": 1.0,
        "Direction": "maximize"
    })
    
    # Data editor for fast multi-row input (supports Enter key navigation & unformatted decimals)
    edited_df = st.sidebar.data_editor(
        weights_df_init,
        column_config={
            "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, format=None, step=None), 
            "Direction": st.column_config.SelectboxColumn("Direction", options=["maximize", "minimize"])
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Extract edited values back to dictionaries
    weights = dict(zip(edited_df["Criterion"], edited_df["Weight"]))
    directions = dict(zip(edited_df["Criterion"], edited_df["Direction"]))
    
    submit_button = st.sidebar.button("Calculate AURA", type="primary", use_container_width=True)
        
    @st.dialog("Weightage Verification")
    def confirm_weight_warning(t_weight):
        st.warning(f"The total weightage is **{t_weight:g}**, which is not exactly equal to 1.")
        st.write("Do you want to proceed with the calculation anyway, or go back to adjust the weightage?")
        col1, col2 = st.columns(2)
        if col1.button("Proceed anyway", type="primary", use_container_width=True):
            st.session_state.force_calculate = True
            st.rerun()
        if col2.button("Go back", use_container_width=True):
            st.rerun()

    if "force_calculate" not in st.session_state:
        st.session_state.force_calculate = False

    total_weight = sum(weights.values())

    if submit_button:
        if abs(total_weight - 1.0) > 1e-6:
            confirm_weight_warning(total_weight)
        else:
            st.session_state.force_calculate = True
            
    if st.session_state.force_calculate:
        st.session_state.force_calculate = False
        try:
            with st.spinner("Calculating..."):
                results_df = calculate_aura(numeric_df, weights, directions, alpha, p_metric)
            
            st.subheader("AURA Results & Ranking")
            
            # Formatting results for display
            display_df = results_df.copy()
            
            # Format numeric columns nicely
            cols_to_format = ['Utility Score', 'D+ (PIS)', 'D- (NIS)', 'D_avg (AS)']
            for col in cols_to_format:
                display_df[col] = display_df[col].map('{:.4f}'.format)
                
            st.dataframe(display_df, use_container_width=True)
            
            st.subheader("Top Ranked Alternative")
            top_alt = display_df.index[0]
            top_score = display_df['Utility Score'].iloc[0]
            st.success(f"🏆 **{top_alt}** is the best alternative with a utility score of **{top_score}**.")
            
            # Bar chart of scores
            st.subheader("Utility Scores Visualization")
            chart_data = results_df[['Utility Score']].copy()
            # Sort by rank for the chart
            chart_data = chart_data.sort_values(by='Utility Score', ascending=True)
            st.bar_chart(chart_data)
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")

else:
    st.info("Please upload a decision matrix file to begin.")
    
    # Offer a template download
    st.markdown("### Sample Data Format")
    st.markdown("""
    Your file should look like this (with Alternative names in first column):
    
    | Alternative | Cost | Quality | Durability |
    |---|---|---|---|
    | Car A | 20000 | 8 | 5 |
    | Car B | 25000 | 9 | 7 |
    | Car C | 18000 | 6 | 4 |
    """)

