import streamlit as st
import pandas as pd
import altair as alt
from aura_calculator import calculate_aura
from aras_calculator import calculate_aras
from fuzzy_aras_calculator import calculate_fuzzy_aras
from syai_calculator import calculate_syai
from fuzzy_parser import parse_fuzzy_matrix, parse_fuzzy_weights

st.set_page_config(page_title="MCDM Calculator", layout="wide")

st.title("Multi-Criteria Decision Making Calculator")
st.markdown("""
This application implements three Multi-Criteria Decision Making (MCDM) methods:
1. **Adaptive Utility Ranking Algorithm (AURA)**
2. **Additive Ratio Assessment (ARAS)**
3. **Fuzzy Additive Ratio Assessment (Fuzzy ARAS)**
4. **Simplified Yielded Aggregation Index (SYAI)**

Upload your decision matrix as an Excel or CSV file. The file should have alternatives as rows and criteria as columns.
The first column should contain the names of the alternatives.
""")

st.sidebar.header("Configuration")

# MCDM Method Selection
mcdm_method = st.sidebar.selectbox("Select MCDM Method", ["AURA", "ARAS", "Fuzzy ARAS", "SYAI"])

weight_type = None
fuzzy_matrix_format = None
if mcdm_method == "Fuzzy ARAS":
    fuzzy_matrix_format = st.sidebar.radio("Matrix Values Format", ["Linguistic Terms", "Comma-Separated TFNs"])
    weight_type = st.sidebar.radio("Criteria Weights Type", ["Crisp (Normal)", "Fuzzy"])
    if weight_type == "Fuzzy":
        fuzzy_weight_format = st.sidebar.radio("Fuzzy Weight Format", ["Linguistic Terms", "Comma-Separated TFNs"])
    else:
        fuzzy_weight_format = "Crisp"

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Decision Matrix", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, index_col=0)
    else:
        df = pd.read_excel(uploaded_file, index_col=0)
    
    st.subheader(f"Input Decision Matrix ({mcdm_method})")
    st.dataframe(df)
    
    # Validation & Parsing
    if mcdm_method != "Fuzzy ARAS":
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.error("The uploaded file does not contain numeric data suitable for crisp MCDM.")
            st.stop()
        criteria = numeric_df.columns.tolist()
        matrix_to_calc = numeric_df
    else:
        criteria = df.columns.tolist()
        parsed_df = parse_fuzzy_matrix(df, fuzzy_matrix_format)
        if parsed_df is None:
            st.stop() # Parsing error displayed inside parser
        matrix_to_calc = parsed_df
    
    # Method specific parameters
    if mcdm_method == "AURA":
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
    elif mcdm_method == "SYAI":
        st.sidebar.subheader("SYAI Parameters")
        beta = st.sidebar.slider(
            "Closeness Parameter (β)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Controls preference: >0.5 emphasizes closeness to the ideal solution; <0.5 emphasizes avoiding the anti-ideal solution."
        )
    
    st.sidebar.subheader("Criteria Weights & Directions")
    
    # Initialize dataframe for data editor based on method
    if mcdm_method != "Fuzzy ARAS" or weight_type == "Crisp (Normal)":
        num_criteria = len(criteria)
        default_val = 1.0 / num_criteria if num_criteria > 0 else 1.0
        
        # SYAI supports 'target' direction
        direction_options = ["maximize", "minimize", "target"] if mcdm_method == "SYAI" else ["maximize", "minimize"]
        
        weights_df_init = pd.DataFrame({
            "Criterion": criteria,
            "Weight": default_val,
            "Direction": "maximize",
            "Target Value": 0.0
        })
        
        edited_df = st.sidebar.data_editor(
            weights_df_init,
            column_config={
                "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, format=None, step=None), 
                "Direction": st.column_config.SelectboxColumn("Direction", options=direction_options),
                "Target Value": st.column_config.NumberColumn("Target Value (If 'target')", format=None, step=None)
            },
            hide_index=True,
            use_container_width=True
        )
        weights = dict(zip(edited_df["Criterion"], edited_df["Weight"]))
    else:
        # Fuzzy ARAS Weights Config
        if fuzzy_weight_format == "Linguistic Terms":
            default_weight = "Good"
            st.sidebar.markdown("**Valid terms:** Poor, Fair, Good, Very Good (or P, F, G, VG)")
        else:
            default_weight = "1, 2, 3"
            st.sidebar.markdown("**Format:** `l, m, u` (e.g., `1, 2, 3`)")
            
        weights_df_init = pd.DataFrame({
            "Criterion": criteria,
            "Fuzzy Weight": default_weight,
            "Direction": "maximize"
        })
        
        edited_df = st.sidebar.data_editor(
            weights_df_init,
            column_config={
                "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                "Fuzzy Weight": st.column_config.TextColumn("Fuzzy Weight"),
                "Direction": st.column_config.SelectboxColumn("Direction", options=["maximize", "minimize"])
            },
            hide_index=True,
            use_container_width=True
        )
        
        raw_weights = dict(zip(edited_df["Criterion"], edited_df["Fuzzy Weight"]))
        weights = parse_fuzzy_weights(raw_weights, fuzzy_weight_format)
        if weights is None:
            st.stop()
            
    # Assemble directions dict
    directions = {}
    for _, row in edited_df.iterrows():
        crit = row["Criterion"]
        direction_val = row["Direction"]
        if direction_val == "target" and "Target Value" in row:
            directions[crit] = {"type": "target", "value": row["Target Value"]}
        else:
            directions[crit] = direction_val
    
    submit_button = st.sidebar.button(f"Calculate {mcdm_method}", type="primary", use_container_width=True)
        
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

    # Calculate weight validation only for non-fuzzy methods or crisp weights
    requires_validation = mcdm_method != "Fuzzy ARAS" or weight_type == "Crisp (Normal)"
    if requires_validation:
        total_weight = sum(weights.values())

    if submit_button:
        if requires_validation and abs(total_weight - 1.0) > 1e-6:
            confirm_weight_warning(total_weight)
        else:
            st.session_state.force_calculate = True
            
    if st.session_state.force_calculate:
        st.session_state.force_calculate = False
        try:
            with st.spinner(f"Calculating {mcdm_method}..."):
                if mcdm_method == "AURA":
                    results_df = calculate_aura(matrix_to_calc, weights, directions, alpha, p_metric)
                elif mcdm_method == "ARAS":
                    results_df = calculate_aras(matrix_to_calc, weights, directions)
                elif mcdm_method == "SYAI":
                    results_df = calculate_syai(matrix_to_calc, weights, directions, beta)
                else:
                    results_df = calculate_fuzzy_aras(matrix_to_calc, weights, directions)
            
            st.subheader(f"{mcdm_method} Results & Ranking")
            
            # Formatting results for display
            display_df = results_df.copy()
            
            # Format numeric columns nicely based on methodology
            if mcdm_method == "AURA":
                cols_to_format = ['Utility Score', 'D+ (PIS)', 'D- (NIS)', 'D_avg (AS)']
                score_col = 'Utility Score'
                sort_ascending = True # AURA: Lowest score is best
            elif mcdm_method == "ARAS":
                cols_to_format = ['S (Optimality)', 'K (Utility Degree)']
                score_col = 'K (Utility Degree)'
                sort_ascending = False # ARAS: Highest degree is best
            elif mcdm_method == "SYAI":
                cols_to_format = ['D+ (Dist to Ideal)', 'D- (Dist to Anti-Ideal)', 'Closeness Score (D_i)']
                score_col = 'Closeness Score (D_i)'
                sort_ascending = False # SYAI: Highest score is best
            else:
                cols_to_format = ['S_i (Crisp)', 'K_i (Utility Degree)']
                score_col = 'K_i (Utility Degree)'
                sort_ascending = False # Fuzzy ARAS: Highest degree is best
                
            for col in cols_to_format:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map('{:.4f}'.format)
                
            st.dataframe(display_df, use_container_width=True)
            
            st.subheader("Top Ranked Alternative")
            top_alt = display_df.index[0]
            top_score = display_df[score_col].iloc[0]
            st.success(f"🏆 **{top_alt}** is the best alternative with a score/degree of **{top_score}**.")
            
            # Bar chart of scores using Altair for exact tooltip control
            st.subheader(f"{mcdm_method} Scores Visualization")
            chart_data = results_df[[score_col]].copy()
            # Sort by appropriate ascending order for the chart
            chart_data = chart_data.sort_values(by=score_col, ascending=sort_ascending)
            
            # Reset index to properly label the Alternatives
            chart_data = chart_data.reset_index()
            alt_col_name = chart_data.columns[0]
            if alt_col_name == 'index':
                chart_data.rename(columns={'index': 'Alternative'}, inplace=True)
                alt_col_name = 'Alternative'
                
            # Create explicit Altair chart to guarantee tooltips
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(f"{alt_col_name}:N", sort=None, title="Alternative"),
                y=alt.Y(f"{score_col}:Q", title=score_col),
                tooltip=[alt.Tooltip(f"{alt_col_name}:N", title="Alternative"), 
                         alt.Tooltip(f"{score_col}:Q", title="Score", format=".4f")]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")

else:
    st.info("Please upload a decision matrix file to begin.")
    
    # Offer a template download
    st.markdown("### Sample Data Format (Crisp AURA/ARAS)")
    st.markdown("""
    | Alternative | Cost | Quality | Durability |
    |---|---|---|---|
    | Car A | 20000 | 8 | 5 |
    | Car B | 25000 | 9 | 7 |
    """)
    st.markdown("### Sample Data Format (Fuzzy ARAS - Linguistic)")
    st.markdown("""
    | Alternative | Cost | Quality | Durability |
    |---|---|---|---|
    | Car A | High | Good | Fair |
    | Car B | Very High | Very Good | Good |
    """)
    st.markdown("### Sample Data Format (Fuzzy ARAS - Comma Separated TFN)")
    st.markdown("""
    | Alternative | Cost | Quality | Durability |
    |---|---|---|---|
    | Car A | 18000, 20000, 22000 | 7, 8, 9 | 4, 5, 6 |
    | Car B | 23000, 25000, 26000 | 8, 9, 10| 6, 7, 8 |
    """)
