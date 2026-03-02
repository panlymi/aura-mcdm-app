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
    st.markdown("*You can edit the values below before running the calculation.*")
    df = st.data_editor(df, use_container_width=True, num_rows="fixed")
    
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
                    results_df, steps_dict = calculate_aura(matrix_to_calc, weights, directions, alpha, p_metric, return_steps=True)
                elif mcdm_method == "ARAS":
                    results_df = calculate_aras(matrix_to_calc, weights, directions)
                elif mcdm_method == "SYAI":
                    results_df, steps_dict = calculate_syai(matrix_to_calc, weights, directions, beta, return_steps=True)
                else:
                    results_df, steps_dict = calculate_fuzzy_aras(matrix_to_calc, weights, directions, return_steps=True)
            
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
            if alt_col_name != 'Alternative':
                chart_data.rename(columns={alt_col_name: 'Alternative'}, inplace=True)
                alt_col_name = 'Alternative'
                
            # Create explicit Altair chart to guarantee tooltips
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(f"{alt_col_name}:N", sort=None, title="Alternative"),
                y=alt.Y(f"{score_col}:Q", title=score_col),
                tooltip=[alt.Tooltip(f"{alt_col_name}:N", title="Alternative"), 
                         alt.Tooltip(f"{score_col}:Q", title="Score", format=".4f")]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
            
            # Show Detailed Steps for AURA if method is AURA
            if mcdm_method == "AURA":
                st.subheader(f"Step-by-Step AURA Calculations")
                st.markdown("This section details the internal calculations along with their formulas so researchers can verify the results themselves.")
                
                with st.expander("Step 1: Normalized Decision Matrix", expanded=False):
                    st.markdown(r'''
                    **Formula:**
                    - For Beneficial Criteria (Maximize): $r_{ij} = \frac{x_{ij} - \min(x_{ij})}{\max(x_{ij}) - \min(x_{ij})}$
                    - For Non-Beneficial Criteria (Minimize): $r_{ij} = \frac{\max(x_{ij}) - x_{ij}}{\max(x_{ij}) - \min(x_{ij})}$
                    ''')
                    st.dataframe(steps_dict['Step 1: Normalized Decision Matrix'], use_container_width=True)

                with st.expander("Step 2: Weighted Normalized Matrix", expanded=False):
                    st.markdown(r'''
                    **Formula:** $v_{ij} = r_{ij} \times w_j$
                    
                    *(where $w_j$ is the weight for criterion $j$)*
                    ''')
                    st.dataframe(steps_dict['Step 2: Weighted Normalized Matrix'], use_container_width=True)

                with st.expander("Step 3: Ideal Solutions", expanded=False):
                    st.markdown(r'''
                    **Formulas:**
                    - **PIS (Positive Ideal Solution):** maximum value in each column of $v_{ij}$
                    - **NIS (Negative Ideal Solution):** minimum value in each column of $v_{ij}$
                    - **AS (Average Solution):** average value in each column of $v_{ij}$
                    ''')
                    pis_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['PIS (Positive Ideal Solution)']], index=['PIS'])
                    nis_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['NIS (Negative Ideal Solution)']], index=['NIS'])
                    as_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['AS (Average Solution)']], index=['AS'])
                    st.dataframe(pd.concat([pis_df, nis_df, as_df]), use_container_width=True)

                with st.expander("Step 4: Distance Calculations", expanded=False):
                    st.markdown(r'''
                    **Raw Distances:**
                    Calculate distance to PIS ($d^+$), NIS ($d^-$) and AS ($d_{avg}$). Let $c_j$ refer to the solution to compare against.
                    ''')
                    st.markdown(rf"- If $p=1$ (Manhattan): $d_i = \sum_j |v_{{ij}} - c_j|$")
                    st.markdown(rf"- If $p=2$ (Euclidean): $d_i = \sqrt{{\sum_j (v_{{ij}} - c_j)^2}}$")
                    
                    st.markdown("**1. Raw Distances:**")
                    st.dataframe(steps_dict['Step 4a: Raw Distances'], use_container_width=True)
                    
                    st.markdown(r'''
                    **2. Corrected Distances:**
                    To handle extreme values, AURA introduces a correction penalty factor:
                    $D_i = d_i + \sigma d_i^2$, where $\sigma = \max(d) - \min(d)$
                    ''')
                    st.markdown(r"**Correction Factors ($\sigma$):**")
                    st.json(steps_dict['Step 4b: Correction Factors'])
                    st.markdown("**Corrected Distances ($D^+, D^-, D_{avg}$):**")
                    st.dataframe(steps_dict['Step 4b: Corrected Distances'], use_container_width=True)

                with st.expander("Step 5: Final Utility Score & Ranking", expanded=False):
                    st.markdown(r'''
                    **Formula:**
                    $$U_i = \frac{\alpha (D^+_i - D^-_i) + (1 - \alpha) D^{avg}_i}{2}$$
                    
                    *(where $\alpha$ is the balance parameter)*
                    ''')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Final Utility Scores:**")
                        st.dataframe(steps_dict['Step 5: Final Utility Score'], use_container_width=True)
                    with col2:
                        st.markdown("**Final Result and Ranking:**")
                        st.dataframe(steps_dict['Step 6: Final Result and Ranking'][['Rank', 'Utility Score']], use_container_width=True)
            
            # Show Detailed Steps for SYAI if method is SYAI
            if mcdm_method == "SYAI":
                st.subheader(f"Step-by-Step SYAI Calculations")
                st.markdown("This section details the internal calculations along with their formulas so researchers can verify the results themselves.")
                
                with st.expander("Step 1: Normalized Decision Matrix", expanded=False):
                    st.markdown(r'''
                    **Formula:**
                    $N_{ij} = C + (1 - C) \times \left(1 - \frac{|x_{ij} - x^*|}{R_j}\right)$
                    
                    **Where:**
                    - $C = 0.01$ (Fixed constant)
                    - $R_j = \max(x_j) - \min(x_j)$ (Range of criterion $j$)
                    - $x^*$ is the ideal point:
                        - $\max(x_j)$ for beneficial criteria (maximize)
                        - $\min(x_j)$ for non-beneficial criteria (minimize)
                        - Target value $T$ for goal criteria
                    ''')
                    st.dataframe(steps_dict['Step 1: Normalized Decision Matrix'], use_container_width=True)

                with st.expander("Step 2: Weighted Normalized Matrix", expanded=False):
                    st.markdown(r'''
                    **Formula:** $v_{ij} = N_{ij} \times w_j$
                    
                    *(where $w_j$ is the normalized weight for criterion $j$)*
                    ''')
                    st.dataframe(steps_dict['Step 2: Weighted Normalized Matrix'], use_container_width=True)

                with st.expander("Step 3: Ideal Solutions", expanded=False):
                    st.markdown(r'''
                    **Formulas:**
                    - **$A^+$ (Yielded-Ideal Solution):** maximum value in each column of $v_{ij}$
                    - **$A^-$ (Anti-Ideal Solution):** minimum value in each column of $v_{ij}$
                    ''')
                    a_plus_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['A+ (Yielded-Ideal Solution)']], index=['A+ (Ideal)'])
                    a_minus_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['A- (Anti-Ideal Solution)']], index=['A- (Anti-Ideal)'])
                    st.dataframe(pd.concat([a_plus_df, a_minus_df]), use_container_width=True)

                with st.expander("Step 4: Distances to Ideal Solutions", expanded=False):
                    st.markdown(r'''
                    **Formulas:**
                    - **Distance to Yielded-Ideal ($D^+_i$):** $D^+_i = \sum_j |v_{ij} - A^+_j|$
                    - **Distance to Anti-Ideal ($D^-_i$):** $D^-_i = \sum_j |v_{ij} - A^-_j|$
                    ''')
                    st.dataframe(steps_dict['Step 4: Distances to Ideal Solutions'], use_container_width=True)

                with st.expander("Step 5: Final Closeness Score & Ranking", expanded=False):
                    st.markdown(r'''
                    **Formula:**
                    $$D_i = \frac{(1 - \beta) D^-_i}{\beta D^+_i + (1 - \beta) D^-_i}$$
                    
                    *(where $\beta$ is the closeness parameter. Higher score implies better rank)*
                    ''')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Final Closeness Scores:**")
                        st.dataframe(steps_dict['Step 5: Final Closeness Score'], use_container_width=True)
                    with col2:
                        st.markdown("**Final Result and Ranking:**")
                        st.dataframe(steps_dict['Step 6: Final Result and Ranking'][['Rank', 'Closeness Score (D_i)']], use_container_width=True)
            
            # Show Detailed Steps for Fuzzy ARAS if method is Fuzzy ARAS
            if mcdm_method == "Fuzzy ARAS":
                st.subheader(f"Step-by-Step Fuzzy ARAS Calculations")
                st.markdown("This section details the internal calculations along with their formulas so researchers can verify the results themselves.")
                
                with st.expander("Step 0: Fuzzy Weights", expanded=False):
                    st.markdown("**Weights converted to Triangular Fuzzy Numbers:**")
                    st.dataframe(steps_dict['Step 0: Fuzzy Weights'], use_container_width=True)

                with st.expander("Step 1: Decision Matrix with Optimal TFN", expanded=False):
                    st.markdown(r'''
                    **Determining the Optimal Alternative ($x_0$):**
                    - For Beneficial Criteria (Maximize): $x_{0j} = (\max_i l_{ij}, \max_i m_{ij}, \max_i u_{ij})$
                    - For Non-Beneficial Criteria (Minimize): $x_{0j} = (\min_i l_{ij}, \min_i m_{ij}, \min_i u_{ij})$
                    ''')
                    st.dataframe(steps_dict['Step 1: Decision Matrix with Optimal TFN ($x_0$)'], use_container_width=True)

                with st.expander("Step 2: Normalized Fuzzy Decision Matrix", expanded=False):
                    st.markdown(r'''
                    **Normalization Formulas:**
                    - **Beneficial:** $\tilde{r}_{ij} = (\frac{l_{ij}}{\sum u_{ij}}, \frac{m_{ij}}{\sum m_{ij}}, \frac{u_{ij}}{\sum l_{ij}})$
                    - **Non-Beneficial:** $\tilde{r}_{ij} = (\frac{1/u_{ij}}{\sum (1/l_{ij})}, \frac{1/m_{ij}}{\sum (1/m_{ij})}, \frac{1/l_{ij}}{\sum (1/u_{ij})})$
                    ''')
                    st.dataframe(steps_dict['Step 2: Normalized Fuzzy Decision Matrix'], use_container_width=True)

                with st.expander("Step 3 & 4: Weighted Matrix and Fuzzy Optimality Function", expanded=False):
                    st.markdown(r'''
                    **Weighted Matrix:** $\tilde{v}_{ij} = \tilde{r}_{ij} \times \tilde{w}_j$
                    **Fuzzy Optimality Function ($S_i$):** $\tilde{S}_i = \sum_{j} \tilde{v}_{ij}$
                    ''')
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown("**Weighted Normalized Matrix:**")
                        st.dataframe(steps_dict['Step 3: Weighted Normalized Fuzzy Decision Matrix'], use_container_width=True)
                    with col2:
                        st.markdown("**Fuzzy $S_i$:**")
                        st.dataframe(steps_dict['Step 4: Fuzzy Optimality Function ($S_i$)'], use_container_width=True)

                with st.expander("Step 5 & 6: Defuzzification and Utility Degree", expanded=False):
                    st.markdown(r'''
                    **Defuzzification (Center of Area):** $S_i = \frac{l + m + u}{3}$
                    **Utility Degree ($K_i$):** $K_i = \frac{S_i}{S_0}$
                    *(where $S_0$ is the crisp optimality function of the optimal alternative)*
                    ''')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Defuzzified $S_i$:**")
                        st.dataframe(steps_dict['Step 5: Defuzzified Crisp $S_i$'], use_container_width=True)
                        st.info(f"**Optimal $S_0$ (Crisp):** {steps_dict['Step 5: Optimal $S_0$ (Crisp)']:.4f}")
                    with col2:
                        st.markdown("**Final Result and Ranking:**")
                        st.dataframe(steps_dict['Step 7: Final Result and Ranking'][['Rank', 'K_i (Utility Degree)', 'S_i (Crisp)']], use_container_width=True)
            
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
