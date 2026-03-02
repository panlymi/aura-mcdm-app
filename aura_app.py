import streamlit as st
import pandas as pd
import altair as alt
from aura_calculator import calculate_aura
from aras_calculator import calculate_aras
from fuzzy_aras_calculator import calculate_fuzzy_aras
from syai_calculator import calculate_syai
from arie_calculator import calculate_arie
from fuzzy_parser import parse_fuzzy_matrix, parse_fuzzy_weights

# Function to generate sample CSV templates
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title="MCDM Calculator", layout="wide", page_icon="📊")

st.title("Multi-Criteria Decision Making (MCDM) Calculator 📊")
st.markdown("""
This application implements five highly effective MCDM methods:
1. **Adaptive Utility Ranking Algorithm (AURA)**
2. **Additive Ratio Assessment (ARAS)**
3. **Fuzzy Additive Ratio Assessment (Fuzzy ARAS)**
4. **Simplified Yielded Aggregation Index (SYAI)**
5. **Adaptive Ranking with Ideal Evaluation (ARIE)**
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Global Settings")
mcdm_method = st.sidebar.selectbox("Select Method", ["AURA", "ARAS", "Fuzzy ARAS", "SYAI", "ARIE"])

weight_type = None
fuzzy_matrix_format = None
if mcdm_method == "Fuzzy ARAS":
    st.sidebar.subheader("Fuzzy ARAS Settings")
    fuzzy_matrix_format = st.sidebar.radio("Matrix Values Format", ["Linguistic Terms", "Comma-Separated TFNs"])
    weight_type = st.sidebar.radio("Criteria Weights Type", ["Crisp (Normal)", "Fuzzy"])
    if weight_type == "Fuzzy":
        fuzzy_weight_format = st.sidebar.radio("Fuzzy Weight Format", ["Linguistic Terms", "Comma-Separated TFNs"])
    else:
        fuzzy_weight_format = "Crisp"

# Method specific parameters in sidebar
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
elif mcdm_method == "ARIE":
    st.sidebar.subheader("ARIE Parameters")
    gamma = st.sidebar.slider(
        "Sensitivity Parameter (γ)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Controls how sharply deviations from benchmarks are penalized. γ=1 is linear, γ>1 is risk-averse, γ<1 is risk-seeking."
    )
    kappa = st.sidebar.slider(
        "Balancing Parameter (κ)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Trades-off importance between being close to ideal (κ > 0.5) and far from worst (κ < 0.5)."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Decision Matrix", type=["xlsx", "csv"], help="First column should be alternative names. Rows are alternatives, columns are criteria.")

# --- SESSION STATE INITIALIZATION ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "steps_dict" not in st.session_state:
    st.session_state.steps_dict = None
if "force_calculate" not in st.session_state:
    st.session_state.force_calculate = False
if "normalize_weights" not in st.session_state:
    st.session_state.normalize_weights = False

# Reset calculation state when method or file changes
if "prev_method" not in st.session_state:
    st.session_state.prev_method = mcdm_method
if st.session_state.prev_method != mcdm_method:
    st.session_state.calculated = False
    st.session_state.prev_method = mcdm_method

if "prev_file" not in st.session_state:
    st.session_state.prev_file = None if uploaded_file is None else uploaded_file.name
current_filename = None if uploaded_file is None else uploaded_file.name
if st.session_state.prev_file != current_filename:
    st.session_state.calculated = False
    st.session_state.prev_file = current_filename

# --- MAIN APP LOGIC ---
if uploaded_file is None:
    st.info("👈 Please upload a decision matrix file in the sidebar to begin.")
    
    st.markdown("### 📥 Download Sample Templates")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Crisp Data (AURA/ARAS/SYAI/ARIE)**")
        df_crisp = pd.DataFrame({
            "Alternative": ["Car A", "Car B", "Car C"],
            "Cost": [20000, 25000, 18000],
            "Quality": [8, 9, 6],
            "Durability": [5, 7, 4]
        })
        st.download_button("⬇️ Download Crisp Template", convert_df_to_csv(df_crisp), "crisp_template.csv", "text/csv", use_container_width=True)
        st.dataframe(df_crisp, hide_index=True)
        
    with col2:
        st.markdown("**Fuzzy ARAS (Linguistic)**")
        df_ling = pd.DataFrame({
            "Alternative": ["Car A", "Car B", "Car C"],
            "Cost": ["High", "Very High", "Moderate"],
            "Quality": ["Good", "Very Good", "Fair"],
            "Durability": ["Fair", "Good", "Poor"]
        })
        st.download_button("⬇️ Download Linguistic Template", convert_df_to_csv(df_ling), "linguistic_template.csv", "text/csv", use_container_width=True)
        st.dataframe(df_ling, hide_index=True)
        
    with col3:
        st.markdown("**Fuzzy ARAS (TFN)**")
        df_tfn = pd.DataFrame({
            "Alternative": ["Car A", "Car B"],
            "Cost": ["18, 20, 22", "23, 25, 26"],
            "Quality": ["7, 8, 9", "8, 9, 10"],
            "Durability": ["4, 5, 6", "6, 7, 8"]
        })
        st.download_button("⬇️ Download TFN Template", convert_df_to_csv(df_tfn), "tfn_template.csv", "text/csv", use_container_width=True)
        st.dataframe(df_tfn, hide_index=True)

else:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, index_col=0)
    else:
        df = pd.read_excel(uploaded_file, index_col=0)
        
    # Validation & Parsing (moved out of UI components to happen first)
    is_valid = True
    matrix_to_calc = None
    if mcdm_method != "Fuzzy ARAS":
        for col in df.columns:
            if df[col].dtype == object or df[col].dtype == str:
                try:
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)
                except ValueError:
                    pass
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            st.error("The uploaded file does not contain numeric data suitable for crisp MCDM.")
            is_valid = False
        else:
            criteria = numeric_df.columns.tolist()
            matrix_to_calc = numeric_df
    else:
        criteria = df.columns.tolist()
        parsed_df = parse_fuzzy_matrix(df, fuzzy_matrix_format)
        if parsed_df is None:
            is_valid = False
            # Parsing error is displayed inside parse_fuzzy_matrix
        else:
            matrix_to_calc = parsed_df

    if is_valid:
        # Create Tabs
        tab_setup, tab_results, tab_steps = st.tabs(["📝 Data Setup & Configuration", "📊 Results & Rankings", "🧮 Detailed Steps"])
        
        with tab_setup:
            st.subheader("1. Verify Decision Matrix")
            st.markdown("*You can dynamically edit the values below before running the calculation.*")
            df = st.data_editor(df, use_container_width=True, num_rows="fixed")
            
            # Re-apply parsing on the dynamically edited df
            if mcdm_method != "Fuzzy ARAS":
                numeric_df = df.select_dtypes(include=['number'])
                matrix_to_calc = numeric_df
            else:
                matrix_to_calc = parse_fuzzy_matrix(df, fuzzy_matrix_format)
            
            st.subheader("2. Configure Criteria Weights & Directions")
            
            weights = None
            directions = None
            
            if mcdm_method != "Fuzzy ARAS" or weight_type == "Crisp (Normal)":
                num_criteria = len(criteria)
                default_val = 1.0 / num_criteria if num_criteria > 0 else 1.0
                
                with st.expander("💡 Bulk Quick-Fill Weights", expanded=False):
                    st.markdown("Paste all weights separated by commas, spaces, or tabs.")
                    bulk_weights_input = st.text_area("Bulk Weights", key="bulk_weights", label_visibility="collapsed", help="e.g. 0.2, 0.3, 0.1, 0.4")
                    parsed_weights = []
                    if bulk_weights_input:
                        import re
                        try:
                            parsed_weights = [float(x) for x in re.split(r'[,\s]+', bulk_weights_input.strip()) if x]
                            if len(parsed_weights) == num_criteria:
                                st.success("Weights parsed successfully!")
                            else:
                                st.warning(f"Expected {num_criteria} weights, found {len(parsed_weights)}.")
                                parsed_weights = []
                        except ValueError:
                            st.error("Invalid numbers detected. Please use format like: 0.2, 0.3, 0.1")
                            parsed_weights = []

                weight_init_values = [default_val] * num_criteria
                if len(parsed_weights) == num_criteria:
                    weight_init_values = parsed_weights

                direction_options = ["maximize", "minimize", "target"] if mcdm_method in ["SYAI", "ARIE"] else ["maximize", "minimize"]
                
                weights_df_init = pd.DataFrame({
                    "Criterion": criteria,
                    "Weight": weight_init_values,
                    "Direction": "maximize",
                    "Target Value": 0.0
                })
                
                edited_weights_df = st.data_editor(
                    weights_df_init,
                    column_config={
                        "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                        "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, format="%.4f", step=0.01), 
                        "Direction": st.column_config.SelectboxColumn("Direction", options=direction_options),
                        "Target Value": st.column_config.NumberColumn("Target Value (If 'target')", format="%.4f", step=None)
                    },
                    hide_index=True,
                    use_container_width=True
                )
                weights = dict(zip(edited_weights_df["Criterion"], edited_weights_df["Weight"]))
                
                directions = {}
                for _, row in edited_weights_df.iterrows():
                    crit = row["Criterion"]
                    direction_val = row["Direction"]
                    if direction_val == "target" and "Target Value" in row:
                        directions[crit] = {"type": "target", "value": row["Target Value"]}
                    else:
                        directions[crit] = direction_val
                        
            else:
                num_criteria = len(criteria)
                if fuzzy_weight_format == "Linguistic Terms":
                    default_weight = "Good"
                    help_text = "e.g. G, VG, F, P"
                    st.markdown("**Valid terms:** Poor, Fair, Good, Very Good (or P, F, G, VG)")
                else:
                    default_weight = "1, 2, 3"
                    help_text = "e.g. 1 2 3; 4 5 6; 7 8 9 (Use semicolons to separate criteria)"
                    st.markdown("**Format:** `l, m, u` (e.g., `1, 2, 3`)")
                    
                with st.expander("💡 Bulk Quick-Fill Weights", expanded=False):
                    st.markdown("Paste all weights separated by commas (for linguistic) or semicolons (for TFNs).")
                    bulk_fuzzy_input = st.text_area("Bulk Fuzzy Weights", key="bulk_fuzzy_weights", label_visibility="collapsed", help=help_text)
                    parsed_fuzzy = []
                    if bulk_fuzzy_input:
                        import re
                        if fuzzy_weight_format == "Linguistic Terms":
                            parsed_fuzzy = [x.strip() for x in re.split(r'[,\s]+', bulk_fuzzy_input.strip()) if x.strip()]
                        else:
                            parsed_fuzzy = [x.strip() for x in bulk_fuzzy_input.split(";") if x.strip()]
                        
                        if len(parsed_fuzzy) == num_criteria:
                            st.success("Weights parsed successfully!")
                        else:
                            st.warning(f"Expected {num_criteria} weights, found {len(parsed_fuzzy)}.")
                            parsed_fuzzy = []
                            
                weight_init_values = [default_weight] * num_criteria
                if len(parsed_fuzzy) == num_criteria:
                    weight_init_values = parsed_fuzzy
                    
                weights_df_init = pd.DataFrame({
                    "Criterion": criteria,
                    "Fuzzy Weight": weight_init_values,
                    "Direction": "maximize"
                })
                
                edited_weights_df = st.data_editor(
                    weights_df_init,
                    column_config={
                        "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                        "Fuzzy Weight": st.column_config.TextColumn("Fuzzy Weight"),
                        "Direction": st.column_config.SelectboxColumn("Direction", options=["maximize", "minimize"])
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                raw_weights = dict(zip(edited_weights_df["Criterion"], edited_weights_df["Fuzzy Weight"]))
                weights = parse_fuzzy_weights(raw_weights, fuzzy_weight_format)
                
                directions = dict(zip(edited_weights_df["Criterion"], edited_weights_df["Direction"]))

            # --- Calculation Trigger ---
            @st.dialog("Weightage Verification")
            def confirm_weight_warning(t_weight):
                st.warning(f"The total weightage is **{t_weight:g}**, which is not exactly equal to 1.")
                st.write("Do you want to proceed anyway, normalize the weights to 1, or go back?")
                c1, c2, c3 = st.columns(3)
                if c1.button("Proceed anyway", use_container_width=True):
                    st.session_state.force_calculate = True
                    st.rerun()
                if c2.button("Normalize & Proceed", type="primary", use_container_width=True):
                    st.session_state.normalize_weights = True
                    st.session_state.force_calculate = True
                    st.rerun()
                if c3.button("Go back", use_container_width=True):
                    st.rerun()

            requires_validation = mcdm_method != "Fuzzy ARAS" or weight_type == "Crisp (Normal)"
            total_weight = sum(weights.values()) if weights and requires_validation else None
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button(f"🚀 Run {mcdm_method} Calculation", type="primary", use_container_width=True)
            
            if submit_button:
                if weights is None:
                    # When parser fails, weights are None. We shouldn't proceed.
                    pass
                elif matrix_to_calc is None:
                    # Matrix parsing failed
                    pass
                elif requires_validation and abs(total_weight - 1.0) > 1e-6:
                    if total_weight == 0:
                        st.error("Total weight is 0. Please enter valid weights.")
                    else:
                        confirm_weight_warning(total_weight)
                else:
                    st.session_state.force_calculate = True

            # Perform Calculation
            if st.session_state.force_calculate:
                st.session_state.force_calculate = False
                
                if st.session_state.normalize_weights and requires_validation and total_weight != 0:
                    st.session_state.normalize_weights = False
                    weights = {k: v / total_weight for k, v in weights.items()}
                
                try:
                    with st.spinner(f"Running {mcdm_method}..."):
                        if mcdm_method == "AURA":
                            res_df, steps = calculate_aura(matrix_to_calc, weights, directions, alpha, p_metric, return_steps=True)
                        elif mcdm_method == "ARAS":
                            res_df = calculate_aras(matrix_to_calc, weights, directions)
                            steps = {} 
                        elif mcdm_method == "SYAI":
                            res_df, steps = calculate_syai(matrix_to_calc, weights, directions, beta, return_steps=True)
                        elif mcdm_method == "ARIE":
                            res_df, steps = calculate_arie(matrix_to_calc, weights, directions, gamma, kappa, return_steps=True)
                        else:
                            res_df, steps = calculate_fuzzy_aras(matrix_to_calc, weights, directions, return_steps=True)
                        
                        st.session_state.calculated = True
                        st.session_state.results_df = res_df
                        st.session_state.steps_dict = steps
                        
                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")
                    st.session_state.calculated = False

        # --- RESULTS TAB ---
        with tab_results:
            if not st.session_state.calculated:
                st.info("Results will appear here after you run the calculation in the Setup tab.")
            else:
                st.subheader(f"🏆 {mcdm_method} Ranking Results")
                
                res_df = st.session_state.results_df
                
                if mcdm_method == "AURA":
                    cols_to_format = ['Utility Score', 'D+ (PIS)', 'D- (NIS)', 'D_avg (AS)']
                    score_col = 'Utility Score'
                    sort_ascending = True 
                    unit = "Utility"
                elif mcdm_method == "ARAS":
                    cols_to_format = ['S (Optimality)', 'K (Utility Degree)']
                    score_col = 'K (Utility Degree)'
                    sort_ascending = False 
                    unit = "Degree"
                elif mcdm_method == "SYAI":
                    cols_to_format = ['D+ (Dist to Ideal)', 'D- (Dist to Anti-Ideal)', 'Closeness Score (D_i)']
                    score_col = 'Closeness Score (D_i)'
                    sort_ascending = False 
                    unit = "Score"
                elif mcdm_method == "ARIE":
                    cols_to_format = ['Sim_best', 'Sim_worst', 'Relative Closeness (RC_i)']
                    score_col = 'Relative Closeness (RC_i)'
                    sort_ascending = False 
                    unit = "Closeness"
                else:
                    cols_to_format = ['S_i (Crisp)', 'K_i (Utility Degree)']
                    score_col = 'K_i (Utility Degree)'
                    sort_ascending = False 
                    unit = "Degree"

                display_df = res_df.copy()
                for col in cols_to_format:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].map('{:.4f}'.format)
                
                top_alt = display_df.index[0]
                top_score = float(display_df[score_col].iloc[0])
                
                if len(display_df) > 1:
                    runner_up = display_df.index[1]
                    runner_up_score = float(display_df[score_col].iloc[1])
                    gap = abs(top_score - runner_up_score)
                else:
                    runner_up = "N/A"
                    gap = 0.0

                # Metrics Dashboard
                dash_c1, dash_c2, dash_c3 = st.columns(3)
                dash_c1.metric(label="🥇 Top Ranked Alternative", value=str(top_alt))
                dash_c2.metric(label=f"⭐ Winning {unit}", value=f"{top_score:.4f}")
                if len(display_df) > 1:
                    dash_c3.metric(label="🥈 1st to 2nd Place Gap", value=f"+{gap:.4f}" if sort_ascending else f"-{gap:.4f}", delta=f"{gap:.4f} diff", delta_color="off")
                
                st.markdown("---")
                
                st.subheader("Performance Bar Chart")
                chart_data = res_df[[score_col]].copy()
                chart_data = chart_data.sort_values(by=score_col, ascending=sort_ascending)
                chart_data = chart_data.reset_index()
                alt_col_name = chart_data.columns[0]
                if alt_col_name != 'Alternative':
                    chart_data.rename(columns={alt_col_name: 'Alternative'}, inplace=True)
                    alt_col_name = 'Alternative'
                    
                chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color='#4A90E2').encode(
                    x=alt.X(f"{alt_col_name}:N", sort=None, title="Alternative", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y(f"{score_col}:Q", title=score_col),
                    tooltip=[alt.Tooltip(f"{alt_col_name}:N", title="Alternative"), 
                             alt.Tooltip(f"{score_col}:Q", title="Score", format=".4f")]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)
                
                st.subheader("Full Rankings Data")
                st.dataframe(display_df, use_container_width=True)

        # --- DETAILED STEPS TAB ---
        with tab_steps:
            if not st.session_state.calculated:
                st.info("Steps will appear here after you run the calculation in the Setup tab.")
            elif mcdm_method == "ARAS":
                st.info("Detailed step-by-step breakdown is currently not implemented for plain ARAS. Ranking results are available.")
            else:
                steps_dict = st.session_state.steps_dict
                st.subheader(f"Step-by-Step {mcdm_method} Calculations")
                st.markdown("This section details the internal math so researchers can verify the results manually.")
                
                if mcdm_method == "AURA":
                    with st.expander("Step 1: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        - For Beneficial Criteria (Maximize): $r_{ij} = \frac{x_{ij} - \min(x_{ij})}{\max(x_{ij}) - \min(x_{ij})}$
                        - For Non-Beneficial Criteria (Minimize): $r_{ij} = \frac{\max(x_{ij}) - x_{ij}}{\max(x_{ij}) - \min(x_{ij})}$
                        ''')
                        st.dataframe(steps_dict.get('Step 1: Normalized Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Weighted Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** $v_{ij} = r_{ij} \times w_j$
                        *(where $w_j$ is the weight for criterion $j$)*
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Weighted Normalized Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Ideal Solutions", expanded=False):
                        st.markdown(r'''
                        **Formulas:**
                        - **PIS (Positive Ideal Solution):** maximum value in each column of $v_{ij}$
                        - **NIS (Negative Ideal Solution):** minimum value in each column of $v_{ij}$
                        - **AS (Average Solution):** average value in each column of $v_{ij}$
                        ''')
                        try:
                            pis_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['PIS (Positive Ideal Solution)']], index=['PIS'])
                            nis_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['NIS (Negative Ideal Solution)']], index=['NIS'])
                            as_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['AS (Average Solution)']], index=['AS'])
                            st.dataframe(pd.concat([pis_df, nis_df, as_df]), use_container_width=True)
                        except KeyError: pass

                    with st.expander("Step 4: Distance Calculations", expanded=False):
                        st.markdown(r'''
                        **Raw Distances:**
                        Calculate distance to PIS ($d^+$), NIS ($d^-$) and AS ($d_{avg}$). Let $c_j$ refer to the solution to compare against.
                        ''')
                        st.markdown(rf"- If $p=1$ (Manhattan): $d_i = \sum_j |v_{{ij}} - c_j|$")
                        st.markdown(rf"- If $p=2$ (Euclidean): $d_i = \sqrt{{\sum_j (v_{{ij}} - c_j)^2}}$")
                        
                        st.markdown("**1. Raw Distances:**")
                        st.dataframe(steps_dict.get('Step 4a: Raw Distances', pd.DataFrame()), use_container_width=True)
                        
                        st.markdown(r'''
                        **2. Corrected Distances:**
                        To handle extreme values, AURA introduces a correction penalty factor:
                        $D_i = d_i + \sigma d_i^2$, where $\sigma = \max(d) - \min(d)$
                        ''')
                        st.markdown(r"**Correction Factors ($\sigma$):**")
                        st.json(steps_dict.get('Step 4b: Correction Factors', {}))
                        st.markdown("**Corrected Distances ($D^+, D^-, D_{avg}$):**")
                        st.dataframe(steps_dict.get('Step 4b: Corrected Distances', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 5: Final Utility Score & Ranking", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $$U_i = \frac{\alpha (D^+_i - D^-_i) + (1 - \alpha) D^{avg}_i}{2}$$
                        *(where $\alpha$ is the balance parameter)*
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Final Utility Scores:**")
                            st.dataframe(steps_dict.get('Step 5: Final Utility Score', pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 6: Final Result and Ranking'][['Rank', 'Utility Score']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "SYAI":
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
                        st.dataframe(steps_dict.get('Step 1: Normalized Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Weighted Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** $v_{ij} = N_{ij} \times w_j$
                        *(where $w_j$ is the normalized weight for criterion $j$)*
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Weighted Normalized Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Ideal Solutions", expanded=False):
                        st.markdown(r'''
                        **Formulas:**
                        - **$A^+$ (Yielded-Ideal Solution):** maximum value in each column of $v_{ij}$
                        - **$A^-$ (Anti-Ideal Solution):** minimum value in each column of $v_{ij}$
                        ''')
                        try:
                            a_plus_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['A+ (Yielded-Ideal Solution)']], index=['A+ (Ideal)'])
                            a_minus_df = pd.DataFrame([steps_dict['Step 3: Ideal Solutions']['A- (Anti-Ideal Solution)']], index=['A- (Anti-Ideal)'])
                            st.dataframe(pd.concat([a_plus_df, a_minus_df]), use_container_width=True)
                        except KeyError: pass

                    with st.expander("Step 4: Distances to Ideal Solutions", expanded=False):
                        st.markdown(r'''
                        **Formulas:**
                        - **Distance to Yielded-Ideal ($D^+_i$):** $D^+_i = \sum_j |v_{ij} - A^+_j|$
                        - **Distance to Anti-Ideal ($D^-_i$):** $D^-_i = \sum_j |v_{ij} - A^-_j|$
                        ''')
                        st.dataframe(steps_dict.get('Step 4: Distances to Ideal Solutions', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 5: Final Closeness Score & Ranking", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $$D_i = \frac{(1 - \beta) D^-_i}{\beta D^+_i + (1 - \beta) D^-_i}$$
                        *(where $\beta$ is the closeness parameter. Higher score implies better rank)*
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Final Closeness Scores:**")
                            st.dataframe(steps_dict.get('Step 5: Final Closeness Score', pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 6: Final Result and Ranking'][['Rank', 'Closeness Score (D_i)']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "ARIE":
                    with st.expander("Step 1: Decision Matrix", expanded=False):
                        st.markdown("Original Decision Matrix $X = [x_{ij}]$")
                        st.dataframe(steps_dict.get('Step 1: Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Normalization Formulas:**
                        - **Max-type (Benefit):** $r_{ij} = \frac{x_{ij}}{x_j^{max}}$
                        - **Min-type (Cost):** $r_{ij} = \frac{x_j^{min}}{x_{ij}}$
                        - **Target-type (Goal):** $r_{ij} = 1 - \frac{|x_{ij} - x_j^T|}{\max(|x_j^{max} - x_j^T|, |x_j^{min} - x_j^T|)}$
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Normalized Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Weighted Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** $v_{ij} = w_j \cdot r_{ij}$
                        *(where $w_j$ is the normalized weight for criterion $j$)*
                        ''')
                        st.dataframe(steps_dict.get('Step 3: Weighted Normalized Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 4: Compute Similarity to Ideal and Anti-Ideal Solutions", expanded=False):
                        st.markdown(r'''
                        **1. Ideal and Anti-Ideal Solutions:**
                        - **Ideal Solution:** $v_j^{max} = \max_i v_{ij}$
                        - **Anti-Ideal Solution:** $v_j^{min} = \min_i v_{ij}$
                        ''')
                        st.dataframe(steps_dict.get('Step 4a: Ideal and Anti-Ideal Solutions', pd.DataFrame()), use_container_width=True)
                        
                        st.markdown(r'''
                        **2. Similarity Computation:**
                        - **Similarity to Ideal ($Sim_i^{best}$):** $Sim_i^{best} = \sum_{j=1}^n \left( \frac{v_{ij}}{v_j^{max}} \right)^\gamma$
                        - **Similarity to Anti-Ideal ($Sim_i^{worst}$):** $Sim_i^{worst} = \sum_{j=1}^n \left( \frac{v_j^{min}}{v_{ij}} \right)^\gamma$
                        *(where $\gamma$ is the sensitivity parameter)*
                        ''')
                        st.dataframe(steps_dict.get('Step 4b: Similarity Computations', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 5: Compute Relative Closeness & Ranking", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $$RC_i = \frac{\kappa \cdot Sim_i^{best}}{\kappa \cdot Sim_i^{best} + (1 - \kappa) \cdot Sim_i^{worst}}$$
                        *(where $\kappa$ is the balancing parameter. Higher $RC_i$ score implies better rank)*
                        ''')
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown("**Relative Closeness Scores:**")
                            try:
                                st.dataframe(steps_dict['Step 5: Final Result and Ranking'][['Sim_best', 'Sim_worst', 'Relative Closeness (RC_i)']], use_container_width=True)
                            except KeyError: pass
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 5: Final Result and Ranking'][['Rank', 'Relative Closeness (RC_i)']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "Fuzzy ARAS":
                    with st.expander("Step 0: Fuzzy Weights", expanded=False):
                        st.markdown("**Weights converted to Triangular Fuzzy Numbers:**")
                        st.dataframe(steps_dict.get('Step 0: Fuzzy Weights', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 1: Decision Matrix with Optimal TFN", expanded=False):
                        st.markdown(r'''
                        **Determining the Optimal Alternative ($x_0$):**
                        - For Beneficial Criteria (Maximize): $x_{0j} = (\max_i l_{ij}, \max_i m_{ij}, \max_i u_{ij})$
                        - For Non-Beneficial Criteria (Minimize): $x_{0j} = (\min_i l_{ij}, \min_i m_{ij}, \min_i u_{ij})$
                        ''')
                        st.dataframe(steps_dict.get('Step 1: Decision Matrix with Optimal TFN ($x_0$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Normalized Fuzzy Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Normalization Formulas:**
                        - **Beneficial:** $\tilde{r}_{ij} = (\frac{l_{ij}}{\sum u_{ij}}, \frac{m_{ij}}{\sum m_{ij}}, \frac{u_{ij}}{\sum l_{ij}})$
                        - **Non-Beneficial:** $\tilde{r}_{ij} = (\frac{1/u_{ij}}{\sum (1/l_{ij})}, \frac{1/m_{ij}}{\sum (1/m_{ij})}, \frac{1/l_{ij}}{\sum (1/u_{ij})})$
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Normalized Fuzzy Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3 & 4: Weighted Matrix and Fuzzy Optimality Function", expanded=False):
                        st.markdown(r'''
                        **Weighted Matrix:** $\tilde{v}_{ij} = \tilde{r}_{ij} \times \tilde{w}_j$
                        **Fuzzy Optimality Function ($S_i$):** $\tilde{S}_i = \sum_{j} \tilde{v}_{ij}$
                        ''')
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown("**Weighted Normalized Matrix:**")
                            st.dataframe(steps_dict.get('Step 3: Weighted Normalized Fuzzy Decision Matrix', pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Fuzzy $S_i$:**")
                            st.dataframe(steps_dict.get('Step 4: Fuzzy Optimality Function ($S_i$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 5 & 6: Defuzzification and Utility Degree", expanded=False):
                        st.markdown(r'''
                        **Defuzzification (Center of Area):** $S_i = \frac{l + m + u}{3}$
                        **Utility Degree ($K_i$):** $K_i = \frac{S_i}{S_0}$
                        *(where $S_0$ is the crisp optimality function of the optimal alternative)*
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Defuzzified $S_i$:**")
                            st.dataframe(steps_dict.get('Step 5: Defuzzified Crisp $S_i$', pd.DataFrame()), use_container_width=True)
                            try:
                                st.info(f"**Optimal $S_0$ (Crisp):** {steps_dict['Step 5: Optimal $S_0$ (Crisp)']:.4f}")
                            except KeyError: pass
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 7: Final Result and Ranking'][['Rank', 'K_i (Utility Degree)', 'S_i (Crisp)']], use_container_width=True)
                            except KeyError: pass
