import streamlit as st
import pandas as pd
import altair as alt
from fuzzy_parser import parse_fuzzy_matrix, parse_fuzzy_weights
import numpy as np
from entropy_calculator import calculate_entropy_weights
from merec_calculator import calculate_merec_weights
from mcdm.analysis import calculate_method, compare_methods as run_method_comparison
from mcdm.criteria import CriterionType, METHOD_CAPABILITIES
from mcdm.presentation import RESULT_PRESENTATION
from mcdm.ranking import natural_sort_key
from mcdm.research import (
    generate_dirichlet_weights,
    rank_acceptability_table,
    simulate_aura_weights,
    summarize_rank_simulation,
    types_and_targets_from_directions,
)
from mcdm.state import calculation_fingerprint
from mcdm.validation import (
    MCDMValidationError,
    validate_crisp_matrix,
    validate_fuzzy_matrix,
    validate_weights,
)

# Function to generate sample CSV templates
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title="MCDM Calculator", layout="wide", page_icon="📊")

st.title("Multi-Criteria Decision Making (MCDM) Calculator 📊")
st.markdown("""
This application implements nine MCDM options:
- **AURA** (Adaptive Utility Ranking Algorithm)
- **ARAS** (Additive Ratio Assessment - Crisp & Fuzzy)
- **SYAI** (Simplified Yielded Aggregation Index)
- **ARIE** (Adaptive Ranking with Ideal Evaluation)
- **MOORA** (Multi-Objective Optimization on the basis of Ratio Analysis)
- **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution)
- **SAW** (Simple Additive Weighting)
- **VIKOR** (VlseKriterijumska Optimizacija I Kompromisno Resenje)

### Instructions:
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("Configuration")

st.sidebar.subheader("Method Selection")
mcdm_method = st.sidebar.selectbox(
    "Choose MCDM Method", 
    ["AURA", "ARAS", "Fuzzy ARAS", "SYAI", "ARIE", "MOORA", "TOPSIS", "SAW", "VIKOR"]
)

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

# Initialize default parameters for all methods to support Comparative Analysis
alpha = 0.5
p_metric = 1
beta = 0.5
gamma = 1.0
kappa = 0.5
v_param = 0.5

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
elif mcdm_method == "VIKOR":
    st.sidebar.subheader("VIKOR Parameters")
    v_param = st.sidebar.slider(
        "Weight of strategy of 'majority of criteria' (v)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="v > 0.5: voting by majority. v = 0.5: consensus. v < 0.5: veto."
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
if "ewm_steps" not in st.session_state:
    st.session_state.ewm_steps = None
if "merec_steps" not in st.session_state:
    st.session_state.merec_steps = None
if "calculation_fingerprint" not in st.session_state:
    st.session_state.calculation_fingerprint = None
if "monte_carlo_result" not in st.session_state:
    st.session_state.monte_carlo_result = None
if "monte_carlo_fingerprint" not in st.session_state:
    st.session_state.monte_carlo_fingerprint = None

# Reset calculation state when method or file changes
if "prev_method" not in st.session_state:
    st.session_state.prev_method = mcdm_method
if st.session_state.prev_method != mcdm_method:
    st.session_state.calculated = False
    st.session_state.results_df = None
    st.session_state.steps_dict = None
    st.session_state.calculation_fingerprint = None
    st.session_state.monte_carlo_result = None
    st.session_state.monte_carlo_fingerprint = None
    st.session_state.prev_method = mcdm_method

if "prev_file" not in st.session_state:
    st.session_state.prev_file = None if uploaded_file is None else uploaded_file.name
current_filename = None if uploaded_file is None else uploaded_file.name
if st.session_state.prev_file != current_filename:
    st.session_state.calculated = False
    st.session_state.results_df = None
    st.session_state.steps_dict = None
    st.session_state.calculation_fingerprint = None
    st.session_state.monte_carlo_result = None
    st.session_state.monte_carlo_fingerprint = None
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
    
    # Apply natural sorting to the index (alternatives) so A1, A2, ..., A10 instead of A1, A10, A2
    df = df.loc[sorted(df.index, key=natural_sort_key)]
        
    # Validation & Parsing (moved out of UI components to happen first)
    is_valid = True
    matrix_to_calc = None
    criteria = df.columns.tolist()
    try:
        if mcdm_method != "Fuzzy ARAS":
            matrix_to_calc = validate_crisp_matrix(df)
        else:
            parsed_df = parse_fuzzy_matrix(df, fuzzy_matrix_format)
            matrix_to_calc, _ = validate_fuzzy_matrix(parsed_df)
    except MCDMValidationError as exc:
        st.error(str(exc))
        is_valid = False

    if is_valid:
        # Create Tabs
        tab_setup, tab_results, tab_steps, tab_sensitivity, tab_monte_carlo, tab_compare = st.tabs(
            [
                "📝 Data Setup & Configuration",
                "📊 Results & Rankings",
                "🧮 Detailed Steps",
                "📈 Sensitivity Analysis",
                "Monte Carlo Simulation",
                "⚖️ Comparative Analysis",
            ]
        )
        
        with tab_setup:
            st.subheader("1. Verify Decision Matrix")
            st.markdown("*You can dynamically edit the values below before running the calculation.*")
            df = st.data_editor(df, use_container_width=True, num_rows="fixed")
            
            # Re-apply parsing on the dynamically edited df
            try:
                if mcdm_method != "Fuzzy ARAS":
                    matrix_to_calc = validate_crisp_matrix(df)
                else:
                    parsed_df = parse_fuzzy_matrix(df, fuzzy_matrix_format)
                    matrix_to_calc, _ = validate_fuzzy_matrix(parsed_df)
            except MCDMValidationError as exc:
                st.error(str(exc))
                matrix_to_calc = None
            
            st.subheader("2. Configure Criteria Weights & Directions")
            
            weights = None
            directions = None
            
            if mcdm_method != "Fuzzy ARAS" or weight_type == "Crisp (Normal)":
                num_criteria = len(criteria)
                default_val = 1.0 / num_criteria if num_criteria > 0 else 1.0
                
                weight_calc_method = st.radio("Weight Calculation Method", 
                                              options=["Manual / Equal Weights", "Entropy Weight Method (Objective)", "MEREC (Objective)"], 
                                              horizontal=True)
                
                capabilities = METHOD_CAPABILITIES[mcdm_method.upper()]
                direction_options = ["maximize", "minimize"]
                if CriterionType.TARGET in capabilities:
                    direction_options.append("target")
                parsed_weights = []
                
                if weight_calc_method == "Manual / Equal Weights":
                    with st.expander("💡 Bulk Quick-Fill Weights", expanded=False):
                        st.markdown("Paste all weights separated by commas, spaces, or tabs.")
                        bulk_weights_input = st.text_area("Bulk Weights", key="bulk_weights", label_visibility="collapsed", help="e.g. 0.2, 0.3, 0.1, 0.4")
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
                
                weights_df_init = pd.DataFrame({
                    "Criterion": criteria,
                    "Weight": weight_init_values,
                    "Direction": "maximize",
                    "Target Value": 0.0
                })
                
                if weight_calc_method in ["Entropy Weight Method (Objective)", "MEREC (Objective)"]:
                    if weight_calc_method == "Entropy Weight Method (Objective)":
                        st.info("EWM evaluates the variation in criteria data to assign weights objectively. You only need to configure the Directions below.")
                        ewm_normalization = st.selectbox(
                            "EWM Normalization Method",
                            ["Simple Proportions (P_ij = x_ij / sum_x)", "Shifted Min-Max (Min-Max + 0.001)", "Strict Min-Max (Current)"],
                            help="Select the normalization formula used for entropy probability calculation."
                        )
                    else:
                        st.info("MEREC determines weights by evaluating the removal effect of each criterion. You only need to configure the Directions below.")
                        
                    edited_weights_df = st.data_editor(
                        weights_df_init,
                        column_config={
                            "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                            "Direction": st.column_config.SelectboxColumn("Direction", options=direction_options),
                            "Target Value": st.column_config.NumberColumn("Target Value (If 'target')", format="%.4f", step=None)
                        },
                        hide_index=True,
                        disabled=["Weight"],
                        use_container_width=True
                    )
                else:
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
                
                directions = {}
                for _, row in edited_weights_df.iterrows():
                    crit = row["Criterion"]
                    direction_val = row["Direction"]
                    if direction_val == "target" and "Target Value" in row:
                        directions[crit] = {"type": "target", "value": row["Target Value"]}
                    else:
                        directions[crit] = direction_val
                        
                if weight_calc_method == "Entropy Weight Method (Objective)":
                    if ewm_normalization.startswith("Simple"):
                        ewm_method_code = "simple"
                    elif ewm_normalization.startswith("Shifted"):
                        ewm_method_code = "shifted"
                    else:
                        ewm_method_code = "standard"
                        
                    try:
                        ewm_weights, ewm_steps = calculate_entropy_weights(matrix_to_calc, directions, method=ewm_method_code)
                        weights = ewm_weights
                        st.session_state.ewm_steps = ewm_steps
                        st.session_state.merec_steps = None
                        st.markdown("**Calculated Entropy Weights:**")
                        ewm_df = pd.DataFrame(list(weights.items()), columns=["Criterion", "Calculated Weight"])
                        st.dataframe(ewm_df.style.format({"Calculated Weight": "{:.4f}"}), use_container_width=True, hide_index=True)
                    except (MCDMValidationError, ValueError) as exc:
                        st.error(str(exc))
                        weights = None
                elif weight_calc_method == "MEREC (Objective)":
                    try:
                        merec_weights, merec_steps = calculate_merec_weights(matrix_to_calc, directions)
                        weights = merec_weights
                        st.session_state.merec_steps = merec_steps
                        st.session_state.ewm_steps = None
                        st.markdown("**Calculated MEREC Weights:**")
                        merec_df = pd.DataFrame(list(weights.items()), columns=["Criterion", "Calculated Weight"])
                        st.dataframe(merec_df.style.format({"Calculated Weight": "{:.4f}"}), use_container_width=True, hide_index=True)
                    except (MCDMValidationError, ValueError) as exc:
                        st.error(str(exc))
                        weights = None
                else:
                    weights = dict(zip(edited_weights_df["Criterion"], edited_weights_df["Weight"]))
                    st.session_state.ewm_steps = None
                    st.session_state.merec_steps = None
                        
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
                directions = dict(zip(edited_weights_df["Criterion"], edited_weights_df["Direction"]))
                try:
                    weights = parse_fuzzy_weights(raw_weights, fuzzy_weight_format)
                except MCDMValidationError as exc:
                    st.error(str(exc))
                    weights = None

            # --- Calculation Trigger ---
            requires_validation = mcdm_method != "Fuzzy ARAS" or weight_type == "Crisp (Normal)"
            if weights and requires_validation and matrix_to_calc is not None:
                try:
                    raw_total = sum(float(value) for value in weights.values())
                    weights = validate_weights(weights, matrix_to_calc.columns, normalize=True)
                    if abs(raw_total - 1.0) > 1e-6:
                        st.info(
                            f"Weights summed to {raw_total:g} and were automatically normalized to 1. "
                            "This keeps rankings invariant to weight scale."
                        )
                except (MCDMValidationError, TypeError, ValueError) as exc:
                    st.error(str(exc))
                    weights = None

            parameters = {
                "alpha": alpha,
                "p": p_metric,
                "beta": beta,
                "gamma": gamma,
                "kappa": kappa,
                "v": v_param,
            }
            current_fingerprint = None
            if weights is not None and matrix_to_calc is not None and directions is not None:
                try:
                    current_fingerprint = calculation_fingerprint(
                        method=mcdm_method,
                        matrix=matrix_to_calc,
                        weights=weights,
                        directions=directions,
                        parameters=parameters,
                    )
                    previous_fingerprint = st.session_state.calculation_fingerprint
                    if previous_fingerprint and previous_fingerprint != current_fingerprint:
                        st.session_state.calculated = False
                        st.session_state.results_df = None
                        st.session_state.steps_dict = None
                        st.session_state.monte_carlo_result = None
                        st.session_state.monte_carlo_fingerprint = None
                except (MCDMValidationError, ValueError) as exc:
                    st.error(str(exc))
            if current_fingerprint is None and st.session_state.calculated:
                st.session_state.calculated = False
                st.session_state.results_df = None
                st.session_state.steps_dict = None
                st.session_state.monte_carlo_result = None
                st.session_state.monte_carlo_fingerprint = None
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.button(f"🚀 Run {mcdm_method} Calculation", type="primary", use_container_width=True)
            
            if submit_button:
                if weights is None:
                    st.error("Please correct the criterion weights before calculating.")
                elif matrix_to_calc is None:
                    st.error("Please correct the decision matrix before calculating.")
                else:
                    st.session_state.force_calculate = True

            # Perform Calculation
            if st.session_state.force_calculate:
                st.session_state.force_calculate = False

                try:
                    with st.spinner(f"Running {mcdm_method}..."):
                        res_df, steps = calculate_method(
                            mcdm_method,
                            matrix_to_calc,
                            weights,
                            directions,
                            parameters=parameters,
                            return_steps=True,
                        )
                        
                        st.session_state.calculated = True
                        st.session_state.results_df = res_df
                        st.session_state.steps_dict = steps
                        st.session_state.calculation_fingerprint = current_fingerprint
                        st.session_state.monte_carlo_result = None
                        st.session_state.monte_carlo_fingerprint = None
                        
                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")
                    st.session_state.calculated = False
                    st.session_state.calculation_fingerprint = None

        # --- RESULTS TAB ---
        with tab_results:
            if not st.session_state.calculated:
                st.info("Results will appear here after you run the calculation in the Setup tab.")
            else:
                st.subheader(f"🏆 {mcdm_method} Ranking Results")
                
                res_df = st.session_state.results_df
                result_meta = RESULT_PRESENTATION[mcdm_method.upper()]
                cols_to_format = result_meta.format_columns
                score_col = result_meta.score_column
                sort_ascending = result_meta.score_ascending
                unit = result_meta.unit

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
                
                # Apply natural sorting to the alternatives
                unique_alts = sorted(chart_data[alt_col_name].unique(), key=natural_sort_key)
                    
                chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color='#4A90E2').encode(
                    x=alt.X(f"{alt_col_name}:N", sort=list(chart_data[alt_col_name]), title="Alternative", axis=alt.Axis(labelAngle=-45)),
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
            else:
                steps_dict = st.session_state.steps_dict
                
                if st.session_state.get('ewm_steps'):
                    ewm_steps = st.session_state.ewm_steps
                    st.subheader("Entropy Weight Method (Objective) Calculations")
                    with st.expander("EWM Step 2: Normalized Data", expanded=False):
                        st.markdown(r'''
                        **Normalization Strategy:**
                        - **Simple Proportions:** $P_{ij} = x_{ij} / \sum x_{ij}$ (skips standard Min-Max)
                        - **Min-Max (Strict/Shifted):** Scales $x_{ij}$ to $[0, 1]$ based on maximize/minimize rules.
                        ''')
                        st.dataframe(ewm_steps.get("Step 2: Normalized Data", pd.DataFrame()), use_container_width=True)
                    with st.expander("EWM Step 3 & 4: Probability and Information Entropy", expanded=False):
                        st.markdown(r'''
                        **1. Probability ($P_{ij}$):** Proportions of the values (shifted by $+0.001$ if using Shifted Min-Max).
                        **2. Information Entropy ($e_j$):** 
                        $$e_j = -k \sum_{i=1}^{m} P_{ij} \ln(P_{ij})$$
                        *(where $k = 1 / \ln(m)$, $m$ is the number of alternatives, and $0 \ln(0)$ is treated as $0$)*
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Probability ($p_{ij}$)**")
                            st.dataframe(ewm_steps.get("Step 3: Proportion / Probability", pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Information Entropy ($e_j$)**")
                            e_df = pd.DataFrame.from_dict(ewm_steps.get("Step 4: Information Entropy (e_j)", {}), orient='index', columns=['e_j'])
                            st.dataframe(e_df, use_container_width=True)
                    with st.expander("EWM Step 5 & 6: Diversification and Final Weights", expanded=False):
                        st.markdown(r'''
                        **1. Degree of Diversification ($d_j$):** $d_j = 1 - e_j$
                        **2. Final Entropy Weight ($w_j$):** $w_j = \frac{d_j}{\sum d_j}$
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Degree of Diversification ($d_j = 1 - e_j$)**")
                            d_df = pd.DataFrame.from_dict(ewm_steps.get("Step 5: Degree of Diversification (d_j)", {}), orient='index', columns=['d_j'])
                            st.dataframe(d_df, use_container_width=True)
                        with c2:
                            st.markdown("**Final Entropy Weights ($w_j$)**")
                            w_df = pd.DataFrame.from_dict(ewm_steps.get("Step 6: Final Entropy Weights", {}), orient='index', columns=['w_j'])
                            st.dataframe(w_df, use_container_width=True)
                            
                if st.session_state.get('merec_steps'):
                    merec_steps = st.session_state.merec_steps
                    st.subheader("MEREC (Objective Weighting) Calculations")
                    with st.expander("MEREC Step 2: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Normalization:**
                        - **Beneficial ($N_{ij}$):** $\frac{\min_k x_{kj}}{x_{ij}}$
                        - **Non-Beneficial ($N_{ij}$):** $\frac{x_{ij}}{\max_k x_{kj}}$
                        ''')
                        st.dataframe(merec_steps.get("Step 2: Normalized Decision Matrix (N)", pd.DataFrame()), use_container_width=True)
                    with st.expander("MEREC Step 3 & 4: Overall Performance & Removal Performance", expanded=False):
                        st.markdown(r'''
                        **1. Logarithmic Penalty:** $|\ln(N_{ij})|$
                        **2. Overall Performance ($S_i$):** $S_i = \ln \left( 1 + \left( \frac{1}{m} \sum_{j} |\ln(N_{ij})| \right) \right)$
                        **3. Performance Without Criterion ($S'_{ij}$):** $S'_{ij} = \ln \left( 1 + \left( \frac{1}{m} \sum_{k \neq j} |\ln(N_{ik})| \right) \right)$
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Overall Performance ($S_i$)**")
                            st.dataframe(merec_steps.get("Step 3: Overall Performance (S_i)", pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Performance w/o Criterion ($S'_{ij}$)**")
                            st.dataframe(merec_steps.get("Step 4: Performance Without Criterion (S'_ij)", pd.DataFrame()), use_container_width=True)
                    with st.expander("MEREC Step 5 & 6: Removal Effects and Final Weights", expanded=False):
                        st.markdown(r'''
                        **1. Removal Effect ($E_j$):** $E_j = \sum_{i} |S'_{ij} - S_i|$
                        **2. Final Weight ($w_j$):** $w_j = \frac{E_j}{\sum_k E_k}$
                        ''')
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Removal Effects ($E_j$)**")
                            st.dataframe(merec_steps.get("Step 5: Removal Effects (E_j)", pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Final MEREC Weights ($w_j$)**")
                            st.dataframe(merec_steps.get("Step 6: Final Weights (w_j)", pd.DataFrame()), use_container_width=True)
                            
                st.subheader(f"Step-by-Step {mcdm_method} Calculations")
                st.markdown("This section details the internal math so researchers can verify the results manually.")
                
                if mcdm_method == "AURA":
                    with st.expander("Step 1: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $$r_{ij} = 1 - \frac{|x_{ij} - r_j|}{\max(x_j) - \min(x_j)}$$
                        
                        **Where $r_j$ is the reference value:**
                        - For Beneficial Criteria (Maximize): $r_j = \max(x_j)$
                        - For Non-Beneficial Criteria (Minimize): $r_j = \min(x_j)$
                        - For Target Criteria: $r_j = \text{Target Value}$
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
                        st.dataframe(
                            steps_dict.get('Step 1: Normalized Decision Matrix', pd.DataFrame()),
                            use_container_width=True,
                            column_config={col: st.column_config.NumberColumn(format="%.4f") for col in matrix_to_calc.columns}
                        )

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

                elif mcdm_method == "ARAS":
                    with st.expander("Step 1: Decision Matrix with Optimal Alternative ($x_0$)", expanded=False):
                        st.markdown(r'''
                        **Determining Optimal Alternative:**
                        - For Maximize Criteria: $x_{0j} = \max_i x_{ij}$
                        - For Minimize Criteria: $x_{0j} = \min_i x_{ij}$
                        ''')
                        st.dataframe(steps_dict.get('Step 1b: Decision Matrix with Optimal Alternative ($x_0$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Normalization Formulas:**
                        - **Benefit Criteria:** $\overline{x}_{ij} = \frac{x_{ij}}{\sum_{i=0}^{m} x_{ij}}$
                        - **Cost Criteria:** $\overline{x}_{ij} = \frac{1/x_{ij}}{\sum_{i=0}^{m} (1/x_{ij})}$
                        ''')
                        st.dataframe(steps_dict.get(r'Step 2: Normalized Decision Matrix ($\overline{x}_{ij}$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Weighted Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** $\hat{x}_{ij} = \overline{x}_{ij} \times w_j$
                        ''')
                        st.dataframe(steps_dict.get(r'Step 3: Weighted Normalized Matrix ($\hat{x}_{ij}$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 4 & 5: Optimality Function & Utility Degree", expanded=False):
                        st.markdown(r'''
                        **Optimality Function ($S_i$):** $S_i = \sum_{j=1}^{n} \hat{x}_{ij}$
                        **Utility Degree ($K_i$):** $K_i = \frac{S_i}{S_0}$
                        *(where $S_0$ is the optimality function for the optimal alternative)*
                        ''')
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown("**Optimality Function ($S_i$):**")
                            st.dataframe(steps_dict.get('Step 4: Optimality Function ($S_i$)', pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 5: Final Result and Ranking'][['Rank', 'K (Utility Degree)', 'S (Optimality)']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "MOORA":
                    with st.expander("Step 1: Decision Matrix", expanded=False):
                        st.markdown("Original Decision Matrix $X$")
                        st.dataframe(steps_dict.get('Step 1: Original Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Ratio Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $x^*_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x_{ij}^2}}$
                        ''')
                        st.dataframe(steps_dict.get(r'Step 2: Ratio Normalized Matrix ($x^*_{ij}$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Weighted Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** $v_{ij} = w_j \times x^*_{ij}$
                        ''')
                        st.dataframe(steps_dict.get(r'Step 3: Weighted Normalized Matrix ($v_{ij}$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 4 & 5: Normalized Assessment Value & Ranking", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $y_i = \sum_{j \in Maximize} v_{ij} - \sum_{j \in Minimize} v_{ij}$
                        *(Higher $y_i$ score is better)*
                        ''')
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**Assessment Value ($y_i$):**")
                            st.dataframe(steps_dict.get('Step 4: Normalized Assessment Value ($y_i$)', pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 5: Final Result and Ranking'][['Rank', 'y_i (Assessment Value)']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "TOPSIS":
                    with st.expander("Step 1: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula (Vector Normalization):**
                        $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x_{ij}^2}}$$
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Normalized Decision Matrix ($r_{ij}$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Weighted Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** 
                        $$v_{ij} = r_{ij} \times w_j$$
                        ''')
                        st.dataframe(steps_dict.get('Step 3: Weighted Normalized Matrix ($v_{ij}$)', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Ideal and Anti-Ideal Solutions", expanded=False):
                        st.markdown(r'''
                        **Formulas:**
                        - **PIS ($A^+$):** $\max v_{ij}$ for beneficial criteria, $\min v_{ij}$ for non-beneficial.
                        - **NIS ($A^-$):** $\min v_{ij}$ for beneficial criteria, $\max v_{ij}$ for non-beneficial.
                        ''')
                        st.dataframe(steps_dict.get('Step 4: Ideal and Anti-Ideal Solutions', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 4: Separation Measures", expanded=False):
                        st.markdown(r'''
                        **Formulas (Euclidean Distance):**
                        - **Distance to PIS ($D^+_i$):** $\sqrt{\sum_j (v_{ij} - A^+_j)^2}$
                        - **Distance to NIS ($D^-_i$):** $\sqrt{\sum_j (v_{ij} - A^-_j)^2}$
                        ''')
                        st.dataframe(steps_dict.get('Step 5: Separation Measures', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 5: Relative Closeness & Ranking", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $$C_i = \frac{D^-_i}{D^+_i + D^-_i}$$
                        *(Higher $C_i$ score is better)*
                        ''')
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**Relative Closeness ($C_i$):**")
                            st.dataframe(steps_dict.get('Step 6: Relative Closeness', pd.DataFrame()), use_container_width=True)
                        with c2:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 7: Final Result and Ranking'][['Rank', 'Relative Closeness (C_i)']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "SAW":
                    with st.expander("Step 1: Normalized Decision Matrix", expanded=False):
                        st.markdown(r'''
                        **Normalization Formulas:**
                        - **Beneficial (Maximize):** $r_{ij} = \frac{x_{ij}}{x_j^{max}}$
                        - **Non-Beneficial (Minimize):** $r_{ij} = \frac{x_j^{min}}{x_{ij}}$
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Normalized Decision Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Weighted Normalized Matrix", expanded=False):
                        st.markdown(r'''
                        **Formula:** 
                        $$v_{ij} = r_{ij} \times w_j$$
                        ''')
                        st.dataframe(steps_dict.get('Step 3: Weighted Normalized Matrix', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: Final Score & Ranking", expanded=False):
                        st.markdown(r'''
                        **Formula:**
                        $$V_i = \sum_{j=1}^{n} v_{ij}$$
                        *(Higher $V_i$ score is better)*
                        ''')
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 4: Final Result and Ranking'][['Rank', 'V_i (SAW Score)']], use_container_width=True)
                            except KeyError: pass

                elif mcdm_method == "VIKOR":
                    with st.expander("Step 1: Best (f*) and Worst (f-) Values", expanded=False):
                        st.markdown(r'''
                        **Determining Ideal/Anti-Ideal values:**
                        - For Benefit Criteria (Maximize): $f_j^* = \max x_{ij}$ ; $f_j^- = \min x_{ij}$
                        - For Cost Criteria (Minimize):    $f_j^* = \min x_{ij}$ ; $f_j^- = \max x_{ij}$
                        ''')
                        st.dataframe(steps_dict.get('Step 2: Best (f*) and Worst (f-) Values', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 2: Utility ($S_i$) and Regret ($R_i$) Measures", expanded=False):
                        st.markdown(r'''
                        **Weighted Normalized Distance:** 
                        - Benefit: $w_j \times \frac{f_j^* - x_{ij}}{f_j^* - f_j^-}$
                        - Cost:    $w_j \times \frac{x_{ij} - f_j^*}{f_j^- - f_j^*}$ = $w_j \times \frac{f_j^* - x_{ij}}{f_j^* - f_j^-}$ (equivalent form in code)
                        ''')
                        st.dataframe(steps_dict.get('Step 3: Weighted Normalized Distance Matrix', pd.DataFrame()), use_container_width=True)
                        st.markdown(r'''
                        **Formulas:**
                        - **Utility ($S_i$):** $S_i = \sum_{j=1}^n w_j \frac{f_j^* - x_{ij}}{f_j^* - f_j^-}$
                        - **Regret ($R_i$):** $R_i = \max_j \left[w_j \frac{f_j^* - x_{ij}}{f_j^* - f_j^-}\right]$
                        ''')
                        st.dataframe(steps_dict.get('Step 4: Utility (S_i) and Regret (R_i) Measures', pd.DataFrame()), use_container_width=True)

                    with st.expander("Step 3: VIKOR Index ($Q_i$) & Ranking", expanded=False):
                        st.markdown(r'''
                        **Boundary values:**
                        $S^* = \min S_i$, $S^- = \max S_i$
                        $R^* = \min R_i$, $R^- = \max R_i$
                        ''')
                        st.json(steps_dict.get('Step 5: VIKOR Index (Q_i) Parameters', {}))
                        
                        st.markdown(r'''
                        **Formula:**
                        $$Q_i = v \frac{S_i - S^*}{S^- - S^*} + (1 - v) \frac{R_i - R^*}{R^- - R^*}$$
                        *(Smaller $Q_i$ score is better)*
                        ''')
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**Final Result and Ranking:**")
                            try:
                                st.dataframe(steps_dict['Step 6: Final Result and Ranking'][['Rank', 'Q_i (VIKOR Index)']], use_container_width=True)
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

        # --- SENSITIVITY ANALYSIS TAB ---
        with tab_sensitivity:
            if not st.session_state.calculated:
                st.info("Sensitivity analysis will appear here after you run the calculation in the Setup tab.")
            else:
                st.subheader(f"📈 Sensitivity Analysis ({mcdm_method})")
                st.markdown("Analyze how changes in criteria weights or method parameters affect the final rankings.")
                
                # 1. Weight Sensitivity Analysis
                if mcdm_method == "Fuzzy ARAS" and weight_type == "Fuzzy":
                    st.warning("Weight Sensitivity Analysis is currently supported for Crisp weights only.")
                else:
                    st.markdown("### ⚖️ Weight Sensitivity Analysis")
                    st.markdown("Select a criterion to vary its weight from 0.0 to 1.0. Other criteria weights will be adjusted proportionally to maintain a sum of 1.0.")
                    
                    base_weights = weights.copy()
                    
                    if len(base_weights) > 1:
                        selected_criterion = st.selectbox("Select Criterion to Vary", list(base_weights.keys()))
                        
                        sensitivity_steps = 11
                        w_range = np.linspace(0.0, 1.0, sensitivity_steps)
                        
                        sensitivity_results = []
                        sensitivity_error = None
                        original_weights = base_weights.copy()
                        w_k_orig = original_weights[selected_criterion]
                        score_col_sens = RESULT_PRESENTATION[mcdm_method.upper()].score_column
                        
                        progress_text = "Running Weight Sensitivity Analysis. Please wait."
                        my_bar = st.progress(0, text=progress_text)
                        
                        for i, w_k_new in enumerate(w_range):
                            new_weights = {}
                            for crit, w_val in original_weights.items():
                                if crit == selected_criterion:
                                    new_weights[crit] = w_k_new
                                else:
                                    if w_k_orig == 1.0:
                                        new_weights[crit] = (1.0 - w_k_new) / (len(original_weights) - 1)
                                    else:
                                        new_weights[crit] = w_val * (1.0 - w_k_new) / (1.0 - w_k_orig)
                            
                            try:
                                temp_res = calculate_method(
                                    mcdm_method,
                                    matrix_to_calc,
                                    new_weights,
                                    directions,
                                    parameters=parameters,
                                )
                                
                                for alt_idx in temp_res.index:
                                    sensitivity_results.append({
                                        'Weight': w_k_new,
                                        'Alternative': alt_idx,
                                        'Score': temp_res.loc[alt_idx, score_col_sens]
                                    })
                            except Exception as exc:
                                sensitivity_error = str(exc)
                                
                            my_bar.progress((i + 1) / sensitivity_steps, text=progress_text)
                            
                        my_bar.empty()
                        if sensitivity_error and not sensitivity_results:
                            st.warning(f"Sensitivity analysis could not run: {sensitivity_error}")
                        
                        if sensitivity_results:
                            sens_df = pd.DataFrame(sensitivity_results)
                            unique_alts_sens = sorted(sens_df['Alternative'].unique(), key=natural_sort_key)
                            
                            st.markdown(f"**Impact of varying '{selected_criterion}' weight on Alternative Scores:**")
                            chart = alt.Chart(sens_df).mark_line(point=True).encode(
                                x=alt.X('Weight:Q', title=f"Weight of '{selected_criterion}' (0 to 1)"),
                                y=alt.Y('Score:Q', title="Score", scale=alt.Scale(zero=False)),
                                color=alt.Color('Alternative:N', sort=unique_alts_sens, legend=alt.Legend(orient='right')),
                                tooltip=['Alternative', alt.Tooltip('Weight', format='.2f'), alt.Tooltip('Score', format='.4f')]
                            ).properties(height=400).interactive()
                            st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Weight Sensitivity Analysis requires at least 2 criteria.")
                        
                # 2. Parameter Sensitivity Analysis
                if mcdm_method in ["AURA", "SYAI", "ARIE", "VIKOR"]:
                    st.markdown("---")
                    st.markdown(f"### ⚙️ {mcdm_method} Parameter Sensitivity")
                    
                    param_results = []
                    
                    if mcdm_method == "AURA":
                        st.markdown("**Varying Balance Parameter (α) from 0.0 to 1.0:**")
                        param_range = np.linspace(0.0, 1.0, 11)
                        param_name = "Alpha (α)"
                        score_col_sens = 'Utility Score'
                        for p_val in param_range:
                            try:
                                temp_res = calculate_method(
                                    "AURA", matrix_to_calc, base_weights, directions,
                                    parameters={**parameters, "alpha": p_val},
                                )
                                for alt_idx in temp_res.index:
                                    param_results.append({'Parameter': p_val, 'Alternative': alt_idx, 'Score': temp_res.loc[alt_idx, score_col_sens]})
                            except Exception: pass
                            
                    elif mcdm_method == "SYAI":
                        st.markdown("**Varying Closeness Parameter (β) from 0.0 to 1.0:**")
                        param_range = np.linspace(0.0, 1.0, 11)
                        param_name = "Beta (β)"
                        score_col_sens = 'Closeness Score (D_i)'
                        for p_val in param_range:
                            try:
                                temp_res = calculate_method(
                                    "SYAI", matrix_to_calc, base_weights, directions,
                                    parameters={**parameters, "beta": p_val},
                                )
                                for alt_idx in temp_res.index:
                                    param_results.append({'Parameter': p_val, 'Alternative': alt_idx, 'Score': temp_res.loc[alt_idx, score_col_sens]})
                            except Exception: pass
                            
                    elif mcdm_method == "ARIE":
                        st.markdown("**Varying Sensitivity Parameter (γ) from 0.5 to 3.0 (fixing κ):**")
                        param_range = np.linspace(0.5, 3.0, 11)
                        param_name = "Gamma (γ)"
                        score_col_sens = 'Relative Closeness (RC_i)'
                        for p_val in param_range:
                            try:
                                temp_res = calculate_method(
                                    "ARIE", matrix_to_calc, base_weights, directions,
                                    parameters={**parameters, "gamma": p_val},
                                )
                                for alt_idx in temp_res.index:
                                    param_results.append({'Parameter': p_val, 'Alternative': alt_idx, 'Score': temp_res.loc[alt_idx, score_col_sens]})
                            except Exception: pass
                            
                    elif mcdm_method == "VIKOR":
                        st.markdown("**Varying Weight Parameter (v) from 0.0 (Veto) to 1.0 (Majority Voting):**")
                        param_range = np.linspace(0.0, 1.0, 11)
                        param_name = "v (Majority Weight)"
                        score_col_sens = 'Q_i (VIKOR Index)'
                        for p_val in param_range:
                            try:
                                temp_res = calculate_method(
                                    "VIKOR", matrix_to_calc, base_weights, directions,
                                    parameters={**parameters, "v": p_val},
                                )
                                for alt_idx in temp_res.index:
                                    param_results.append({'Parameter': p_val, 'Alternative': alt_idx, 'Score': temp_res.loc[alt_idx, score_col_sens]})
                            except Exception: pass
                            
                    if param_results:
                        p_df = pd.DataFrame(param_results)
                        unique_alts_param = sorted(p_df['Alternative'].unique(), key=natural_sort_key)
                        p_chart = alt.Chart(p_df).mark_line(point=True).encode(
                            x=alt.X('Parameter:Q', title=param_name),
                            y=alt.Y('Score:Q', title="Score", scale=alt.Scale(zero=False)),
                            color=alt.Color('Alternative:N', sort=unique_alts_param, legend=alt.Legend(orient='right')),
                            tooltip=['Alternative', alt.Tooltip('Parameter', format='.2f'), alt.Tooltip('Score', format='.4f')]
                        ).properties(height=400).interactive()
                        st.altair_chart(p_chart, use_container_width=True)
                        
                    if mcdm_method == "ARIE":
                        st.markdown("**Varying Balancing Parameter (κ) from 0.0 to 1.0 (fixing γ):**")
                        param_results_k = []
                        param_range_k = np.linspace(0.0, 1.0, 11)
                        for k_val in param_range_k:
                            try:
                                temp_res = calculate_method(
                                    "ARIE", matrix_to_calc, base_weights, directions,
                                    parameters={**parameters, "kappa": k_val},
                                )
                                for alt_idx in temp_res.index:
                                    param_results_k.append({'Parameter': k_val, 'Alternative': alt_idx, 'Score': temp_res.loc[alt_idx, score_col_sens]})
                            except Exception: pass
                        if param_results_k:
                            p_df_k = pd.DataFrame(param_results_k)
                            p_chart_k = alt.Chart(p_df_k).mark_line(point=True).encode(
                                x=alt.X('Parameter:Q', title="Kappa (κ)"),
                                y=alt.Y('Score:Q', title="Score", scale=alt.Scale(zero=False)),
                                color=alt.Color('Alternative:N', legend=alt.Legend(orient='right')),
                                tooltip=['Alternative', alt.Tooltip('Parameter', format='.2f'), alt.Tooltip('Score', format='.4f')]
                            ).properties(height=400).interactive()
                            st.altair_chart(p_chart_k, use_container_width=True)

        # --- MONTE CARLO SIMULATION TAB ---
        with tab_monte_carlo:
            if mcdm_method != "AURA":
                st.info(
                    "The canonical Monte Carlo workflow is currently available for AURA. "
                    "Select AURA and run the baseline calculation to use it."
                )
            elif not st.session_state.calculated:
                st.info(
                    "Run the AURA baseline calculation in the Setup tab before starting "
                    "a Monte Carlo simulation."
                )
            else:
                st.subheader("AURA Monte Carlo Rank Robustness")
                st.markdown(
                    "Randomly sample criterion-weight combinations and recalculate AURA "
                    "to estimate how stable the baseline ranking is. The decision matrix, "
                    "criterion directions, targets, alpha, and distance metric remain fixed."
                )

                control_col1, control_col2, control_col3 = st.columns(3)
                with control_col1:
                    mc_iterations = st.selectbox(
                        "Iterations",
                        options=[250, 1_000, 5_000, 10_000],
                        index=1,
                        key="mc_iterations",
                        help="Larger runs give smoother estimates but take longer.",
                    )
                with control_col2:
                    mc_seed = st.number_input(
                        "Random seed",
                        min_value=0,
                        max_value=4_294_967_295,
                        value=42,
                        step=1,
                        key="mc_seed",
                        help="Use the same seed to reproduce the same simulation.",
                    )
                with control_col3:
                    mc_mode = st.selectbox(
                        "Weight sampling",
                        options=[
                            "Global robustness (uniform simplex)",
                            "Local uncertainty (around current weights)",
                        ],
                        key="mc_sampling_mode",
                    )

                is_local_sampling = mc_mode.startswith("Local")
                mc_concentration = st.number_input(
                    "Local concentration",
                    min_value=1.0,
                    max_value=500.0,
                    value=50.0,
                    step=5.0,
                    disabled=not is_local_sampling,
                    key="mc_concentration",
                    help=(
                        "Higher values keep sampled weights closer to the current weights. "
                        "This setting is only used for local sampling."
                    ),
                )

                current_weight_values = np.asarray(
                    [float(weights[column]) for column in matrix_to_calc.columns], dtype=float
                )
                local_weights_valid = bool(np.all(current_weight_values > 0))
                if is_local_sampling and not local_weights_valid:
                    st.warning(
                        "Local sampling requires every current criterion weight to be greater "
                        "than zero. Use global sampling or adjust the zero weights."
                    )

                run_monte_carlo = st.button(
                    "Run Monte Carlo Simulation",
                    type="primary",
                    use_container_width=True,
                    disabled=is_local_sampling and not local_weights_valid,
                    key="run_monte_carlo",
                )

                if run_monte_carlo:
                    try:
                        with st.spinner(
                            f"Running {mc_iterations:,} AURA simulations with seed {mc_seed}..."
                        ):
                            baseline_result = st.session_state.results_df.reindex(
                                matrix_to_calc.index
                            )
                            baseline_ranks = pd.to_numeric(
                                baseline_result["Rank"], errors="raise"
                            ).to_numpy(dtype=int)
                            alternative_names = [str(value) for value in matrix_to_calc.index]
                            criterion_names = [str(value) for value in matrix_to_calc.columns]
                            criteria_types, target_values = types_and_targets_from_directions(
                                matrix_to_calc.columns.tolist(), directions
                            )

                            center_weights = (
                                current_weight_values if is_local_sampling else None
                            )
                            sampled_weights = generate_dirichlet_weights(
                                matrix_to_calc.shape[1],
                                int(mc_iterations),
                                seed=int(mc_seed),
                                center_weights=center_weights,
                                concentration=float(mc_concentration),
                            )
                            rank_matrix, correlations = simulate_aura_weights(
                                matrix_to_calc,
                                sampled_weights,
                                criteria_types,
                                target_val=target_values,
                                alpha=float(parameters["alpha"]),
                                p=int(parameters["p"]),
                                baseline_ranks=baseline_ranks,
                            )

                            summary_df = summarize_rank_simulation(
                                alternative_names, baseline_ranks, rank_matrix
                            )
                            acceptability_df = rank_acceptability_table(
                                alternative_names, rank_matrix
                            )
                            rank_one = acceptability_df.loc[
                                acceptability_df["Rank"] == 1,
                                ["Alternative", "Probability_Pct"],
                            ].rename(columns={"Probability_Pct": "Rank_1_Freq_Pct"})
                            summary_df = summary_df.merge(
                                rank_one, on="Alternative", how="left", validate="one_to_one"
                            )
                            summary_df = summary_df[
                                [
                                    "Alternative",
                                    "Baseline_Rank",
                                    "Rank_1_Freq_Pct",
                                    "Mean_Rank",
                                    "Rank_SD",
                                    "Min_Rank",
                                    "Max_Rank",
                                    "Top_5_Freq_Pct",
                                    "Bottom_3_Freq_Pct",
                                ]
                            ]

                            finite_correlations = correlations[np.isfinite(correlations)]
                            average_spearman = (
                                float(finite_correlations.mean())
                                if finite_correlations.size
                                else float("nan")
                            )
                            baseline_winner_positions = np.flatnonzero(baseline_ranks == 1)
                            winner_retention = float(
                                np.mean(
                                    np.any(
                                        rank_matrix[:, baseline_winner_positions] == 1,
                                        axis=1,
                                    )
                                )
                                * 100
                            )

                            rank_samples_df = pd.DataFrame(
                                rank_matrix,
                                columns=alternative_names,
                                index=np.arange(1, int(mc_iterations) + 1),
                            ).rename_axis("Iteration").reset_index()
                            weight_samples_df = pd.DataFrame(
                                sampled_weights,
                                columns=criterion_names,
                                index=np.arange(1, int(mc_iterations) + 1),
                            ).rename_axis("Iteration").reset_index()

                            st.session_state.monte_carlo_result = {
                                "summary": summary_df,
                                "acceptability": acceptability_df,
                                "rank_samples": rank_samples_df,
                                "weight_samples": weight_samples_df,
                                "average_spearman": average_spearman,
                                "winner_retention": winner_retention,
                                "iterations": int(mc_iterations),
                                "seed": int(mc_seed),
                                "mode": mc_mode,
                                "concentration": (
                                    float(mc_concentration) if is_local_sampling else None
                                ),
                            }
                            st.session_state.monte_carlo_fingerprint = (
                                st.session_state.calculation_fingerprint
                            )
                    except (MCDMValidationError, ValueError, KeyError) as exc:
                        st.error(f"Monte Carlo simulation could not run: {exc}")

                monte_carlo_result = st.session_state.monte_carlo_result
                result_matches_baseline = (
                    monte_carlo_result is not None
                    and st.session_state.monte_carlo_fingerprint
                    == st.session_state.calculation_fingerprint
                )
                if result_matches_baseline:
                    st.markdown("---")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    average_spearman = monte_carlo_result["average_spearman"]
                    metric_col1.metric(
                        "Average Spearman correlation",
                        f"{average_spearman:.4f}"
                        if np.isfinite(average_spearman)
                        else "N/A",
                    )
                    metric_col2.metric(
                        "Baseline winner retains Rank 1",
                        f"{monte_carlo_result['winner_retention']:.2f}%",
                    )
                    most_stable = monte_carlo_result["summary"].sort_values(
                        ["Rank_SD", "Baseline_Rank"]
                    ).iloc[0]
                    metric_col3.metric("Most stable alternative", most_stable["Alternative"])
                    metric_col4.metric(
                        "Simulations", f"{monte_carlo_result['iterations']:,}"
                    )

                    settings_text = (
                        f"Seed {monte_carlo_result['seed']} | "
                        f"{monte_carlo_result['mode']}"
                    )
                    if monte_carlo_result["concentration"] is not None:
                        settings_text += (
                            f" | concentration {monte_carlo_result['concentration']:.1f}"
                        )
                    st.caption(settings_text)

                    st.markdown("### Rank stability summary")
                    st.dataframe(
                        monte_carlo_result["summary"],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("### Rank acceptability heatmap")
                    st.caption(
                        "Each cell is the percentage of simulations in which an alternative "
                        "attained that rank. Hover for exact probabilities."
                    )
                    alternative_order = monte_carlo_result["summary"][
                        "Alternative"
                    ].tolist()
                    rank_order = list(range(1, len(alternative_order) + 1))
                    acceptability_chart = (
                        alt.Chart(monte_carlo_result["acceptability"])
                        .mark_rect()
                        .encode(
                            x=alt.X("Rank:O", sort=rank_order, title="Simulated rank"),
                            y=alt.Y(
                                "Alternative:N",
                                sort=alternative_order,
                                title="Alternative",
                            ),
                            color=alt.Color(
                                "Probability_Pct:Q",
                                title="Probability (%)",
                                scale=alt.Scale(scheme="blues", domain=[0, 100]),
                            ),
                            tooltip=[
                                "Alternative:N",
                                "Rank:O",
                                alt.Tooltip(
                                    "Probability_Pct:Q",
                                    title="Probability (%)",
                                    format=".2f",
                                ),
                            ],
                        )
                        .properties(height=max(300, 30 * len(alternative_order)))
                    )
                    st.altair_chart(acceptability_chart, use_container_width=True)

                    st.markdown("### Download simulation data")
                    download_col1, download_col2, download_col3, download_col4 = st.columns(4)
                    download_col1.download_button(
                        "Summary CSV",
                        convert_df_to_csv(monte_carlo_result["summary"]),
                        "aura_monte_carlo_summary.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    download_col2.download_button(
                        "Rank acceptability CSV",
                        convert_df_to_csv(monte_carlo_result["acceptability"]),
                        "aura_rank_acceptability.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    download_col3.download_button(
                        "Raw ranks CSV",
                        convert_df_to_csv(monte_carlo_result["rank_samples"]),
                        "aura_monte_carlo_ranks.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    download_col4.download_button(
                        "Sampled weights CSV",
                        convert_df_to_csv(monte_carlo_result["weight_samples"]),
                        "aura_monte_carlo_weights.csv",
                        "text/csv",
                        use_container_width=True,
                    )

        # --- COMPARATIVE ANALYSIS TAB ---
        with tab_compare:
            if not st.session_state.calculated:
                st.info("Comparative analysis will appear here after you run the calculation in the Setup tab.")
            elif mcdm_method == "Fuzzy ARAS":
                st.warning("Comparative Analysis is available for crisp decision matrices. Please select a crisp method in the global settings to use this feature.")
            else:
                st.subheader("⚖️ Comparative Analysis")
                st.markdown("Select multiple MCDM methods below to compare their final rankings side-by-side using the current matrix and criteria weights.")
                
                available_compare_methods = ["AURA", "ARAS", "SYAI", "ARIE", "MOORA", "TOPSIS", "SAW", "VIKOR"]
                selected_compare_methods = st.multiselect(
                    "Select Methods to Compare",
                    available_compare_methods,
                    default=available_compare_methods,
                )
                
                if selected_compare_methods:
                    try:
                        comp_df, excluded_methods = run_method_comparison(
                            selected_compare_methods,
                            matrix_to_calc,
                            weights,
                            directions,
                            parameters=parameters,
                        )
                    except Exception as exc:
                        st.error(f"Comparative analysis could not run: {exc}")
                        comp_df, excluded_methods = pd.DataFrame(), {}

                    for excluded_method, reason in excluded_methods.items():
                        st.info(f"**{excluded_method} excluded:** {reason}")

                    active_compare_methods = list(comp_df.columns)
                    if not comp_df.empty:
                        
                        st.markdown("### 🏆 Method Ranking Comparison")
                        st.dataframe(comp_df,
                            use_container_width=True,
                            column_config={
                                method: st.column_config.NumberColumn(
                                    method, format="%d", help=f"Rank according to {method}"
                                ) 
                                for method in comp_df.columns
                            }
                        )
                        
                        # Display Winning Alternatives count
                        st.markdown("### 🥇 Top Performing Alternatives")
                        winning_alts = comp_df[comp_df == 1].count(axis=1) # count how many times each alternative got Rank 1
                        winning_alts = winning_alts[winning_alts > 0].sort_values(ascending=False)
                        if not winning_alts.empty:
                            winner_df = winning_alts.reset_index()
                            winner_df.columns = ['Alternative', 'Times Ranked #1']
                            winner_chart = alt.Chart(winner_df).mark_bar(color='#FFD700').encode(
                                x=alt.X('Alternative:N', sort='-y'),
                                y=alt.Y('Times Ranked #1:Q', title="Times Ranked #1"),
                                tooltip=['Alternative', 'Times Ranked #1']
                            ).properties(height=250, title="Number of #1 Ranks per Alternative")
                            st.altair_chart(winner_chart, use_container_width=True)
                        else:
                            st.info("No single alternative clearly dominates rank #1 across all methods.")
                            
                        # Bump chart visualizing rank shifts
                        comp_df_reset = comp_df.reset_index()
                        if comp_df_reset.columns[0] != 'Alternative':
                            comp_df_reset.rename(columns={comp_df_reset.columns[0]: 'Alternative'}, inplace=True)
                        comp_melted = comp_df_reset.melt(id_vars='Alternative', var_name='Method', value_name='Rank')
                        
                        unique_alts_comp = sorted(comp_melted['Alternative'].unique(), key=natural_sort_key)
                        
                        chart = alt.Chart(comp_melted).mark_line(point=alt.OverlayMarkDef(filled=False, fill="white", size=100)).encode(
                            x=alt.X('Method:N', title='MCDM Method', sort=active_compare_methods),
                            y=alt.Y('Rank:O', title='Rank', sort='descending'),
                            color=alt.Color('Alternative:N', sort=unique_alts_comp, legend=alt.Legend(title="Alternatives", orient='right')),
                            tooltip=['Alternative', 'Method', 'Rank']
                        ).properties(height=400, title="Alternative Ranking Shifts Across Methods").interactive()
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Spearman Rank Correlation Heatmap
                        if len(active_compare_methods) > 1:
                            st.markdown("---")
                            st.markdown("### 🔗 Mathematical Rank Correlation (Spearman's $\\rho$)")
                            st.markdown("A higher value (closer to 1.0) indicates that the methods produce broadly similar relative ranking sequences.")
                            correlation_matrix = comp_df.corr(method='spearman')
                            
                            # Prepare data for Altair
                            corr_reset = correlation_matrix.reset_index().melt('index')
                            corr_reset.columns = ['Method 1', 'Method 2', 'Correlation']
                            
                            base = alt.Chart(corr_reset).encode(
                                x=alt.X('Method 1:O', sort=active_compare_methods),
                                y=alt.Y('Method 2:O', sort=active_compare_methods),
                            )

                            heatmap = base.mark_rect().encode(
                                color=alt.Color('Correlation:Q', scale=alt.Scale(domain=[-1, 1], scheme='redblue'), title="Spearman's ρ"),
                                tooltip=['Method 1', 'Method 2', alt.Tooltip('Correlation', format='.3f')]
                            )

                            text = base.mark_text(baseline='middle').encode(
                                text=alt.Text('Correlation:Q', format='.2f'),
                                color=alt.condition(
                                    alt.datum.Correlation > 0.5,
                                    alt.value('white'),
                                    alt.value('black')
                                )
                            )

                            corr_chart = (heatmap + text).properties(height=400, title="Method vs Method Spearman's Rank Correlation")
                            st.altair_chart(corr_chart, use_container_width=True)

