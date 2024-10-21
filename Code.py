
# Functions for Δp Calculation in CODE1
def calculate_dp_code1(rho_mix, mu_mix, rho_cal, mu_cal, q, q_cal, a_tilde, z, y, x):
    delta_p = (rho_mix / rho_cal)**z * (mu_cal / mu_mix)**y * rho_mix * a_tilde * (q / q_cal)**x
    return delta_p

# Functions for Δp Calculation in CODE2
def calculate_rho_mix(alpha_oil, alpha_water, alpha_gas, rho_oil, rho_water, rho_gas, a, b, c):
    return alpha_oil**a * rho_oil + alpha_water**b * rho_water + alpha_gas**c * rho_gas

def calculate_mu_mix(alpha_oil, alpha_water, alpha_gas, mu_oil, mu_water, mu_gas, d, e, f):
    return alpha_oil**d * mu_oil + alpha_water**e * mu_water + alpha_gas**f * mu_gas

def calculate_dp_code2(rho_mix, mu_mix, rho_cal, mu_cal, a_AICD, q, y, x):
    return (rho_mix**2 / rho_cal) * (mu_cal / mu_mix)**y * a_AICD * q**x

# Cost function for Parameter Optimization in CODE2
def cost_function(params, datasets, rho_oil, rho_water, rho_gas, mu_oil, mu_water, mu_gas, rho_cal, mu_cal):
    a, b, c, d, e, f, x, y, a_AICD = params
    total_cost = 0
    for data in datasets:
        alpha_oil, alpha_water, alpha_gas, rates, experimental_dps = data
        for rate, exp_dp in zip(rates, experimental_dps):
            rho_mix = calculate_rho_mix(alpha_oil, alpha_water, alpha_gas, rho_oil, rho_water, rho_gas, a, b, c)
            mu_mix = calculate_mu_mix(alpha_oil, alpha_water, alpha_gas, mu_oil, mu_water, mu_gas, d, e, f)
            calc_dp = calculate_dp_code2(rho_mix, mu_mix, rho_cal, mu_cal, a_AICD, rate, y, x)
            total_cost += (calc_dp - exp_dp)**2
    return total_cost

# Optimization function for CODE2

def optimize_parameters(datasets, rho_oil, rho_water, rho_gas, mu_oil, mu_water, mu_gas, rho_cal, mu_cal):
    initial_guess = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
    result = minimize(cost_function, initial_guess, args=(datasets, rho_oil, rho_water, rho_gas, mu_oil, mu_water, mu_gas, rho_cal, mu_cal), bounds=bounds)
    return result.x

# Plotting function for optimization results in CODE2

import plotly.graph_objs as go

def plot_optimization_results(datasets, optimized_params, rho_oil, rho_water, rho_gas, mu_oil, mu_water, mu_gas, rho_cal, mu_cal):
    a, b, c, d, e, f, x, y, a_AICD = optimized_params
    fig = go.Figure()

    for i, (alpha_oil, alpha_water, alpha_gas, rates, experimental_dps) in enumerate(datasets):
        calculated_dps = []
        for rate in rates:
            rho_mix = calculate_rho_mix(alpha_oil, alpha_water, alpha_gas, rho_oil, rho_water, rho_gas, a, b, c)
            mu_mix = calculate_mu_mix(alpha_oil, alpha_water, alpha_gas, mu_oil, mu_water, mu_gas, d, e, f)
            calc_dp = calculate_dp_code2(rho_mix, mu_mix, rho_cal, mu_cal, a_AICD, rate, y, x)
            calculated_dps.append(calc_dp)

        # Add experimental data as points
        fig.add_trace(go.Scatter(
            x=rates,
            y=experimental_dps,
            mode='markers',
            name=f'Experimental Δp (Dataset {i+1})'
        ))

        # Add calculated data as a line
        fig.add_trace(go.Scatter(
            x=rates,
            y=calculated_dps,
            mode='lines',
            name=f'Calculated Δp (Dataset {i+1})'
        ))

    fig.update_layout(
        title='Pressure Drop vs Rate',
        xaxis_title='Rate (m³/d)',
        yaxis_title='Pressure Drop Δp (bar)',
        legend_title="Legend",
        template="plotly_white"
    )
    st.plotly_chart(fig)

    st.write("Optimized parameters:")
    st.write(f"a = {a:.4f}")
    st.write(f"b = {b:.4f}")
    st.write(f"c = {c:.4f}")
    st.write(f"d = {d:.4f}")
    st.write(f"e = {e:.4f}")
    st.write(f"f = {f:.4f}")
    st.write(f"x = {x:.4f}")
    st.write(f"y = {y:.4f}")
    st.write(f"a_AICD = {a_AICD:.4f} bar·m³/kg")




# Main application function
def main():
    st.title("Δp Calculator and Parameter Optimization for (A)ICD/AICV")

    st.sidebar.header("Select Mode")
    mode = st.sidebar.radio("Mode", ["Δp Calculation", "Parameter Optimization"])

    if mode == "Δp Calculation":
        st.subheader("Δp Calculator for (A)ICD/AICV")

        st.header("Equations:")
        st.latex(r"""
        \Delta p = \left(\frac{\rho_{mix}}{\rho_{cal}}\right)^z \left(\frac{\mu_{cal}}{\mu_{mix}}\right)^y \rho_{mix} \tilde{a} \left(\frac{q}{q_{cal}}\right)^x
        """)
        st.latex(r"""
        \rho_{mix} = \alpha_{oil}^a \rho_{oil} + \alpha_{water}^b \rho_{water} + \alpha_{gas}^c \rho_{gas}
        """)
        st.latex(r"""
        \mu_{mix} = \alpha_{oil}^d \mu_{oil} + \alpha_{water}^e \mu_{water} + \alpha_{gas}^f \mu_{gas}
        """)

        st.header("Parameters:")
        st.latex(r"\rho_{oil}: \text{Oil Density (kg/m}^3\text{)}")
        st.latex(r"\mu_{oil}: \text{Oil Viscosity (cp)}")
        st.latex(r"q_{cal}: \text{Calibration Rate (m}^3\text{/d)}")
        st.latex(r"\rho_{cal}: \text{Calibration Density (kg/m}^3\text{)}")
        st.latex(r"\mu_{cal}: \text{Calibration Viscosity (cp)}")
        st.latex(r"\tilde{a}: \text{Device Strength (bar} \cdot \text{m}^3\text{/kg)}")
        st.latex(r"z, y, x: \text{Density, Viscosity and Rate Exponents}")
        st.latex(r"a, d: \text{Oil Density and Viscosity Exponents}")
        st.latex(r"\rho_{gas}, \mu_{gas}, c, f: \text{Gas Density, Viscosity and its Exponents (if OIL+GAS model)}")
        st.latex(r"\rho_{water}, \mu_{water}, b, e: \text{Water Density, Viscosity and its Exponents (if OIL+WATER model)}")

        compare_mode = st.checkbox("Enable Comparison Mode")

        if compare_mode:
            selected_models = st.multiselect("Select Models to Compare", ["OIL+GAS", "OIL+WATER", "OIL+WATER+GAS"], ["OIL+GAS", "OIL+WATER"])
        else:
            model = st.selectbox("Select the Model", ["OIL+GAS", "OIL+WATER", "OIL+WATER+GAS"])
            selected_models = [model]

        st.header("Enter Common Parameters")
        rho_oil = st.number_input("Oil Density (kg/m³)", min_value=0.0, value=845.0, format="%.2f", key='rho_oil')
        mu_oil = st.number_input("Oil Viscosity (cp)", min_value=0.0, value=5.6, format="%.2f", key='mu_oil')
        q_cal = st.number_input("Calibration Rate (m³/d)", min_value=0.0, value=1.0, format="%.2f", key='q_cal')
        rho_cal = st.number_input("Calibration Density (kg/m³)", min_value=0.0, value=1000.0, format="%.2f", key='rho_cal')
        mu_cal = st.number_input("Calibration Viscosity (cp)", min_value=0.0, value=1.0, format="%.2f", key='mu_cal')
        a_tilde = st.number_input("Device Strength (bar·m³/kg)", min_value=0.0, value=0.00052, format="%.5f", key='a_tilde')
        z = st.number_input("z", min_value=0.0, value=6.1, format="%.2f", key='z')
        y = st.number_input("y", min_value=0.0, value=3.35, format="%.2f", key='y')
        x = st.number_input("x", min_value=0.0, value=3.6, format="%.2f", key='x')
        a = st.number_input("a", min_value=0.0, value=0.5, format="%.2f", key='a')
        d = st.number_input("d", min_value=0.0, value=1.2, format="%.2f", key='d')

        rates = np.linspace(0, 21.6, 40)  # Rates from 0 to 21.6 m³/d

        fig = go.Figure()
        all_dp_results = []

        if "OIL+GAS" in selected_models:
            st.header("Enter Gas Parameters")
            rho_gas = st.number_input("Gas Density (kg/m³)", min_value=0.0, value=124.0, format="%.2f", key='rho_gas_og')
            mu_gas = st.number_input("Gas Viscosity (cp)", min_value=0.0, value=0.018, format="%.3f", key='mu_gas_og')
            c = st.number_input("c", min_value=0.0, value=0.6, format="%.2f", key='c_og')
            f = st.number_input("f", min_value=0.0, value=1.0, format="%.2f", key='f_og')

            gvfs = np.arange(0, 1.1, 0.1)  # GVF from 0 to 100%
            dp_results = []

            for gvf in gvfs:
                dp_for_gvf = []
                for q in rates:
                    rho_mix = ((1 - gvf)**a * rho_oil) + (gvf**c * rho_gas)
                    mu_mix = ((1 - gvf)**d * mu_oil) + (gvf**f * mu_gas)
                    dp = calculate_dp_code1(rho_mix, mu_mix, rho_cal, mu_cal, q, q_cal, a_tilde, z, y, x)
                    dp_for_gvf.append(dp)
                dp_results.append(dp_for_gvf)

            all_dp_results.append(dp_results)

            for i, gvf in enumerate(gvfs):
                fig.add_trace(go.Scatter(x=rates, y=dp_results[i], mode='lines+markers', name=f'GVF={gvf:.1f} (OIL+GAS)'))

        if "OIL+WATER" in selected_models:
            st.header("Enter Water Parameters")
            rho_water = st.number_input("Water Density (kg/m³)", min_value=0.0, value=1012.0, format="%.2f", key='rho_water_ow')
            mu_water = st.number_input("Water Viscosity (cp)", min_value=0.0, value=0.66, format="%.2f", key='mu_water_ow')
            b = st.number_input("b", min_value=0.0, value=2.19, format="%.2f", key='b_ow')
            e = st.number_input("e", min_value=0.0, value=0.54, format="%.2f", key='e_ow')

            wcs = np.arange(0, 1.1, 0.1)  # WC from 0 to 100%
            dp_results = []

            for wc in wcs:
                dp_for_wc = []
                for q in rates:
                    rho_mix = ((1 - wc)**a * rho_oil) + (wc**b * rho_water)
                    mu_mix = ((1 - wc)**d * mu_oil) + (wc**e * mu_water)
                    dp = calculate_dp_code1(rho_mix, mu_mix, rho_cal, mu_cal, q, q_cal, a_tilde, z, y, x)
                    dp_for_wc.append(dp)
                dp_results.append(dp_for_wc)

            all_dp_results.append(dp_results)

            for i, wc in enumerate(wcs):
                fig.add_trace(go.Scatter(x=rates, y=dp_results[i], mode='lines+markers', name=f'WC={wc:.1f} (OIL+WATER)'))

        if "OIL+WATER+GAS" in selected_models:
            st.header("Enter Gas and Water Parameters")
            rho_gas = st.number_input("Gas Density (kg/m³)", min_value=0.0, value=124.0, format="%.2f", key='rho_gas_owg')
            mu_gas = st.number_input("Gas Viscosity (cp)", min_value=0.0, value=0.018, format="%.2f", key='mu_gas_owg')
            c = st.number_input("c", min_value=0.0, value=0.6, format="%.2f", key='c_owg')
            f = st.number_input("f", min_value=0.0, value=1.0, format="%.2f", key='f_owg')

            rho_water = st.number_input("Water Density (kg/m³)", min_value=0.0, value=1012.0, format="%.2f", key='rho_water_owg')
            mu_water = st.number_input("Water Viscosity (cp)", min_value=0.0, value=0.66, format="%.2f", key='mu_water_owg')
            b = st.number_input("b", min_value=0.0, value=2.19, format="%.2f", key='b_owg')
            e = st.number_input("e", min_value=0.0, value=0.54, format="%.2f", key='e_owg')

            wvf = st.slider("Water Volume Fraction (WVF)", 0.0, 1.0, 0.0, 0.1, key='wvf_owg')
            gvf = st.slider("Gas Volume Fraction (GVF)", 0.0, 1.0 - wvf, 0.0, 0.1, key='gvf_owg')
            ovf = 1.0 - wvf - gvf

            dp_results = []

            for q in rates:
                rho_mix = (ovf**a * rho_oil) + (wvf**b * rho_water) + (gvf**c * rho_gas)
                mu_mix = (ovf**d * mu_oil) + (wvf**e * mu_water) + (gvf**f * mu_gas)
                dp = calculate_dp_code1(rho_mix, mu_mix, rho_cal, mu_cal, q, q_cal, a_tilde, z, y, x)
                dp_results.append(dp)

            all_dp_results.append(dp_results)

            fig.add_trace(go.Scatter(x=rates, y=dp_results, mode='lines+markers', name=f'WVF={wvf:.1f}, GVF={gvf:.1f} (OIL+WATER+GAS)'))

        fig.update_layout(title='Δp vs Rate',
                          xaxis_title='Rate (m³/d)',
                          yaxis_title='Δp (bar)',
                          yaxis=dict(range=[0, 40]))

        st.plotly_chart(fig)

        st.header("Download Results")
        all_results_df = pd.DataFrame()

        for model_idx, model_name in enumerate(selected_models):
            model_results = all_dp_results[model_idx]
            for scenario_idx, scenario_results in enumerate(model_results):
                df = pd.DataFrame({"Rate (m³/d)": rates, f"Δp (bar) {model_name} Scenario {scenario_idx+1}": scenario_results})
                all_results_df = pd.concat([all_results_df, df], axis=1)

        st.download_button(label="Download data as CSV", data=all_results_df.to_csv(index=False), file_name='results.csv')

    elif mode == "Parameter Optimization":
        st.subheader("Parameter Optimization")

        st.write("### Constant Parameters")
        rho_oil = st.number_input("ρ_oil (kg/m³)", value=850.0)
        rho_water = st.number_input("ρ_water (kg/m³)", value=1000.0)
        rho_gas = st.number_input("ρ_gas (kg/m³)", value=2.0)
        mu_oil = st.number_input("μ_oil (cp)", value=10.0)
        mu_water = st.number_input("μ_water (cp)", value=1.0)
        mu_gas = st.number_input("μ_gas (cp)", value=0.01)
        rho_cal = st.number_input("ρ_cal (kg/m³)", value=950.0)
        mu_cal = st.number_input("μ_cal (cp)", value=5.0)

        n_datasets = st.number_input("Number of Datasets", min_value=1, max_value=10, value=2, step=1)
        datasets = []

        for i in range(n_datasets):
            st.write(f"### Dataset {i+1}")
            alpha_oil = st.slider(f'α_oil (Dataset {i+1})', 0.0, 1.0, 0.3)
            alpha_water = st.slider(f'α_water (Dataset {i+1})', 0.0, 1.0, 0.5)
            alpha_gas = st.slider(f'α_gas (Dataset {i+1})', 0.0, 1.0, 0.2)
            rates = st.text_input(f'Rates (m³/d) (Dataset {i+1})', '50, 60, 70, 80, 90, 100')
            experimental_dps = st.text_input(f'Exp. Δp (bar) (Dataset {i+1})', '0.8, 0.85, 0.9, 0.95, 1.0, 1.05')

            rates = list(map(float, rates.split(',')))
            experimental_dps = list(map(float, experimental_dps.split(',')))

            datasets.append((alpha_oil, alpha_water, alpha_gas, rates, experimental_dps))

        if st.button("Run Optimization"):
            optimized_params = optimize_parameters(datasets, rho_oil, rho_water, rho_gas, mu_oil, mu_water, mu_gas, rho_cal, mu_cal)
            plot_optimization_results(datasets, optimized_params, rho_oil, rho_water, rho_gas, mu_oil, mu_water, mu_gas, rho_cal, mu_cal)

if __name__ == "__main__":
    main()


