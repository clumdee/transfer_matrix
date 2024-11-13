# for calculation
import numpy as np
import pandas as pd
from itertools import product

# for visualization and interaction
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# from ipywidgets import interactive, Label, HTML, HBox, VBox


####################
# CALCULATION
def transfer_matrix_layer(delta_now, n1, n2, theta1, theta2):
    """
    Calculate transfer-matrix of each layer.
    """

    ## TE component
    # propagation matrix
    prp_te = np.array([[np.exp(-1j * delta_now), 0], [0, np.exp(1j * delta_now)]])

    # reflection matrix
    r_te = (n1 * np.cos(theta1) - n2 * np.cos(theta2)) / (
        n1 * np.cos(theta1) + n2 * np.cos(theta2)
    )
    t_te = 2 * n1 * np.cos(theta1) / (n1 * np.cos(theta1) + n2 * np.cos(theta2))
    rlc_te = 1 / t_te * np.array([[1, r_te], [r_te, 1]])

    # combined matrix on that layer
    m_te = np.matmul(prp_te, rlc_te)

    ## TM component
    # propagation matrix -- same as TE
    prp_tm = np.array([[np.exp(-1j * delta_now), 0], [0, np.exp(1j * delta_now)]])

    # reflection matrix
    r_tm = (n2 * np.cos(theta1) - n1 * np.cos(theta2)) / (
        n2 * np.cos(theta1) + n1 * np.cos(theta2)
    )
    t_tm = 2 * n1 * np.cos(theta1) / (n2 * np.cos(theta1) + n1 * np.cos(theta2))
    rlc_tm = 1 / t_tm * np.array([[1, r_tm], [r_tm, 1]])

    # combined matrix on that layer
    m_tm = np.matmul(prp_tm, rlc_tm)

    return m_te, m_tm


def transfer_matrix(delta, n, theta):
    """
    Calculate the combined transfer-matrix of wave propagation.
    """
    # starting matrices
    m_te_all = np.identity(2)
    m_tm_all = np.identity(2)

    # combine transfer matrix of all layers
    for delta_now, n1, n2, theta1, theta2 in zip(
        delta[:-1], n[:-1], n[1:], theta[:-1], theta[1:]
    ):

        # transfer matrix of each layer
        m_te, m_tm = transfer_matrix_layer(delta_now, n1, n2, theta1, theta2)

        # multiply to the previous matrix
        m_te_all = np.matmul(m_te_all, m_te)
        m_tm_all = np.matmul(m_tm_all, m_tm)

    return m_te_all, m_tm_all


def reflectance_transmitance(delta, n, theta):
    """
    Get optical reflactance and transmittance (TE and TM).
    Valid when first and last layers are non-lossy materials.
    """
    # get m_te_all and m_tm_all
    m_te_all, m_tm_all = transfer_matrix(delta, n, theta)

    # TE reflection/transmission coefficients
    reflect_te = m_te_all[1, 0] / m_te_all[0, 0]
    transmt_te = 1 / m_te_all[0, 0]
    # TE reflectance and transmittance
    r_power_te = np.abs(reflect_te) ** 2
    t_power_te = (
        np.abs(transmt_te) ** 2
        * np.real(n[-1] * np.cos(theta[-1]))
        / np.real(n[0] * np.cos(theta[0]))
    )

    # TM reflection/transmission coefficients
    reflect_tm = m_tm_all[1, 0] / m_tm_all[0, 0]
    transmt_tm = 1 / m_tm_all[0, 0]
    # TM reflectance and transmittance
    r_power_tm = np.abs(reflect_tm) ** 2
    t_power_tm = (
        np.abs(transmt_tm) ** 2
        * np.real(n[-1] * np.cos(theta[-1]))
        / np.real(n[0] * np.cos(theta[0]))
    )

    return r_power_te, t_power_te, r_power_tm, t_power_tm


def get_params(thickness, n, wavelength, theta_inc):
    """
    Get varying parameters due to changing wavelength and theta_inc.
    """
    # calculate wavenumber in all layers
    k = 2 * np.pi * n / wavelength

    # wavenumber in incidence material
    k_inc = k[0]

    # tagent component of wavenumber
    theta_inc_rad = np.deg2rad(theta_inc)
    k_tan = k_inc * np.sin(theta_inc_rad)

    # angle of incidence in all layers
    theta = np.arcsin(k_tan / k)

    # propagating component of wavenumber in all layers
    #     k_prop        = np.sqrt(np.square(k) - np.square(k_tan))
    #      or -- k_prop        = k*np.cos(theta)

    # phase gain in all layers
    delta = k * np.cos(theta) * thickness

    return delta, theta


def get_reflectance_transmittance(thickness, n, wavelength, theta_inc):
    """
    Get optical reflactance and transmittance (TE, TM, and total).
    """

    delta, theta = get_params(thickness, n, wavelength, theta_inc)
    r_power_te, t_power_te, r_power_tm, t_power_tm = reflectance_transmitance(
        delta, n, theta
    )

    return (
        r_power_te,
        t_power_te,
        r_power_tm,
        t_power_tm,
        (r_power_te + r_power_tm) / 2,
        (t_power_te + t_power_tm) / 2,
    )


def get_excitation_conidtion_df():
    """
    Excitation conditions to initialize dataframe for calculation.
    Currently wavelength_range and theta_inc are fixed -- will allow user to update in future version.
    """
    # currently wavelength_range and theta_inc are fixed -- will allow user to update in future version
    wavelength_range = range(400, 701, 1)
    theta_inc_range = range(0, 90, 1)

    excitation_df = pd.DataFrame(
        list(product(wavelength_range, theta_inc_range)),
        columns=["wavelength", "theta_inc"],
    )

    return excitation_df


def get_RT_all_conds(excitation_df, n, thickness):
    """
    Calculate reflectance and transmittance for all excitation conditions and return RT dataframe.
    """

    response_df = pd.DataFrame(
        excitation_df.apply(
            lambda x: get_reflectance_transmittance(
                thickness, n, x.wavelength, x.theta_inc
            ),
            axis=1,
        ).tolist(),
        columns=["R_TE", "T_TE", "R_TM", "T_TM", "R_Total", "T_Total"],
    )

    RT_df = pd.concat([excitation_df, response_df], axis=1)

    return RT_df


####################
# PLOT
def plot_heatmaps(RT_df):
    """
    Return heatmaps of reflectance and transmittance.
    Also with option to select excitation mode to display (TE, TM, Total).
    """

    fig = go.FigureWidget(
        make_subplots(rows=1, cols=2, subplot_titles=("Reflectance", "Transmittance"))
    )

    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="Angle of Incidence (degree)")

    fig.update_layout(height=400)  # , width=800)

    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                name="Optical Reflectance and Transmittance",
                active=0,
                x=0.5,
                xanchor="center",
                y=1.2,
                yanchor="top",
                buttons=list(
                    [
                        dict(
                            label="Total",
                            method="update",
                            args=[
                                {"visible": [False, False, True, False, False, True]},
                                #                                {"title": "TM Response"}
                            ],
                        ),
                        dict(
                            label="TE",
                            method="update",
                            args=[
                                {"visible": [True, False, False, True, False, False]},
                                #                                {"title": "Total Response"}
                            ],
                        ),
                        dict(
                            label="TM",
                            method="update",
                            args=[
                                {"visible": [False, True, False, False, True, False]},
                                #                                {"title": "TE Response"}
                            ],
                        ),
                    ]
                ),
            )
        ]
    )

    if RT_df is None:
        return fig

    for mode in ["R_TE", "R_TM", "R_Total"]:
        heatmap = pd.pivot_table(
            RT_df, values=mode, index="theta_inc", columns="wavelength"
        )
        fig.add_trace(
            go.Heatmap(
                z=heatmap.values,
                x=heatmap.columns,
                y=heatmap.index,
                colorscale="hot",
                name=mode,
                zmin=0,
                zmax=1,
                showscale=False,
            ),
            row=1,
            col=1,
        )

    for mode in ["T_TE", "T_TM", "T_Total"]:
        heatmap = pd.pivot_table(
            RT_df, values=mode, index="theta_inc", columns="wavelength"
        )
        fig.add_trace(
            go.Heatmap(
                z=heatmap.values,
                x=heatmap.columns,
                y=heatmap.index,
                colorscale="hot",
                name=mode,
                zmin=0,
                zmax=1,
                showscale=True,
            ),
            row=1,
            col=2,
        )

    return fig


####################
# GET USER INPUTS
def get_n_and_thickness(n_real, n_imag, thickness):
    """
    Get user inputs of refractive indices (real and imaginary) and layer thicknesses (nm).
    """

    # convert to arguments to lists
    n_real_list = [float(n.strip()) for n in n_real.split(";")]
    n_imag_list = [float(n.strip()) for n in n_imag.split(";")]
    thickness_list = [float(thickness.strip()) for thickness in thickness.split(";")]

    # check number of input layers
    if len(n_real_list) != len(n_imag_list):
        raise Exception(
            "Numbers of real and imaginary part of refractive indices are not equal."
        )
    if len(n_real_list) != len(thickness_list):
        raise Exception("Numbers of refractive indices and thicknesses are not equal.")

    # get n and thickness
    n = np.array(
        [(n_real + 1j * n_imag) for n_real, n_imag in zip(n_real_list, n_imag_list)]
    )
    thickness = np.array(thickness_list)

    # pad front and back with n=1 and thickness=0 (air)
    n = np.pad(n, 1, "constant", constant_values=1)
    thickness = np.pad(thickness, 1, "constant", constant_values=0)

    return (n, thickness)


####################
# UPDATE INPUT, CALCULATE, RETURN HEATMAP
def update_layers_cal(n_real, n_imag, thickness):
    """
    Get user specified information of thin film layers (refractive indeices and thicknesses).
    Plot heatmaps of reflectance and transmittance.
    Return RT dataframe for all excitation conditions.
    """

    print("Calculation in progress ...")

    # get user specified information of thin film layers
    n, thickness = get_n_and_thickness(n_real, n_imag, thickness)

    # define excitation condition -- currently wavelength 400-700nm and theta_inc 0-89 deg
    excitation_df = get_excitation_conidtion_df()

    # get RT dataframe
    RT_df = get_RT_all_conds(excitation_df, n, thickness)

    print("Calculation completed.")

    return RT_df
