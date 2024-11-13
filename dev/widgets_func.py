import numpy as np
from dev.cal_func import update_layers_cal, plot_heatmaps
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ipywidgets import interactive, Label, HTML, HBox, VBox, Button


f_heatmap = go.FigureWidget(
    make_subplots(rows=1, cols=2, subplot_titles=("Reflectance", "Transmittance"))
)


####################
# Initialize interactive_box
def get_interactive_box(
    wavelength_range=range(400, 701, 1), theta_inc_range=range(0, 90, 1)
):

    # UI to update thin film layers
    update_layers = interactive(
        update_layers_cal,
        {"manual": True, "manual_name": "Update layers"},
        n_real="Real{n}",
        n_imag="Imag{n}",
        thickness="Thickness (nm)",
    )

    # fixed wavelength for now
    # need to also update get_excitation_conidtion_df() when user type in these two parameters
    #     wavelength_range = range(400, 701, 1)
    #     theta_inc_range  = range(0, 90, 1)

    # set initial plots before user provides information of thin film layers

    if update_layers.result is None:

        #### RT across wavelength range
        # Reflectance across range of wavelength -- initially all 0
        f_R_lambda = go.FigureWidget(
            data=[
                go.Scatter(
                    x=np.array(wavelength_range),
                    y=np.zeros(len(wavelength_range)),
                    name=mode,
                )
                for mode in ["R_TE", "R_TM", "R_Total"]
            ],
            layout=go.Layout(
                width=800,
                legend_orientation="h",
                legend={"x": 0, "y": 1.1},
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance (a.u.)",
                yaxis={"range": [-0.1, 1.1]},
            ),
        )

        # Transmittance across range of wavelength -- initially all 1
        f_T_lambda = go.FigureWidget(
            data=[
                go.Scatter(
                    x=np.array(wavelength_range),
                    y=np.ones(len(wavelength_range)),
                    name=mode,
                )
                for mode in ["T_TE", "T_TM", "T_Total"]
            ],
            layout=go.Layout(
                width=800,
                legend_orientation="h",
                legend={"x": 0, "y": 1.1},
                xaxis_title="Wavelength (nm)",
                yaxis_title="Transmittance (a.u.)",
                yaxis={"range": [-0.1, 1.1]},
            ),
        )

        #### RT across theta_inc range
        # Reflectance across range of angle of incidencec -- initially all 0
        f_R_theta = go.FigureWidget(
            data=[
                go.Scatter(
                    x=np.array(theta_inc_range),
                    y=np.zeros(len(theta_inc_range)),
                    name=mode,
                )
                for mode in ["R_TE", "R_TM", "R_Total"]
            ],
            layout=go.Layout(
                width=800,
                legend_orientation="h",
                legend={"x": 0, "y": 1.1},
                xaxis_title="Angle of Incidence (degree)",
                yaxis_title="Reflectance (a.u.)",
                yaxis={"range": [-0.1, 1.1]},
            ),
        )

        # Transmittance across range of angle of incidencec -- initially all 1
        f_T_theta = go.FigureWidget(
            data=[
                go.Scatter(
                    x=np.array(theta_inc_range),
                    y=np.ones(len(theta_inc_range)),
                    name=mode,
                )
                for mode in ["T_TE", "T_TM", "T_Total"]
            ],
            layout=go.Layout(
                width=800,
                legend_orientation="h",
                legend={"x": 0, "y": 1.1},
                xaxis_title="Angle of Incidence (degree)",
                yaxis_title="Transmittance (a.u.)",
                yaxis={"range": [-0.1, 1.1]},
            ),
        )

    ########################################
    # function to plot RT at different theta_inc
    def update_RT_theta(theta_inc):
        if update_layers.result is None:
            pass
        else:
            RT_df = update_layers.result
            for selected_data, mode in zip(
                f_R_lambda.data, ["R_TE", "R_TM", "R_Total"]
            ):
                selected_data.y = RT_df.loc[RT_df.theta_inc == theta_inc, mode]
            for selected_data, mode in zip(
                f_T_lambda.data, ["T_TE", "T_TM", "T_Total"]
            ):
                selected_data.y = RT_df.loc[RT_df.theta_inc == theta_inc, mode]

    # UI to update theta_inc
    theta_slider = interactive(update_RT_theta, theta_inc=(1, 89, 1))

    ########################################
    # function to plot RT at different wavelength
    def update_RT_wavelength(wavelength):
        if update_layers.result is None:
            pass
        else:
            RT_df = update_layers.result
            for selected_data, mode in zip(f_R_theta.data, ["R_TE", "R_TM", "R_Total"]):
                selected_data.y = RT_df.loc[RT_df.wavelength == wavelength, mode]
            for selected_data, mode in zip(f_T_theta.data, ["T_TE", "T_TM", "T_Total"]):
                selected_data.y = RT_df.loc[RT_df.wavelength == wavelength, mode]

    # UI to update lambda
    lambda_slider = interactive(update_RT_wavelength, wavelength=(400, 700, 10))

    ########################################

    # header messages
    header_text = "Interactive heatmaps and plots of optical reflectance and transmittance through thin films using transfer-matrix calculation."
    header = HTML(value="<{size}>{text}</{size}>".format(text=header_text, size="h2"))
    description_text = "Assume light propagates from <u><b>air</b></u> to <b>specified thin film layers</b>, then exits to <u><b>air</b></u>."
    description = HTML(
        value="<{size}>{text}</{size}>".format(text=description_text, size="h3")
    )

    # description to update thin film layers
    update_layers_inst = (
        "Update layers and perform calculation. Note that it could take a while."
    )
    update_layers_note = HTML(
        value="<{size}>{text}</{size}>".format(text=update_layers_inst, size="h3")
    )

    update_layers_n_real = "List real part of refractive indices of materials from the front to the back of a thin film stack separated by <b>;</b> (e.g. 1.5; 1.33; 1.5)."
    update_layers_n_imag = "List imaginary part of refractive indices of materials separated by <b>;</b> (e.g. 0.1; 0; 0.05, positive and negative values for absorber and amplifier, respectively)."
    update_layers_thickness = "List thickness of materials in <b>nanometer</b> separated by <b>;</b> (e.g. 500; 800; 500)."
    update_layers_text = (
        f"{update_layers_n_real}<br>{update_layers_n_imag}<br>{update_layers_thickness}"
    )
    update_layers_html = HTML(value=update_layers_text)

    # RT across range of wavelength at difference theta_inc
    theta_slide_text = "Move <b>theta_inc slider</b> to see reflectance and transmittance at different angle of incidence."
    theta_slide_html = HTML(
        value="<{size}>{text}</{size}>".format(text=theta_slide_text, size="h3")
    )
    hb1 = HBox((f_R_lambda, f_T_lambda))
    vb1 = VBox((theta_slide_html, theta_slider, hb1))

    # RT across range of theta_inc at difference wavelength
    lambda_slide_text = "Move <b>wavelength slider</b> to see reflectance and transmittance at different wavelength."
    lambda_slide_html = HTML(
        value="<{size}>{text}</{size}>".format(text=lambda_slide_text, size="h3")
    )
    hb2 = HBox((f_R_theta, f_T_theta))
    vb2 = VBox((lambda_slide_html, lambda_slider, hb2))

    ########################################
    # function to replace RT heatmap with new calculation
    def heatmap_update(b):
        global f_heatmap
        RT_heatmap = plot_heatmaps(update_layers.result)
        f_heatmap.layout = RT_heatmap.layout
        f_heatmap.data = []
        for i in range(len(RT_heatmap.data)):
            f_heatmap.add_trace(RT_heatmap.data[i])

    # UI to update RT heatmap
    button = Button(description="Plot heatmaps")
    button.on_click(heatmap_update)

    ########################################

    interactive_box = VBox(
        (
            header,
            description,
            Label("#" * 100),
            update_layers_note,
            update_layers_html,
            update_layers,
            button,
            f_heatmap,
            Label("#" * 100),
            vb1,
            Label("#" * 100),
            vb2,
            Label("#" * 100),
        )
    )
    return interactive_box
