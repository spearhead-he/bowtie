import filecmp
import math
import matplotlib
import os
import pytest
import numpy as np
import pandas as pd

from bowtie import bowtie as bow
from bowtie import bowtie_util as btutil

"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=bowtie/tests/baseline bowtie/tests/test.py

To run the tests locally, go to the base directory of the repository and run:
pytest -rP --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html --cov=bowtie bowtie/tests/test.py
"""

@pytest.mark.mpl_image_compare(remove_text=True, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
def test_bowtie():
    response_df = pd.read_csv("boxcar_responses.csv", index_col="incident_energy")
    #
    energy_min = 0.01
    energy_max = 50.
    bowtie = bow.Bowtie(energy_min=energy_min, energy_max=energy_max, data=response_df)
    #
    gamma_min = -5.5
    gamma_max = -2.5
    num_of_spectra = 100
    spectra = bow.Spectra(gamma_min=gamma_min, gamma_max=gamma_max, gamma_steps=num_of_spectra)
    #
    spectra.produce_power_law_spectra(response_df)
    #
    channel = "boxcar1"
    b1_results = bowtie.bowtie_analysis(channel=channel, spectra=spectra, plot=False)

    assert b1_results['geometric_factor'] == 0.02783532881086404
    assert b1_results['geometric_factor_errors'] == {'gfup': np.float64(0.00015185377096835553), 'gflo': np.float64(0.00010571189187651153)}
    assert b1_results['effective_energy'] == np.float64(0.0809)
    # assert b1_results['fig'] == 
    # assert b1_results['axes'] == 

    new_gamma_min = -3.5
    new_gamma_max = -1.5
    spectra.set_spectral_indices(gamma_min=new_gamma_min, gamma_max=new_gamma_max)
    spectra.produce_power_law_spectra(response_df=response_df)

    new_b1_results = bowtie.bowtie_analysis(channel=channel, spectra=spectra, plot=False)

    assert new_b1_results['geometric_factor'] == 0.02945031431749762

    assert math.isclose(new_b1_results['geometric_factor_errors']['gfup'], np.float64(8.885965640134663e-05))
    assert math.isclose(new_b1_results['geometric_factor_errors']['gflo'], np.float64(5.596433236966167e-05))
    assert math.isclose(new_b1_results['effective_energy'], np.float64(0.0824))
    # assert new_b1_results['fig'] == 
    # assert new_b1_results['axes'] ==

    all_channels_results = bowtie.bowtie_analysis_full_stack(spectra=spectra, plot=True)

    assert len(all_channels_results) == response_df.shape[1]

    # Finally check the last result (which now is for an integral channel)
    integral_channel = "boxcar4"
    b4_results = bowtie.bowtie_analysis(channel=integral_channel, spectra=spectra, 
                                        plot=False, bowtie_method="integral")

    assert math.isclose(b4_results['geometric_factor'], 0.9927950947299083)
    assert math.isclose(b4_results['geometric_factor_errors']['gfup'], np.float64(0.0030737165893933716))
    assert math.isclose(b4_results['geometric_factor_errors']['gflo'], np.float64(0.003365956433684758))
    assert math.isclose(b4_results['threshold_energy'], np.float64(4.9804))

    # test saving
    filename = "test.csv"
    column_names = response_df.columns
    btutil.save_results(results=all_channels_results, filename=filename, column_names=column_names, save_figures=True)

    # Compare the two files
    filecmp.cmp('test.csv', 'bowtie/tests/test_org.csv')

    for i in range(1,5):
        assert os.path.exists(f'boxcar{i}_bowtie.png')

    # return last produced fig for mpl pytest
    return all_channels_results[3]['fig']
