#!/usr/bin/env python3
"""
The Bowtie package. This file contains the Bowtie class, which runs bowtie analysis on 
response function data, given a range of spectra.
"""
__author__ = "Christian Palmroos"
__credits__ = ["Christian Palmroos", "Philipp Oleynik"]

import os
import pandas as pd

from . import bowtie_util as btutil
from . import bowtie_calc
from . import validations as validate
from .spectra import Spectra

RESPONSE_INDEX_COL = "incident_energy"

class Bowtie:
    """
    Contains the functionality needed to run bowtie analysis.
    """

    def __init__(self, energy_min:float, energy_max:float, data:pd.DataFrame,
                sigma:int=3) -> None:

        if isinstance(data,pd.DataFrame):
            self.data = data
            self.response_matrix = btutil.assemble_response_matrix(response_df=data)
        else:
            raise TypeError(f"Data needs to be a pandas dataframe, {type(data)} was passed!")

        self.energy_min = energy_min
        self.energy_max = energy_max
        self.sigma = sigma


    def set_energy_range(self, energy_min:float, energy_max:float) -> None:
        """
        Sets the limits of the energy range.
        """
        self.energy_min = energy_min
        self.energy_max = energy_max


    def bowtie_analysis(self, channel:str, spectra:Spectra, plot:bool=False,
                        geom_factor_confidence:float=0.9, bowtie_method:str="differential") -> dict:
        """
        Runs bowtie analysis to a channel.

        Parameters:
        -----------
        channel : {str}
        spectra : {bowtie.Spectra}
        plot : {bool}
        geom_factor_confidence : {float} Float between 0-1.
        bowtie_method : {str} Method of bowtie analysis, either 'differential' (default) or 'integral'.
        """

        # Check that the channel is valid
        validate.validate_channel(channel=channel, dataframe=self.data)

        # Check that the input spectra is valid (also that it has a power law spectra)
        validate.validate_spectra(spectra=spectra)

        # Validate bowtie mewthod
        validate.validate_bowtie_method(bowtie_method=bowtie_method)

        # Choose the correct response from the matrix
        for response in self.response_matrix:
            if response["name"]==channel:
                response_dict = response

        use_integral_bowtie = False if bowtie_method=="differential" else True

        # The bowtie_results are in order:
        # Geometric factor (G \Delta E) in cm2srMeV : {float}
        # Geometric factor errors : {dict} with keys ["gfup", "gflo"]
        # The effective energy of the channel in MeV : {float}
        # The channel effective lower boundary in MeV : {float} 
        # The channel effective upper boundary in MeV : {float}
        # if plot, also returns fig and axes
        bowtie_results = bowtie_calc.calculate_bowtie_gf(response_data=response_dict, spectra=spectra,
                                                    emin=self.energy_min, emax=self.energy_max,
                                                    use_integral_bowtie=use_integral_bowtie,
                                                    sigma=self.sigma, plot=plot, gfactor_confidence_level=geom_factor_confidence,
                                                    return_gf_stddev=True, channel=channel)

        # Collect the results to a dictionary for easier handling
        energy_id = "effective_energy" if bowtie_method=="differential" else "threshold_energy"
        result_dict = {}
        result_names = ["geometric_factor", "geometric_factor_errors", energy_id,\
                        "effective_lower_boundary", "effective_upper_boundary"]
        
        # Attach effective lower and upper boundary to class attributes. Let's not return them
        # in the dictionary to avoid confusion.
        self.effective_energy_low = bowtie_results[3]
        self.effective_energy_high = bowtie_results[4]

        if plot:
            result_names.append("fig")
            result_names.append("axes")

        for i, res in enumerate(bowtie_results):

            # Handle gf errors here
            if i==1:
                res["gfup"] -= bowtie_results[0]
                res["gflo"] -= bowtie_results[0]
                res["gflo"] = -res["gflo"]

            if result_names[i] not in ("effective_lower_boundary", "effective_upper_boundary"):
                result_dict[result_names[i]] = res

        return result_dict


    def bowtie_analysis_full_stack(self, spectra:Spectra, plot:bool=False,
                                   geom_factor_confidence:float=0.9, bowtie_method:str="differential") -> list[dict]:
        """
        Wrapper for bowtie_analysis(). Runs bowtie_analysis() for all of the 
        given channels in bowtie.data
        """

        all_bowtie_results = []

        for channel in self.data.columns:

            new_result = self.bowtie_analysis(channel=channel, spectra=spectra, plot=plot,
                                              geom_factor_confidence=geom_factor_confidence, bowtie_method=bowtie_method)

            all_bowtie_results.append(new_result)

        return all_bowtie_results



def main(filename:str):

    return True

if __name__ == "__main__":

    dev_directory = "_development"
    filename = "side0_p_response_functions.csv"

    main(filename=f"{dev_directory}{os.sep}{filename}")
