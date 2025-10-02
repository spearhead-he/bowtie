#!/usr/bin/env python3
"""
Utility functions to aid with the Bowtie package.
"""
__author__ = "Christian Palmroos"
__credits__ = ["Christian Palmroos", "Philipp Oleynik"]

import numpy as np
import pandas as pd


def read_npy_vault(vault_name):
    """
    Reads in either the 'array_vault_e_256' or 'array_vault_p_256'

    Parameters:
    -----------
    vault_name : {str}

    Returns
    ----------
    particles_shot : {np.ndarray}
    particles_response : {np.ndarray}
    energy_grid : {dict}
    radiation_area : {float}
    """

    # The number of particles shot in a simulation of all energy bins
    particles_shot = np.load(f"{vault_name}/particles_Shot.npy")

    # The number of particles detected per particle channel in all energy bins
    particles_response = np.load(f"{vault_name}/particles_Respo.npy")

    other_params = np.load(f"{vault_name}/other_params.npy")

    # The total number of energy bins
    nstep = int(other_params[0])

    # The radiation area (isotropically radiating sphere) around the Geant4 instrument model in cm2
    radiation_area = other_params[2]

    # Midpoints of the energy bins in MeV
    energy_midpoint = np.load(f"{vault_name}/energy_Mid.npy")

    # High cuts of the energy bins in MeV
    energy_toppoint = np.load(f"{vault_name}/energy_Cut.npy")

    # The energy bin widths in MeV
    energy_channel_width = np.load(f"{vault_name}/energy_Width.npy")

    # An energy grid in the format compatible with the output of a function in the bowtie package
    energy_grid = { "nstep": nstep, 
                    "midpt": energy_midpoint,
                    "ehigh": energy_toppoint, 
                    "enlow": energy_toppoint - energy_channel_width,
                    "binwd": energy_channel_width }

    return particles_shot, particles_response, energy_grid, radiation_area


def assemble_response_matrix(response_df) ->list[dict]:
    """
    Assembles the response matrix needed by 'calculate_bowtie_gf()' from
    an input dataframe.
    """
    
    response_matrix = []
    for col in response_df.columns:

        response_matrix.append({
            "name": col,
            "grid": {"midpt" : response_df.index.values,
                     "nstep" : len(response_df.index)},
            "resp": response_df[col].values
        })
    
    return response_matrix


def calculate_response_matrix(particles_shot, particles_response, energy_grid:dict,
                             radiation_area:float, side:int,
                             channel_start:int, channel_stop:int,
                             contamination:bool=False, sum_channels:bool=False):
    """
    This function only applies for BepiColombo / SIXS-P energy chanel configuration, and should NOT be 
    used for the calculation of any other particle instrument's response matrix.
    
    Parameters:
    -----------
    particles_shot : {np.ndarray}
    particles_response : {np.ndarray}
    energy_grid : {dict}
    radiation_area : {float}
    channel_start : {int}
    channel_stop : {int}
    side : {int}

    contamination : {bool} optional, default False
    sum_channels : {bool} optional, default False
    
    Returns: 
    --------
    response_matrix : {list[dict]} 
    """

    if sum_channels:
        step = 2
        channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "P1+P2", "P2", "P3+P4", "P4", "P5+P6", "P6", "P7+P8", "P8", "P9"]
    else:
        step = 1
        if contamination:
            channel_names = ["O", "EP1", "EP2", "EP3", "EP4", "EP5", "EP6", "EP7", "PE1", "PE2", "PE3", "PE4", "PE5", "PE6", "PE7", "PE8", "PE9"]

        # The normal case: no summing channels and no contamination.
        else:
            channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", \
                             "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

    response_matrix = []
    normalize_to_area = 1.0 / ((particles_shot + 1) / radiation_area) * np.pi

    for i in range(channel_start, channel_stop, step):

        if not sum_channels:
            resp_cache = particles_response[:, i, side] * normalize_to_area
            resp_error = np.sqrt(particles_response[:, i, side]) * normalize_to_area

        else:
            if i < channel_stop-1:
                resp_cache1 = particles_response[:, i, side] * normalize_to_area
                resp_cache2 = particles_response[:, i+1, side] * normalize_to_area

                # Sum element-wise over the slices of the two channels
                resp_cache = np.add(resp_cache1, resp_cache2)
                resp_error = np.sqrt(np.add(particles_response[:, i, side],particles_response[:, i+1, side])) * normalize_to_area
            else:
                resp_cache = particles_response[:, i, side] * normalize_to_area

        response_matrix.append({
            "name"  : channel_names[i],
            "grid"  : energy_grid,
            "resp"  : resp_cache,  # The channel response
            "error" : resp_error
        })

    return response_matrix


def save_results(results, filename, column_names=None, save_figures=False):
    """
    Saves the results Bowtie.bowtie_analysis or Bowtie.bowtie_analysis_full_stack
    
    Parameters:
    -----------
    results : {list[dict]} 
    
    filename : {str} Name for the csv table.
    
    column_names : {list[str]} Names for the csv table columns. Optional
    
    save_figures : {bool} Saves the figures in .png format. Optional
    """

    # The indices (rows) of the output file
    INDICES = ["geometric_factor", "gf_error_up", "gf_error_low", "effective_energy"]
    FIGNAME = "bowtie.png"

    # Make sure that results are a list of dictionaries to iterate over
    if isinstance(results,dict):
        results = [results]

    # Handle column names. If single string, make sure it's contained in a list.
    # If not given, procude a range of integers for placeholders
    if isinstance(column_names,str):
        column_names = [column_names]

    if column_names is None:
        column_names = range(len(results))

    data = np.empty((len(INDICES),len(results)))

    # Loop through the list of results:
    for i, res in enumerate(results):
        
        data[0,i] = res["geometric_factor"]
        data[1,i] = res["geometric_factor_errors"]["gfup"]
        data[2,i] = res["geometric_factor_errors"]["gflo"]
        try:
            data[3,i] = res["effective_energy"]
        except KeyError:
            print("At least one integral bowtie result in the results. The threshold energy will appear in the 'effective_energy' column.")
            data[3,i] = res["threshold_energy"]
        
        if save_figures:
            
            res["fig"].savefig(f"{column_names[i]}_{FIGNAME}", 
                               transparent=False, facecolor="white", 
                               bbox_inches="tight")
    
    result_df = pd.DataFrame(data=data, index=INDICES, columns=column_names)

    filename = filename if filename[-3:] == "csv" else f"{filename}.csv"
    result_df.to_csv(filename)


def main():

    print("This library of functions is a part of the Bowtie analysis package.")

if __name__ == "__main__":

    main()
