
"""
This file contains validations functions.
"""
__author__ = "Christian Palmroos"
__credits__ = ["Christian Palmroos", "Philipp Oleynik"]

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

# The importing of Spectra is hidden from the interpreter at runtime. Spectra
# is imported so that the input type of the validate_spectra() can be signaled.
if TYPE_CHECKING:
    from .spectra import Spectra

def validate_response_df_and_grid(response_df:pd.DataFrame, energy_grid:np.ndarray) -> None:

    # Neither was provided -> Error
    if response_df is None and energy_grid is None:
        raise KeyError("Either the argument 'response_df' or 'energy_grid must be provided!")

    # energy_grid was not provided -> response_df must be a correct input
    elif energy_grid is None:
        if not isinstance(response_df,pd.DataFrame):
            raise TypeError(f"Argument 'reponse_df' must be a Pandas DataFrame, but {type(response_df)} was provided.")

    # response_df was not provided -> energy_grid must be a correct input
    elif response_df is None:
        if not isinstance(energy_grid,np.ndarray):
            raise TypeError(f"Argument 'energy_grid' must be a Numpy array, but {type(energy_grid)} was provided.")

    # Both were provided -> response_df takes precedence
    else:
        if not isinstance(response_df,pd.DataFrame):
            raise TypeError(f"Argument 'reponse_df' must be a Pandas DataFrame, but {type(response_df)} was provided.")



def validate_spectra(spectra:"Spectra") -> None:

    from .spectra import Spectra

    if not isinstance(spectra,Spectra): 
        raise TypeError(f"Input spectra needs to be a Spectra-type of object, not {type(spectra)}!")
    if not hasattr(spectra, "power_law_spectra"):
        raise AttributeError("Produce power law spectra with Spectra.produce_power_law_spectra() to calculate bowtie!")


def validate_channel(channel:str|int, dataframe:pd.DataFrame) -> None:

    if channel not in dataframe.columns:
        raise ValueError(f"Channel {channel} not found in response dataframe!")


def validate_bowtie_method(bowtie_method:str) -> None:

    VALID_BOWTIE_METHODS = ("differential", "integral")

    if not isinstance(bowtie_method, str):
        raise TypeError("Argument 'bowtie_method' must be of type string!")

    if bowtie_method not in VALID_BOWTIE_METHODS:
        raise ValueError(f"Argument 'bowtie_method' not among the valid inputs! Valid inputs are: {VALID_BOWTIE_METHODS}")

