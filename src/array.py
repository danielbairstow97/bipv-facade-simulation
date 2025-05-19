from typing import Optional
import pandas as pd
import numpy as np

from pvlib.pvsystem import Array
from src.view_profile import ViewProfile


class FacadeArray(Array):
    """
    Extension of pvlib.pvsystem.Array to allow custom facade shading logic
    via a ViewProfile.
    """

    facade_profile: Optional[ViewProfile]

    def __init__(self, *args, facade_profile: Optional[ViewProfile] = None, **kwargs):
        """
        Parameters
        ----------
        facade_profile : ViewProfile, optional
            Profile describing the obstruction of sunlight due to surrounding
            structures or horizons.
        """
        super().__init__(*args, **kwargs)
        self.facade_profile = facade_profile

    def get_irradiance(
        self,
        solar_zenith: pd.Series,
        solar_azimuth: pd.Series,
        dni: pd.Series,
        ghi: pd.Series,
        dhi: pd.Series,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply facade shading logic before computing total irradiance.

        Returns
        -------
        pd.DataFrame
            Irradiance on the tilted surface with facade-based shading applied.
        """
        # Apply the custom facade profile to mask irradiance inputs
        if self.facade_profile is not None:
            _, dni, ghi, dhi = self.facade_profile.apply(
                solar_zenith, solar_azimuth, dni, ghi, dhi
            )

        # Use the base Array method for final irradiance calculation
        return super().get_irradiance(
            solar_zenith, solar_azimuth, dni, ghi, dhi, **kwargs
        )
