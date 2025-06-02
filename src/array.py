from typing import Optional
import pandas as pd
import numpy as np

from pvlib.pvsystem import Array
from src.view_profile import ViewProfile


class SurfaceArray(Array):
    """
    Extension of pvlib.pvsystem.Array to allow custom surface shading logic
    via a ViewProfile.
    """

    surface_profile: Optional[ViewProfile]

    def __init__(self, *args, surface_profile: Optional[ViewProfile] = None, **kwargs):
        """
        Parameters
        ----------
        surface_profile : ViewProfile, optional
            Profile describing the obstruction of sunlight due to surrounding
            structures or horizons.
        """
        super().__init__(*args, **kwargs)
        self.surface_profile = surface_profile

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
        Apply surface shading logic before computing total irradiance.

        Returns
        -------
        pd.DataFrame
            Irradiance on the tilted surface with surface-based shading applied.
        """
        # Apply the custom surface profile to mask irradiance inputs
        if self.surface_profile is not None:
            _, dni, ghi, dhi = self.surface_profile.apply(
                solar_zenith, solar_azimuth, dni, ghi, dhi
            )

        # Use the base Array method for final irradiance calculation
        return super().get_irradiance(
            solar_zenith, solar_azimuth, dni, ghi, dhi, **kwargs
        )
