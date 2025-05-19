from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pvlib.irradiance import get_total_irradiance
from src.view_profile import ViewProfile, PANORAMIC
from src.site import Site


class Facade:
    """
    Represents a building facade, including its orientation, tilt, and
    obstruction profile, and provides methods to compute solar irradiance.
    """

    orientation: float
    tilt: float
    site: Site
    facade_profile: ViewProfile

    def __init__(
        self,
        orientation_azimuth: float,
        site,
        tilt: float = 90,
        profile: ViewProfile = PANORAMIC
    ):
        self.orientation = orientation_azimuth
        self.tilt = tilt
        self.site = site

        # Combine facade and site horizon profile
        self.facade_profile = profile.rotate(
            orientation_azimuth).combine(site.horizon_profile)

    def solve_irradiance(
        self,
        solar_position: pd.DataFrame,
        site_irradiance: pd.DataFrame,
        tilt: Optional[float] = None,
        orientation: Optional[float] = None
    ) -> 'FacadeResult':
        """
        Compute POA irradiance for the facade given solar position and site data.
        """
        tilt = tilt or self.tilt
        orientation = orientation or self.orientation

        # Mask irradiance using the profile
        blocked_elevation_at_dt, dni, ghi, dhi = self.facade_profile.apply(
            solar_position.zenith,
            solar_position.azimuth,
            site_irradiance.dni,
            site_irradiance.ghi,
            site_irradiance.dhi
        )

        poa_irradiance = get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=orientation,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi
        )

        return FacadeResult(
            poa_irradiance, orientation, tilt, self.site,
            self.facade_profile, blocked_elevation_at_dt
        )

    def find_optimal_tilt(
        self,
        tilt_range=range(0, 100, 10),
        mask_indirect: bool = True
    ) -> Tuple[float, Dict[int, 'FacadeResult']]:
        """
        Brute-force search for optimal facade tilt angle based on annual POA energy.
        """
        best_tilt = None
        best_total = -np.inf
        results: Dict[int, FacadeResult] = {}

        for tilt in tilt_range:
            irradiance = self.SolveIrradiance(
                self.site.GenerateSolarResources()[0],
                self.site.GenerateSolarResources()[1],
                tilt=tilt
            )

            results[tilt] = irradiance
            total = irradiance.Total()
            if total > best_total:
                best_total = total
                best_tilt = tilt

        return best_tilt, results


class FacadeResult:
    """
    Result object holding irradiance values and metadata for a facade orientation.
    """

    irradiance_df: pd.DataFrame
    tilt: float
    orientation: float
    site: any  # Again, replace with `Site` if available
    facade_view: ViewProfile

    def __init__(
        self,
        irradiance_df: pd.DataFrame,
        orientation: float,
        tilt: float,
        site,
        facade_view: ViewProfile,
        blocked_elevation_at_dt: pd.Series
    ):
        self.irradiance_df = irradiance_df
        self.tilt = tilt
        self.orientation = orientation
        self.site = site
        self.facade_view = facade_view

    def total(self) -> float:
        """
        Return the total annual POA irradiance (Wh/m²).
        """
        return self.irradiance_df["poa_global"].sum()

    def plot_yearly_irradiance(
        self,
        figure: Optional[plt.Figure] = None,
        show: bool = True,
        label: str = 'Daily POA Irradiance',
        title: str = 'Daily Plane of Array Irradiance'
    ):
        """
        Plot daily irradiance totals for the entire year.
        """
        daily_irradiance = self.irradiance_df['poa_global'].resample('D').sum()

        if figure is None:
            figure = plt.figure(figsize=(10, 4))
            show = False

        plt.figure(figure)
        plt.plot(daily_irradiance.index, daily_irradiance.values, label=label)
        plt.ylabel('Daily Irradiance (Wh/m²)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        if show:
            plt.show()

    def plot_daily_profile(
        self,
        figure: Optional[plt.Figure] = None,
        show: bool = True,
        label: str = 'Average Hourly POA Irradiance',
        title: str = 'Average Hourly Irradiance Profile'
    ) -> pd.Series:
        """
        Plot average daily irradiance profile (hourly means).
        """
        hourly_average = self.irradiance_df['poa_global'].groupby(
            self.irradiance_df.index.hour
        ).mean()

        if figure is None:
            figure = plt.figure(figsize=(10, 4))
            show = False

        plt.figure(figure)
        plt.plot(hourly_average.index, hourly_average.values,
                 marker='o', label=label)
        plt.xlabel('Hour of Day')
        plt.ylabel('Irradiance (W/m²)')
        plt.title(title)
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.legend()
        if show:
            plt.show()

        return hourly_average
