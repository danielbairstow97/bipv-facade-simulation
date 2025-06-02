from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from pvlib.irradiance import get_total_irradiance
from src.view_profile import ViewProfile, PANORAMIC
from src.site import Site


class SurfaceType(Enum):
    UNKNOWN = 1
    FACADE = 2
    BALCONY = 3


class Surface:
    """
    Represents a building facade, including its azimuth, tilt, and
    obstruction profile, and provides methods to compute solar irradiance.
    """
    name: str
    azimuth: float
    tilt: float
    site: Site
    view_profile: ViewProfile
    area_per_level: float
    efficiency: float

    def __init__(
        self,
        name: str,
        type: SurfaceType,
        azimuth: float,
        site: Site,
        area_per_level: float,
        tilt: float = 90,
        view_profile: ViewProfile = PANORAMIC,
        efficiency: float = 0.2,
    ):
        self.name = name
        self.type = type
        self.azimuth = azimuth
        self.tilt = tilt
        self.site = site
        self.area_per_level = area_per_level
        self.efficiency = efficiency

        # Combine facade and site horizon profile
        self.view_profile = view_profile.rotate(
            azimuth).combine(site.horizon_profile)

    def solve_irradiance(
        self,
        solar_position: pd.DataFrame,
        site_irradiance: pd.DataFrame,
        tilt: Optional[float] = None,
        azimuth: Optional[float] = None
    ) -> 'SurfaceResult':
        """
        Compute POA irradiance for the facade given solar position and site data.
        """
        tilt = tilt or self.tilt
        azimuth = azimuth or self.azimuth

        # Mask irradiance using the profile
        blocked_elevation_at_dt, dni, ghi, dhi = self.view_profile.apply(
            solar_position.zenith,
            solar_position.azimuth,
            site_irradiance.dni,
            site_irradiance.ghi,
            site_irradiance.dhi
        )

        poa_irradiance = get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
            dni=dni,
            ghi=ghi,
            dhi=dhi
        )

        return SurfaceResult(
            poa_irradiance, azimuth, tilt, self.site,
            self.view_profile
        )

    def find_optimal_tilt(
        self,
        tilt_range=range(0, 100, 10),
    ) -> Tuple[float, Dict[int, 'SurfaceResult']]:
        """
        Brute-force search for optimal facade tilt angle based on annual POA energy.
        """
        best_tilt = None
        best_total = -np.inf
        results: Dict[int, SurfaceResult] = {}

        position, irrad = self.site.GenerateSolarResources()
        for tilt in tilt_range:
            irradiance = self.SolveIrradiance(
                position,
                irrad,
                tilt=tilt
            )

            results[tilt] = irradiance
            total = irradiance.Total()
            if total > best_total:
                best_total = total
                best_tilt = tilt

        return best_tilt, results


class SurfaceResult:
    """
    Result object holding irradiance values and metadata for a facade azimuth.
    """

    irradiance_df: pd.DataFrame
    tilt: float
    azimuth: float
    site: any  # Again, replace with `Site` if available
    view: ViewProfile

    def __init__(
        self,
        irradiance_df: pd.DataFrame,
        azimuth: float,
        tilt: float,
        site,
        view: ViewProfile,
    ):
        self.irradiance_df = irradiance_df
        self.tilt = tilt
        self.azimuth = azimuth
        self.site = site
        self.view = view

    def total(self) -> float:
        """
        Return the total annual POA irradiance (Wh/m²).
        """
        return self.irradiance_df["poa_global"].sum()

    def plot_total_irradiance(
        self,
        figure: Optional[plt.Figure] = None,
        show: bool = False,
        sample: str = "D",
        label: str = 'POA Irradiance',
        title: str = 'Plane of Array Irradiance',
        plot_fn=plt.plot,
    ):
        """
        Plot irradiance totals for the entire year.
        """
        daily_irradiance = self.irradiance_df['poa_global'].resample(
            sample).sum()

        if figure is None:
            figure = plt.figure(figsize=(10, 4))
            show = True

        plt.figure(figure)
        plot_fn(daily_irradiance.index, daily_irradiance.values, label=label)
        plt.ylabel('Irradiance (Wh/m²)')
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
        title: str = 'Average Hourly Irradiance Profile',
        plot_fn=plt.plot,
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
        plot_fn(hourly_average.index, hourly_average.values,
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
