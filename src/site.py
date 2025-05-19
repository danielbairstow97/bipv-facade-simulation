import pandas as pd
from pvlib.location import Location
from pvlib.iotools import get_pvgis_horizon

from src.view_profile import ViewProfile


class Site:
    """
    Represents a geographic site used for solar resource modeling.
    """

    latitude: float
    longitude: float
    location: Location
    horizon_profile: ViewProfile

    def __init__(self, longitude: float, latitude: float, tz: str):
        """
        Initialize the site with location and terrain horizon data.

        Args:
            longitude (float): Longitude of the site.
            latitude (float): Latitude of the site.
            tz (str): Timezone string (e.g., 'Australia/Brisbane').
        """
        self.latitude = latitude
        self.longitude = longitude

        self.location = Location(latitude, longitude, tz=tz)

        # Retrieve 360° elevation horizon profile from PVGIS
        horizon_data, _ = get_pvgis_horizon(latitude, longitude)
        self.horizon_profile = ViewProfile(
            azimuth=horizon_data.index, elevation=horizon_data
        )

    def generate_clearsky(
        self,
        dt_start: str = '2024-01-01',
        dt_end: str = '2024-12-31 23:00:00',
        tz: str = 'Australia/Brisbane',
        freq: str = '1h',
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate solar position and clear-sky irradiance data.

        Args:
            dt_start (str): Start of the datetime range.
            dt_end (str): End of the datetime range.
            tz (str): Timezone.
            freq (str): Frequency of timestamps (e.g., '1h').

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Solar position and irradiance.
        """
        dt_range = pd.date_range(dt_start, dt_end, freq=freq, tz=tz)

        solar_position = self.location.get_solarposition(dt_range)
        irradiance = self.location.get_clearsky(
            dt_range)  # Clear-sky model

        return solar_position, irradiance

    def generate_from_weather(self, epw_data, epw_metadata, dt_start: str = '2024-01-01', dt_end: str = '2024-12-31 23:00:00', freq: str = '1h'):
        return epw_data.between_datetime(dt_start, dt_end, inclusive='both', axis=0)
