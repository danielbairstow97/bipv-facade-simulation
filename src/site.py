import pandas as pd
from pvlib.location import Location
from pvlib.iotools import get_pvgis_horizon, get_pvgis_tmy
from pvlib.iotools import read_epw

from src.view_profile import ViewProfile


TMY_COLUMNS = ['temp_air', 'relative_humidity', 'ghi', 'dni',
               'dhi', 'IR(h)', 'wind_speed', 'wind_direction', 'pressure']


class Site:
    """
    Represents a geographic site used for solar resource modeling.
    """

    latitude: float
    longitude: float
    altitude: float
    location: Location
    horizon_profile: ViewProfile
    tmy: pd.DataFrame

    def __init__(self, longitude: float, latitude: float, tz: str, tmy_path=None):
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

        if tmy_path is None:
            # Get Typical Meteorological Year for location
            tmy, _, inputs, medatada = get_pvgis_tmy(
                longitude=longitude, latitude=latitude, usehorizon=False)

            # # Convert to consistent time format
            tmy.index = tmy.index.tz_convert(tz)
            tmy.index = tmy.index + pd.DateOffset(year=2013)
            tmy = tmy.sort_index()
            tmy.index.name = 'Time'
            self.tmy = tmy[['ghi', 'dni', 'dhi']]
            self.altitude = inputs['location']['elevation']
        else:
            tmy, metadata = read_epw(tmy_path)
            tmy.index = tmy.index + pd.DateOffset(year=2013)
            tmy = tmy.sort_index()
            tmy.index.name = 'Time'
            self.tmy = tmy[['ghi', 'dni', 'dhi']]

            self.altitude = metadata['altitude']
            self.latitude = metadata['latitude']
            self.longitude = metadata['longitude']

        # Retrieve 360° elevation horizon profile from PVGIS
        horizon_data, _ = get_pvgis_horizon(
            latitude, longitude)
        self.horizon_profile = ViewProfile(
            azimuth=horizon_data.index, elevation=horizon_data
        )

    def get_solar(self):
        return self.location.get_solarposition(self.tmy.index), self.tmy

    def plot_daily_irradiance(self):
        avg = self.tmy.groupby(self.tmy.index.hour).mean()
        avg.plot.area(start=0, end=24)

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
