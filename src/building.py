from src.site import Site
from src.surface import Surface, SurfaceType
from src.view_profile import profile_from_str

import matplotlib.pyplot as plt
import pandas as pd


class Azimuth(float):
    def rotate(self, degrees):
        return (self + degrees) % 360


class Building:
    name: str
    site: Site
    surfaces: dict[str, Surface]
    levels: int
    azimuth: Azimuth

    def __init__(self, name: str, azimuth: float, site: Site, levels: int):
        self.name = name
        self.azimuth = Azimuth(azimuth)
        self.site = site
        self.levels = levels
        self.surfaces = {}

    def add_surface(self, name, type: SurfaceType, azimuth: float, view_profile, tilt, area_per_level, efficiency):
        self.surfaces[name] = Surface(
            name=name,
            type=type,
            azimuth=self.azimuth.rotate(azimuth),
            site=self.site,
            view_profile=view_profile,
            tilt=tilt,
            area_per_level=area_per_level,
            efficiency=efficiency,
        )

    def load_surfaces_from_csv(self, file_path):
        df = pd.read_csv(file_path)

        for idx, row in df.iterrows():
            self.add_surface(
                row.Name,
                SurfaceType[row.Type],
                row.Azimuth,
                profile_from_str(row.Profile),
                row.Tilt,
                row['Capture Area (m2)'],
                row['Efficiency']
            )

    def calculate_irradiance(self):
        results = {}
        position, irradiance = self.site.get_solar()
        for name, surface in self.surfaces.items():
            results[name] = surface.solve_irradiance(position, irradiance)

        self.results = results
        self.global_irradiance = pd.concat({
            k: res.irradiance_df['poa_global'] for k, res in results.items()}, axis=1)

        # Calculate the captured irradiance using surface area and efficiency
        to_concat = {}
        for name, surface in self.surfaces.items():
            captured = self.results[name].irradiance_df['poa_global'] * \
                surface.area_per_level * surface.efficiency

            to_concat[(name, surface.type)] = captured

        captured = pd.concat(to_concat, axis=1) * self.levels
        # Step 1: Reset column MultiIndex into DataFrame
        captured.columns.names = ['SurfaceName', 'SurfaceType']

        # Step 2: Convert wide format to long format
        df_long = captured.stack(level=[0, 1]).reset_index()
        df_long.columns = ['Datetime', 'SurfaceName',
                           'SurfaceType', 'Generation (kWh)']
        df_long['Generation (kWh)'] = df_long['Generation (kWh)'] / 1000

        self.generated_power = df_long
        self.captured_irradiance = captured

    def plot_irradiance_profile(self):
        avg = self.global_irradiance.groupby(
            self.global_irradiance.index.hour).mean()

        avg.plot(
            title='Building Available Irradiance (W/m^2)',
            stacked=True,
            xlabel='Hour',
            ylabel='Irradiance (W/m^2)'
        )

    def plot_available_irradiance(self, **kwargs):
        avg = self.captured_irradiance.groupby(
            self.captured_irradiance.index.hour).mean()

        avg.plot.area(
            title='Captured Irradiance (W)',
            stacked=True,
            xlabel='Hour',
            ylabel='Irradiance (W)',
            figsize=(16, 4),
            **kwargs,
        )

    def plot_captured_total(self, **kwargs):
        sm = self.captured_irradiance.sum().plot.bar(
            title='Yearly Captured Irradiance (Wh)',
            xlabel='Surface',
            ylabel='Irradiance (W)',
            rot=90,
            figsize=(16, 4),
            **kwargs,
        )

    def plot_yearly_total(self):
        figure = plt.figure(figsize=(10, 4))
        surfaces = self.results.keys()
        totals = [res.total() for _, res in self.results.items()]

        plt.bar(surfaces, totals)

    def plot_irradiance_profile(self):
        figure = plt.figure(figsize=(10, 4))

        for name, result in self.results.items():
            result.plot_daily_profile(
                figure=figure,
                label=name,
                plot_fn=plt.stackplot
            )

    def plot_level_irradiance(self):
        return
