from src.site import Site
from src.building import Building
from src.surface import SurfaceType
from src.view_profile import BACKED


def load_torbreck():
    latitude = -27.48637226
    longitude = 153.01796166
    tz = 'Australia/Brisbane'

    site = Site(longitude=longitude, latitude=latitude, tz=tz,
                tmy_path='/workspace/src/torbreck/tmy_2005_2023.epw')

    torbreck = Building('Torbreck', 0.0, site, 13)
    torbreck.load_surfaces_from_csv(
        '/workspace/src/torbreck/surface_definitions.csv')

    position, irradiance = site.get_solar()
    torbreck.calculate_irradiance()

    return torbreck
