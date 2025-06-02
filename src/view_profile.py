import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default azimuth grid for 360° resolution
AZIMUTH_360 = np.arange(360)


class ViewProfile:
    """
    Represents a facade's visibility or shading profile across azimuth angles.
    """

    azimuth: np.ndarray
    elevation: np.ndarray

    def __init__(self, azimuth: np.ndarray, elevation: np.ndarray):
        """
        Initialize a profile over 360 degrees by interpolating elevation values.

        Args:
            azimuth (array-like): Azimuth angles in degrees.
            elevation (array-like): Elevation angles in degrees.
        """
        self.azimuth = AZIMUTH_360
        self.elevation = np.interp(AZIMUTH_360, azimuth, elevation)

    def mirror(self) -> "ViewProfile":
        return ViewProfile(AZIMUTH_360, np.flip(self.elevation))

    def rotate(self, degrees: float) -> "ViewProfile":
        """
        Rotate the azimuth profile by a given number of degrees clockwise.

        Args:
            degrees (float): Degrees to rotate.

        Returns:
            ViewProfile: Rotated profile.
        """
        rotated_azimuth = (self.azimuth + degrees) % 360
        sort_idx = np.argsort(rotated_azimuth)

        return ViewProfile(
            azimuth=rotated_azimuth[sort_idx],
            elevation=self.elevation[sort_idx]
        )

    def combine(self, profile2: "ViewProfile") -> "ViewProfile":
        """
        Combine two profiles by taking the maximum elevation at each azimuth.

        Args:
            profile2 (ViewProfile): Another shading profile.

        Returns:
            ViewProfile: Combined profile.
        """
        return ViewProfile(
            AZIMUTH_360,
            np.max([self.elevation, profile2.elevation], axis=0)
        )

    def apply(
        self,
        solar_zenith: pd.Series,
        solar_azimuth: pd.Series,
        dni: pd.Series,
        ghi: pd.Series,
        dhi: pd.Series
    ) -> tuple[pd.Series, np.ndarray, np.ndarray, pd.Series]:
        """
        Apply the shading profile to irradiance data.

        Args:
            solar_zenith (Series): Solar zenith angles in degrees.
            solar_azimuth (Series): Solar azimuth angles in degrees.
            dni (Series): Direct Normal Irradiance (W/m²).
            ghi (Series): Global Horizontal Irradiance (W/m²).
            dhi (Series): Diffuse Horizontal Irradiance (W/m²).

        Returns:
            Tuple of:
                blocked_elevation_at_dt (Series),
                masked_dni (np.ndarray),
                adjusted_ghi (np.ndarray),
                unchanged_dhi (Series)
        """
        blocked_elevation_at_dt = pd.Series(
            np.interp(solar_azimuth, self.azimuth, self.elevation),
            index=solar_azimuth.index
        )

        # Mask DNI where the sun is blocked by the profile
        masked_dni = np.where((90 - solar_zenith) >
                              blocked_elevation_at_dt, dni, 0)

        # Adjust GHI to equal DHI when DNI is zero
        adjusted_ghi = np.where(masked_dni == 0, dhi, ghi)

        return blocked_elevation_at_dt, masked_dni, adjusted_ghi, dhi

    def plot_radial(self, label: str = 'Facade blocked elevation', title: str = 'Radial Facade Profile', ax=None, show=True, log=True, log_base=10):
        """
            Plot the view as a polar plot with azimuth on theta (in radians) and
            log-scaled elevation on the radial axis. North is at the top (0° azimuth).

            Parameters
            ----------
            label : str
                Legend label for the view.
            title : str
                Plot title.
            ax : matplotlib.axes._subplots.PolarAxesSubplot, optional
                Axes object to plot into.
            show : bool
                Whether to call plt.show().
            log : bool
                Whether to plot elevation in log scale.
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location("N")  # 0 degrees at top (North)
            ax.set_theta_direction(-1)       # Clockwise direction

        theta = np.radians(self.azimuth)
        radius = self.elevation

        radial_ticks = [1, 5, 30, 90]
        radial_labels = [f"{r}°" for r in radial_ticks]

        # Avoid log(0); add a small offset and clip to min
        if log:
            radius = np.clip(radius, 1e-3, 90)
            radial_ticks.insert(0, 1e-3)
            radial_labels.insert(0, f"0°")

            radius = np.emath.logn(log_base, radius)
            radial_ticks = np.emath.logn(log_base, radial_ticks)

        ax.plot(theta, radius, label=label)
        theta_full = np.append(theta, theta[0])
        radius_full = np.append(radius, radius[0])

        # Fill under the curve
        ax.fill(theta_full, radius_full, alpha=0.3)

        ax.set_title(title, va='bottom')
        ax.legend(loc='upper right')

        # Formatting plot
        ax.set_rgrids(radial_ticks, labels=radial_labels)
        ax.set_rlabel_position(135)  # Move radial labels away from North
        ax.set_theta_offset(np.pi / 2)  # Optional: keeps 0° pointing up

        if show:
            plt.show()

        return ax

    def plot_unwrapped(
        self,
        figure=None,
        label: str = 'Facade blocked elevation',
        show: bool = True,
        title: str = 'Facade View Blockage'
    ):
        """
        Plot the facade profile in azimuth vs elevation format.
        """
        if figure is None:
            show = True
            figure = plt.figure(figsize=(10, 4))

        plt.plot(self.azimuth, self.elevation, marker='o', label=label)
        plt.xlabel('Azimuth')
        plt.ylabel('Elevation (degrees)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        if show:
            plt.show()


# Common static profiles
PANORAMIC = ViewProfile([0, 360], [0, 0])  # Full sky view
BACKED = ViewProfile(
    [0, 90, 91, 270, 271, 360], [0, 0, 90, 90, 0, 0]
)  # Unobstructed view infront, wall behind with obstructed view
NW_BALCONY = ViewProfile(
    [0, 12.36, 12.361, 90, 90.1, 270, 270.1], [0, 0, 49.5, 79.639, 90, 90, 0]
)  # NW corner balcony facing north with the north east facing into a facade of the building
NE_BALCONY = NW_BALCONY.mirror()


def profile_from_str(str):
    match str:
        case 'BACKED':
            return BACKED
        case 'PANORAMIC':
            return PANORAMIC
        case 'NE_BALCONY':
            return NE_BALCONY
        case 'NW_BALCONY':
            return NW_BALCONY
