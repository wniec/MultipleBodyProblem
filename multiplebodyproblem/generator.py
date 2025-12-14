import numpy as np
from astroquery.gaia import Gaia


def _estimate_mass(mag_abs):
    # sun magnitude
    sun_mag = 4.67

    delta_mag = sun_mag - mag_abs
    luminosity = 10 ** (delta_mag / 2.5)

    # Relationship Mass-Luminosity: L ~ M^3.5
    try:
        mass = luminosity ** (1 / 3.5)
    except:
        mass = 0.1

    # truncation for extreme cases
    if mass > 150:
        mass = 150
    if mass < 0.08:
        mass = 0.08
    return round(mass, 4)


def generate_nearby_stars(n: int):

    query = f"""
    SELECT TOP {n}
        source_id,
        ra, dec, parallax, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE parallax > 0 
    AND parallax_over_error > 10
    ORDER BY parallax DESC
    """

    job = Gaia.launch_job_async(query)
    stars = job.get_results().to_pandas()

    # Light years distance
    stars["distance_ly"] = (1000 / stars["parallax"]) * 3.26156

    # Conversion from spherical to cartesian coordinates
    ra_rad = np.radians(stars["ra"])
    dec_rad = np.radians(stars["dec"])
    dist = stars["distance_ly"]

    stars["x_ly"] = dist * np.cos(dec_rad) * np.cos(ra_rad)
    stars["y_ly"] = dist * np.cos(dec_rad) * np.sin(ra_rad)
    stars["z_ly"] = dist * np.sin(dec_rad)

    stars["abs_mag_g"] = stars["phot_g_mean_mag"] + 5 * np.log10(
        stars["parallax"] / 100
    )

    stars["mass_solar"] = stars["abs_mag_g"].apply(_estimate_mass)

    output_df = stars[["x_ly", "y_ly", "z_ly", "mass_solar"]]
    output_df.columns = ["X", "Y", "Z", "solar_masses"]
    return output_df
