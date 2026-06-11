"""
PV-DIAL — Dynamic Pipeline System
===================================
Users define their own pipelines by choosing models.
The system builds and runs them automatically.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy  as np
import pandas as pd
import pvlib
from pvlib import irradiance, atmosphere, temperature, pvsystem, location

# ── File path ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMY_FILE = os.path.join(BASE_DIR, "tmy_6.944_79.856_2005_2020.csv")

# ── Site configuration ─────────────────────────────────────────────────────
SITE = {
    "latitude":        6.939,
    "longitude":       79.854,
    "altitude":        11.0,
    "surface_tilt":    6.0,
    "surface_azimuth": 180.0,
    "albedo":          0.20,
}

# ── Component selection ────────────────────────────────────────────────────
MODULE_NAME   = "Canadian_Solar_Inc__CS6K_300MS"
INVERTER_NAME = "SMA_America__SB5000TL_US_22__240V_"
N_MODULES     = 20   # 10 per string x 2 strings


# =============================================================================
# PART 1 — ALL MODEL FUNCTIONS
# =============================================================================

def _erbs(ghi, solpos, times, **kw):
    r = irradiance.erbs(ghi=ghi, zenith=solpos["apparent_zenith"],
                        datetime_or_doy=times)
    return {"dni": r["dni"].clip(0), "dhi": r["dhi"].clip(0)}

def _disc(ghi, solpos, times, pressure=None, **kw):
    r = irradiance.disc(ghi=ghi, solar_zenith=solpos["apparent_zenith"],
                        datetime_or_doy=times, pressure=pressure)
    c = irradiance.complete_irradiance(solar_zenith=solpos["apparent_zenith"],
                                       ghi=ghi, dni=r["dni"], dhi=None)
    return {"dni": r["dni"].clip(0), "dhi": c["dhi"].clip(0)}

def _boland(ghi, solpos, times, **kw):
    r = irradiance.boland(ghi=ghi, solar_zenith=solpos["apparent_zenith"],
                          datetime_or_doy=times)
    return {"dni": r["dni"].clip(0), "dhi": r["dhi"].clip(0)}

def _haydavies(s1, solpos, ghi, **kw):
    dni_extra = irradiance.get_extra_radiation(solpos.index)
    r = irradiance.get_total_irradiance(
        surface_tilt=SITE["surface_tilt"], surface_azimuth=SITE["surface_azimuth"],
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=s1["dni"], ghi=ghi, dhi=s1["dhi"],
        dni_extra=dni_extra, albedo=SITE["albedo"], model="haydavies")
    return r.to_dict(orient="series")

def _perez(s1, solpos, ghi, **kw):
    dni_extra = irradiance.get_extra_radiation(solpos.index)
    airmass   = atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    r = irradiance.get_total_irradiance(
        surface_tilt=SITE["surface_tilt"], surface_azimuth=SITE["surface_azimuth"],
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=s1["dni"], ghi=ghi, dhi=s1["dhi"],
        dni_extra=dni_extra, airmass=airmass, albedo=SITE["albedo"],
        model="perez", model_perez="allsitescomposite1990")
    return r.to_dict(orient="series")

def _isotropic(s1, solpos, ghi, **kw):
    r = irradiance.get_total_irradiance(
        surface_tilt=SITE["surface_tilt"], surface_azimuth=SITE["surface_azimuth"],
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=s1["dni"], ghi=ghi, dhi=s1["dhi"],
        albedo=SITE["albedo"], model="isotropic")
    return r.to_dict(orient="series")

def _sapm_temp(s2, temp_air, wind_speed, **kw):
    p = temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    return temperature.sapm_cell(
        poa_global=s2["poa_global"], temp_air=temp_air,
        wind_speed=wind_speed, a=p["a"], b=p["b"], deltaT=p["deltaT"])

def _faiman(s2, temp_air, wind_speed, **kw):
    return temperature.faiman(
        poa_global=s2["poa_global"], temp_air=temp_air,
        wind_speed=wind_speed, u0=25.0, u1=6.84)

def _noct(s2, temp_air, wind_speed, module_params=None, **kw):
    t_noct = module_params["T_NOCT"] if module_params is not None else 45.3
    return temperature.noct_sam(
        poa_global=s2["poa_global"], temp_air=temp_air,
        wind_speed=wind_speed, noct=t_noct, module_efficiency=0.185)

def _pvwatts_dc(s2, t_cell, module_params, **kw):
    return pvsystem.pvwatts_dc(
        g_poa_effective=s2["poa_global"], temp_cell=t_cell,
        pdc0=module_params["STC"] * N_MODULES,
        gamma_pdc=module_params["gamma_r"] / 100,
        temp_ref=25.0).clip(lower=0)

def _singlediode(s2, t_cell, module_params, **kw):
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_cec(
        effective_irradiance=s2["poa_global"], temp_cell=t_cell,
        alpha_sc=module_params["alpha_sc"], a_ref=module_params["a_ref"],
        I_L_ref=module_params["I_L_ref"], I_o_ref=module_params["I_o_ref"],
        R_sh_ref=module_params["R_sh_ref"], R_s=module_params["R_s"],
        Adjust=module_params["Adjust"])
    sd = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth, method="lambertw")
    return (sd["p_mp"] * N_MODULES).clip(lower=0)

def _sapm_dc(s2, t_cell, **kw):
    san_mod = pvsystem.retrieve_sam("SandiaMod")["Silevo_Triex_U300_Black__2014_"]
    out = pvsystem.sapm(effective_irradiance=s2["poa_global"],
                        temp_cell=t_cell, module=san_mod)
    return (out["p_mp"] * N_MODULES).clip(lower=0)

def _sandia_inv(pdc, inverter_params, **kw):
    return pvsystem.inverter.sandia(
        v_dc=inverter_params["Vdco"], p_dc=pdc,
        inverter=inverter_params).clip(lower=0)

def _pvwatts_inv(pdc, inverter_params, **kw):
    return pvsystem.inverter.pvwatts(
        pdc=pdc, pdc0=inverter_params["Pdco"],
        eta_inv_nom=0.96, eta_inv_ref=0.9637).clip(lower=0)


# =============================================================================
# PART 2 — MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {
    "S1": {
        "erbs":        _erbs,
        "disc":        _disc,
        "boland":      _boland,
    },
    "S2": {
        "haydavies":   _haydavies,
        "perez":       _perez,
        "isotropic":   _isotropic,
    },
    "S3": {
        "sapm":        _sapm_temp,
        "faiman":      _faiman,
        "noct":        _noct,
    },
    "S4": {
        "pvwatts":     _pvwatts_dc,
        "singlediode": _singlediode,
        "sapm":        _sapm_dc,
    },
    "S5": {
        "sandia":      _sandia_inv,
        "pvwatts":     _pvwatts_inv,
    },
}


def list_available_models():
    print("\n" + "=" * 50)
    print("  AVAILABLE MODELS")
    print("=" * 50)
    names = {
        "S1": "Stage 1 - Irradiance Decomposition",
        "S2": "Stage 2 - POA Transposition",
        "S3": "Stage 3 - Temperature Modeling",
        "S4": "Stage 4 - DC Power Modeling",
        "S5": "Stage 5 - AC Conversion",
    }
    for stage, models in MODEL_REGISTRY.items():
        print(f"\n  {names[stage]}")
        for name in models:
            print(f"      '{name}'")
    print()


# =============================================================================
# PART 3 — PIPELINE CLASS
# =============================================================================

class Pipeline:

    def __init__(self, name, config):
        self.name   = name
        self.config = config
        self._validate()

    def _validate(self):
        for stage in ["S1", "S2", "S3", "S4", "S5"]:
            if stage not in self.config:
                raise ValueError(f"Pipeline '{self.name}': missing stage {stage}")
            model = self.config[stage]
            if model not in MODEL_REGISTRY[stage]:
                available = list(MODEL_REGISTRY[stage].keys())
                raise ValueError(
                    f"Pipeline '{self.name}': unknown model '{model}' "
                    f"for {stage}. Available: {available}")

    def run(self, df, solpos, module_params, inverter_params):
        print(f"  Running: {self.name}  "
              f"[{self.config['S1']} -> {self.config['S2']} -> "
              f"{self.config['S3']} -> {self.config['S4']} -> "
              f"{self.config['S5']}]")

        s1 = MODEL_REGISTRY["S1"][self.config["S1"]](
            ghi=df["ghi"], solpos=solpos, times=df.index,
            pressure=df.get("pressure"))

        s2 = MODEL_REGISTRY["S2"][self.config["S2"]](
            s1=s1, solpos=solpos, ghi=df["ghi"])

        s3 = MODEL_REGISTRY["S3"][self.config["S3"]](
            s2=s2, temp_air=df["temp_air"], wind_speed=df["wind_speed"],
            module_params=module_params)

        s4 = MODEL_REGISTRY["S4"][self.config["S4"]](
            s2=s2, t_cell=s3, module_params=module_params)

        s5 = MODEL_REGISTRY["S5"][self.config["S5"]](
            pdc=s4, inverter_params=inverter_params)

        return {
            "name":     self.name,
            "config":   self.config,
            "S1_dni":   s1["dni"],
            "S1_dhi":   s1["dhi"],
            "S2_poa":   s2["poa_global"],
            "S3_tcell": s3,
            "S4_pdc":   s4,
            "S5_pac":   s5,
        }


# =============================================================================
# PART 4 — PIPELINE FACTORY
# =============================================================================

class PipelineFactory:

    @staticmethod
    def build(config):
        return Pipeline(name=config.get("name", "Unnamed"), config=config)

    @staticmethod
    def build_many(configs):
        return [PipelineFactory.build(c) for c in configs]


# =============================================================================
# PART 5 — DATA LOADING
# =============================================================================

def load_tmy(filepath):
    print("-" * 60)
    print("Loading PVGIS TMY data")
    print("-" * 60)
    with open(filepath, "r") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("time(UTC)"))
    df = pd.read_csv(filepath, skiprows=header_idx, nrows=8760)
    df["datetime"] = pd.to_datetime(df["time(UTC)"], format="%Y%m%d:%H%M", utc=True)
    df = df.set_index("datetime").drop(columns=["time(UTC)"])
    df = df.rename(columns={
        "G(h)": "ghi", "Gb(n)": "dni", "Gd(h)": "dhi",
        "T2m": "temp_air", "WS10m": "wind_speed",
        "SP": "pressure", "RH": "relative_humidity",
    })
    df[["ghi", "dni", "dhi"]] = df[["ghi", "dni", "dhi"]].clip(lower=0)
    assert len(df) == 8760
    print(f"  Rows loaded : {len(df)}")
    print(f"  Annual GHI  : {df['ghi'].sum()/1000:.2f} kWh/m2")
    print(f"  Mean Tamb   : {df['temp_air'].mean():.2f} C")
    print()
    return df


def compute_solar_position(times):
    loc = location.Location(
        latitude=SITE["latitude"], longitude=SITE["longitude"],
        altitude=SITE["altitude"], tz="UTC")
    return loc.get_solarposition(times)


def load_components():
    print("-" * 60)
    print("Loading component databases")
    print("-" * 60)
    mod = pvsystem.retrieve_sam("CECMod")[MODULE_NAME]
    inv = pvsystem.retrieve_sam("CECInverter")[INVERTER_NAME]
    print(f"  Module   : {MODULE_NAME}  ({mod['STC']:.0f} W)")
    print(f"  Inverter : {INVERTER_NAME}  ({inv['Paco']:.0f} W AC)")
    print()
    return mod, inv


# =============================================================================
# PART 6 — RESULTS
# =============================================================================

def print_results(results):
    print()
    print("=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n  {'Pipeline':<30} {'Annual AC (kWh)':>16}   Models")
    print("  " + "-" * 66)
    for r in results:
        kwh   = r["S5_pac"].sum() / 1000
        combo = (f"{r['config']['S1']} | {r['config']['S2']} | "
                 f"{r['config']['S3']} | {r['config']['S4']} | "
                 f"{r['config']['S5']}")
        print(f"  {r['name']:<30} {kwh:>16.2f}   {combo}")

    print(f"\n  {'Pair':<40} {'Diff (kWh)':>12} {'Diff (%)':>10}")
    print("  " + "-" * 64)
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            kwhA = results[i]["S5_pac"].sum() / 1000
            kwhB = results[j]["S5_pac"].sum() / 1000
            diff = abs(kwhA - kwhB)
            pct  = diff / kwhA * 100
            pair = f"{results[i]['name']}  vs  {results[j]['name']}"
            print(f"  {pair:<40} {diff:>12.2f} {pct:>10.2f}%")

    print(f"\n  Stage-wise means (daytime hours only)")
    print(f"  {'Stage':<12}", end="")
    for r in results:
        print(f"  {r['name']:>18}", end="")
    print()
    print("  " + "-" * (12 + 20 * len(results)))
    for label, key, unit in [
        ("S1 DNI",   "S1_dni",   "W/m2"),
        ("S2 POA",   "S2_poa",   "W/m2"),
        ("S3 Tcell", "S3_tcell", "C"),
        ("S4 PDC",   "S4_pdc",   "W"),
        ("S5 PAC",   "S5_pac",   "W"),
    ]:
        print(f"  {label:<12}", end="")
        for r in results:
            v   = r[key]
            val = v[v > 0].mean() if (v > 0).sum() > 0 else v.mean()
            print(f"  {val:>18.2f}", end="")
        print(f"  {unit}")
    print()


def save_results(results):
    frames = {}
    for r in results:
        n = r["name"].replace(" ", "_")
        for key in ["S1_dni", "S1_dhi", "S2_poa", "S3_tcell", "S4_pdc", "S5_pac"]:
            frames[f"{n}_{key}"] = r[key]
    out  = pd.DataFrame(frames)
    path = os.path.join(BASE_DIR, "tmy_6.944_79.856_2005_2020.csv")
    out.to_csv(path)
    print(f"  Saved -> {path}")
    return out


# =============================================================================
# PART 7 — USER SECTION
# This is the only part you edit.
# =============================================================================

def run():
    print()
    print("=" * 60)
    print("  PV-DIAL — Dynamic Pipeline System")
    print(f"  pvlib {pvlib.__version__}  |  Colombo, Sri Lanka")
    print("=" * 60)

    list_available_models()

    # ------------------------------------------------------------------
    # EDIT THIS SECTION — define your pipelines here
    # Each dict is one pipeline.
    # Change model names using the list printed above.
    # Add more dicts to create more pipelines.
    # ------------------------------------------------------------------

    user_pipeline_configs = [

        {
            "name": "Pipeline A",
            "S1": "erbs",
            "S2": "haydavies",
            "S3": "sapm",
            "S4": "pvwatts",
            "S5": "sandia",
        },

        {
            "name": "Pipeline B",
            "S1": "disc",
            "S2": "perez",
            "S3": "faiman",
            "S4": "singlediode",
            "S5": "sandia",
        },

        {
            "name": "Pipeline C",
            "S1": "boland",
            "S2": "isotropic",
            "S3": "noct",
            "S4": "sapm",
            "S5": "pvwatts",
        },

    ]

    # ------------------------------------------------------------------
    # DO NOT EDIT BELOW THIS LINE
    # ------------------------------------------------------------------

    df                              = load_tmy(TMY_FILE)
    module_params, inverter_params  = load_components()

    print("-" * 60)
    print("Computing solar position")
    print("-" * 60)
    solpos = compute_solar_position(df.index)
    print(f"  Computed for {len(solpos)} timesteps\n")

    print("-" * 60)
    print("Building pipelines")
    print("-" * 60)
    pipelines = PipelineFactory.build_many(user_pipeline_configs)
    print(f"  {len(pipelines)} pipelines built\n")

    print("-" * 60)
    print("Executing pipelines")
    print("-" * 60)
    all_results = []
    for pipeline in pipelines:
        result = pipeline.run(df, solpos, module_params, inverter_params)
        all_results.append(result)

    print_results(all_results)

    print("-" * 60)
    print("Saving outputs")
    print("-" * 60)
    save_results(all_results)

    print()
    print("Done.")
    print()

    return all_results


# =============================================================================
# THIS MUST BE THE LAST TWO LINES OF THE FILE — DO NOT CHANGE
# =============================================================================

if __name__ == "__main__":
    run()