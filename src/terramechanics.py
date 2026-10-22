"""
Module 1: Rotary-percussive terramechanics (Moon vs. Mars), no tuning factors.

Models (all coefficients come from config/system_params.yaml with literature sources;
all design quantities come from src/design.py):

Rotary cutting - Detournay & Defourny (1992) drag-bit model, bit radius a, depth of cut
per revolution delta, intrinsic specific energy eps, wearflat contact stress sigma and
total wearflat length l (rounded WC edges):
    W_f = sigma l a                       (WOB carried by the wearflats = rotary threshold)
    delta = (W - W_f) / (zeta a eps)       for W > W_f, else 0 (bit only rubs)
    T_bit = a^2 eps delta / 2 + mu a min(W, W_f) / 2
    delta <= cutter protrusion (crown bottoms out; excess WOB is carried in bearing)
    ROP_rot = delta N / 60

Percussion - energy balance with percussive specific energy SE_p = c_p UCS:
    ROP_perc = eta_t E f / (A_hole SE_p)     (requires W >= W_min to keep the bit seated)

Cuttings removal - retained cuttings friction (centrifugal + gravity) in the sampling
flutes, plus Janssen plug resistance once the bite exceeds the flute capacity z_crit:
    L_plug = (z_bite - z_crit) A_hole B / A_transport
    sigma_plug = sigma_0 exp(4 mu_w K L_plug / D_h),   K = 1 - sin(phi)  (Jaky),
    sigma_0 = 2 c_r tan(45 + phi/2)  (unconfined strength of the remoulded cuttings)
    T_plug = sigma_plug A_transport r_m tan(alpha + phi_f)     (screw torque)

Mechanical specific energy - Teale (1965):
    MSE = W/A + (2 pi N/60 T + eta_t E f) / (A ROP)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.design import (
    EXPORTS_DATA_DIR,
    PROJECT_ROOT,
    derive_all,
    layer_ucs_pa,
    mohr_coulomb_ucs_pa,
    load_system_params,
)

__all__ = ["PROJECT_ROOT", "EXPORTS_DATA_DIR", "load_system_params", "Physics"]


class Physics:
    """Bundles config + derived design for one environment."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, design: Optional[Dict[str, Any]] = None,
                 env_id: str = "moon") -> None:
        self.p = params or load_system_params()
        self.d = design or derive_all(self.p)
        self.env_id = env_id
        self.env = self.p["environments"][env_id]
        a = self.p["auger_geometry"]
        self.cm = self.p["cutting_model"]
        self.r_bit = float(a["gauge_diameter_m"]) / 2
        self.r_o = float(a["outer_diameter_m"]) / 2
        self.a_hole = math.pi * self.r_bit ** 2
        self.g = float(self.env["gravity_m_s2"])
        self.mu_w = float(a["wall_friction_coeff_soil"])
        self.mu_f = float(a["wall_friction_coeff_auger"])
        aug = self.d["auger"]
        self.z_crit = float(aug["choking_onset_m"])
        self.wearflat = float(aug["wearflat_mm"]) / 1e3
        self.alpha_tr = math.radians(float(aug["transport_helix_deg"]))
        self.p_tr = float(aug["transport_pitch_mm"]) / 1e3
        d_tube = float(a["tube_outer_diameter_m"])
        t_tr = float(a["transport_flight_thickness_m"])
        self.a_transport = (math.pi / 4 * ((2 * self.r_o) ** 2 - d_tube ** 2) * (1 - t_tr / self.p_tr)
                            + math.pi / 4 * ((2 * self.r_bit) ** 2 - (2 * self.r_o) ** 2))
        h_ch = self.r_o - d_tube / 2
        w_ch = self.p_tr - t_tr
        self.d_h = 4 * h_ch * w_ch / (2 * (h_ch + w_ch))
        self.r_m_tr = 0.5 * (self.r_o + d_tube / 2)
        self.l_samp = float(aug["sampling_length_mm"]) / 1e3
        self.root_r = float(aug["root_od_mm"]) / 2e3
        hm = self.d["hammer"]
        self.e_imp = float(hm["impact_energy_j"])
        self.f_blow = float(hm["blow_frequency_hz"])
        self.w_min_perc = float(hm["min_wob_n"])
        self.p_perc_el = float(hm["elec_power_w"])
        rot = self.p["actuators"]["rotary"]
        self.eta_rot = float(rot["motor_efficiency"]) * float(rot["gear_efficiency"])
        self.p_nl = float(rot["no_load_power_w"])

    # --------------------------------------------------------------------------
    def layer(self, z: float) -> Dict[str, Any]:
        zc = min(max(z, 0.0), 1.0)
        for L in self.env["stratigraphy"]:
            if L["z_start_m"] <= zc < L["z_end_m"]:
                return L
        return self.env["stratigraphy"][-1]

    def cutting(self, z: float, wob: float, rpm: float, percussion: bool) -> Dict[str, float]:
        L = self.layer(z)
        ucs = layer_ucs_pa(L)
        eps = float(self.cm["intrinsic_energy_to_ucs"]) * ucs
        sigma = float(self.cm["wearflat_contact_stress_to_ucs"]) * ucs
        zeta = float(self.cm["zeta_force_ratio"])
        mu = float(self.cm["wearflat_friction_coeff"])
        a = self.r_bit
        w_f = sigma * self.wearflat * a
        delta = max(0.0, wob - w_f) / (zeta * a * eps)
        delta_max = float(self.p["auger_geometry"]["cutter_protrusion_m"])
        w_bearing = 0.0
        if delta > delta_max:  # crown face bottoms out: excess WOB carried in bearing on the crown
            w_bearing = (delta - delta_max) * zeta * a * eps
            delta = delta_max
        rop_rot = delta * rpm / 60.0
        t_bit = 0.5 * a * a * eps * delta + 0.5 * mu * a * (min(wob, w_f) + w_bearing)
        rop_perc = 0.0
        perc_ok = percussion and wob >= self.w_min_perc
        if perc_ok:
            se_p = float(self.cm["percussive_se_to_ucs"]) * ucs
            rop_perc = float(self.cm["impact_transfer_efficiency"]) * self.e_imp * self.f_blow / (self.a_hole * se_p)
        return {
            "layer": L["name"], "ucs_pa": ucs, "w_threshold_n": w_f, "delta_m": delta,
            "rop_rot_m_s": rop_rot, "rop_perc_m_s": rop_perc,
            "rop_m_s": min(rop_rot + rop_perc, float(self.p["actuators"]["z_axis_feed"]["max_drill_feed_m_s"])),
            "torque_bit_nm": t_bit, "percussion_active": perc_ok,
            "p_face_w": t_bit * rpm * 2 * math.pi / 60 + (float(self.cm["impact_transfer_efficiency"]) * self.e_imp * self.f_blow if perc_ok else 0.0),
        }

    def removal_torque(self, z: float, bite_pen: float, rpm: float) -> Dict[str, float]:
        """Torque to rotate the cuttings-laden auger: retained-cuttings friction + Janssen plug."""
        L = self.layer(z)
        bulk = float(self.cm["cuttings_bulking_factor"])
        rho_b = float(L["density_kg_m3"]) / bulk
        om = rpm * 2 * math.pi / 60
        filled = min(self.l_samp, bite_pen * self.a_hole * bulk / max(1e-9, self.a_transport))
        # Retained cuttings pressed on the wall by centrifugal force; gravity load on the flights
        h_f = self.r_o - self.root_r
        p_wall = rho_b * om * om * self.r_o * h_f
        t_ret = self.mu_w * p_wall * (2 * math.pi * self.r_o * filled) * self.r_o
        w_cut = rho_b * self.g * self.a_hole * bite_pen
        t_ret += self.mu_f * w_cut * 0.5 * (self.r_o + self.root_r)
        # Janssen plug once the bite exceeds the sampling-flute capacity
        t_plug, l_plug = 0.0, 0.0
        if bite_pen > self.z_crit:
            l_plug = (bite_pen - self.z_crit) * self.a_hole * bulk / self.a_transport
            k_j = 1 - math.sin(math.radians(float(L["friction_angle_deg"])))
            # Plug starts moving at the unconfined strength of remoulded (loose) cuttings
            sigma0 = mohr_coulomb_ucs_pa(float(self.cm["remoulded_cuttings_cohesion_kpa"]) * 1e3,
                                         float(L["friction_angle_deg"]))
            sigma = sigma0 * math.exp(min(60.0, 4 * self.mu_w * k_j * l_plug / self.d_h))
            phi_f = math.atan(self.mu_f)
            t_plug = sigma * self.a_transport * self.r_m_tr * math.tan(self.alpha_tr + phi_f)
        return {"torque_retained_nm": t_ret, "torque_plug_nm": t_plug, "plug_length_m": l_plug}

    def operating_point(self, z: float, bite_pen: float, wob: float, rpm: float, percussion: bool) -> Dict[str, float]:
        c = self.cutting(z, wob, rpm, percussion)
        r = self.removal_torque(z, bite_pen, rpm)
        om = rpm * 2 * math.pi / 60
        t_tot = c["torque_bit_nm"] + r["torque_retained_nm"] + r["torque_plug_nm"]
        p_cut = c["torque_bit_nm"] * om
        p_rem = (r["torque_retained_nm"] + r["torque_plug_nm"]) * om
        p_perc_mech = self.e_imp * self.f_blow if c["percussion_active"] else 0.0
        p_rot_el = t_tot * om / self.eta_rot + (self.p_nl if rpm > 0 else 0.0)
        p_perc_el = self.p_perc_el if c["percussion_active"] else 0.0
        rop = c["rop_m_s"]
        mech = p_cut + p_rem + float(self.cm["impact_transfer_efficiency"]) * p_perc_mech
        mse = (wob / self.a_hole + mech / (self.a_hole * rop)) if rop > 0 else float("inf")
        return {**c, **r, "torque_total_nm": t_tot, "p_cutting_mech_w": p_cut, "p_removal_mech_w": p_rem,
                "p_percussion_mech_w": p_perc_mech, "p_rotary_elec_w": p_rot_el, "p_percussion_elec_w": p_perc_el,
                "p_drill_elec_w": p_rot_el + p_perc_el, "mse_pa": mse}


# ------------------------------------------------------------------------------
# Curves for the website
# ------------------------------------------------------------------------------
def generate_curves(params: Optional[Dict[str, Any]] = None, design: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = params or load_system_params()
    d = design or derive_all(p)
    rpm = float(p["actuators"]["rotary"]["nominal_rpm"])
    depths = np.round(np.linspace(0.01, 0.99, 50), 3)
    out: Dict[str, Any] = {"depths_m": depths.tolist()}
    for env_id in p["environments"]:
        ph = Physics(p, d, env_id)
        wob = d["wob"][env_id]["wob_setpoint_n"]
        prof = {k: [] for k in ("ucs_mpa", "w_threshold_n", "rop_rotary_mm_s", "rop_rp_mm_s",
                                "torque_nm", "p_cutting_w", "p_removal_w", "p_percussion_w", "p_drill_elec_w", "mse_mj_m3")}
        for z in depths:
            r_only = ph.operating_point(float(z), 0.05, wob, rpm, False)
            rp = ph.operating_point(float(z), 0.05, wob, rpm, True)
            prof["ucs_mpa"].append(round(rp["ucs_pa"] / 1e6, 4))
            prof["w_threshold_n"].append(round(rp["w_threshold_n"], 2))
            prof["rop_rotary_mm_s"].append(round(r_only["rop_m_s"] * 1e3, 4))
            prof["rop_rp_mm_s"].append(round(rp["rop_m_s"] * 1e3, 4))
            prof["torque_nm"].append(round(rp["torque_total_nm"], 4))
            prof["p_cutting_w"].append(round(rp["p_cutting_mech_w"], 3))
            prof["p_removal_w"].append(round(rp["p_removal_mech_w"], 4))
            prof["p_percussion_w"].append(round(rp["p_percussion_mech_w"], 3))
            prof["p_drill_elec_w"].append(round(rp["p_drill_elec_w"], 2))
            prof["mse_mj_m3"].append(round(min(rp["mse_pa"], 1e12) / 1e6, 3))
        bites = np.round(np.linspace(0.01, 0.20, 77), 4)
        z_ref = 0.9
        choke_t = [round(ph.operating_point(z_ref, float(b), wob, rpm, True)["torque_total_nm"], 4) for b in bites]
        prof["choking"] = {"z_ref_m": z_ref, "bite_m": bites.tolist(), "torque_nm": choke_t,
                           "z_crit_m": ph.z_crit, "torque_fault_nm": d["torque_fault_nm"]}
        out[env_id] = prof
    return out


def main() -> None:
    EXPORTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = load_system_params()
    d = derive_all(p)
    curves = generate_curves(p, d)
    (EXPORTS_DATA_DIR / "terramechanics_curves.json").write_text(json.dumps(curves, indent=2))
    print("=== Module 1: Terramechanics curves ===")
    for env_id in p["environments"]:
        c = curves[env_id]
        print(f"  {env_id}: rotary threshold {min(c['w_threshold_n']):.2f}-{max(c['w_threshold_n']):.1f} N vs WOB_set "
              f"{d['wob'][env_id]['wob_setpoint_n']} N | R-P ROP {min(c['rop_rp_mm_s']):.3f}-{max(c['rop_rp_mm_s']):.2f} mm/s")


if __name__ == "__main__":
    main()
