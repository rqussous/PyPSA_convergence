from __future__ import annotations

import io
import hashlib
import importlib
import platform
import tempfile
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency in runtime
    go = None


@st.cache_resource(show_spinner=False)
def _get_network_module():
    # Delay importing network.py (and pypsa) until diagnostics are actually needed.
    return importlib.import_module("network")


def _run_with_captured_output(func, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _run_and_capture(func, *args, **kwargs) -> tuple[str, Any]:
    """Run func, capture its stdout, and also return its return value."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return buffer.getvalue(), result


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "core_done": False,
        "localization_done": False,
        "ramp_done": False,
        "opt_done": False,
        "physical_done": False,
        "core_output": "",
        "localization_output": "",
        "ramp_output": "",
        "opt_output": "",
        "physical_output": "",
        "localization_df": None,
        "ramp_df": None,
        "v_violation_df": None,
        "loading_df": None,
        "balance_df": None,
        "q_df": None,
        "run_summary_path": None,
        "recent_csv_folders": [],
        "network_load_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def _load_csv_preview(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _find_network_root(search_dir: Path) -> Path | None:
    direct_candidate = search_dir / "network.csv"
    if direct_candidate.exists():
        return search_dir

    candidates = sorted(search_dir.rglob("network.csv"), key=lambda path: len(path.parts))
    if not candidates:
        return None
    return candidates[0].parent


def _extract_uploaded_network_zip(uploaded_file) -> Path:
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    extract_dir = Path(tempfile.gettempdir()) / "pypsa_convergence_uploads" / digest
    marker_path = extract_dir / ".complete"

    if not marker_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(extract_dir)
        marker_path.write_text("ok", encoding="utf-8")

    network_root = _find_network_root(extract_dir)
    if network_root is None:
        raise ValueError("Uploaded zip does not contain a network.csv file.")
    return network_root


def _pick_directory_with_tkinter(initial_dir: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=initial_dir or str(Path.cwd()))
        root.destroy()
        return str(selected) if selected else None
    except Exception:
        return None


def _update_recent_folders(folder: str, max_items: int = 8) -> None:
    current = [str(p) for p in st.session_state.get("recent_csv_folders", []) if str(p).strip()]
    folder = str(folder).strip()
    if not folder:
        return
    updated = [folder] + [p for p in current if p != folder]
    st.session_state["recent_csv_folders"] = updated[:max_items]


def _render_folder_picker(default_network_folder: Path) -> str:
    st.sidebar.subheader("Network folder picker")

    if not default_network_folder.exists():
        st.sidebar.info(
            "Bundled network data is not available in this deployment. "
            "Upload a zipped CSV folder or enter a server path manually."
        )

    uploaded_network = st.sidebar.file_uploader(
        "Upload zipped network CSV folder",
        type=["zip"],
        help="Upload a zip containing your exported PyPSA CSV network folder, including network.csv.",
    )

    if uploaded_network is not None:
        try:
            extracted_path = _extract_uploaded_network_zip(uploaded_network)
            st.sidebar.success(f"Using uploaded network: {extracted_path.name}")
            st.session_state["csv_folder_input"] = str(extracted_path)
            _update_recent_folders(str(extracted_path))
        except Exception as exc:
            st.sidebar.error(f"Upload could not be read: {exc}")

    default_value = str(default_network_folder)
    if "csv_folder_input" not in st.session_state:
        st.session_state["csv_folder_input"] = default_value
    if not st.session_state.get("recent_csv_folders"):
        st.session_state["recent_csv_folders"] = [default_value]

    recent = [str(p) for p in st.session_state.get("recent_csv_folders", []) if str(p).strip()]
    if st.session_state["csv_folder_input"] not in recent:
        recent = [st.session_state["csv_folder_input"]] + recent
    recent = list(dict.fromkeys(recent))
    st.session_state["recent_csv_folders"] = recent[:8]

    selected_recent = st.sidebar.selectbox("Recent folders", options=st.session_state["recent_csv_folders"])
    can_browse_local_folder = platform.system() in {"Windows", "Darwin"}
    col_a, col_b, col_c = st.sidebar.columns(3)
    with col_a:
        if st.button("Use recent", use_container_width=True):
            st.session_state["csv_folder_input"] = selected_recent
            st.rerun()
    with col_b:
        browse_clicked = st.button(
            "Browse",
            use_container_width=True,
            disabled=not can_browse_local_folder,
            help="Native folder picker is available for local Windows/macOS runs.",
        )
        if browse_clicked and can_browse_local_folder:
            selected = _pick_directory_with_tkinter(st.session_state.get("csv_folder_input", default_value))
            if selected:
                st.session_state["csv_folder_input"] = selected
                _update_recent_folders(selected)
                st.rerun()
            else:
                st.info("No folder selected. You can still type the path manually.")
    with col_c:
        if st.button("Default", use_container_width=True):
            st.session_state["csv_folder_input"] = default_value
            _update_recent_folders(default_value)
            st.rerun()

    if not can_browse_local_folder:
        st.sidebar.caption("Browse is disabled in cloud/Linux deployments. Use upload, recent, or manual path.")

    return st.sidebar.text_input(
        "Network CSV folder",
        key="csv_folder_input",
        help="Use Browse for a native folder picker (local runs), pick from recent, or type manually.",
    )


def _sidebar_controls(default_network_folder: Path) -> dict:
    st.sidebar.header("Inputs")

    csv_folder = _render_folder_picker(default_network_folder)

    n_snapshots = st.sidebar.number_input(
        "Snapshots to analyze",
        min_value=1,
        max_value=8760,
        value=4,
        step=1,
    )

    st.sidebar.header("Localization")
    z_threshold = st.sidebar.number_input(
        "Switch impedance threshold [Ohm]",
        min_value=1e-6,
        max_value=1.0,
        value=1e-3,
        step=1e-4,
        format="%.6f",
    )

    st.sidebar.header("Ramp Test")
    ramp_start = st.sidebar.number_input("Ramp start", value=0.1, step=0.1, format="%.3f")
    ramp_stop = st.sidebar.number_input("Ramp stop", value=1.0, step=0.1, format="%.3f")
    ramp_step = st.sidebar.number_input("Ramp step", value=0.1, step=0.05, format="%.3f")

    st.sidebar.header("Optimization")
    solver = st.sidebar.text_input("Solver", value="highs")

    st.sidebar.header("Risk scoring")
    voltage_weight = st.sidebar.slider("Voltage deviation weight", 0.0, 1.0, 0.55, 0.05)
    structure_weight = st.sidebar.slider("Structural flags weight", 0.0, 1.0, 0.35, 0.05)
    convergence_weight = st.sidebar.slider("Convergence/ramp weight", 0.0, 1.0, 0.10, 0.05)

    st.sidebar.header("Physical Checks (LV)")
    v_min = st.sidebar.number_input("V_min [pu]", min_value=0.50, max_value=1.00, value=0.90, step=0.01, format="%.2f",
        help="Lower voltage limit. EN 50160 / typical LV: 0.90 pu.")
    v_max = st.sidebar.number_input("V_max [pu]", min_value=1.00, max_value=1.50, value=1.10, step=0.01, format="%.2f",
        help="Upper voltage limit. EN 50160 / typical LV: 1.10 pu.")
    s_max_warn = st.sidebar.number_input("Loading warn [pu]", min_value=0.10, max_value=1.00, value=0.80, step=0.05, format="%.2f",
        help="Branch loading warning threshold as fraction of s_nom.")
    s_max_fail = st.sidebar.number_input("Loading fail [pu]", min_value=0.10, max_value=2.00, value=1.00, step=0.05, format="%.2f",
        help="Branch loading fail threshold. 1.00 = at rated capacity.")
    balance_tol = st.sidebar.number_input("Balance tolerance", min_value=0.001, max_value=0.50, value=0.05, step=0.01, format="%.3f",
        help="Max allowed |gen-load|/load ratio before flagging imbalance.")
    q_tol = st.sidebar.number_input("Q limit tol [pu]", min_value=0.001, max_value=0.20, value=0.05, step=0.005, format="%.3f",
        help="Proximity band for classifying a generator as 'at Q limit'.")

    st.sidebar.header("Network loading")
    auto_load = st.sidebar.checkbox(
        "Auto-load selected network",
        value=False,
        help="If enabled, the app loads the selected network automatically when the path changes.",
    )
    col_load, col_reload = st.sidebar.columns(2)
    with col_load:
        load_now = st.button("Load", type="primary", use_container_width=True)
    with col_reload:
        reload_now = st.button("Reload", use_container_width=True)

    return {
        "csv_folder": csv_folder,
        "n_snapshots": int(n_snapshots),
        "z_threshold": float(z_threshold),
        "ramp_start": float(ramp_start),
        "ramp_stop": float(ramp_stop),
        "ramp_step": float(ramp_step),
        "solver": solver,
        "voltage_weight": float(voltage_weight),
        "structure_weight": float(structure_weight),
        "convergence_weight": float(convergence_weight),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "s_max_warn": float(s_max_warn),
        "s_max_fail": float(s_max_fail),
        "balance_tol": float(balance_tol),
        "q_tol": float(q_tol),
        "auto_load": bool(auto_load),
        "load_now": bool(load_now),
        "reload_now": bool(reload_now),
    }


def _get_work_paths(base_dir: Path) -> dict:
    diagnostics_dir = base_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return {
        "localization_csv": diagnostics_dir / "localization_report.csv",
        "ramp_csv": diagnostics_dir / "pf_ramp_report.csv",
        "diagnostics_dir": diagnostics_dir,
    }


def _load_network_once(csv_folder: Path, force_reload: bool = False):
    if force_reload:
        st.session_state.pop("loaded_network", None)
        st.session_state.pop("loaded_folder", None)

    if "loaded_network" not in st.session_state or st.session_state.get("loaded_folder") != str(csv_folder):
        net_diag = _get_network_module()
        st.session_state["loaded_network"] = net_diag.load_network(csv_folder)
        st.session_state["loaded_folder"] = str(csv_folder)
    return st.session_state["loaded_network"]


def _show_console_output(title: str, output: str) -> None:
    if title.strip():
        st.markdown(f"### {title}")
    if output.strip():
        st.code(output, language="text")
    else:
        st.info("No output returned.")


def _run_core_checks(network_obj, snapshots) -> str:
    net_diag = _get_network_module()
    output_parts = [_run_with_captured_output(net_diag.run_consistency_check, network_obj)]
    output_parts.append(_run_with_captured_output(net_diag.run_data_sanity_checks, network_obj))
    output_parts.append(_run_with_captured_output(net_diag.run_lpf_angle_check, network_obj, snapshots))
    output_parts.append(_run_with_captured_output(net_diag.run_structural_checks, network_obj))
    output_parts.append(_run_with_captured_output(net_diag.run_pf_checks, network_obj, snapshots))
    output = "\n".join(output_parts)
    return output


def _run_localization(network_obj, snapshots, z_threshold: float, localization_csv: Path) -> tuple[str, pd.DataFrame | None]:
    net_diag = _get_network_module()
    output = _run_with_captured_output(
        net_diag.run_network_localization,
        network_obj,
        snapshots,
        z_threshold,
        localization_csv,
    )

    if localization_csv.exists():
        data = _load_csv_preview(str(localization_csv))
        return output, data
    return output, None


def _run_ramp_test(network_obj, snapshots, settings: dict, ramp_csv: Path) -> tuple[str, pd.DataFrame | None]:
    net_diag = _get_network_module()
    output = _run_with_captured_output(
        net_diag.run_pf_ramp_test,
        network_obj,
        snapshots,
        settings["ramp_start"],
        settings["ramp_stop"],
        settings["ramp_step"],
        ramp_csv,
    )

    if ramp_csv.exists():
        data = _load_csv_preview(str(ramp_csv))
        return output, data
    return output, None


def _run_optimize(network_obj, snapshots, solver: str) -> str:
    net_diag = _get_network_module()
    output = _run_with_captured_output(net_diag.run_optimize_smoke_test, network_obj, snapshots, solver)
    return output


def _run_physical(
    network_obj, snapshots, settings: dict, diagnostics_dir: Path,
) -> tuple[str, dict[str, pd.DataFrame | None]]:
    net_diag = _get_network_module()
    output, results = _run_and_capture(
        net_diag.run_physical_checks,
        network_obj,
        snapshots,
        v_min=settings["v_min"],
        v_max=settings["v_max"],
        s_max_warn=settings["s_max_warn"],
        s_max_fail=settings["s_max_fail"],
        tol_ratio=settings["balance_tol"],
        q_tol_pu=settings["q_tol"],
        report_dir=diagnostics_dir,
    )
    return output, results


def _render_workflow_help() -> None:
    st.markdown("### Guided Workflow")
    st.info(
        "Run checks top-to-bottom for best triage: Core Checks -> Localization -> Ramp Test -> Optimization. "
        "The interactive network view below is always visible and becomes more informative after checks finish."
    )
    with st.expander("What each step does", expanded=True):
        st.markdown(
            "1. Core Checks: validates consistency, data quality, LPF angle differences, structural integrity, and PF convergence.\n"
            "2. Localization: finds switch-like clusters and buses with strongest voltage divergence.\n"
            "3. Ramp Test: increases loading/generation scale to detect the first non-converging operating point.\n"
            "4. Optimization: executes a solver smoke test to catch optimization setup problems early."
        )


def _status_label(done: bool) -> str:
    return "Complete" if done else "Pending"


def _render_detailed_check_guide() -> None:
    st.markdown("### How to Interpret Each Check")
    st.caption(
        "Reference guide for what each stage computes, why it matters, how to read outputs, and what to do next."
    )

    with st.expander("1) Core Checks: consistency, data sanity, angles, structure, PF convergence", expanded=False):
        st.markdown(
            "**What runs**\n"
            "- `consistency_check()` for PyPSA model coherence.\n"
            "- Data sanity scan for NaN/Inf and tiny non-zero values.\n"
            "- LPF angle-difference check on analyzed snapshots.\n"
            "- Structural checks for islanded buses, slack setup, and zero impedance values.\n"
            "- Non-linear PF checks with and without LPF seed.\n\n"
            "**Why this is done**\n"
            "- It catches model/data issues early so later diagnostics are meaningful.\n\n"
            "**How to read results**\n"
            "- Any NaN/Inf indicates a data issue that should be fixed first.\n"
            "- Very large angle differences indicate stressed or inconsistent operation.\n"
            "- Zero r/x counts indicate potentially numerically fragile branches.\n"
            "- `NOT converged` means PF failed for one or more snapshots.\n\n"
            "**What to do next**\n"
            "- Fix critical data/topology issues, then rerun Core Checks before deeper steps."
        )

    with st.expander("2) Localization: where instability is concentrated", expanded=False):
        st.markdown(
            "**What runs**\n"
            "- Switch-like lines are identified using the impedance threshold.\n"
            "- Bus clusters connected by near-zero impedance are computed.\n"
            "- PF voltage state is analyzed and top diverging buses are reported.\n\n"
            "**Why this is done**\n"
            "- It localizes likely problematic regions instead of inspecting the full network blindly.\n\n"
            "**How to read results**\n"
            "- High `abs_deviation` means stronger voltage abnormality at a bus.\n"
            "- Large `cluster_size` can indicate switch-connected propagation effects.\n"
            "- Many switch-only connected buses can indicate weakly anchored topology areas.\n\n"
            "**What to do next**\n"
            "- Start investigation from top localized buses and their neighboring elements."
        )

    with st.expander("3) Ramp Test: robustness margin and failure onset", expanded=False):
        st.markdown(
            "**What runs**\n"
            "- Demand/generation setpoints are scaled from ramp start to ramp stop.\n"
            "- PF convergence and final errors are tracked per scale factor.\n\n"
            "**Why this is done**\n"
            "- It estimates the stability margin and the first stress point that breaks PF convergence.\n\n"
            "**How to read results**\n"
            "- `all_converged=False` means the operating point is unstable for at least one snapshot.\n"
            "- The first failed `scale_factor` is a practical robustness threshold.\n"
            "- Increasing `max_final_error` before failure is an early warning trend.\n\n"
            "**What to do next**\n"
            "- Correlate failure onset with top risky buses/branches in the network triage view."
        )

    with st.expander("4) Optimization Smoke Test: solver-path sanity check", expanded=False):
        st.markdown(
            "**What runs**\n"
            "- A lightweight `optimize()` call over selected snapshots with the chosen solver.\n\n"
            "**Why this is done**\n"
            "- It validates solver configuration and optimization-path readiness before larger studies.\n\n"
            "**How to read results**\n"
            "- Success indicates basic model-solver compatibility.\n"
            "- Failure usually indicates solver availability/options or formulation consistency issues.\n\n"
            "**What to do next**\n"
            "- Verify solver setup and rerun Core Checks if optimization continues to fail."
        )

    with st.expander("How this connects to the interactive network triage", expanded=False):
        st.markdown(
            "- Bus risk voltage component uses **physical violation ratio** when Physical Checks have run, "
            "otherwise falls back to localization abs_deviation.\n"
            "- Branch risk includes **thermal loading** (max_loading_pu from Physical Checks) as a 20% component.\n"
            "- Use top-problem tables to move from symptoms to likely root-cause regions quickly."
        )

    with st.expander("5) Physical Feasibility Checks: voltage, loading, balance, Q", expanded=False):
        st.markdown(
            "**What runs**\n"
            "- A single shared non-linear PF (LPF-seeded) to get post-convergence state.\n"
            "- **Voltage violation check**: per-bus, per-snapshot v_mag_pu vs [v_min, v_max] bands.\n"
            "- **Branch loading check**: apparent power vs s_nom for all rated lines and transformers.\n"
            "- **Nodal balance check**: total generation vs total load using setpoint data per snapshot.\n"
            "- **Q-limit consistency check**: generator reactive output vs q_min/q_max after PF.\n\n"
            "**Why this is done**\n"
            "- Numerical PF convergence alone does not guarantee physical acceptability.\n"
            "- These checks verify the solution is electrically valid under EN 50160 / LV standards.\n\n"
            "**How to read results**\n"
            "- Voltage: FAIL = any snapshot outside limits; persistent = >=50% of snapshots.\n"
            "- Loading: WARN at the warning threshold, FAIL at the fail threshold; check s_nom coverage.\n"
            "- Balance: imbalance_ratio = |gen-load|/load per snapshot; FAIL > tolerance.\n"
            "- Q: AT_Q_MIN / AT_Q_MAX flags; saturation_ratio is fraction of snapshots at a limit.\n\n"
            "**What to do next**\n"
            "- Voltage violations: check topology connectivity, transformer tap settings, and load/gen imbalance near flagged buses.\n"
            "- Loading violations: check s_nom data accuracy, then consider reconductoring or reactive compensation.\n"
            "- Balance issues: verify completeness of generator/load data; may indicate missing assets.\n"
            "- Q saturation: consider adjusting generator reactive capability or adding reactive compensation.\n\n"
            "**Reproducibility**\n"
            "- `run_summary.json` records thresholds, snapshot IDs, and per-check pass/fail for archiving."
        )


def _collect_structural_metrics(network_obj, z_threshold: float) -> pd.DataFrame:
    buses = network_obj.buses.copy()
    if buses.empty:
        return pd.DataFrame(columns=["bus", "x", "y", "switch_degree", "normal_degree", "islanded", "switch_only", "zero_rx_incident"])

    lines = network_obj.lines.copy()
    if not lines.empty:
        r_series = pd.to_numeric(lines["r"], errors="coerce").fillna(0.0) if "r" in lines.columns else pd.Series(0.0, index=lines.index)
        x_series = pd.to_numeric(lines["x"], errors="coerce").fillna(0.0) if "x" in lines.columns else pd.Series(0.0, index=lines.index)
        lines["z_total"] = np.sqrt(r_series**2 + x_series**2)
        switch_lines = lines[lines["z_total"] < z_threshold]
        normal_lines = lines[lines["z_total"] >= z_threshold]
        zero_rx_lines = lines[(r_series == 0.0) | (x_series == 0.0)]
    else:
        switch_lines = pd.DataFrame(columns=["bus0", "bus1"])
        normal_lines = pd.DataFrame(columns=["bus0", "bus1"])
        zero_rx_lines = pd.DataFrame(columns=["bus0", "bus1"])

    def _degree_from_edges(edge_df: pd.DataFrame) -> pd.Series:
        if edge_df.empty:
            return pd.Series(0, index=buses.index, dtype=float)
        counts = pd.concat([edge_df["bus0"], edge_df["bus1"]], ignore_index=True).value_counts()
        return counts.reindex(buses.index, fill_value=0).astype(float)

    switch_degree = _degree_from_edges(switch_lines)
    normal_degree = _degree_from_edges(normal_lines)
    zero_rx_incident = _degree_from_edges(zero_rx_lines)

    connected = set()
    if not network_obj.lines.empty:
        connected |= set(network_obj.lines.bus0) | set(network_obj.lines.bus1)
    if not network_obj.transformers.empty:
        connected |= set(network_obj.transformers.bus0) | set(network_obj.transformers.bus1)
    if not network_obj.generators.empty:
        connected |= set(network_obj.generators.bus)
    if not network_obj.loads.empty:
        connected |= set(network_obj.loads.bus)
    if not network_obj.storage_units.empty:
        connected |= set(network_obj.storage_units.bus)

    islanded = pd.Series([1.0 if b not in connected else 0.0 for b in buses.index], index=buses.index)
    switch_only = ((normal_degree == 0) & (switch_degree > 0)).astype(float)

    x_vals = pd.to_numeric(buses["x"], errors="coerce") if "x" in buses.columns else pd.Series(np.nan, index=buses.index)
    y_vals = pd.to_numeric(buses["y"], errors="coerce") if "y" in buses.columns else pd.Series(np.nan, index=buses.index)

    return pd.DataFrame(
        {
            "bus": buses.index,
            "x": x_vals.to_numpy(),
            "y": y_vals.to_numpy(),
            "switch_degree": switch_degree.to_numpy(),
            "normal_degree": normal_degree.to_numpy(),
            "islanded": islanded.to_numpy(),
            "switch_only": switch_only.to_numpy(),
            "zero_rx_incident": zero_rx_incident.to_numpy(),
        }
    )


def _ramp_penalty_from_report(ramp_df: pd.DataFrame | None) -> float:
    if ramp_df is None or ramp_df.empty or "all_converged" not in ramp_df.columns:
        return 0.0
    converged = ramp_df["all_converged"].astype(bool)
    fail_ratio = 1.0 - float(converged.mean())
    if "scale_factor" in ramp_df.columns and (~converged).any():
        first_failed = float(ramp_df.loc[~converged, "scale_factor"].min())
        threshold_penalty = max(0.0, min(1.0, (1.0 - first_failed)))
        return min(1.0, 0.7 * fail_ratio + 0.3 * threshold_penalty)
    return min(1.0, fail_ratio)


def _bus_risk_table(network_obj, settings: dict) -> pd.DataFrame:
    base_df = _collect_structural_metrics(network_obj, settings["z_threshold"])

    loc_df = st.session_state.get("localization_df")
    if isinstance(loc_df, pd.DataFrame) and not loc_df.empty and "bus" in loc_df.columns:
        overlay_cols = [c for c in ["bus", "abs_deviation", "cluster_size", "v_mag_pu", "v_ang_deg"] if c in loc_df.columns]
        base_df = base_df.merge(loc_df[overlay_cols], on="bus", how="left")
    if "abs_deviation" not in base_df.columns:
        base_df["abs_deviation"] = 0.0
    base_df["abs_deviation"] = pd.to_numeric(base_df["abs_deviation"], errors="coerce").fillna(0.0)

    # Physical check overlay — use violation_ratio when available (higher quality signal)
    v_df = st.session_state.get("v_violation_df")
    if isinstance(v_df, pd.DataFrame) and not v_df.empty and "bus" in v_df.columns:
        v_cols = [c for c in ["bus", "violation_ratio", "worst_v_mag_pu", "n_violations"] if c in v_df.columns]
        base_df = base_df.merge(v_df[v_cols], on="bus", how="left")
        base_df["violation_ratio"] = pd.to_numeric(base_df.get("violation_ratio", 0.0), errors="coerce").fillna(0.0)
        voltage_component = base_df["violation_ratio"].clip(0.0, 1.0)
    else:
        # Fallback: normalized abs_deviation from localization (only first snapshot)
        voltage_component = (base_df["abs_deviation"] / 0.5).clip(0.0, 1.0)

    ramp_penalty = _ramp_penalty_from_report(st.session_state.get("ramp_df"))
    core_output = str(st.session_state.get("core_output", ""))
    core_penalty = 1.0 if "NOT converged" in core_output else 0.0
    convergence_penalty = max(ramp_penalty, core_penalty)

    structure_component = (
        0.5 * base_df["islanded"]
        + 0.3 * base_df["switch_only"]
        + 0.2 * (base_df["zero_rx_incident"] / 3.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    convergence_component = pd.Series(convergence_penalty, index=base_df.index)

    raw_weights = np.array(
        [
            settings["voltage_weight"],
            settings["structure_weight"],
            settings["convergence_weight"],
        ],
        dtype=float,
    )
    if raw_weights.sum() <= 0:
        norm_weights = np.array([0.55, 0.35, 0.10], dtype=float)
    else:
        norm_weights = raw_weights / raw_weights.sum()

    base_df["voltage_component"] = voltage_component
    base_df["structure_component"] = structure_component
    base_df["convergence_component"] = convergence_component
    base_df["risk_score"] = (
        norm_weights[0] * voltage_component
        + norm_weights[1] * structure_component
        + norm_weights[2] * convergence_component
    )
    base_df["risk_score"] = base_df["risk_score"].clip(0.0, 1.0)
    return base_df.sort_values("risk_score", ascending=False)


def _edge_risk_table(network_obj, bus_risk: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    if network_obj.lines.empty and network_obj.transformers.empty:
        return pd.DataFrame(columns=["kind", "name", "bus0", "bus1", "edge_risk", "near_switch", "zero_rx", "x0", "y0", "x1", "y1"])

    bus_xy = pd.DataFrame(index=network_obj.buses.index)
    bus_xy["x"] = pd.to_numeric(network_obj.buses["x"], errors="coerce") if "x" in network_obj.buses.columns else np.nan
    bus_xy["y"] = pd.to_numeric(network_obj.buses["y"], errors="coerce") if "y" in network_obj.buses.columns else np.nan
    bus_score = bus_risk.set_index("bus")["risk_score"] if not bus_risk.empty else pd.Series(dtype=float)

    # Loading data from physical checks (keyed by branch name string)
    loading_df = st.session_state.get("loading_df")
    loading_map: dict[str, float] = {}
    if isinstance(loading_df, pd.DataFrame) and "name" in loading_df.columns and "max_loading_pu" in loading_df.columns:
        valid_loading = loading_df.dropna(subset=["max_loading_pu"])
        loading_map = {str(k): float(v) for k, v in zip(valid_loading["name"], valid_loading["max_loading_pu"])}

    rows: list[dict[str, Any]] = []
    if not network_obj.lines.empty:
        lines = network_obj.lines.copy()
        r_series = pd.to_numeric(lines["r"], errors="coerce").fillna(0.0) if "r" in lines.columns else pd.Series(0.0, index=lines.index)
        x_series = pd.to_numeric(lines["x"], errors="coerce").fillna(0.0) if "x" in lines.columns else pd.Series(0.0, index=lines.index)
        lines["z_total"] = np.sqrt(r_series**2 + x_series**2)
        for idx, row in lines.iterrows():
            b0 = row.get("bus0")
            b1 = row.get("bus1")
            if b0 not in bus_xy.index or b1 not in bus_xy.index:
                continue
            if pd.isna(bus_xy.at[b0, "x"]) or pd.isna(bus_xy.at[b0, "y"]) or pd.isna(bus_xy.at[b1, "x"]) or pd.isna(bus_xy.at[b1, "y"]):
                continue
            near_switch = float(row["z_total"] < z_threshold)
            r_val = float(row.get("r", 0.0)) if pd.notna(row.get("r", 0.0)) else 0.0
            x_val = float(row.get("x", 0.0)) if pd.notna(row.get("x", 0.0)) else 0.0
            zero_rx = float((r_val == 0.0) or (x_val == 0.0))
            score = max(float(bus_score.get(b0, 0.0)), float(bus_score.get(b1, 0.0)))
            loading_score = min(1.0, loading_map.get(str(idx), 0.0))
            edge_risk = min(1.0, 0.50 * score + 0.20 * near_switch + 0.10 * zero_rx + 0.20 * loading_score)
            rows.append(
                {
                    "kind": "line",
                    "name": str(idx),
                    "bus0": b0,
                    "bus1": b1,
                    "near_switch": near_switch,
                    "zero_rx": zero_rx,
                    "loading_pu": round(loading_map.get(str(idx), 0.0), 3),
                    "edge_risk": edge_risk,
                    "x0": float(bus_xy.at[b0, "x"]),
                    "y0": float(bus_xy.at[b0, "y"]),
                    "x1": float(bus_xy.at[b1, "x"]),
                    "y1": float(bus_xy.at[b1, "y"]),
                }
            )

    if not network_obj.transformers.empty:
        for idx, row in network_obj.transformers.iterrows():
            b0 = row.get("bus0")
            b1 = row.get("bus1")
            if b0 not in bus_xy.index or b1 not in bus_xy.index:
                continue
            if pd.isna(bus_xy.at[b0, "x"]) or pd.isna(bus_xy.at[b0, "y"]) or pd.isna(bus_xy.at[b1, "x"]) or pd.isna(bus_xy.at[b1, "y"]):
                continue
            score = max(float(bus_score.get(b0, 0.0)), float(bus_score.get(b1, 0.0)))
            loading_score = min(1.0, loading_map.get(str(idx), 0.0))
            rows.append(
                {
                    "kind": "transformer",
                    "name": str(idx),
                    "bus0": b0,
                    "bus1": b1,
                    "near_switch": 0.0,
                    "zero_rx": 0.0,
                    "loading_pu": round(loading_map.get(str(idx), 0.0), 3),
                    "edge_risk": min(1.0, 0.70 * score + 0.30 * loading_score),
                    "x0": float(bus_xy.at[b0, "x"]),
                    "y0": float(bus_xy.at[b0, "y"]),
                    "x1": float(bus_xy.at[b1, "x"]),
                    "y1": float(bus_xy.at[b1, "y"]),
                }
            )

    return pd.DataFrame(rows).sort_values("edge_risk", ascending=False)


def _render_network_plot(network_obj, settings: dict, height: int = 600, show_title: bool = True) -> None:
    """Render interactive network plot with configurable height and title."""
    if show_title:
        st.markdown("### Interactive Network Triage")
        st.caption(
            "Baseline topology is always shown. After checks run, risk overlays combine voltage deviation, structural flags, and convergence/ramp signals."
        )

    bus_risk = _bus_risk_table(network_obj, settings)
    edge_risk = _edge_risk_table(network_obj, bus_risk, settings["z_threshold"])

    with st.expander("How risk score is computed", expanded=False):
        st.markdown(
            "Risk score per bus = weighted sum of: voltage violation ratio (physical checks) or normalized deviation "
            "(localization fallback) + structural penalties + convergence/ramp penalty.\n\n"
            "Branch risk = 50% bus risk + 20% switch-like flag + 10% zero-r/x + 20% thermal loading (physical checks)."
        )

    col1, col2, col3 = st.columns(3)
    min_score = col1.slider("Min bus risk to display", 0.0, 1.0, 0.0, 0.05, key=f"min_score_{height}")
    top_n_bus = int(col2.number_input("Top problematic buses", min_value=5, max_value=200, value=20, step=5, key=f"top_bus_{height}"))
    top_n_edge = int(col3.number_input("Top problematic branches", min_value=5, max_value=200, value=20, step=5, key=f"top_edge_{height}"))

    plot_buses = bus_risk[bus_risk["risk_score"] >= min_score].copy()
    plot_buses = plot_buses.dropna(subset=["x", "y"])

    if go is None:
        st.warning("Interactive edge plotting needs plotly. Install it with: pip install plotly")
        if not plot_buses.empty:
            st.map(plot_buses.rename(columns={"x": "lon", "y": "lat"})[["lat", "lon"]])
    else:
        fig = go.Figure()

        if not edge_risk.empty:
            low = edge_risk[edge_risk["edge_risk"] < 0.6]
            high = edge_risk[edge_risk["edge_risk"] >= 0.6]

            for subset, color, width, name in [
                (low, "rgba(120,120,120,0.35)", 1, "Branches"),
                (high, "rgba(220,50,32,0.75)", 2, "High-risk branches"),
            ]:
                if subset.empty:
                    continue
                xs: list[float | None] = []
                ys: list[float | None] = []
                for _, row in subset.iterrows():
                    xs.extend([row["x0"], row["x1"], None])
                    ys.extend([row["y0"], row["y1"], None])
                fig.add_trace(
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line={"color": color, "width": width},
                        name=name,
                        hoverinfo="skip",
                    )
                )

        if not plot_buses.empty:
            hover_text = (
                "Bus: "
                + plot_buses["bus"].astype(str)
                + "<br>Risk score: "
                + plot_buses["risk_score"].map("{:.3f}".format)
                + "<br>Voltage component: "
                + plot_buses["voltage_component"].map("{:.3f}".format)
                + "<br>Structure component: "
                + plot_buses["structure_component"].map("{:.3f}".format)
                + "<br>Convergence component: "
                + plot_buses["convergence_component"].map("{:.3f}".format)
            )
            fig.add_trace(
                go.Scattergl(
                    x=plot_buses["x"],
                    y=plot_buses["y"],
                    mode="markers",
                    name="Buses",
                    marker={
                        "size": 8,
                        "color": plot_buses["risk_score"],
                        "colorscale": "YlOrRd",
                        "cmin": 0,
                        "cmax": 1,
                        "line": {"width": 0.5, "color": "#2f2f2f"},
                        "colorbar": {"title": "Risk"},
                    },
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            height=height,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
            xaxis_title="x / longitude",
            yaxis_title="y / latitude",
            legend={"orientation": "h", "y": 1.02, "x": 0},
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)


def _render_overview_tab(network_obj, snapshots, settings: dict, paths: dict) -> None:
    """Overview tab: summary metrics, step status, compact plot, and action buttons."""
    st.markdown("### Network Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Buses", len(network_obj.buses))
    col1.metric("Lines", len(network_obj.lines))
    col2.metric("Transformers", len(network_obj.transformers))
    col2.metric("Generators", len(network_obj.generators))
    col3.metric("Loads", len(network_obj.loads))
    col3.metric("Storage Units", len(network_obj.storage_units))
    st.caption(f"Total snapshots: {len(network_obj.snapshots)} | Analyzed: {len(snapshots)}")

    st.markdown("### Step Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Core Checks", _status_label(bool(st.session_state.get("core_done"))))
    col2.metric("Localization", _status_label(bool(st.session_state.get("localization_done"))))
    col3.metric("Ramp Test", _status_label(bool(st.session_state.get("ramp_done"))))
    col4.metric("Optimization", _status_label(bool(st.session_state.get("opt_done"))))
    st.caption(f"Physical Checks: {_status_label(bool(st.session_state.get('physical_done')))}")

    st.markdown("### Interactive Network Triage (Compact View)")
    _render_network_plot(network_obj, settings, height=350, show_title=False)

    st.markdown("### Actions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        run_core = st.button("Run Core Checks", type="primary", use_container_width=True)
    with col2:
        run_localize = st.button("Run Localization", use_container_width=True)
    with col3:
        run_ramp = st.button("Run Ramp Test", use_container_width=True)
    with col4:
        run_opt = st.button("Run Optimization", use_container_width=True)
    col5, col6 = st.columns([1, 3])
    with col5:
        run_physical = st.button("Run Physical Checks", type="primary", use_container_width=True,
            help="Voltage violations, thermal loading, nodal balance, Q-limit consistency.")

    # Recommendation
    if not st.session_state.get("core_done"):
        st.info("Recommended: Run Core Checks first to validate the network")
    elif not st.session_state.get("localization_done"):
        st.info("Recommended: Run Localization to identify problem areas")
    elif not st.session_state.get("physical_done"):
        st.info("Recommended: Run Physical Checks for voltage/loading/balance/Q validation")
    elif not st.session_state.get("ramp_done"):
        st.info("Recommended: Run Ramp Test to assess stability margin")
    elif not st.session_state.get("opt_done"):
        st.info("Recommended: Run Optimization to verify solver setup")

    # Handle button clicks
    if run_core:
        with st.spinner("Running core checks..."):
            st.session_state["core_output"] = _run_core_checks(network_obj.copy(), snapshots)
            st.session_state["core_done"] = True
            st.rerun()

    if run_localize:
        with st.spinner("Running localization..."):
            output, data = _run_localization(network_obj.copy(), snapshots, settings["z_threshold"], paths["localization_csv"])
            st.session_state["localization_output"] = output
            st.session_state["localization_df"] = data
            st.session_state["localization_done"] = True
            st.rerun()

    if run_ramp:
        with st.spinner("Running ramp test..."):
            output, data = _run_ramp_test(network_obj.copy(), snapshots, settings, paths["ramp_csv"])
            st.session_state["ramp_output"] = output
            st.session_state["ramp_df"] = data
            st.session_state["ramp_done"] = True
            st.rerun()

    if run_opt:
        with st.spinner("Running optimization smoke test..."):
            st.session_state["opt_output"] = _run_optimize(network_obj.copy(), snapshots, settings["solver"])
            st.session_state["opt_done"] = True
            st.rerun()

    if run_physical:
        with st.spinner("Running physical feasibility checks (voltage, loading, balance, Q)..."):
            output, results = _run_physical(network_obj.copy(), snapshots, settings, paths["diagnostics_dir"])
            st.session_state["physical_output"] = output
            st.session_state["v_violation_df"] = results.get("voltage")
            st.session_state["loading_df"] = results.get("loading")
            st.session_state["balance_df"] = results.get("balance")
            st.session_state["q_df"] = results.get("q_consistency")
            st.session_state["run_summary_path"] = str(paths["diagnostics_dir"] / "run_summary.json")
            st.session_state["physical_done"] = True
            st.rerun()


def _render_analysis_tab(network_obj, settings: dict) -> None:
    """Analysis tab: detailed network plot and risk tables side-by-side."""
    _render_network_plot(network_obj, settings, height=600, show_title=True)

    st.divider()
    st.markdown("### Detailed Risk Analysis")

    bus_risk = _bus_risk_table(network_obj, settings)
    edge_risk = _edge_risk_table(network_obj, bus_risk, settings["z_threshold"])

    # Table controls
    col1, col2, col3 = st.columns(3)
    min_score = col1.slider("Filter by risk score", 0.0, 1.0, 0.0, 0.05, key="analysis_min_score")
    top_n_bus = int(col2.number_input("Top buses to display", min_value=5, max_value=100, value=15, key="analysis_top_bus"))
    top_n_edge = int(col3.number_input("Top branches to display", min_value=5, max_value=100, value=15, key="analysis_top_edge"))

    bus_risk_filtered = bus_risk[bus_risk["risk_score"] >= min_score]

    # Side-by-side tables
    colb, cole = st.columns(2)
    with colb:
        st.markdown("#### Top Problematic Buses")
        st.dataframe(
            bus_risk_filtered.head(top_n_bus)[
                [
                    "bus",
                    "risk_score",
                    "abs_deviation",
                    "islanded",
                    "switch_only",
                    "zero_rx_incident",
                    "switch_degree",
                    "normal_degree",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with cole:
        st.markdown("#### Top Problematic Branches")
        if edge_risk.empty:
            st.info("No branch data available.")
        else:
            edge_cols = ["kind", "name", "bus0", "bus1", "edge_risk", "near_switch", "zero_rx"]
            if "loading_pu" in edge_risk.columns:
                edge_cols.append("loading_pu")
            st.dataframe(
                edge_risk.head(top_n_edge)[edge_cols],
                use_container_width=True,
                hide_index=True,
            )


def _render_logs_tab() -> None:
    """Logs tab: displays check outputs in collapsible sections."""
    st.markdown("### Check Outputs & Logs")

    checks_available = any([
        st.session_state.get("core_done"),
        st.session_state.get("localization_done"),
        st.session_state.get("ramp_done"),
        st.session_state.get("opt_done")
    ])

    if not checks_available:
        st.info("Run checks from the Overview tab to see outputs here.")
        return

    if st.session_state.get("core_done") and st.session_state.get("core_output"):
        with st.expander("Core Checks Output", expanded=True):
            _show_console_output("", str(st.session_state.get("core_output")))

    if st.session_state.get("localization_done") and st.session_state.get("localization_output"):
        with st.expander("Localization Output", expanded=False):
            _show_console_output("", str(st.session_state.get("localization_output")))
            if isinstance(st.session_state.get("localization_df"), pd.DataFrame):
                st.dataframe(st.session_state["localization_df"], use_container_width=True, hide_index=True)
                localization_csv = Path(__file__).parent / "diagnostics" / "localization_report.csv"
                if localization_csv.exists():
                    st.download_button(
                        label="Download localization_report.csv",
                        data=localization_csv.read_bytes(),
                        file_name=localization_csv.name,
                        mime="text/csv",
                        key="dl_loc"
                    )

    if st.session_state.get("ramp_done") and st.session_state.get("ramp_output"):
        with st.expander("Ramp Test Output", expanded=False):
            _show_console_output("", str(st.session_state.get("ramp_output")))
            if isinstance(st.session_state.get("ramp_df"), pd.DataFrame):
                st.dataframe(st.session_state["ramp_df"], use_container_width=True, hide_index=True)
                ramp_csv = Path(__file__).parent / "diagnostics" / "pf_ramp_report.csv"
                if ramp_csv.exists():
                    st.download_button(
                        label="Download pf_ramp_report.csv",
                        data=ramp_csv.read_bytes(),
                        file_name=ramp_csv.name,
                        mime="text/csv",
                        key="dl_ramp"
                    )

    if st.session_state.get("opt_done") and st.session_state.get("opt_output"):
        with st.expander("Optimization Smoke Test Output", expanded=False):
            _show_console_output("", str(st.session_state.get("opt_output")))

    if st.session_state.get("physical_done") and st.session_state.get("physical_output"):
        with st.expander("Physical Checks Output", expanded=False):
            _show_console_output("", str(st.session_state.get("physical_output")))
            diag_dir = Path(__file__).parent / "diagnostics"
            for fname, label, key in [
                ("voltage_violations.csv", "Download voltage_violations.csv", "dl_vv"),
                ("branch_loading.csv", "Download branch_loading.csv", "dl_bl"),
                ("nodal_balance.csv", "Download nodal_balance.csv", "dl_nb"),
                ("q_consistency.csv", "Download q_consistency.csv", "dl_qc"),
                ("run_summary.json", "Download run_summary.json", "dl_rs"),
            ]:
                fpath = diag_dir / fname
                if fpath.exists():
                    st.download_button(
                        label=label, data=fpath.read_bytes(), file_name=fname,
                        mime="text/csv" if fname.endswith(".csv") else "application/json",
                        key=key,
                    )


def _render_physical_tab() -> None:
    """Physical Checks tab: summarised violation tables and run summary."""
    st.markdown("### Physical Feasibility Results")
    if not st.session_state.get("physical_done"):
        st.info("Run Physical Checks from the Overview tab to see results here.")
        return

    v_df = st.session_state.get("v_violation_df")
    loading_df = st.session_state.get("loading_df")
    balance_df = st.session_state.get("balance_df")
    q_df = st.session_state.get("q_df")

    # Summary badges
    def _badge(df, col, threshold, kind="any") -> str:
        if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
            return "No data"
        if kind == "any":
            n = int((df[col] > 0).sum())
        else:
            n = int(df[col].sum())
        return f"{n} flagged" if n > 0 else "PASS"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Voltage violations", _badge(v_df, "n_violations", 0))
    c2.metric("Overloaded branches", _badge(loading_df, "n_fail", 0))

    # Balance badge: count FAIL rows
    if isinstance(balance_df, pd.DataFrame) and not balance_df.empty and "pass_flag" in balance_df.columns:
        n_balance_fail = int((balance_df["pass_flag"] == "FAIL").sum())
        balance_label = f"{n_balance_fail} snapshots" if n_balance_fail > 0 else "PASS"
    else:
        balance_label = "No data"
    c3.metric("Balance issues", balance_label)

    if isinstance(q_df, pd.DataFrame) and not q_df.empty:
        q_events = pd.Series(0, index=q_df.index, dtype=float)
        if "n_at_qmax" in q_df.columns:
            q_events = q_events + pd.to_numeric(q_df["n_at_qmax"], errors="coerce").fillna(0.0)
        if "n_at_qmin" in q_df.columns:
            q_events = q_events + pd.to_numeric(q_df["n_at_qmin"], errors="coerce").fillna(0.0)
        q_label = f"{int((q_events > 0).sum())} flagged" if int((q_events > 0).sum()) > 0 else "PASS"
    else:
        q_label = "No data"
    c4.metric("Q saturation events", q_label)

    st.divider()

    tab_v, tab_l, tab_b, tab_q, tab_summary = st.tabs(
        ["Voltage", "Loading", "Balance", "Q Limits", "Summary JSON"]
    )

    with tab_v:
        st.markdown("#### Voltage Violations (per bus)")
        if isinstance(v_df, pd.DataFrame) and not v_df.empty and "n_violations" in v_df.columns:
            violating = v_df[v_df["n_violations"] > 0].sort_values("n_violations", ascending=False)
            if violating.empty:
                st.success("No buses outside voltage limits.")
            else:
                st.warning(f"{len(violating)} bus(es) outside voltage limits.")
                if go is not None and "bus" in violating.columns:
                    top_v = violating.head(30)
                    fig_v = go.Figure(
                        data=[
                            go.Bar(
                                x=top_v["bus"].astype(str),
                                y=pd.to_numeric(top_v["n_violations"], errors="coerce").fillna(0.0),
                                marker_color="#d62728",
                                name="Violations",
                            )
                        ]
                    )
                    fig_v.update_layout(
                        height=320,
                        margin={"l": 10, "r": 10, "t": 20, "b": 10},
                        xaxis_title="Bus",
                        yaxis_title="Number of violating snapshots",
                    )
                    st.plotly_chart(fig_v, use_container_width=True)
                st.dataframe(violating, use_container_width=True, hide_index=True)
        else:
            st.info("No voltage data available.")

    with tab_l:
        st.markdown("#### Branch Thermal Loading")
        if isinstance(loading_df, pd.DataFrame) and not loading_df.empty and "pass_flag" in loading_df.columns:
            rated = loading_df[loading_df["pass_flag"].isin(["PASS", "WARN", "FAIL"])].sort_values(
                "max_loading_pu", ascending=False, na_position="last"
            )
            overloaded = rated[rated["pass_flag"].isin(["WARN", "FAIL"])]
            if overloaded.empty:
                st.success("No branches exceeding warning threshold.")
            else:
                n_fail = int((rated["pass_flag"] == "FAIL").sum())
                st.warning(f"{len(overloaded)} branch(es) above warning threshold ({n_fail} at FAIL).")

            if go is not None and not rated.empty and "name" in rated.columns and "max_loading_pu" in rated.columns:
                top_l = rated.head(30).copy()
                color_map = {"PASS": "#2ca02c", "WARN": "#ff7f0e", "FAIL": "#d62728"}
                bar_colors = [color_map.get(str(v), "#7f7f7f") for v in top_l["pass_flag"]]
                fig_l = go.Figure(
                    data=[
                        go.Bar(
                            x=top_l["name"].astype(str),
                            y=pd.to_numeric(top_l["max_loading_pu"], errors="coerce").fillna(0.0),
                            marker_color=bar_colors,
                            name="Max loading [pu]",
                        )
                    ]
                )
                fig_l.add_hline(y=1.0, line_dash="dash", line_color="#d62728")
                fig_l.update_layout(
                    height=320,
                    margin={"l": 10, "r": 10, "t": 20, "b": 10},
                    xaxis_title="Branch",
                    yaxis_title="Max loading [pu]",
                )
                st.plotly_chart(fig_l, use_container_width=True)

            cols_to_show = [
                c for c in ["kind", "name", "bus0", "bus1", "s_nom_mva", "max_loading_pu", "mean_loading_pu", "n_fail", "pass_flag"]
                if c in rated.columns
            ]
            if cols_to_show:
                st.dataframe(rated.head(30)[cols_to_show], use_container_width=True, hide_index=True)
            else:
                st.dataframe(rated.head(30), use_container_width=True, hide_index=True)
        else:
            st.info("No loading data available.")

    with tab_b:
        st.markdown("#### Active Power Balance (per snapshot)")
        if isinstance(balance_df, pd.DataFrame) and not balance_df.empty:
            n_fail = int((balance_df["pass_flag"] == "FAIL").sum()) if "pass_flag" in balance_df.columns else 0
            if n_fail == 0:
                st.success("Generation-demand balance within tolerance for all snapshots.")
            else:
                st.warning(f"{n_fail} snapshot(s) outside balance tolerance.")

            if go is not None and "snapshot" in balance_df.columns and "imbalance_ratio" in balance_df.columns:
                balance_plot = balance_df.copy()
                y_vals = pd.to_numeric(balance_plot["imbalance_ratio"], errors="coerce").fillna(0.0)
                marker_colors = "#d62728" if n_fail > 0 else "#2ca02c"
                fig_b = go.Figure(
                    data=[
                        go.Scatter(
                            x=balance_plot["snapshot"].astype(str),
                            y=y_vals,
                            mode="lines+markers",
                            line={"color": "#1f77b4"},
                            marker={"color": marker_colors, "size": 8},
                            name="Imbalance ratio",
                        )
                    ]
                )
                fig_b.update_layout(
                    height=320,
                    margin={"l": 10, "r": 10, "t": 20, "b": 10},
                    xaxis_title="Snapshot",
                    yaxis_title="|gen-load| / load",
                )
                st.plotly_chart(fig_b, use_container_width=True)

            st.dataframe(balance_df, use_container_width=True, hide_index=True)
        else:
            st.info("No balance data available.")

    with tab_q:
        st.markdown("#### Reactive Power / Q-Limit Consistency")
        if isinstance(q_df, pd.DataFrame) and not q_df.empty:
            if "pass_flag" in q_df.columns:
                flagged = q_df[q_df["pass_flag"] != "PASS"]
            else:
                flagged = q_df
            if flagged.empty:
                st.success("No Q limit saturation detected.")
            else:
                n_pers = int((q_df["saturation_ratio"] >= 0.5).sum()) if "saturation_ratio" in q_df.columns else 0
                st.warning(f"{len(flagged)} generator(s) with Q limit events, {n_pers} persistently saturated.")

            if go is not None and "name" in q_df.columns and ("n_at_qmax" in q_df.columns or "n_at_qmin" in q_df.columns):
                q_plot = q_df.copy().head(30)
                qmax_vals = pd.to_numeric(q_plot.get("n_at_qmax", 0), errors="coerce").fillna(0.0)
                qmin_vals = pd.to_numeric(q_plot.get("n_at_qmin", 0), errors="coerce").fillna(0.0)
                fig_q = go.Figure()
                fig_q.add_trace(go.Bar(x=q_plot["name"].astype(str), y=qmax_vals, name="At Q max", marker_color="#d62728"))
                fig_q.add_trace(go.Bar(x=q_plot["name"].astype(str), y=qmin_vals, name="At Q min", marker_color="#1f77b4"))
                fig_q.update_layout(
                    barmode="stack",
                    height=320,
                    margin={"l": 10, "r": 10, "t": 20, "b": 10},
                    xaxis_title="Generator",
                    yaxis_title="Number of saturated snapshots",
                )
                st.plotly_chart(fig_q, use_container_width=True)

            st.dataframe(flagged if not flagged.empty else q_df, use_container_width=True, hide_index=True)
        else:
            st.info("No Q-limit data available.")

    # Summary artifact download
    summary_path = Path(__file__).parent / "diagnostics" / "run_summary.json"
    with tab_summary:
        if summary_path.exists():
            st.markdown("#### Reproducibility Artifact")
            with open(summary_path) as f:
                import json as _json
                summary_data = _json.load(f)
            st.json(summary_data)
            st.download_button("Download run_summary.json", data=summary_path.read_bytes(),
                file_name="run_summary.json", mime="application/json", key="dl_summary_tab")
        else:
            st.info("run_summary.json is not available yet.")


def _render_help_tab() -> None:
    """Help tab: workflow guide and reference documentation."""
    _render_workflow_help()

    st.divider()
    _render_detailed_check_guide()

    st.divider()
    st.markdown("### Quick Start Guide")
    st.markdown(
        "1. Confirm the network CSV folder in the left sidebar.\n"
        "2. Adjust snapshot count, impedance threshold, voltage/loading thresholds, and risk weights.\n"
        "3. Switch to the **Overview** tab to see the compact network visualization.\n"
        "4. Run checks sequentially from Overview: Core → Localization → Ramp → Physical → Optimization.\n"
        "5. Switch to **Physical Checks** tab for voltage/loading/balance/Q violation summaries.\n"
        "6. Switch to **Analysis** tab for detailed risk tables and full-size visualization.\n"
        "7. Check the **Logs** tab for detailed outputs and download CSV reports.\n"
        "8. Download `run_summary.json` from the Physical Checks tab for research reproducibility."
    )


def main() -> None:
    st.set_page_config(page_title="PyPSA Diagnostics GUI", layout="wide")
    _init_state()

    st.title("PyPSA Convergence Diagnostics")
    st.caption("Analyze network convergence, stability, and optimization readiness—compact, tab-based interface.")

    default_folder = Path(__file__).parent / "network_postprocessed"
    settings = _sidebar_controls(default_folder)
    base_dir = Path(__file__).parent
    paths = _get_work_paths(base_dir)

    csv_folder = Path(settings["csv_folder"]).expanduser()
    if not csv_folder.exists():
        st.error(
            f"Network folder not found: {csv_folder}. "
            "For Streamlit deployment, upload a zipped CSV folder from the sidebar or configure an accessible path."
        )
        st.stop()

    _update_recent_folders(str(csv_folder))

    current_loaded_folder = st.session_state.get("loaded_folder")
    needs_load = current_loaded_folder != str(csv_folder) or "loaded_network" not in st.session_state
    should_reload = bool(settings.get("reload_now", False))
    should_load = bool(settings.get("load_now", False)) or should_reload or (bool(settings.get("auto_load", False)) and needs_load)

    if should_load:
        try:
            with st.spinner("Loading network data..."):
                _load_network_once(csv_folder, force_reload=should_reload)
            st.session_state["network_load_error"] = ""
            st.success("Network loaded successfully.")
        except Exception as exc:
            st.session_state["network_load_error"] = str(exc)

    network_obj = st.session_state.get("loaded_network")
    if network_obj is None or st.session_state.get("loaded_folder") != str(csv_folder):
        if st.session_state.get("network_load_error"):
            st.error(f"Failed to load network: {st.session_state['network_load_error']}")
        st.info("Select a network path and click Load in the sidebar to start analysis.")
        st.stop()

    snapshots = network_obj.snapshots[: max(0, settings["n_snapshots"])]

    # Create tabs
    tab_overview, tab_physical, tab_analysis, tab_logs, tab_help = st.tabs(
        ["Overview", "Physical Checks", "Analysis", "Logs", "Help"]
    )

    with tab_overview:
        _render_overview_tab(network_obj, snapshots, settings, paths)

    with tab_physical:
        _render_physical_tab()

    with tab_analysis:
        _render_analysis_tab(network_obj, settings)

    with tab_logs:
        _render_logs_tab()

    with tab_help:
        _render_help_tab()


if __name__ == "__main__":
    main()
