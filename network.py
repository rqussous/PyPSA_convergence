from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
	import pypsa
except Exception as exc:  # pragma: no cover - import diagnostics
	raise SystemExit(
		"Failed to import pypsa. This is usually an environment dependency issue. "
		f"Original error: {exc}"
	) from exc


def load_network(csv_folder: Path) -> pypsa.Network:
	network = pypsa.Network()
	network.import_from_csv_folder(csv_folder)
	return network


def print_header(title: str) -> None:
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def run_consistency_check(network: pypsa.Network) -> None:
	print_header("1) Network consistency_check()")
	try:
		network.consistency_check()
		print("consistency_check() executed. Inspect warnings above, if any.")
	except Exception as exc:
		print(f"consistency_check() failed: {exc}")


def _count_nan_inf(df: pd.DataFrame) -> tuple[int, int]:
	if df.empty:
		return 0, 0
	values = df.select_dtypes(include=[np.number]).to_numpy(copy=False)
	if values.size == 0:
		return 0, 0
	nan_count = int(np.isnan(values).sum())
	inf_count = int(np.isinf(values).sum())
	return nan_count, inf_count


def run_data_sanity_checks(network: pypsa.Network) -> None:
	print_header("2) Data sanity (NaN/Inf and tiny non-zero values)")

	static_tables = [
		"buses",
		"lines",
		"transformers",
		"generators",
		"loads",
		"storage_units",
		"links",
	]
	time_series_tables = [
		("generators_t", "p_max_pu"),
		("generators_t", "p_min_pu"),
		("loads_t", "p_set"),
		("storage_units_t", "p_set"),
	]

	for name in static_tables:
		table = getattr(network, name, pd.DataFrame())
		nan_count, inf_count = _count_nan_inf(table)
		print(f"{name:<15} NaN={nan_count:<8} Inf={inf_count:<8} rows={len(table)}")

	for container, attr in time_series_tables:
		container_obj = getattr(network, container, None)
		table = getattr(container_obj, attr, pd.DataFrame()) if container_obj is not None else pd.DataFrame()
		nan_count, inf_count = _count_nan_inf(table)
		print(
			f"{container}.{attr:<20} NaN={nan_count:<8} Inf={inf_count:<8} shape={table.shape}"
		)

	p_max_pu = getattr(network.generators_t, "p_max_pu", pd.DataFrame())
	if not p_max_pu.empty:
		tiny_nonzero_mask = (p_max_pu.abs() < 1e-3) & (p_max_pu != 0)
		tiny_nonzero = int(tiny_nonzero_mask.sum().sum())
		print(f"Tiny non-zero values in generators_t.p_max_pu (<1e-3): {tiny_nonzero}")
		if tiny_nonzero:
			print("Tip: clip tiny values to 0, e.g. p_max_pu = p_max_pu.clip(lower=1e-3) where needed.")


def _angle_diff_stats(network: pypsa.Network, snapshot) -> pd.Series:
	if network.lines.empty:
		return pd.Series(dtype=float)
	angle_diff = pd.Series(
		network.buses_t.v_ang.loc[snapshot, network.lines.bus0].values
		- network.buses_t.v_ang.loc[snapshot, network.lines.bus1].values,
		index=network.lines.index,
		dtype=float,
	)
	return angle_diff * 180 / np.pi


def run_lpf_angle_check(network: pypsa.Network, snapshots: Iterable) -> None:
	print_header("3) LPF angle-difference check")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return

	network.lpf(snapshots=snapshots)
	first_snapshot = snapshots[0]
	angle_deg = _angle_diff_stats(network, first_snapshot)
	if angle_deg.empty:
		print("No lines available for angle difference check.")
		return

	summary = angle_deg.describe()
	max_abs = float(angle_deg.abs().max())
	print("Line angle difference stats [deg] for first checked snapshot:")
	print(summary.to_string())
	print(f"Max abs angle difference: {max_abs:.2f} deg")
	if max_abs > 40:
		print("WARNING: max abs angle difference > 40 deg (possible PF convergence risk).")
	else:
		print("OK: max abs angle difference <= 40 deg.")


def run_structural_checks(network: pypsa.Network) -> None:
	print_header("4) Structural checks (islands, slack, impedances)")

	bus_index = set(network.buses.index)
	buses_in_branches = set(network.lines.bus0) | set(network.lines.bus1)
	buses_in_branches |= set(network.transformers.bus0) | set(network.transformers.bus1)

	if not network.links.empty:
		buses_in_branches |= set(network.links.bus0)
		for col in [c for c in network.links.columns if c.startswith("bus") and c != "bus0"]:
			buses_in_branches |= set(network.links[col].dropna())

	if not network.generators.empty:
		buses_in_branches |= set(network.generators.bus)
	if not network.loads.empty:
		buses_in_branches |= set(network.loads.bus)
	if not network.storage_units.empty:
		buses_in_branches |= set(network.storage_units.bus)

	islanded_buses = sorted(bus_index - buses_in_branches)
	print(f"Potential islanded buses (no connected asset): {len(islanded_buses)}")
	if islanded_buses:
		print(f"First 10 islanded buses: {islanded_buses[:10]}")

	slack_gens = network.generators.index[network.generators.control == "Slack"] if "control" in network.generators else []
	print(f"Generators with control='Slack': {len(slack_gens)}")

	if not network.lines.empty:
		line_zero_x = int((network.lines.x.fillna(0) == 0).sum()) if "x" in network.lines else 0
		line_zero_r = int((network.lines.r.fillna(0) == 0).sum()) if "r" in network.lines else 0
		print(f"Lines with zero x: {line_zero_x}; lines with zero r: {line_zero_r}")

	if not network.transformers.empty:
		trafo_zero_x = int((network.transformers.x.fillna(0) == 0).sum()) if "x" in network.transformers else 0
		trafo_zero_r = int((network.transformers.r.fillna(0) == 0).sum()) if "r" in network.transformers else 0
		print(f"Transformers with zero x: {trafo_zero_x}; transformers with zero r: {trafo_zero_r}")


def _find_switch_clusters(network: pypsa.Network, z_threshold: float) -> dict[str, str]:
	"""Return {bus_name: cluster_id} grouping buses joined by near-zero-impedance lines."""
	lines = network.lines.copy()
	lines["z_total"] = np.sqrt(lines["r"] ** 2 + lines["x"] ** 2)
	switch_lines = lines[lines["z_total"] < z_threshold]

	# Union-Find to merge buses that share a switch-like line
	parent: dict[str, str] = {bus: bus for bus in network.buses.index}

	def find(x: str) -> str:
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x

	def union(a: str, b: str) -> None:
		ra, rb = find(a), find(b)
		if ra != rb:
			parent[ra] = rb

	for _, row in switch_lines.iterrows():
		if row["bus0"] in parent and row["bus1"] in parent:
			union(row["bus0"], row["bus1"])

	return {bus: find(bus) for bus in network.buses.index}


def run_network_localization(
	network: pypsa.Network,
	snapshots: Iterable,
	z_threshold: float,
	report_path: Path | None,
) -> None:
	print_header("7) Network localization (switch clusters & diverging bus trace)")
	snapshots = list(snapshots)

	# --- Switch cluster analysis ---
	lines = network.lines.copy()
	lines["z_total"] = np.sqrt(lines["r"] ** 2 + lines["x"] ** 2)
	switch_lines = lines[lines["z_total"] < z_threshold]
	normal_lines = lines[lines["z_total"] >= z_threshold]

	print(f"Switch-like lines (|Z| < {z_threshold} Ohm): {len(switch_lines)} / {len(lines)}")
	print(f"  min Z = {switch_lines['z_total'].min():.2e} Ohm" if not switch_lines.empty else "  none")
	print(f"  max Z = {switch_lines['z_total'].max():.2e} Ohm" if not switch_lines.empty else "")

	clusters = _find_switch_clusters(network, z_threshold)
	cluster_sizes = pd.Series(clusters).value_counts()
	non_trivial = cluster_sizes[cluster_sizes > 1]
	print(f"Switch-merged bus clusters: {len(cluster_sizes)} total, {len(non_trivial)} with >1 bus")
	if not non_trivial.empty:
		print(f"  Largest cluster: {non_trivial.iloc[0]} buses (root bus: {non_trivial.index[0]})")
		print(f"  Top 5 cluster sizes: {non_trivial.head().to_dict()}")

	# --- Run PF and capture voltage state ---
	if not snapshots:
		print("No snapshots to run PF for.")
		return

	network.lpf(snapshots=snapshots)
	pf_result = network.pf(snapshots=snapshots, use_seed=True)

	# --- Bus voltage divergence analysis ---
	v_mag = network.buses_t.v_mag_pu
	if v_mag.empty or snapshots[0] not in v_mag.index:
		print("No v_mag_pu available after PF.")
		return

	snap = snapshots[0]
	v_row = v_mag.loc[snap]
	v_ang_row = (
		(network.buses_t.v_ang.loc[snap] * 180 / np.pi)
		if snap in network.buses_t.v_ang.index
		else pd.Series(dtype=float)
	)

	n_extreme = int(((v_row < 0.5) | (v_row > 1.5)).sum())
	print(f"\nAfter PF ({snap}):")
	print(f"  Buses with extreme v_mag_pu (<0.5 or >1.5): {n_extreme} / {len(v_row)}")
	print(f"  v_mag_pu range: {v_row.min():.3e} to {v_row.max():.3e}")

	# Worst diverging buses (most negative or largest absolute deviation)
	v_abs_dev = (v_row - 1.0).abs()
	worst_buses = v_abs_dev.nlargest(20).index

	# Build neighborhood adjacency from all lines + transformers
	adj: dict[str, list[str]] = {bus: [] for bus in network.buses.index}
	for _, row in network.lines.iterrows():
		adj[row["bus0"]].append(row["bus1"])
		adj[row["bus1"]].append(row["bus0"])
	for _, row in network.transformers.iterrows():
		adj[row["bus0"]].append(row["bus1"])
		adj[row["bus1"]].append(row["bus0"])

	# For each worst bus, describe its 1-hop neighbors and connected elements
	print(f"\nTop {len(worst_buses)} most-diverged buses and their direct neighbours:")
	report_rows: list[dict] = []
	for bus in worst_buses:
		v_val = float(v_row[bus])
		v_ang_val = float(v_ang_row[bus]) if bus in v_ang_row.index else np.nan
		neighbours = list(dict.fromkeys(adj[bus]))  # deduplicated
		switch_count = int(switch_lines[
			(switch_lines["bus0"] == bus) | (switch_lines["bus1"] == bus)
		].shape[0])
		normal_count = int(normal_lines[
			(normal_lines["bus0"] == bus) | (normal_lines["bus1"] == bus)
		].shape[0])
		gen_count = int((network.generators.bus == bus).sum())
		load_count = int((network.loads.bus == bus).sum())
		cluster_root = clusters.get(bus, bus)
		cluster_sz = int(cluster_sizes.get(cluster_root, 1))

		print(
			f"  Bus {bus!r:50s} v={v_val:+.3e}  v_ang={v_ang_val:+.1f}deg  "
			f"switch_lines={switch_count}  normal_lines={normal_count}  "
			f"gens={gen_count}  loads={load_count}  cluster_sz={cluster_sz}"
		)
		report_rows.append({
			"bus": bus,
			"v_mag_pu": v_val,
			"v_ang_deg": v_ang_val,
			"abs_deviation": float(v_abs_dev[bus]),
			"switch_lines_connected": switch_count,
			"normal_lines_connected": normal_count,
			"generators": gen_count,
			"loads": load_count,
			"cluster_size": cluster_sz,
			"cluster_root": cluster_root,
			"neighbours": ";".join(neighbours[:10]),
		})

	# Summary: how many switch-cluster buses have no normal-line connection?
	only_switch_buses = [
		b for b in network.buses.index
		if int(normal_lines[(normal_lines["bus0"] == b) | (normal_lines["bus1"] == b)].shape[0]) == 0
		and int(switch_lines[(switch_lines["bus0"] == b) | (switch_lines["bus1"] == b)].shape[0]) > 0
	]
	print(f"\nBuses connected ONLY via switch-like lines (no normal branch): {len(only_switch_buses)}")

	if report_path is not None and report_rows:
		report_df = pd.DataFrame(report_rows)
		report_path.parent.mkdir(parents=True, exist_ok=True)
		report_df.to_csv(report_path, index=False)
		print(f"Saved localization report to: {report_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _report_pf_result(label: str, pf_result) -> None:
	try:
		converged = pf_result["converged"]
		if hasattr(converged, "to_numpy"):
			ok = bool(np.all(converged.to_numpy(dtype=bool)))
		else:
			ok = bool(converged)

		if ok:
			print(f"{label}: converged for all checked snapshots")
		else:
			print(f"{label}: NOT converged for all checked snapshots")
			if hasattr(converged, "to_string"):
				print("Converged flags:")
				print(converged.to_string())

		error = pf_result.get("error", None)
		if error is not None and hasattr(error, "to_string"):
			print("Final PF error:")
			print(error.to_string())
	except Exception:
		print(f"{label}: completed (could not parse convergence flags from return object)")


def _extract_pf_series(pf_result) -> tuple[bool, pd.Series, pd.Series]:
	converged = pf_result["converged"]
	error = pf_result.get("error", pd.Series(dtype=float))

	if isinstance(converged, pd.DataFrame):
		converged_series = converged.fillna(False).all(axis=1)
	elif isinstance(converged, pd.Series):
		converged_series = converged.fillna(False).astype(bool)
	else:
		converged_series = pd.Series([bool(converged)])

	all_ok = bool(converged_series.all()) if not converged_series.empty else False

	if isinstance(error, pd.DataFrame):
		error_series = pd.to_numeric(error.stack(dropna=False), errors="coerce")
	elif isinstance(error, pd.Series):
		error_series = pd.to_numeric(error, errors="coerce")
	else:
		error_series = pd.Series([error], dtype=float)

	return all_ok, converged_series, error_series


def _scale_time_series_values(network: pypsa.Network, scale_factor: float) -> None:
	# Scale demand/generation set-points to identify the loadability threshold of PF.
	for comp_name in ["loads_t", "generators_t", "storage_units_t"]:
		comp = getattr(network, comp_name, None)
		if comp is None:
			continue
		for attr in ["p_set", "q_set"]:
			df = getattr(comp, attr, None)
			if isinstance(df, pd.DataFrame) and not df.empty:
				setattr(comp, attr, df * scale_factor)

	for comp_name in ["loads", "generators", "storage_units"]:
		comp = getattr(network, comp_name, None)
		if comp is None or comp.empty:
			continue
		for attr in ["p_set", "q_set"]:
			if attr in comp.columns:
				comp[attr] = comp[attr] * scale_factor


def run_pf_ramp_test(  # noqa: PLR0912
	network: pypsa.Network,
	snapshots: Iterable,
	start: float,
	stop: float,
	step: float,
	report_path: Path | None,
) -> None:
	print_header("6) PF load ramp test")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return

	if step <= 0:
		print("Invalid ramp step. It must be > 0.")
		return

	factors: list[float] = []
	value = start
	while value <= stop + 1e-12:
		factors.append(round(value, 6))
		value += step

	if not factors:
		print("No ramp factors generated.")
		return

	rows: list[dict] = []
	first_failed_factor: float | None = None

	for factor in factors:
		n = network.copy()
		_scale_time_series_values(n, factor)

		status = "ok"
		all_ok = False
		failed_snapshots: list[str] = []
		max_error = np.nan

		try:
			n.lpf(snapshots=snapshots)
			pf_result = n.pf(snapshots=snapshots, use_seed=True)
			all_ok, converged, error = _extract_pf_series(pf_result)
			failed_snapshots = [str(idx) for idx, ok in converged.items() if not bool(ok)]
			if not error.empty:
				max_error = float(pd.to_numeric(error, errors="coerce").max())
		except Exception as exc:
			status = f"exception: {exc}"

		if not all_ok and first_failed_factor is None:
			first_failed_factor = factor

		rows.append(
			{
				"scale_factor": factor,
				"all_converged": all_ok,
				"status": status,
				"failed_snapshot_count": len(failed_snapshots),
				"failed_snapshots": ";".join(failed_snapshots),
				"max_final_error": max_error,
			}
		)

	report = pd.DataFrame(rows)
	print(report.to_string(index=False))

	if first_failed_factor is None:
		print("Ramp result: PF converged for all tested scale factors.")
	else:
		print(f"Ramp result: first non-converging scale factor = {first_failed_factor}")

	if report_path is not None:
		report_path.parent.mkdir(parents=True, exist_ok=True)
		report.to_csv(report_path, index=False)
		print(f"Saved ramp report to: {report_path}")


def run_pf_checks(network: pypsa.Network, snapshots: Iterable) -> None:
	print_header("5) Non-linear PF convergence checks")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return

	try:
		result = network.pf(snapshots=snapshots, use_seed=False)
		_report_pf_result("PF without seed", result)
	except Exception as exc:
		print(f"PF without seed: failed -> {exc}")

	try:
		network.lpf(snapshots=snapshots)
		result = network.pf(snapshots=snapshots, use_seed=True)
		_report_pf_result("PF with LPF seed", result)
	except Exception as exc:
		print(f"PF with LPF seed: failed -> {exc}")


def run_optimize_smoke_test(network: pypsa.Network, snapshots: Iterable, solver_name: str) -> None:
	print_header("7) Optimization smoke test (optional)")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return

	network_small = network.copy()
	try:
		result = network_small.optimize(
			snapshots=snapshots,
			solver_name=solver_name,
			solver="ipm",
			run_crossover="off",
			random_seed=123,
		)
		print(f"Optimization result: {result}")
	except TypeError:
		# Fallback for PyPSA versions with different optimize signatures.
		try:
			result = network_small.optimize(snapshots=snapshots, solver_name=solver_name)
			print(f"Optimization result (fallback call): {result}")
		except Exception as exc:
			print(f"Optimization failed (fallback call): {exc}")
	except Exception as exc:
		print(f"Optimization failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Physical feasibility checks (research-grade, LV-oriented)
# ─────────────────────────────────────────────────────────────────────────────

def _total_injection_per_snapshot(
	comp: pd.DataFrame,
	comp_t,
	attr: str,
	snapshots: list,
) -> pd.Series:
	"""Sum a setpoint attribute across all components for each snapshot.

	Combines time-varying values from comp_t with static values from comp for
	components not represented in the time series.
	"""
	result = pd.Series(0.0, index=snapshots, dtype=float)
	ts = getattr(comp_t, attr, None)
	ts_components: set = set()
	if isinstance(ts, pd.DataFrame) and not ts.empty:
		valid_snaps = [s for s in snapshots if s in ts.index]
		if valid_snaps:
			ts_valid = ts.reindex(snapshots).fillna(0.0)
			numeric = ts_valid.apply(pd.to_numeric, errors="coerce").fillna(0.0)
			result = result.add(numeric.sum(axis=1), fill_value=0.0)
		ts_components = set(ts.columns)
	if not comp.empty and attr in comp.columns:
		static = comp.loc[~comp.index.isin(ts_components), attr]
		static_val = float(pd.to_numeric(static, errors="coerce").fillna(0.0).sum())
		result = result + static_val
	return result


def run_voltage_violation_check(
	network: pypsa.Network,
	snapshots: Iterable,
	v_min: float = 0.90,
	v_max: float = 1.10,
	run_pf: bool = True,
) -> pd.DataFrame:
	"""Check per-bus, per-snapshot voltage limit violations after non-linear PF.

	Returns a DataFrame with one row per bus: n_below_vmin, n_above_vmax,
	n_violations, violation_ratio, worst_v_mag_pu, worst_snapshot, pass_flag.
	"""
	print_header("8) Voltage violation check")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return pd.DataFrame()

	if run_pf:
		try:
			network.lpf(snapshots=snapshots)
			network.pf(snapshots=snapshots, use_seed=True)
		except Exception as exc:
			print(f"WARNING: PF failed during voltage check: {exc}")
			print("Results may be based on pre-convergence state.")

	v_mag = getattr(network.buses_t, "v_mag_pu", pd.DataFrame())
	if v_mag.empty:
		print("No v_mag_pu time series available (PF not run or failed).")
		return pd.DataFrame()

	available_snaps = [s for s in snapshots if s in v_mag.index]
	if not available_snaps:
		print("No matching snapshots in v_mag_pu.")
		return pd.DataFrame()

	v_sub = v_mag.loc[available_snaps]
	n_snaps = len(available_snaps)

	rows = []
	for bus in v_sub.columns:
		v_series = pd.to_numeric(v_sub[bus], errors="coerce")
		n_below = int((v_series < v_min).sum())
		n_above = int((v_series > v_max).sum())
		n_viol = n_below + n_above
		worst_idx = v_series.sub(1.0).abs().idxmax()
		worst_v = float(v_series[worst_idx])
		rows.append({
			"bus": bus,
			"n_below_vmin": n_below,
			"n_above_vmax": n_above,
			"n_violations": n_viol,
			"violation_ratio": round(n_viol / max(n_snaps, 1), 4),
			"worst_v_mag_pu": round(worst_v, 5),
			"worst_snapshot": str(worst_idx),
			"pass_flag": "FAIL" if n_viol > 0 else "PASS",
		})

	df = pd.DataFrame(rows)
	n_buses_failing = int((df["n_violations"] > 0).sum())
	n_persistent = int((df["violation_ratio"] >= 0.5).sum())
	print(f"Voltage bands: v_min={v_min} pu, v_max={v_max} pu | Snapshots analyzed: {n_snaps}")
	print(f"Buses with any violation:     {n_buses_failing} / {len(df)}")
	print(f"  Below v_min (>=1 snapshot): {int((df['n_below_vmin'] > 0).sum())}")
	print(f"  Above v_max (>=1 snapshot): {int((df['n_above_vmax'] > 0).sum())}")
	print(f"  Persistent (>=50% snaps):   {n_persistent}")
	if n_buses_failing > 0:
		print("\nTop voltage violating buses:")
		top = df[df["n_violations"] > 0].nlargest(10, "n_violations")
		print(top[["bus", "n_violations", "violation_ratio", "worst_v_mag_pu"]].to_string(index=False))
	if n_buses_failing == 0:
		print("\nRESULT: PASS - No voltage limit violations.")
	elif n_persistent > 0:
		print(f"\nRESULT: FAIL - {n_persistent} bus(es) persistently outside voltage limits.")
	else:
		print(f"\nRESULT: WARNING - {n_buses_failing} bus(es) have transient voltage violations.")
	return df


def run_branch_loading_check(
	network: pypsa.Network,
	snapshots: Iterable,
	s_max_warn: float = 0.80,
	s_max_fail: float = 1.00,
	run_pf: bool = True,
) -> pd.DataFrame:
	"""Check thermal loading for lines and transformers against s_nom.

	Apparent power is computed as sqrt(P^2 + Q^2) when Q flows are available
	from a completed non-linear PF; otherwise |P| is used conservatively.

	Returns a DataFrame with one row per branch: kind, name, bus0, bus1,
	s_nom_mva, max_loading_pu, mean_loading_pu, n_warn, n_fail, pass_flag.
	"""
	print_header("9) Branch thermal loading check")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return pd.DataFrame()

	if run_pf:
		try:
			network.lpf(snapshots=snapshots)
			network.pf(snapshots=snapshots, use_seed=True)
		except Exception as exc:
			print(f"WARNING: PF failed during loading check: {exc}")

	rows: list[dict] = []

	def _process(comp: pd.DataFrame, comp_t, kind: str) -> None:
		if comp.empty:
			return
		p0_ts = getattr(comp_t, "p0", pd.DataFrame())
		q0_ts = getattr(comp_t, "q0", pd.DataFrame())
		available_snaps = [s for s in snapshots if s in p0_ts.index] if not p0_ts.empty else []

		for name in comp.index:
			s_nom_raw = comp.at[name, "s_nom"] if "s_nom" in comp.columns else np.nan
			s_nom = float(pd.to_numeric(s_nom_raw, errors="coerce")) if pd.notna(s_nom_raw) else np.nan
			b0 = comp.at[name, "bus0"] if "bus0" in comp.columns else ""
			b1 = comp.at[name, "bus1"] if "bus1" in comp.columns else ""

			if pd.isna(s_nom) or s_nom <= 0:
				rows.append({"kind": kind, "name": str(name), "bus0": b0, "bus1": b1,
					"s_nom_mva": np.nan, "max_loading_pu": np.nan, "mean_loading_pu": np.nan,
					"n_warn": 0, "n_fail": 0, "pass_flag": "NO_RATING"})
				continue

			if not available_snaps or name not in p0_ts.columns:
				rows.append({"kind": kind, "name": str(name), "bus0": b0, "bus1": b1,
					"s_nom_mva": round(s_nom, 4), "max_loading_pu": np.nan, "mean_loading_pu": np.nan,
					"n_warn": 0, "n_fail": 0, "pass_flag": "NO_FLOW_DATA"})
				continue

			p_series = pd.to_numeric(p0_ts.loc[available_snaps, name], errors="coerce").fillna(0.0)
			if not q0_ts.empty and name in q0_ts.columns:
				q_series = pd.to_numeric(q0_ts.loc[available_snaps, name], errors="coerce").fillna(0.0)
				s_series = np.sqrt(p_series**2 + q_series**2)
			else:
				s_series = p_series.abs()

			loading = s_series / s_nom
			max_l = float(loading.max())
			mean_l = float(loading.mean())
			n_warn = int((loading >= s_max_warn).sum())
			n_fail = int((loading >= s_max_fail).sum())
			flag = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")
			rows.append({"kind": kind, "name": str(name), "bus0": b0, "bus1": b1,
				"s_nom_mva": round(s_nom, 4), "max_loading_pu": round(max_l, 4),
				"mean_loading_pu": round(mean_l, 4), "n_warn": n_warn, "n_fail": n_fail,
				"pass_flag": flag})

	_process(network.lines, network.lines_t, "line")
	_process(network.transformers, network.transformers_t, "transformer")

	df = pd.DataFrame(rows)
	if df.empty:
		print("No branches to check.")
		return df

	no_rating = int((df["pass_flag"] == "NO_RATING").sum())
	no_data = int((df["pass_flag"] == "NO_FLOW_DATA").sum())
	n_fail = int((df["pass_flag"] == "FAIL").sum())
	n_warn = int((df["pass_flag"] == "WARN").sum())
	valid = df.dropna(subset=["max_loading_pu"])

	print(f"Loading thresholds: WARN at {s_max_warn:.0%}, FAIL at {s_max_fail:.0%} of s_nom")
	print(f"Branches: {len(df)} total | {no_rating} no s_nom rating | {no_data} no flow data")
	if not valid.empty:
		print(f"Max loading (rated+flow data): {valid['max_loading_pu'].max():.3f} pu")
	print(f"FAIL (>= s_max_fail): {n_fail} | WARN (>= {s_max_warn:.0%}): {n_warn}")
	if not valid.empty:
		top = valid.nlargest(10, "max_loading_pu")
		print("\nTop loaded branches:")
		print(top[["kind", "name", "s_nom_mva", "max_loading_pu", "n_fail", "pass_flag"]].to_string(index=False))
	if n_fail == 0 and n_warn == 0:
		print("\nRESULT: PASS - No overloaded branches.")
	elif n_fail > 0:
		print(f"\nRESULT: FAIL - {n_fail} branch(es) exceed thermal rating.")
	else:
		print(f"\nRESULT: WARNING - {n_warn} branch(es) exceed warning threshold.")
	return df


def run_nodal_balance_check(
	network: pypsa.Network,
	snapshots: Iterable,
	tol_ratio: float = 0.05,
) -> pd.DataFrame:
	"""Check system-level active power balance per snapshot.

	Computes total generation (generators + net storage) vs total load demand
	using setpoint data. Flags snapshots where |gen - load| / load > tol_ratio.
	This is a model-data quality check, independent of PF convergence.

	Returns a DataFrame with one row per snapshot.
	"""
	print_header("10) Nodal active power balance check")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return pd.DataFrame()

	gen_p = _total_injection_per_snapshot(network.generators, network.generators_t, "p_set", snapshots)
	load_p = _total_injection_per_snapshot(network.loads, network.loads_t, "p_set", snapshots)
	# For storage: positive p_set = charging (withdrawing from grid), negative = discharging
	stor_p = _total_injection_per_snapshot(network.storage_units, network.storage_units_t, "p_set", snapshots)

	rows = []
	for snap in snapshots:
		g = float(gen_p.get(snap, 0.0) or 0.0)
		l = float(load_p.get(snap, 0.0) or 0.0)
		s = float(stor_p.get(snap, 0.0) or 0.0)
		# Net generation supply = generators - storage net charging
		net_gen = g - s
		imbalance = net_gen - l
		if abs(l) > 1e-9:
			imbalance_ratio = abs(imbalance) / abs(l)
		else:
			imbalance_ratio = abs(imbalance)
		rows.append({
			"snapshot": str(snap),
			"total_gen_mw": round(g, 4),
			"total_load_mw": round(l, 4),
			"net_storage_mw": round(s, 4),
			"net_generation_mw": round(net_gen, 4),
			"imbalance_mw": round(imbalance, 4),
			"imbalance_ratio": round(imbalance_ratio, 4),
			"pass_flag": "FAIL" if imbalance_ratio > tol_ratio else "PASS",
		})

	df = pd.DataFrame(rows)
	n_fail = int((df["pass_flag"] == "FAIL").sum())
	max_imbalance = float(df["imbalance_ratio"].max()) if not df.empty else 0.0
	print(f"Balance tolerance: {tol_ratio:.0%} of total load | Snapshots: {len(df)}")
	print(f"Max imbalance ratio: {max_imbalance:.3%} | Snapshots failing: {n_fail} / {len(df)}")
	print("\nBalance per snapshot:")
	print(df[["snapshot", "total_gen_mw", "total_load_mw", "imbalance_mw", "pass_flag"]].to_string(index=False))
	if n_fail == 0:
		print("\nRESULT: PASS - Active power dispatch is balanced within tolerance.")
	else:
		print(f"\nRESULT: WARNING - {n_fail} snapshot(s) show generation-demand imbalance.")
		print("  Check for missing generators/loads or inconsistent setpoint data.")
	return df


def run_reactive_power_consistency_check(
	network: pypsa.Network,
	snapshots: Iterable,
	q_tol_pu: float = 0.05,
	run_pf: bool = True,
) -> pd.DataFrame:
	"""Check reactive power output against generator Q limits after non-linear PF.

	Detects generators that are chronically saturated at their Q limits, which
	indicates insufficient reactive capability or reactive demand misconfiguration.

	Returns a DataFrame with one row per generator that has Q data.
	"""
	print_header("11) Reactive power / Q-limit consistency check")
	snapshots = list(snapshots)
	if not snapshots:
		print("No snapshots available.")
		return pd.DataFrame()

	if run_pf:
		try:
			network.lpf(snapshots=snapshots)
			network.pf(snapshots=snapshots, use_seed=True)
		except Exception as exc:
			print(f"WARNING: PF failed during Q check: {exc}")

	q_ts = getattr(network.generators_t, "q", pd.DataFrame())
	if q_ts.empty:
		print("No generators_t.q time series available (requires completed non-linear PF).")
		print("RESULT: SKIPPED - Q consistency check requires post-PF reactive power data.")
		return pd.DataFrame()

	available_snaps = [s for s in snapshots if s in q_ts.index]
	if not available_snaps:
		print("No matching snapshots in generators_t.q.")
		return pd.DataFrame()

	q_sub = q_ts.loc[available_snaps]
	gens = network.generators

	rows = []
	for gen in q_sub.columns:
		if gen not in gens.index:
			continue
		q_series = pd.to_numeric(q_sub[gen], errors="coerce").dropna()
		if q_series.empty:
			continue
		n_snaps = len(q_series)

		# Resolve Q limits — try per-unit columns first, then absolute
		def _get_limit(col_pu, col_abs):
			if col_pu in gens.columns:
				v = gens.at[gen, col_pu]
			elif col_abs in gens.columns:
				v = gens.at[gen, col_abs]
			else:
				return np.nan
			return float(pd.to_numeric(v, errors="coerce")) if pd.notna(v) else np.nan

		q_min = _get_limit("q_min_pu", "q_min")
		q_max = _get_limit("q_max_pu", "q_max")

		q_mean = float(q_series.mean())
		q_abs_max = float(q_series.abs().max())

		tol_min = q_tol_pu * abs(q_min) if not np.isnan(q_min) else 0.0
		tol_max = q_tol_pu * abs(q_max) if not np.isnan(q_max) else 0.0
		n_at_qmin = int((q_series <= (q_min + tol_min)).sum()) if not np.isnan(q_min) else 0
		n_at_qmax = int((q_series >= (q_max - tol_max)).sum()) if not np.isnan(q_max) else 0

		flags = []
		if n_at_qmin > 0:
			flags.append("AT_Q_MIN")
		if n_at_qmax > 0:
			flags.append("AT_Q_MAX")
		pass_flag = ";".join(flags) if flags else "PASS"

		rows.append({
			"generator": gen,
			"q_min": round(q_min, 4) if not np.isnan(q_min) else np.nan,
			"q_max": round(q_max, 4) if not np.isnan(q_max) else np.nan,
			"q_mean": round(q_mean, 4),
			"q_abs_max": round(q_abs_max, 4),
			"n_at_qmin": n_at_qmin,
			"n_at_qmax": n_at_qmax,
			"saturation_ratio": round((n_at_qmin + n_at_qmax) / max(n_snaps, 1), 4),
			"pass_flag": pass_flag,
		})

	df = pd.DataFrame(rows)
	if df.empty:
		print("No generators with Q data to check.")
		return df

	n_flagged = int((df["pass_flag"] != "PASS").sum())
	n_persistent = int((df["saturation_ratio"] >= 0.5).sum())
	print(f"Q limit proximity tolerance: ±{q_tol_pu} pu | Generators with Q data: {len(df)}")
	print(f"With any Q limit event: {n_flagged} | Persistently saturated (>=50% snaps): {n_persistent}")
	if n_flagged > 0:
		top = df[df["pass_flag"] != "PASS"].nlargest(10, "saturation_ratio")
		print("\nGenerators with Q limit events:")
		print(top[["generator", "q_min", "q_max", "q_mean", "n_at_qmin", "n_at_qmax", "pass_flag"]].to_string(index=False))
	if n_flagged == 0:
		print("\nRESULT: PASS - No Q limit saturation detected.")
	elif n_persistent > 0:
		print(f"\nRESULT: WARNING - {n_persistent} generator(s) persistently saturated at Q limits.")
		print("  This may indicate insufficient reactive power capability for this network state.")
	else:
		print(f"\nRESULT: INFO - {n_flagged} generator(s) occasionally at Q limits (often normal).")
	return df


def write_run_summary(
	path: Path,
	snapshots: Iterable,
	thresholds: dict,
	v_result: pd.DataFrame,
	loading_result: pd.DataFrame,
	balance_result: pd.DataFrame,
	q_result: pd.DataFrame,
) -> None:
	"""Write machine-readable JSON run summary for research reproducibility.

	Records timestamp, thresholds used, snapshot IDs, and pass/fail outcome
	per check so results can be unambiguously re-associated with model inputs.
	"""
	import json
	from datetime import datetime, timezone

	snapshots = list(snapshots)

	def _v_summary(df: pd.DataFrame) -> dict:
		if df.empty:
			return {"status": "no_data"}
		n_viol = int((df["n_violations"] > 0).sum())
		n_pers = int((df["violation_ratio"] >= 0.5).sum())
		worst = float(df["worst_v_mag_pu"].sub(1.0).abs().max()) if "worst_v_mag_pu" in df else None
		return {"buses_violating": n_viol, "persistent_violators": n_pers,
			"worst_v_mag_pu": round(worst, 5) if worst is not None else None,
			"pass": n_viol == 0}

	def _loading_summary(df: pd.DataFrame) -> dict:
		if df.empty:
			return {"status": "no_data"}
		n_fail = int((df["pass_flag"] == "FAIL").sum()) if "pass_flag" in df else 0
		n_warn = int((df["pass_flag"] == "WARN").sum()) if "pass_flag" in df else 0
		valid = df.dropna(subset=["max_loading_pu"]) if "max_loading_pu" in df.columns else pd.DataFrame()
		worst = float(valid["max_loading_pu"].max()) if not valid.empty else None
		return {"branches_fail": n_fail, "branches_warn": n_warn,
			"worst_loading_pu": round(worst, 4) if worst is not None else None,
			"pass": n_fail == 0}

	def _balance_summary(df: pd.DataFrame) -> dict:
		if df.empty:
			return {"status": "no_data"}
		n_fail = int((df["pass_flag"] == "FAIL").sum()) if "pass_flag" in df else 0
		max_imb = float(df["imbalance_ratio"].max()) if "imbalance_ratio" in df.columns else None
		return {"snapshots_fail": n_fail,
			"max_imbalance_ratio": round(max_imb, 4) if max_imb is not None else None,
			"pass": n_fail == 0}

	def _q_summary(df: pd.DataFrame) -> dict:
		if df.empty:
			return {"status": "no_data_or_skipped"}
		n_sat = int((df["pass_flag"] != "PASS").sum()) if "pass_flag" in df else 0
		n_pers = int((df["saturation_ratio"] >= 0.5).sum()) if "saturation_ratio" in df else 0
		return {"generators_with_limit_events": n_sat, "persistently_saturated": n_pers,
			"pass": n_pers == 0}

	vs = _v_summary(v_result)
	ls = _loading_summary(loading_result)
	bs = _balance_summary(balance_result)
	qs = _q_summary(q_result)
	overall = all(d.get("pass", True) for d in [vs, ls, bs, qs])

	summary = {
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"snapshots_analyzed": [str(s) for s in snapshots],
		"n_snapshots": len(snapshots),
		"thresholds": thresholds,
		"overall_pass": overall,
		"checks": {
			"voltage_violations": vs,
			"branch_loading": ls,
			"nodal_balance": bs,
			"reactive_power": qs,
		},
	}
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w") as f:
		json.dump(summary, f, indent=2)
	print(f"Run summary written to: {path}")


def run_physical_checks(
	network: pypsa.Network,
	snapshots: Iterable,
	v_min: float = 0.90,
	v_max: float = 1.10,
	s_max_warn: float = 0.80,
	s_max_fail: float = 1.00,
	tol_ratio: float = 0.05,
	q_tol_pu: float = 0.05,
	report_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
	"""Run all four physical feasibility checks with a single shared PF run.

	Executes LPF-seeded non-linear PF once, then passes the post-PF network
	to voltage, loading, and Q checks. The balance check uses setpoint data
	and does not depend on PF results.

	Returns a dict with keys: 'voltage', 'loading', 'balance', 'q_consistency'.
	"""
	print_header("Physical Feasibility Checks (Voltage / Loading / Balance / Q)")
	snapshots = list(snapshots)

	pf_ok = False
	try:
		network.lpf(snapshots=snapshots)
		pf_result = network.pf(snapshots=snapshots, use_seed=True)
		all_ok, _, _ = _extract_pf_series(pf_result)
		pf_ok = all_ok
		status = "converged for all snapshots" if pf_ok else "did NOT converge for all snapshots"
		print(f"PF {status}. Physical check results {'are reliable.' if pf_ok else 'may be unreliable for non-converged snapshots.'}")
	except Exception as exc:
		print(f"WARNING: PF failed: {exc}. Physical checks proceeding on available data.")

	v_df = run_voltage_violation_check(network, snapshots, v_min, v_max, run_pf=False)
	loading_df = run_branch_loading_check(network, snapshots, s_max_warn, s_max_fail, run_pf=False)
	balance_df = run_nodal_balance_check(network, snapshots, tol_ratio)
	q_df = run_reactive_power_consistency_check(network, snapshots, q_tol_pu, run_pf=False)

	thresholds = {
		"v_min_pu": v_min, "v_max_pu": v_max,
		"loading_warn_pu": s_max_warn, "loading_fail_pu": s_max_fail,
		"balance_tol_ratio": tol_ratio, "q_tol_pu": q_tol_pu,
	}

	if report_dir is not None:
		rd = Path(report_dir)
		write_run_summary(rd / "run_summary.json", snapshots, thresholds, v_df, loading_df, balance_df, q_df)
		if not v_df.empty:
			v_df.to_csv(rd / "voltage_violations.csv", index=False)
		if not loading_df.empty:
			loading_df.to_csv(rd / "branch_loading.csv", index=False)
		if not balance_df.empty:
			balance_df.to_csv(rd / "nodal_balance.csv", index=False)
		if not q_df.empty:
			q_df.to_csv(rd / "q_consistency.csv", index=False)

	return {"voltage": v_df, "loading": loading_df, "balance": balance_df, "q_consistency": q_df}


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="PyPSA convergence diagnostics")
	parser.add_argument(
		"--csv-folder",
		type=Path,
		default=Path(__file__).parent / "network_postprocessed",
		help="Path to network CSV folder",
	)
	parser.add_argument(
		"--n-snapshots",
		type=int,
		default=4,
		help="Number of initial snapshots to run checks on",
	)
	parser.add_argument(
		"--run-localize",
		action="store_true",
		help="Run network localization (switch clusters + diverging bus trace).",
	)
	parser.add_argument(
		"--z-threshold",
		type=float,
		default=1e-3,
		help="Impedance threshold in Ohm for classifying a line as switch-like.",
	)
	parser.add_argument(
		"--localize-csv",
		type=Path,
		default=Path(__file__).parent / "diagnostics" / "localization_report.csv",
		help="CSV path for localization report.",
	)
	parser.add_argument(
		"--run-optimize",
		action="store_true",
		help="Run optional optimization smoke test",
	)
	parser.add_argument(
		"--solver",
		type=str,
		default="highs",
		help="Solver name for optional optimization smoke test",
	)
	parser.add_argument(
		"--run-ramp-test",
		action="store_true",
		help="Run PF load ramp test from --ramp-start to --ramp-stop.",
	)
	parser.add_argument(
		"--ramp-start",
		type=float,
		default=0.1,
		help="Ramp start factor for p_set/q_set scaling.",
	)
	parser.add_argument(
		"--ramp-stop",
		type=float,
		default=1.0,
		help="Ramp stop factor for p_set/q_set scaling.",
	)
	parser.add_argument(
		"--ramp-step",
		type=float,
		default=0.1,
		help="Ramp step factor for p_set/q_set scaling.",
	)
	parser.add_argument(
		"--report-csv",
		type=Path,
		default=Path(__file__).parent / "diagnostics" / "pf_ramp_report.csv",
		help="Path to export PF ramp diagnostics CSV.",
	)
	parser.add_argument(
		"--run-physical",
		action="store_true",
		help="Run physical feasibility checks (voltage, loading, balance, Q).",
	)
	parser.add_argument("--v-min", type=float, default=0.90, help="Voltage lower limit [pu].")
	parser.add_argument("--v-max", type=float, default=1.10, help="Voltage upper limit [pu].")
	parser.add_argument("--s-max-warn", type=float, default=0.80, help="Loading warning threshold [pu of s_nom].")
	parser.add_argument("--s-max-fail", type=float, default=1.00, help="Loading fail threshold [pu of s_nom].")
	parser.add_argument("--balance-tol", type=float, default=0.05, help="Balance imbalance ratio tolerance.")
	parser.add_argument("--q-tol", type=float, default=0.05, help="Q limit proximity tolerance [pu].")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	csv_folder = args.csv_folder

	if not csv_folder.exists():
		raise FileNotFoundError(f"Network folder not found: {csv_folder}")

	network = load_network(csv_folder)
	snapshots = network.snapshots[: max(0, args.n_snapshots)]

	print_header("Loaded network")
	print(f"Path: {csv_folder}")
	print(f"Buses={len(network.buses)} Lines={len(network.lines)} Transformers={len(network.transformers)}")
	print(f"Generators={len(network.generators)} Loads={len(network.loads)} StorageUnits={len(network.storage_units)}")
	print(f"Total snapshots={len(network.snapshots)} Checked snapshots={len(snapshots)}")

	run_consistency_check(network)
	run_data_sanity_checks(network)
	run_lpf_angle_check(network, snapshots)
	run_structural_checks(network)
	run_pf_checks(network, snapshots)

	if args.run_localize:
		run_network_localization(
			network,
			snapshots,
			z_threshold=args.z_threshold,
			report_path=args.localize_csv,
		)

	if args.run_ramp_test:
		run_pf_ramp_test(
			network,
			snapshots,
			start=args.ramp_start,
			stop=args.ramp_stop,
			step=args.ramp_step,
			report_path=args.report_csv,
		)

	if args.run_optimize:
		run_optimize_smoke_test(network, snapshots, solver_name=args.solver)

	if args.run_physical:
		run_physical_checks(
			network,
			snapshots,
			v_min=args.v_min,
			v_max=args.v_max,
			s_max_warn=args.s_max_warn,
			s_max_fail=args.s_max_fail,
			tol_ratio=args.balance_tol,
			q_tol_pu=args.q_tol,
			report_dir=Path(__file__).parent / "diagnostics",
		)


if __name__ == "__main__":
	main()

	
