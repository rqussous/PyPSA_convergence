# PyPSA Convergence Diagnostics

This repository contains a lightweight Streamlit interface and supporting diagnostics code for analyzing PyPSA network convergence, localization of problematic areas, ramp-test stability, and physical feasibility checks.

## Files

- `streamlit_app.py`: Streamlit user interface for running and reviewing diagnostics.
- `network.py`: diagnostic logic and report generation.

## What the app does

- Runs core consistency and power flow checks.
- Highlights likely problem areas in the network.
- Runs a ramp test to identify stress-related convergence failures.
- Separates physical feasibility results into dedicated tabs for voltage, loading, balance, and Q-limit issues.

## Local setup

Use a Python environment with the packages required by your PyPSA workflow, then run:

```bash
python -m streamlit run streamlit_app.py
```

If you are using a dedicated Conda environment, activate it first and then launch the same command.

## Streamlit Community Cloud deployment

- Set Python to `3.11` in the app's Advanced settings when deploying.
- Keep dependencies in `requirements.txt` at the repository root.
- The app now lazy-loads network data: choose a folder and click **Load** in the sidebar before running checks.
- In cloud/Linux deployments, the native **Browse** folder picker is disabled; use zip upload or a manual path.

## Repository scope

This repository intentionally excludes processed network data and generated diagnostics artifacts from version control.

Ignored paths include:

- `network_postprocessed/`
- `diagnostics/`
- local environment and editor folders such as `.conda/`, `.vscode/`, and `__pycache__/`

## Typical workflow

1. Open the app.
2. Select the network CSV folder.
3. Run checks from the Overview tab.
4. Review separated results in the Physical Checks tab.
5. Export diagnostics files locally when needed.

## Notes

- The repository is meant to contain code and lightweight project metadata, not large derived datasets.
- If you need to share sample data later, add a small sanitized example dataset rather than committing the full processed network folder.

## Co-Author

Co-authored with GitHub Copilot (GPT-5.4).