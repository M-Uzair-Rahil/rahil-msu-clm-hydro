import re
import shutil
import time
from pathlib import Path
from importlib.resources import files


def _print_welcome(Ninit: int, nP: int, nN: int, do_fmax: bool) -> None:
    print("=" * 72)
    print("RMCH | Rahil MSU CLM Hydro")
    print("-" * 72)
    print(f"LHS Samples        : {Ninit}")
    print(f"Hydro P parameters : {nP}")
    print(f"Hydro N parameters : {nN}")
    print(f"FMAX perturbation  : {'Enabled' if do_fmax else 'Disabled'}")
    print("-" * 72)
    print("Initializing ensemble generation...\n")


def lhs(n_samples: int, n_dim: int, rng) -> "object":
    """
    Generate a Latin Hypercube Sampling design in [0, 1].
    """
    import numpy as np

    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.random((n_samples, n_dim))
    a = cut[:-1]
    b = cut[1:]
    h = u * (b - a)[:, None] + a[:, None]
    for j in range(n_dim):
        rng.shuffle(h[:, j])
    return h


def parse_prior_range(x):
    """
    Parse a prior range string like '[0.1, 5]' or '0.1-5'.
    """
    import pandas as pd

    if pd.isna(x):
        return None
    s = str(x).strip()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def find_col(df, candidates):
    """
    Find the first matching column from a candidate list.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def assign_scalar_everywhere(ds, varname: str, value: float):
    """
    Overwrite a variable everywhere with a scalar while preserving dtype and attrs.
    """
    import numpy as np
    import xarray as xr

    var = ds[varname]
    dt = var.dtype
    v = int(np.rint(value)) if np.issubdtype(dt, np.integer) else float(value)
    data = np.full(var.shape, v, dtype=dt)
    ds[varname] = xr.DataArray(data, dims=var.dims, coords=var.coords, attrs=var.attrs)
    return ds


def _load_internal_hydro_csv():
    """
    Load strf_parameters.csv bundled inside the installed package.
    """
    import pandas as pd

    csv_path = files("rahil.data").joinpath("strf_parameters.csv")
    hyd = pd.read_csv(csv_path)
    hyd.columns = [c.strip() for c in hyd.columns]
    return hyd


def generate_lhs(
    Ninit: int = 70,
    seed: int = 42,
    base_surf_dir: str | None = None,
    param_nc_dir: str | None = None,
    output_dir: str = "Calibration/combined",
    location: str = "pe_hydrology",
    iteration: int = 0,
    do_fmax: bool = True,
    fmax_scale_min: float = 0.7,
    fmax_scale_max: float = 1.3,
):
    """
    Generate CLM hydrology calibration ensembles using Latin Hypercube Sampling.

    Parameters
    ----------
    Ninit : int
        Number of initial LHS samples.
    seed : int
        Random seed.
    base_surf_dir : str or None
        Path to base surfdata NetCDF. Required if do_fmax=True.
    param_nc_dir : str
        Path to base CLM parameter NetCDF.
    output_dir : str
        Base output directory.
    location : str
        Label used in generated case IDs and workflow folders.
    iteration : int
        Iteration number.
    do_fmax : bool
        Whether to perturb FMAX in surfdata.
    fmax_scale_min : float
        Minimum FMAX scaling factor.
    fmax_scale_max : float
        Maximum FMAX scaling factor.

    Returns
    -------
    dict
        Dictionary containing key output paths and the generated parameter table.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    from netCDF4 import Dataset
    from tqdm.auto import tqdm

    start_time = time.time()

    if Ninit <= 0:
        raise ValueError("Ninit must be greater than 0.")

    if param_nc_dir is None:
        raise ValueError("param_nc_dir must be provided.")

    if do_fmax and not base_surf_dir:
        raise ValueError("base_surf_dir must be provided when do_fmax=True.")

    if fmax_scale_min > fmax_scale_max:
        raise ValueError("fmax_scale_min must be less than or equal to fmax_scale_max.")

    base_paramfile = Path(param_nc_dir)
    base_surfdata = Path(base_surf_dir) if base_surf_dir else None
    out_base = Path(output_dir) / location / f"iter_{iteration}"

    if not base_paramfile.exists():
        raise FileNotFoundError(f"Base parameter file not found: {base_paramfile}")

    if do_fmax and (base_surfdata is None or not base_surfdata.exists()):
        raise FileNotFoundError(f"Base surfdata file not found: {base_surfdata}")

    combined_param_dir = out_base / "paramfile_combined"
    namelist_txt_dir = out_base / "namelist_txt"
    workflow_dir = out_base / "workflow"
    surfdata_out_dir = out_base / "surfdata_ensemble"

    combined_param_dir.mkdir(parents=True, exist_ok=True)
    namelist_txt_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    if do_fmax:
        surfdata_out_dir.mkdir(parents=True, exist_ok=True)

    hyd = _load_internal_hydro_csv()

    hyd_param_col = find_col(hyd, ["parameter name", "Parameters", "parameter", "Parameter", "name", "Name"])
    hyd_loc_col = find_col(hyd, ["location", "Location", "LOC"])
    hyd_min_col = find_col(hyd, ["min", "Min", "minimum", "Minimum"])
    hyd_max_col = find_col(hyd, ["max", "Max", "maximum", "Maximum"])
    hyd_prior_col = find_col(hyd, ["Prior range", "prior range", "prior", "range", "Range"])

    if hyd_param_col is None or hyd_loc_col is None:
        raise KeyError(
            f"Hydro CSV must contain parameter and location columns. Found: {list(hyd.columns)}"
        )

    hyd[hyd_param_col] = hyd[hyd_param_col].astype(str).str.strip()
    hyd[hyd_loc_col] = hyd[hyd_loc_col].astype(str).str.strip().str.upper()

    mins, maxs = [], []
    for i, row in hyd.iterrows():
        mn = row[hyd_min_col] if hyd_min_col else np.nan
        mx = row[hyd_max_col] if hyd_max_col else np.nan

        if (pd.isna(mn) or pd.isna(mx)) and hyd_prior_col is not None:
            pr = parse_prior_range(row[hyd_prior_col])
            if pr is not None:
                mn, mx = pr

        pname = str(row[hyd_param_col]).strip()

        if pname.upper() == "FMAX":
            mins.append(np.nan)
            maxs.append(np.nan)
            continue

        if pd.isna(mn) or pd.isna(mx):
            raise ValueError(
                f"Missing bounds for hydro row {i}, param={pname}. "
                "Provide min/max or Prior range."
            )

        mins.append(float(mn))
        maxs.append(float(mx))

    hyd["__min__"] = mins
    hyd["__max__"] = maxs

    hyd_P = hyd[(hyd[hyd_loc_col] == "P") & (hyd[hyd_param_col].str.upper() != "FMAX")].copy()
    hyd_N = hyd[(hyd[hyd_loc_col] == "N") & (hyd[hyd_param_col].str.upper() != "FMAX")].copy()

    hydP_params = hyd_P[hyd_param_col].tolist()
    hydN_params = hyd_N[hyd_param_col].tolist()

    _print_welcome(
        Ninit=Ninit,
        nP=len(hydP_params),
        nN=len(hydN_params),
        do_fmax=do_fmax,
    )

    xlb, xub, var_map = [], [], []

    for _, row in hyd_P.iterrows():
        p = row[hyd_param_col]
        xlb.append(float(row["__min__"]))
        xub.append(float(row["__max__"]))
        var_map.append(("HYDRO_P", p))

    for _, row in hyd_N.iterrows():
        p = row[hyd_param_col]
        xlb.append(float(row["__min__"]))
        xub.append(float(row["__max__"]))
        var_map.append(("HYDRO_N", p))

    if do_fmax:
        xlb.append(fmax_scale_min)
        xub.append(fmax_scale_max)
        var_map.append(("FMAX_SCALE", "FMAX_scale"))

    xlb = np.array(xlb, dtype=float)
    xub = np.array(xub, dtype=float)
    nInput = len(xlb)

    print("Joint calibration design:")
    print(f"  hydro P inputs  : {len(hydP_params)}")
    print(f"  hydro N inputs  : {len(hydN_params)}")
    print(f"  FMAX scale input: {1 if do_fmax else 0}")
    print(f"  TOTAL inputs    : {nInput}")
    print(f"  Ninit           : {Ninit}")

    rng = np.random.default_rng(seed)
    X01 = lhs(Ninit, nInput, rng)
    X = X01 * (xub - xlb) + xlb

    test_id_list = [f"{location}_{iteration}_{i:04d}" for i in range(Ninit)]

    colnames = []
    for kind, p in var_map:
        if kind == "HYDRO_P":
            colnames.append(f"HYDRO_P__{p}")
        elif kind == "HYDRO_N":
            colnames.append(f"HYDRO_N__{p}")
        else:
            colnames.append("S__FMAX_scale")

    psets_df = pd.DataFrame(X, columns=colnames, index=test_id_list)

    joint_table = workflow_dir / f"{location}_{iteration}.joint_param_list.txt"
    psets_df.to_csv(joint_table)

    main_run = workflow_dir / f"{location}_{iteration}.main_run.txt"
    with open(main_run, "w", encoding="utf-8") as f:
        f.write("\n".join(psets_df.index.values) + "\n")

    print("\nChecking base parameter file variables...")
    with xr.open_dataset(base_paramfile, decode_times=False) as base:
        missing_hydroP = [p for p in hydP_params if p not in base.variables]

    if missing_hydroP:
        raise KeyError(f"These hydro P params are not in {base_paramfile}: {missing_hydroP}")

    FMAX0 = None
    if do_fmax:
        print("Loading base FMAX field...")
        with xr.open_dataset(base_surfdata) as ds0:
            if "FMAX" not in ds0.variables:
                raise KeyError(f"FMAX not found in surfdata file: {base_surfdata}")
            FMAX0 = ds0["FMAX"].load().values

    hydN_index_file = workflow_dir / f"{location}_{iteration}.hydroN_txt_files.csv"
    hydN_index_lines = ["case_id,hydroN_txt"]

    print("\nWriting per-case outputs...")
    for case_id, row in tqdm(psets_df.iterrows(), total=len(psets_df), desc="Generating cases"):
        with xr.open_dataset(base_paramfile, decode_times=False) as tmp:
            tmp = tmp.load()

        for p in hydP_params:
            v = float(row[f"HYDRO_P__{p}"])
            tmp = assign_scalar_everywhere(tmp, p, v)

        out_nc = combined_param_dir / f"{case_id}.nc"
        tmp.to_netcdf(out_nc, mode="w")
        tmp.close()

        hydroN_txt = namelist_txt_dir / f"{case_id}.hydroN.txt"
        with open(hydroN_txt, "w", encoding="utf-8") as f:
            f.write(f"# N-type hydrology params for case: {case_id}\n")
            for p in hydN_params:
                v = float(row[f"HYDRO_N__{p}"])
                f.write(f"{p} = {v}\n")

        hydN_index_lines.append(f"{case_id},{hydroN_txt}")

        if do_fmax and FMAX0 is not None:
            scale = float(row["S__FMAX_scale"])
            outS = surfdata_out_dir / f"{case_id}.nc"
            shutil.copy2(base_surfdata, outS)

            new_fmax = FMAX0 * scale
            new_fmax[new_fmax > 1.0] = 1.0
            new_fmax[new_fmax < 0.0] = 0.0

            with Dataset(outS, "r+") as nc:
                v = nc.variables["FMAX"]
                v.set_auto_maskandscale(True)
                v[:] = new_fmax

    with open(hydN_index_file, "w", encoding="utf-8") as f:
        f.write("\n".join(hydN_index_lines) + "\n")

    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print("RMCH completed successfully")
    print(f"Total cases generated : {Ninit}")
    print(f"Execution time        : {elapsed:.2f} seconds")
    print("=" * 72)
    print("Thank you for using RMCH — Rahil")

    return {
        "paramfile_dir": str(combined_param_dir),
        "namelist_txt_dir": str(namelist_txt_dir),
        "workflow_dir": str(workflow_dir),
        "hydroN_index_file": str(hydN_index_file),
        "surfdata_dir": str(surfdata_out_dir) if do_fmax else None,
        "joint_param_table": str(joint_table),
        "main_run_file": str(main_run),
        "param_table": psets_df,
    }