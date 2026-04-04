import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Keep this script runnable even if optional GIS deps (osmnx) aren't installed.
# We replicate the scoring constants from `model_latest.py` to avoid importing it.

# walking thresholds prioritizing the accessibility of the elderly
L_a = [0, 400, 800, 1200, 1600, 2000, 2400]
L_f_a = [100, 95, 70, 40, 10, 0, 0]

# restaurant choice weights (raw) and normalized weights
choice_weights_raw = np.array([0.75, 0.45, 0.25, 0.25, 0.225, 0.225, 0.225, 0.225, 0.2, 0.2])
restaurant_sum = float(np.sum(choice_weights_raw))
choice_weights = choice_weights_raw / restaurant_sum

# weights: healthcare, grocery, restaurant (sum), school
weights_array = np.array([4, 3, restaurant_sum, 1], dtype=float) / (restaurant_sum + 3 + 1 + 4)
# weights for multiple-amenity with restaurant depth: healthcare, grocery, 10 restaurant choices, school
weights_array_multi = (
    np.array([4, 3, 0.75, 0.45, 0.25, 0.25, 0.225, 0.225, 0.225, 0.225, 0.2, 0.2, 1], dtype=float)
    / (restaurant_sum + 3 + 1 + 4)
)


def dist_to_score(dists: np.ndarray, breakpoints: list[float], scores: list[float]) -> np.ndarray:
    """
    Piecewise-linear score transform used throughout the repo.
    Equivalent to the PWL mapping defined by (breakpoints, scores).
    """
    x = np.asarray(dists, dtype=float)
    xp = np.asarray(breakpoints, dtype=float)
    fp = np.asarray(scores, dtype=float)
    # np.interp clamps outside range to end values, matching typical PWL behavior.
    return np.interp(x, xp, fp, left=fp[0], right=fp[-1])


def _safe_mean(arr: pd.Series) -> float:
    a = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.nanmean(a))


def _load_nia_names(data_root: str) -> dict[int, str]:
    xlsx = os.path.join(
        data_root,
        "Neighbourhood Improvement Areas - 4326",
        "processed_TSNS 2020 NIA Census Tracts.xlsx",
    )
    if not os.path.exists(xlsx):
        return {}
    try:
        df = pd.read_excel(xlsx)
        # Try common column names first.
        id_candidates = ["area_sh11", "NIA_ID", "nia_id", "AREA_SH11"]
        name_candidates = ["area_na13", "NIA_NAME", "nia_name", "AREA_NA13"]

        id_col = next((c for c in id_candidates if c in df.columns), None)
        name_col = next((c for c in name_candidates if c in df.columns), None)

        # Fallback: pick first column containing 'sh11' and 'na13'
        if id_col is None:
            id_col = next((c for c in df.columns if "sh11" in str(c).lower()), None)
        if name_col is None:
            name_col = next((c for c in df.columns if "na13" in str(c).lower()), None)

        if id_col is None or name_col is None:
            return {}

        out: dict[int, str] = {}
        for _, row in df[[id_col, name_col]].dropna().iterrows():
            try:
                out[int(row[id_col])] = str(row[name_col])
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _infer_k_from_filename(path: str) -> str:
    # assignment_NIA_<id>_<k>.csv where <k> might be "3,3,3,3"
    base = os.path.basename(path)
    if not base.startswith("assignment_NIA_"):
        raise ValueError(f"Unexpected filename: {base}")
    k_part = base.split("assignment_NIA_", 1)[1].rsplit(".csv", 1)[0]
    # k_part is "<nia>_<kstr>"
    try:
        _, k_str = k_part.split("_", 1)
    except ValueError as e:
        raise ValueError(f"Unexpected filename: {base}") from e
    
    return k_str


def _infer_nia_from_filename(path: str) -> int:
    base = os.path.basename(path)
    # assignment_NIA_<nia>_<k>.csv
    middle = base.split("assignment_NIA_", 1)[1].rsplit(".csv", 1)[0]
    nia_str, _ = middle.split("_", 1)
    return int(nia_str)


def _get_num_allocations_from_pickle(sol_folder: str, nia: int, k_str: str) -> float:
    pkl = os.path.join(sol_folder, f"allocation_NIA_{nia}_{k_str}.pkl")
    if not os.path.exists(pkl):
        return float("nan")
    try:
        import pickle

        obj = pickle.load(open(pkl, "rb"))
        # New allocations are stored per-amenity; sum of lengths equals total new facilities.
        total = 0
        for key in [
            "allocate_node_id_grocery",
            "allocate_node_id_restaurant",
            "allocate_node_id_school",
            "allocate_node_id_healthcare",
        ]:
            v = obj.get(key)
            if isinstance(v, list):
                total += len(v)
        return float(total) if total > 0 else float("nan")
    except Exception:
        return float("nan")


def _compute_multiple_obj_from_assignment(df: pd.DataFrame) -> tuple[float, dict[str, float], dict[str, object]]:
    """
    Returns:
      - obj (mean walk score)
      - dist_means (mean distance per amenity/choice)
      - dist_obj_fields (values to place in summary columns)
    """
    # Normalize possible column names across variants.
    col_grocery = "dist_grocery" if "dist_grocery" in df.columns else None
    col_school = "dist_school" if "dist_school" in df.columns else None
    col_healthcare = "dist_healthcare" if "dist_healthcare" in df.columns else None

    # Restaurant can be no-depth ("dist_restaurant") or depth ("0_dist_restaurant", "1_dist_restaurant", ...),
    # sometimes with suffix "_restaurant" (from model_latest helpers).
    col_restaurant_simple = "dist_restaurant" if "dist_restaurant" in df.columns else None

    depth_cols = []
    for c in range(len(choice_weights)):
        for candidate in (f"{c}_dist_restaurant", f"{c}_dist"):
            if candidate in df.columns:
                depth_cols.append(candidate)
                break
        else:
            depth_cols.append(None)

    has_depth = any(c is not None for c in depth_cols)

    if has_depth:
        # Build per-resident weighted distance, padding missing choices with L_a[-2] (i.e., 2400m).
        choices = []
        for c, col in enumerate(depth_cols):
            if col is None:
                choices.append(np.full(len(df), float(L_a[-2]), dtype=float))
            else:
                choices.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        choices = np.vstack(choices)  # (num_choices, num_res)

        dist_g = (
            pd.to_numeric(df[col_grocery], errors="coerce").to_numpy(dtype=float)
            if col_grocery
            else np.full(len(df), float(L_a[-2]), dtype=float)
        )
        dist_s = (
            pd.to_numeric(df[col_school], errors="coerce").to_numpy(dtype=float)
            if col_school
            else np.full(len(df), float(L_a[-2]), dtype=float)
        )
        dist_h = (
            pd.to_numeric(df[col_healthcare], errors="coerce").to_numpy(dtype=float)
            if col_healthcare
            else np.full(len(df), float(L_a[-2]), dtype=float)
        )

        multiple_dist = np.vstack([dist_h, dist_g, *list(choices), dist_s])  # (13, num_res)
        weighted_dist = np.dot(np.array(weights_array_multi), multiple_dist)
        scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
        obj = float(np.nanmean(scores))

        restaurant_means = [
            float(np.nanmean(choices[c])) if choices.shape[1] else float("nan")
            for c in range(choices.shape[0])
        ]
        dist_obj_fields = {
            "dist_obj_grocery": _safe_mean(df[col_grocery]) if col_grocery else float("nan"),
            "dist_obj_restaurant": json.dumps(restaurant_means),
            "dist_obj_school": _safe_mean(df[col_school]) if col_school else float("nan"),
            "dist_obj_healthcare": _safe_mean(df[col_healthcare]) if col_healthcare else float("nan"),
        }
        dist_means = {
            "grocery": dist_obj_fields["dist_obj_grocery"],
            "school": dist_obj_fields["dist_obj_school"],
            "healthcare": dist_obj_fields["dist_obj_healthcare"],
            "restaurant_choice_means": restaurant_means,
        }
        return obj, dist_means, dist_obj_fields

    # No depth: use simple weights_array (healthcare, grocery, restaurant, school).
    dist_g = pd.to_numeric(df[col_grocery], errors="coerce").to_numpy(dtype=float) if col_grocery else None
    dist_r = (
        pd.to_numeric(df[col_restaurant_simple], errors="coerce").to_numpy(dtype=float)
        if col_restaurant_simple
        else None
    )
    dist_s = pd.to_numeric(df[col_school], errors="coerce").to_numpy(dtype=float) if col_school else None
    dist_h = pd.to_numeric(df[col_healthcare], errors="coerce").to_numpy(dtype=float) if col_healthcare else None

    if any(x is None for x in (dist_h, dist_g, dist_r, dist_s)):
        # Can't compute score reliably.
        obj = float("nan")
    else:
        multiple_dist = np.vstack([dist_h, dist_g, dist_r, dist_s])
        weighted_dist = np.dot(np.array(weights_array), multiple_dist)
        scores = dist_to_score(np.array(weighted_dist), L_a, L_f_a)
        obj = float(np.nanmean(scores))

    dist_obj_fields = {
        "dist_obj_grocery": _safe_mean(df[col_grocery]) if col_grocery else float("nan"),
        "dist_obj_restaurant": _safe_mean(df[col_restaurant_simple]) if col_restaurant_simple else float("nan"),
        "dist_obj_school": _safe_mean(df[col_school]) if col_school else float("nan"),
        "dist_obj_healthcare": _safe_mean(df[col_healthcare]) if col_healthcare else float("nan"),
    }
    dist_means = {
        "grocery": dist_obj_fields["dist_obj_grocery"],
        "restaurant": dist_obj_fields["dist_obj_restaurant"],
        "school": dist_obj_fields["dist_obj_school"],
        "healthcare": dist_obj_fields["dist_obj_healthcare"],
    }
    return obj, dist_means, dist_obj_fields


def backfill_model_summaries(
    *,
    results_folder: str,
    model_name: str,
    k_str: str,
    data_root: str,
    overwrite: bool,
) -> tuple[int, int]:
    sol_folder = os.path.join(results_folder, "sol", model_name)
    summary_folder = os.path.join(results_folder, "summary", model_name)
    Path(summary_folder).mkdir(parents=True, exist_ok=True)

    nia_name_map = _load_nia_names(data_root)

    pattern = os.path.join(sol_folder, f"assignment_NIA_*_{k_str}.csv")
    files = sorted(Path(sol_folder).glob(f"assignment_NIA_*_{k_str}.csv"))

    if not files:
        raise FileNotFoundError(f"No assignment files found for pattern: {pattern}")

    written = 0
    skipped = 0

    for p in files:
        nia = _infer_nia_from_filename(str(p))
        k_in_file = _infer_k_from_filename(str(p))
        if k_in_file != k_str:
            continue

        out = os.path.join(summary_folder, f"NIA_{nia}_{k_str}_summary.csv")
        if os.path.exists(out) and not overwrite:
            skipped += 1
            continue

        df = pd.read_csv(p)
        obj, _, dist_obj_fields = _compute_multiple_obj_from_assignment(df)

        # k fields
        k_parts = [int(x) for x in k_str.split(",")] if "," in k_str else [int(k_str)]
        if len(k_parts) == 4:
            k_g, k_r, k_s, k_h = k_parts
        elif len(k_parts) == 3:
            k_g, k_r, k_s = k_parts
            k_h = 0
        else:
            # fall back
            k_g = k_r = k_s = k_h = k_parts[0]

        results_row = {
            "nia_id": nia,
            "nia_name": nia_name_map.get(nia, ""),
            "k_grocery": k_g,
            "k_restaurant": k_r,
            "k_school": k_s,
            "k_healthcare": k_h,
            "obj": obj,
            "dist_obj_grocery": dist_obj_fields["dist_obj_grocery"],
            "dist_obj_restaurant": dist_obj_fields["dist_obj_restaurant"],
            "dist_obj_school": dist_obj_fields["dist_obj_school"],
            "dist_obj_healthcare": dist_obj_fields["dist_obj_healthcare"],
            "solving_time": np.nan,
            "num_res": int(len(df)),
            "num_parking": _get_num_allocations_from_pickle(sol_folder, nia, k_str),
            "num_cur_grocery": np.nan,
            "num_cur_restaurant": np.nan,
            "num_cur_school": np.nan,
            "num_cur_healthcare": np.nan,
            "model_status": "",
        }

        pd.DataFrame([results_row]).to_csv(out, index=False)
        written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing per-NIA *_summary.csv files from saved assignment CSVs."
    )
    parser.add_argument("--results_folder", default="results")
    parser.add_argument("--model_name", required=True, help="e.g., OptMultipleDepth_False_0")
    parser.add_argument("--k_str", required=True, help="e.g., 3,3,3,3")
    parser.add_argument(
        "--data_root",
        default=os.getcwd(),
        help="Project root containing the NIA excel; defaults to current working directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written, skipped = backfill_model_summaries(
        results_folder=args.results_folder,
        model_name=args.model_name,
        k_str=args.k_str,
        data_root=args.data_root,
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote {written} summary files. Skipped {skipped} existing files.")


if __name__ == "__main__":
    main()

