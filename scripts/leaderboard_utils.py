import pandas as pd


def extract_pollutant(item_id: str) -> str:
    """Extract pollutant name from item_id (e.g., 'site_105_..._IMD_CO' -> 'CO')."""
    return item_id.rsplit("_", 1)[-1]


_DATASET_NAME_MAP = {
    "CNEMC": "CNEMC SMALL",
}


# Model category groupings for leaderboard tables.
# Edit these to match your actual model names.
MODEL_GROUPS: dict[str, str] = {
        
    # TSFMs
    "chronos_bolt_base": "TSFMs",
    "chronos2_base": "TSFMs",
    "moirai_base": "TSFMs",
    "moirai2": "TSFMs",
    "TimesFM-1.0": "TSFMs",
    "TimesFM-2.0": "TSFMs",
    "TimesFM-2.5": "TSFMs",
    "TiRex": "TSFMs",
    "visiontspp_base": "TSFMs",
    "sundial_base": "TSFMs",
    "kairos": "TSFMs",
    # ML Baselines
    "patchtst": "ML Baselines",
    "dlinear": "ML Baselines",
    # Statistical Baselines
    "seasonal_naive": "Statistical Baselines",
    "auto_ets": "Statistical Baselines",

}

GROUP_ORDER: list[str] = ["TSFMs", "ML Baselines", "Statistical Baselines"]


def display_dataset(dataset_id: str) -> str:
    """Return a human-readable dataset name, stripping frequency suffix and mapping aliases.

    E.g. 'CPCB/H' -> 'CPCB', 'CNEMC/H' -> 'CNEMC SMALL', 'MY_DS/D' -> 'MY DS'
    """
    name = dataset_id.split("/")[0]
    name = _DATASET_NAME_MAP.get(name, name)
    return name.replace("_", " ")




def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    metric_cols: list = None,
    lower_is_better: bool = True,
    model_groups: dict[str, str] | None = None,
    group_order: list[str] | None = None,
) -> str:
    """
    Convert a DataFrame to a LaTeX table snippet (suitable for \\input{}).

    Formatting per metric column:
      - Bold:      best value
      - Underline: second best
      - Italics:   third best

    If model_groups is provided, rows are sorted by group then alphabetically
    within each group, with a \\multicolumn header row separating groups.
    Models not in model_groups are placed in an "Other" group at the end.
    """
    df = df.reset_index(drop=True).copy()
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c != "model"]

    # Group-based reordering: sort by (group order, model name alphabetically)
    group_labels = None
    if model_groups is not None and "model" in df.columns:
        if group_order is None:
            group_order = GROUP_ORDER
        go_idx = {g: i for i, g in enumerate(group_order)}
        df["_group"] = df["model"].map(model_groups).fillna("Other")
        df["_gord"] = df["_group"].map(lambda g: go_idx.get(g, len(go_idx)))
        df = df.sort_values(["_gord", "model"]).reset_index(drop=True)
        group_labels = df["_group"].tolist()
        df = df.drop(columns=["_group", "_gord"])

    # Determine rank-based formatting per metric column (global across all models)
    cell_fmt: dict[tuple[int, str], str] = {}
    for col in metric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        sorted_idx = vals.sort_values(ascending=lower_is_better).dropna().index.tolist()
        for rank, idx in enumerate(sorted_idx[:3]):
            cell_fmt[(idx, col)] = ["bold", "underline", "italic"][rank]

    def _escape(s: str) -> str:
        # Escape all LaTeX special characters in order (backslash first)
        s = s.replace("\\", "\\textbackslash{}")
        s = s.replace("{", "\\{").replace("}", "\\}")
        s = s.replace("$", "\\$").replace("#", "\\#")
        s = s.replace("^", "\\textasciicircum{}")
        s = s.replace("~", "\\textasciitilde{}")
        s = s.replace("_", "\\_")
        s = s.replace("%", "\\%").replace("&", "\\&")
        s = s.replace("<", "\\textless{}").replace(">", "\\textgreater{}")
        return s

    def _fmt(val, fmt, is_str=False):
        if is_str:
            s = _escape(str(val))
        else:
            try:
                s = f"{float(val):.3f}"
            except (ValueError, TypeError):
                s = str(val)
        if fmt == "bold":
            return f"\\textbf{{{s}}}"
        if fmt == "underline":
            return f"\\underline{{{s}}}"
        if fmt == "italic":
            return f"\\textit{{{s}}}"
        return s

    cols = df.columns.tolist()
    n_cols = len(cols)
    col_spec = "l" + "r" * (n_cols - 1)
    header = " & ".join(_escape(c) for c in cols) + " \\\\"

    body_lines = []
    current_group = None
    for idx, row in df.iterrows():
        if group_labels is not None:
            grp = group_labels[idx]
            if grp != current_group:
                if current_group is not None:
                    body_lines.append("\\midrule")
                body_lines.append(
                    f"\\multicolumn{{{n_cols}}}{{l}}{{{_escape(grp)}}} \\\\"
                )
                current_group = grp
        cells = [_fmt(row[c], cell_fmt.get((idx, c)), is_str=(c not in metric_cols)) for c in cols]
        body_lines.append(" & ".join(cells) + " \\\\")

    return "\n".join([
        "\\begin{table}[H]",
        f"\\caption{{{_escape(caption)}}}",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])