import pandas as pd


def extract_pollutant(item_id: str) -> str:
    """Extract pollutant name from item_id (e.g., 'site_105_..._IMD_CO' -> 'CO')."""
    return item_id.rsplit("_", 1)[-1]


_DATASET_NAME_MAP = {
    "CNEMC": "CNEMC SMALL",
}


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
    table_num: int,
    metric_cols: list = None,
    lower_is_better: bool = True,
) -> str:
    """
    Convert a DataFrame to a LaTeX table snippet (suitable for \\input{}).

    Formatting per metric column:
      - Bold:      best value
      - Underline: second best
      - Italics:   third best

    Caption is placed above the table in 9pt type, centered if it fits on
    one line (<= 60 chars), otherwise flush left, with 0.1in spacing before
    and after. Requires booktabs in the parent document.
    """
    df = df.reset_index(drop=True)
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c != "model"]

    # Determine rank-based formatting per metric column
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
        s = _escape(str(val)) if is_str else str(val)
        if fmt == "bold":
            return f"\\textbf{{{s}}}"
        if fmt == "underline":
            return f"\\underline{{{s}}}"
        if fmt == "italic":
            return f"\\textit{{{s}}}"
        return s

    cols = df.columns.tolist()
    col_spec = "l" + "r" * (len(cols) - 1)
    header = " & ".join(f"\\textbf{{{_escape(c)}}}" for c in cols) + " \\\\"

    body_lines = []
    for idx, row in df.iterrows():
        cells = [_fmt(row[c], cell_fmt.get((idx, c)), is_str=(c not in metric_cols)) for c in cols]
        body_lines.append(" & ".join(cells) + " \\\\")

    caption_align = "centering" if len(caption) <= 60 else "raggedright"
    caption_tex = (
        f"{{\\fontsize{{9}}{{11}}\\selectfont\\{caption_align}"
        f" \\textit{{Table~{table_num}:}} {_escape(caption)}\\par}}"
    )

    return "\n".join([
        "\\begin{table}[h]",
        "\\vspace{0.1in}",
        caption_tex,
        "\\vspace{0.1in}",
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