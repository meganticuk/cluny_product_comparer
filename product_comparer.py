import json
import io
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Product Comparer", layout="wide")


def render_header() -> None:
    st.title("Shopify Collection Product Comparer")
    st.caption(
        "DELETING on the left, KEEPING on the right. "
        "Identify items only in each list and differences in common items."
    )


@st.cache_data(show_spinner=False)
def fetch_json_from_url(url: str, timeout_s: int = 20) -> Any:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def load_json_upload(name: str) -> Optional[Any]:
    upload = st.file_uploader(f"{name}: Upload JSON file", type=["json"], key=f"file_{name}")
    data: Optional[Any] = None
    if upload is not None:
        try:
            bytes_io = io.BytesIO(upload.read())
            data = json.load(bytes_io)
        except Exception as e:
            st.error(f"Failed to parse uploaded {name} JSON: {e}")
    return data


def extract_records(obj: Any, list_path: str) -> List[Dict[str, Any]]:
    # If path is empty and obj is already a list
    if list_path.strip() == "":
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # Try common Shopify key
            if "products" in obj and isinstance(obj["products"], list):
                return [x for x in obj["products"] if isinstance(x, dict)]
            # Or treat values as records if they form a list
            for v in obj.values():
                if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                    return v
        return []

    # Traverse dot-path
    current = obj
    for part in [p for p in list_path.split(".") if p]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return []
    if isinstance(current, list):
        return [x for x in current if isinstance(x, dict)]
    return []


def detect_candidate_keys(records: List[Dict[str, Any]]) -> List[str]:
    if not records:
        return []
    first = records[0]
    candidates = [
        "handle",
        "id",
        "sku",
        "legacy_resource_id",
        "title",
    ]
    present = [k for k in candidates if k in first]
    # Add any other scalar keys present in first record
    for k, v in first.items():
        if isinstance(v, (str, int, float)) and k not in present:
            present.append(k)
    return present


def build_index(records: List[Dict[str, Any]], key_field: str) -> Tuple[Dict[Any, Dict[str, Any]], Set[Any]]:
    index: Dict[Any, Dict[str, Any]] = {}
    duplicates: Set[Any] = set()
    for rec in records:
        key = rec.get(key_field)
        if key is None:
            continue
        if key in index:
            duplicates.add(key)
        else:
            index[key] = rec
    return index, duplicates


def to_preview_df(records: List[Dict[str, Any]], key_field: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[key_field, "title"])  # keep columns stable
    cols: List[str] = [key_field]
    if "title" in records[0]:
        cols.append("title")
    return pd.DataFrame([
        {c: rec.get(c) for c in cols}
        for rec in records
    ])


def dataframe_download(name: str, df: pd.DataFrame, fmt: str = "csv") -> None:
    if df.empty:
        st.info(f"No rows to download for {name}.")
        return
    if fmt == "csv":
        data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {name} CSV",
            data=data,
            file_name=f"{name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )
    else:
        data = df.to_json(orient="records").encode("utf-8")
        st.download_button(
            label=f"Download {name} JSON",
            data=data,
            file_name=f"{name.replace(' ', '_').lower()}.json",
            mime="application/json",
        )


def compare_dicts(a: Dict[str, Any], b: Dict[str, Any], fields: Optional[Set[str]] = None) -> Dict[str, Tuple[Any, Any]]:
    """Return a mapping of field -> (a_value, b_value) where they differ."""
    keys = fields if fields is not None else set(a.keys()) | set(b.keys())
    diff: Dict[str, Tuple[Any, Any]] = {}
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            diff[k] = (av, bv)
    return diff


def render_comparison(
    keeping_records: List[Dict[str, Any]],
    deleting_records: List[Dict[str, Any]],
    key_field: str,
    fields_to_compare: Optional[List[str]] = None,
) -> None:
    keep_index, keep_dupes = build_index(keeping_records, key_field)
    del_index, del_dupes = build_index(deleting_records, key_field)

    keep_keys = set(keep_index.keys())
    del_keys = set(del_index.keys())

    only_in_keeping = sorted(list(keep_keys - del_keys))
    only_in_deleting = sorted(list(del_keys - keep_keys))
    in_both = sorted(list(keep_keys & del_keys))

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Deleting count", len(deleting_records))
    col2.metric("Keeping count", len(keeping_records))
    col3.metric("Only in DELETING", len(only_in_deleting))
    col4.metric("Only in KEEPING", len(only_in_keeping))

    if del_dupes:
        st.warning(f"Duplicate {key_field} in DELETING: {len(del_dupes)} (e.g., {list(del_dupes)[:5]})")
    if keep_dupes:
        st.warning(f"Duplicate {key_field} in KEEPING: {len(keep_dupes)} (e.g., {list(keep_dupes)[:5]})")

    # Tables side-by-side aligned to vertical equator
    st.markdown("---")
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("### DELETING")
        df_only_del = to_preview_df([del_index[k] for k in only_in_deleting], key_field)
        st.dataframe(df_only_del, use_container_width=True, hide_index=True)
        dataframe_download("only_in_deleting", df_only_del)
    with right_col:
        st.markdown("### KEEPING")
        df_only_keep = to_preview_df([keep_index[k] for k in only_in_keeping], key_field)
        st.dataframe(df_only_keep, use_container_width=True, hide_index=True)
        dataframe_download("only_in_keeping", df_only_keep)

    # Differences among common keys
    st.markdown("---")
    st.markdown("### Differences in common products")
    shared_fields: Optional[Set[str]] = None
    if fields_to_compare:
        shared_fields = set(fields_to_compare)
        diff_note = ", ".join(fields_to_compare)
        st.caption(f"Comparing only fields: {diff_note}")

    diff_rows: List[Dict[str, Any]] = []
    for key in in_both:
        a = keep_index[key]
        b = del_index[key]
        diff = compare_dicts(a, b, shared_fields)
        if diff:
            row: Dict[str, Any] = {key_field: key}
            if "title" in a:
                row["title"] = a.get("title")
            for field, (av, bv) in diff.items():
                row[f"del.{field}"] = b.get(field)
                row[f"keep.{field}"] = a.get(field)
            diff_rows.append(row)

    df_diffs = pd.DataFrame(diff_rows)
    if df_diffs.empty:
        st.success("No differences found among common products for the selected fields.")
    else:
        st.dataframe(df_diffs, use_container_width=True, hide_index=True)
        dataframe_download("differences", df_diffs)


def normalize_domain(domain: str) -> str:
    if not domain:
        return "https://www.clunycountrystore.co.uk"
    return domain.rstrip("/")


def build_product_url(handle: Any, domain: str) -> str:
    base = normalize_domain(domain)
    return f"{base}/products/{handle}"


def main() -> None:
    render_header()

    # Two columns: left = DELETING, right = KEEPING
    left, right = st.columns(2)
    with left:
        st.markdown("#### DELETING feed (left)")
        del_data = load_json_upload("DELETING")
        del_list_path = st.text_input("DELETING: JSON path to list", value="products", help="Dot path to array of product objects; leave blank if top-level list")
    with right:
        st.markdown("#### KEEPING feed (right)")
        keep_data = load_json_upload("KEEPING")
        keep_list_path = st.text_input("KEEPING: JSON path to list", value="products", help="Dot path to array of product objects; leave blank if top-level list")

    if keep_data is None or del_data is None:
        st.info("Upload both feeds to proceed.")
        st.stop()

    keeping_records = extract_records(keep_data, keep_list_path)
    deleting_records = extract_records(del_data, del_list_path)

    if not keeping_records and not deleting_records:
        st.error("No records found in either feed. Adjust the JSON path or inputs.")
        st.stop()

    # Key selection
    candidate_keys = list(dict.fromkeys(detect_candidate_keys(keeping_records) + detect_candidate_keys(deleting_records)))
    key_field = st.selectbox(
        "Product key field",
        options=candidate_keys or ["handle", "id"],
        index=(candidate_keys.index("handle") if "handle" in candidate_keys else 0),
        help="Field used to identify products across feeds",
    )

    # Compute items to add (only in DELETING)
    keep_index, _ = build_index(keeping_records, key_field)
    del_index, _ = build_index(deleting_records, key_field)
    only_in_deleting_keys = sorted(list(set(del_index.keys()) - set(keep_index.keys())))

    # Build simple add list with URLs (prefer 'handle' when present)
    domain = st.text_input("Store domain", value="https://www.clunycountrystore.co.uk", help="Used to build product URLs: {domain}/products/{handle}")

    add_rows: List[Dict[str, Any]] = []
    for k in only_in_deleting_keys:
        rec = del_index[k]
        handle_value = rec.get("handle", k)
        add_rows.append({
            "handle": handle_value,
            "title": rec.get("title"),
            "url": build_product_url(handle_value, domain),
        })
    df_add = pd.DataFrame(add_rows, columns=["handle", "title", "url"]) if add_rows else pd.DataFrame(columns=["handle", "title", "url"]) 

    st.markdown("### Products to add to KEEPING")
    st.dataframe(
        df_add,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("URL", help="Opens in a new tab"),
        },
    )
    dataframe_download("add_to_keeping", df_add)

    # Advanced details (optional)
    with st.expander("Advanced comparison (optional)", expanded=False):
        # Optional: choose fields to compare
        example_record = (keeping_records or deleting_records)[0]
        all_fields = sorted([k for k, v in example_record.items() if isinstance(v, (str, int, float, bool))])
        fields_to_compare = st.multiselect(
            "Fields to compare (optional)",
            options=all_fields,
            help="If empty, compares all top-level fields present in either record",
        )
        with st.spinner("Comparing feeds..."):
            render_comparison(keeping_records, deleting_records, key_field, fields_to_compare or None)


if __name__ == "__main__":
    main()
