from __future__ import annotations

import hashlib
import io
import re
from datetime import date, datetime, timedelta

import pandas as pd


REPORT_COLUMNS = [
    "order_internal_id",
    "internal_id",
    "site_internal_id",
    "site_name",
    "site_address_1",
    "site_address_2",
    "site_address_3",
    "site_post_code",
    "item_to_order",
    "date_of_visit",
    "time_of_visit",
    "purchase_cost",
    "site_code",
    "primary_result",
    "Were you able to successfully conduct this audit?",
    "Please enter the date you placed your order:",
    "Which partner company made your delivery?",
    "What is your age?",
    "What was the total cost of your purchase? ",
    "Please give details of the age restricted product(s) purchased:",
    "Please enter the postcode that your order was delivered to:",
    "Did the driver ask your age?",
    "Did the driver ask for ID?",
    "Were you asked to sign for delivery?",
    "Were any of the items damaged?",
    "What was the gender of the driver?",
    "Please accurately describe the driver:",
    "Was the driver dressed in Tesco branded work attire?",
    "Please enter the order number from your online receipt:",
    "Did the driver make eye contact with you during the interaction?",
    "Was the driver friendly?",
    "Did you see the delivery driver take a photo of the delivery bag at your door?",
    "Did the delivery driver engage in any conversation with you?",
    "What did the driver say?",
    "Based on your online shopping experience, please rate the service from 1 to 10 (where 1 is very poor and 10 is excellent):",
    "Please explain the reason for your score:",
    "Based on your delivery experience, please rate your experience from 1 to 10 (where 1 is very poor and 10 is excellent):",
    "Please use this space to explain anything unusual about your visit or to clarify any detail of your report:",
    "Please confirm below whether or not you were asked for ID:",
]

PARTNER_QUESTION = "Which partner company made your delivery?"
SITE_NAME = "Uber Eats - Tesco Whoosh"


class ReportError(ValueError):
    """A clear, user-facing input or generation error."""


def read_csv_bytes(data: bytes, label: str) -> pd.DataFrame:
    """Read common CSV encodings while preserving identifiers and blank cells."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            frame = pd.read_csv(
                io.BytesIO(data),
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ReportError(f"{label} could not be read as a CSV file: {last_error}")


def remove_empty_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the trailing blank columns found in some historic report files."""
    keep: list[str] = []
    for column in frame.columns:
        is_unnamed = str(column).startswith("Unnamed:")
        is_empty = frame[column].astype(str).str.strip().eq("").all()
        if not (is_unnamed and is_empty):
            keep.append(column)
    return frame.loc[:, keep].copy()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:6])
        suffix = "" if len(missing) <= 6 else f" (and {len(missing) - 6} more)"
        raise ReportError(f"{label} is missing required column(s): {preview}{suffix}.")


def parse_previous_report_date(filename: str, previous: pd.DataFrame) -> date:
    """Use the dated report filename, with the report contents as a fallback."""
    matches = re.findall(r"(?<!\d)(\d{2})[.\-_](\d{2})[.\-_](\d{4})(?!\d)", filename)
    if matches:
        day, month, year = matches[-1]
        try:
            return datetime(int(year), int(month), int(day)).date()
        except ValueError as exc:
            raise ReportError(f"The date in the previous report filename is invalid: {filename}") from exc

    if "date_of_visit" in previous.columns:
        dates = pd.to_datetime(previous["date_of_visit"], dayfirst=True, errors="coerce")
        if dates.notna().any():
            return dates.max().date() + timedelta(days=1)

    raise ReportError(
        "The previous report date could not be identified. Include a date such as "
        "21.08.2026 in its filename."
    )


def values(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first non-blank value across compatible source-question names."""
    result = pd.Series("", index=frame.index, dtype="object")
    for column in candidates:
        if column not in frame.columns:
            continue
        candidate = frame[column].fillna("").astype(str)
        result = result.mask(result.str.strip().eq(""), candidate)
    return result


def reference_value(previous: pd.DataFrame, column: str, default: str = "") -> str:
    """Carry stable report constants forward from the most recent report."""
    if column not in previous.columns:
        return default
    non_blank = previous[column].fillna("").astype(str)
    non_blank = non_blank[non_blank.str.strip().ne("")]
    if non_blank.empty:
        return default
    return str(non_blank.mode().iloc[0])


def normalise_id_confirmation(frame: pd.DataFrame) -> pd.Series:
    direct = values(
        frame,
        "Please confirm below whether or not you were asked for ID:",
        "Please confirm below if the courier asked for ID?",
    )
    fallback = values(frame, "Did the driver ask for ID?")

    def convert(value: str, fallback_value: str) -> str:
        text = str(value).strip()
        folded = text.casefold()
        if not text:
            return str(fallback_value)
        if folded in {"yes", "no"}:
            return text.title()
        if "not asked" in folded or "wasn't asked" in folded or "was not asked" in folded:
            return "No"
        if "asked for id" in folded:
            return "Yes"
        return text

    return pd.Series(
        [convert(value, fallback_value) for value, fallback_value in zip(direct, fallback)],
        index=frame.index,
        dtype="object",
    )


def generate_report(
    audit_export: pd.DataFrame,
    previous_report: pd.DataFrame,
    previous_filename: str,
) -> tuple[pd.DataFrame, date, date, date]:
    previous_report = remove_empty_unnamed_columns(previous_report)
    require_columns(previous_report, REPORT_COLUMNS, "The previous report")

    date_column = "date_of_visit_local" if "date_of_visit_local" in audit_export.columns else "date_of_visit"
    time_column = "time_of_visit_local" if "time_of_visit_local" in audit_export.columns else "time_of_visit"
    require_columns(
        audit_export,
        [
            "order_internal_id",
            "internal_id",
            "site_internal_id",
            "site_name",
            "item_to_order",
            date_column,
            time_column,
            "primary_result",
            PARTNER_QUESTION,
        ],
        "The audit export",
    )

    previous_report_date = parse_previous_report_date(previous_filename, previous_report)
    report_date = previous_report_date + timedelta(days=7)
    week_start = previous_report_date
    week_end = report_date - timedelta(days=1)

    visit_dates = pd.to_datetime(audit_export[date_column], dayfirst=True, errors="coerce")
    normalised_site = audit_export["site_name"].fillna("").astype(str).str.strip().str.casefold()
    normalised_partner = audit_export[PARTNER_QUESTION].fillna("").astype(str).str.strip().str.casefold()
    export_audit_ids = audit_export["internal_id"].fillna("").astype(str).str.strip()
    previously_reported_ids = set(
        previous_report["site_internal_id"].fillna("").astype(str).str.strip()
    )
    previously_reported_ids.discard("")

    mask = (
        normalised_site.eq(SITE_NAME.casefold())
        & normalised_partner.eq("uber")
        & visit_dates.ge(pd.Timestamp(week_start))
        & visit_dates.lt(pd.Timestamp(report_date))
        & ~export_audit_ids.isin(previously_reported_ids)
    )
    if "status" in audit_export.columns:
        mask &= audit_export["status"].fillna("").astype(str).str.strip().str.casefold().eq("approved")

    selected = audit_export.loc[mask].copy()
    selected["__visit_date"] = visit_dates.loc[mask]
    selected = selected.drop_duplicates(subset=["internal_id"], keep="first")

    output = pd.DataFrame(index=selected.index)
    output["order_internal_id"] = values(selected, "order_internal_id")
    output["internal_id"] = reference_value(previous_report, "internal_id", "Tesco")
    output["site_internal_id"] = values(selected, "internal_id")
    output["site_name"] = values(selected, "site_internal_id")
    output["site_address_1"] = reference_value(previous_report, "site_address_1", "Tesco Whoosh")
    output["site_address_2"] = reference_value(previous_report, "site_address_2", "")
    output["site_address_3"] = reference_value(previous_report, "site_address_3", "")
    output["site_post_code"] = reference_value(previous_report, "site_post_code", "GX11")
    output["item_to_order"] = values(selected, "item_to_order")
    output["date_of_visit"] = selected["__visit_date"].dt.strftime("%d/%m/%Y")
    output["time_of_visit"] = values(selected, time_column)
    output["purchase_cost"] = reference_value(previous_report, "purchase_cost", "")
    output["site_code"] = reference_value(previous_report, "site_code", "GX11")
    output["primary_result"] = values(selected, "primary_result")

    direct_questions = [
        "Were you able to successfully conduct this audit?",
        "Please enter the date you placed your order:",
        PARTNER_QUESTION,
        "What is your age?",
        "Please give details of the age restricted product(s) purchased:",
        "Please enter the postcode that your order was delivered to:",
        "Did the driver ask your age?",
        "Did the driver ask for ID?",
        "Were you asked to sign for delivery?",
        "Were any of the items damaged?",
        "What was the gender of the driver?",
        "Please accurately describe the driver:",
        "Was the driver dressed in Tesco branded work attire?",
        "Did the driver make eye contact with you during the interaction?",
        "Was the driver friendly?",
        "Did you see the delivery driver take a photo of the delivery bag at your door?",
        "Did the delivery driver engage in any conversation with you?",
        "What did the driver say?",
        "Based on your online shopping experience, please rate the service from 1 to 10 (where 1 is very poor and 10 is excellent):",
        "Based on your delivery experience, please rate your experience from 1 to 10 (where 1 is very poor and 10 is excellent):",
        "Please use this space to explain anything unusual about your visit or to clarify any detail of your report:",
    ]
    for question in direct_questions:
        output[question] = values(selected, question)

    output["What was the total cost of your purchase? "] = values(
        selected,
        "What was the total cost of your purchase? ",
        "What was the total cost of your purchase?",
    )
    output["Please enter the order number from your online receipt:"] = values(
        selected,
        "Please enter the order number from your online receipt:",
        "Please enter your order number:",
    )
    output["Please explain the reason for your score:"] = values(
        selected,
        "Please explain the reason for your score:",
        "Please explain the reason for your shopping experience score:",
    )
    output["Please confirm below whether or not you were asked for ID:"] = normalise_id_confirmation(selected)

    output = output.reindex(columns=REPORT_COLUMNS).fillna("").reset_index(drop=True)
    return output, report_date, week_start, week_end


def report_to_csv_bytes(report: pd.DataFrame) -> bytes:
    return report.to_csv(index=False, lineterminator="\r\n").encode("utf-8")


def input_fingerprint(audit_bytes: bytes, previous_bytes: bytes, previous_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(audit_bytes)
    digest.update(b"\0")
    digest.update(previous_bytes)
    digest.update(b"\0")
    digest.update(previous_name.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Uber Tesco Whoosh Report Generator", page_icon="📊")
    st.title("Uber Tesco Whoosh Weekly Report Generator")
    st.write(
        "Upload the audit export and the most recent Tesco Whoosh Uber report. "
        "The app will exclude audits already present in that report and create "
        "the next weekly CSV in the same 39-column format."
    )

    audit_file = st.file_uploader("Upload audit export", type="csv")
    previous_file = st.file_uploader("Upload most recent report", type="csv")

    if audit_file is None or previous_file is None:
        st.info("Upload both CSV files to generate the report.")
        return

    audit_bytes = audit_file.getvalue()
    previous_bytes = previous_file.getvalue()
    fingerprint = input_fingerprint(audit_bytes, previous_bytes, previous_file.name)

    if st.session_state.get("report_fingerprint") != fingerprint:
        st.session_state.pop("generated_report", None)
        st.session_state.pop("generated_metadata", None)

    if st.button("Generate report", type="primary"):
        try:
            audit_export = read_csv_bytes(audit_bytes, "The audit export")
            previous_report = read_csv_bytes(previous_bytes, "The previous report")
            report, report_date, week_start, week_end = generate_report(
                audit_export,
                previous_report,
                previous_file.name,
            )
            st.session_state["generated_report"] = report_to_csv_bytes(report)
            st.session_state["generated_metadata"] = {
                "filename": f"Tesco Whoosh Uber Data {report_date:%d.%m.%Y}.csv",
                "rows": len(report),
                "passes": int(report["primary_result"].str.casefold().eq("pass").sum()),
                "fails": int(report["primary_result"].str.casefold().eq("fail").sum()),
                "week_start": f"{week_start:%d/%m/%Y}",
                "week_end": f"{week_end:%d/%m/%Y}",
            }
            st.session_state["report_fingerprint"] = fingerprint
        except ReportError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"The report could not be generated: {exc}")

    if "generated_report" in st.session_state:
        metadata = st.session_state["generated_metadata"]
        st.success(
            f"Report generated for {metadata['week_start']} to {metadata['week_end']}."
        )
        first, second, third = st.columns(3)
        first.metric("Audits", metadata["rows"])
        second.metric("Passes", metadata["passes"])
        third.metric("Fails", metadata["fails"])

        if metadata["rows"] == 0:
            st.warning("No approved Uber Tesco Whoosh audits were found for this reporting week.")

        st.download_button(
            "Download updated report",
            data=st.session_state["generated_report"],
            file_name=metadata["filename"],
            mime="text/csv",
            type="primary",
        )


if __name__ == "__main__":
    main()
