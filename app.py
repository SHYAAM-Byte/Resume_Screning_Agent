

import json

import altair as alt
import pandas as pd
import requests
import streamlit as st

from processor import SignalProcessor

OLLAMA_URL = "http://localhost:11434/api/chat"  
OLLAMA_MODEL = "llama3"
MAX_PLOTS = 30            # plot at most this many signals per tab (the CSV always has all of them)
MAX_PLOT_POINTS = 5000    # plot at most this many points per signal (keeps the browser fast)

SYSTEM_PROMPT = """You convert a CAN signal analysis request into ONE JSON object.
Do not explain anything and do not analyse any data. Output JSON only.

The JSON must have exactly these four keys:
"id"           - the CAN arbitration ID as a hex string, written exactly as the user typed it (for example "19" or "1A0"). Use "all" when the user does not name an ID.
"analysis"     - "digital", "counter" or "both". Use "both" when the user asks for both types or does not say which type.
"bytes"        - a list of byte names from "B1","B2","B3","B4","B5","B6","B7","B8". Use all eight when the user does not name bytes. "B1 to B4" means ["B1","B2","B3","B4"].
"min_interval" - a number of seconds when the user asks for a minimum time between state changes ("more than 1 second" -> 1.0, "at least 500 ms" -> 0.5). Use null when there is no time condition.

Examples:

User: In arbitration ID 19 extract digital signals.
{"id": "19", "analysis": "digital", "bytes": ["B1","B2","B3","B4","B5","B6","B7","B8"], "min_interval": null}

User: Find digital signals with more than 1 second between state changes.
{"id": "all", "analysis": "digital", "bytes": ["B1","B2","B3","B4","B5","B6","B7","B8"], "min_interval": 1.0}

User: For ID 1A0 find counter signals.
{"id": "1A0", "analysis": "counter", "bytes": ["B1","B2","B3","B4","B5","B6","B7","B8"], "min_interval": null}

User: Find all digital and counter signals.
{"id": "all", "analysis": "both", "bytes": ["B1","B2","B3","B4","B5","B6","B7","B8"], "min_interval": null}

User: In ID 19 analyze only B1 to B4.
{"id": "19", "analysis": "both", "bytes": ["B1","B2","B3","B4"], "min_interval": null}
"""


def read_csv(uploaded_file):
    """Read the uploaded CSV into a DataFrame. Raises ValueError with a readable message."""
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, dtype=str, skipinitialspace=True, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as err:
        raise ValueError(f"The file could not be read as a CSV ({err}).") from None

def ask_llama(user_text):
    """
    Send the user's sentence to local Llama 3 and return its JSON answer as a dict.
    Raises RuntimeError if Ollama cannot be reached or answers badly,
    and ValueError if the answer is not valid JSON.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "format": "json",            # ask Ollama to force valid JSON
        "stream": False,
        "options": {"temperature": 0},  # 0 = the most predictable answer
    }
    try:
        # (connect timeout, read timeout) - the first answer can be slow while the model loads
        response = requests.post(OLLAMA_URL, json=payload, timeout=(5, 180))
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Start the Ollama app (or run `ollama serve` in a terminal) and try again."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama took too long to answer. Try again in a moment.") from None
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"The request to Ollama failed: {err}") from None

    if response.status_code == 404:
        raise RuntimeError(f"Ollama does not have the model '{OLLAMA_MODEL}'. Run `ollama pull {OLLAMA_MODEL}` first.")
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

    try:
        answer = response.json()["message"]["content"]
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Ollama sent a reply in an unexpected format.") from None
    return parse_json(answer)


def parse_json(text):
    """Turn the AI's answer into a dict. Raises ValueError if it is not a JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Sometimes a model adds words around the JSON - try to cut out the {...} part.
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start:end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError:
            data = None
        if data is None:
            raise ValueError(f"Llama 3 did not return valid JSON. Its answer was: {text[:300]!r}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Llama 3 returned JSON, but not an object: {text[:300]!r}")
    return data

def plot_signal(signal, title, kind):
    """Draw one signal: x = abs_time, y = signal value."""
    signal = signal[["abs_time", "value"]].head(MAX_PLOT_POINTS)
    if kind == "digital":
        # Digital signals are drawn as steps between 0 and 1.
        line = alt.Chart(signal).mark_line(interpolate="step-after", point=True)
        y = alt.Y("value:Q", title="value", scale=alt.Scale(domain=[-0.1, 1.1]), axis=alt.Axis(values=[0, 1], format="d"))
    else:
        line = alt.Chart(signal).mark_line(point=True)
        y = alt.Y("value:Q", title="counter value")
    chart = line.encode(
        x=alt.X("abs_time:Q", title="abs_time (s)"),
        y=y,
        tooltip=["abs_time:Q", "value:Q"],
    ).properties(title=title, height=160, width="container")
    st.altair_chart(chart)


def show_signal_group(kind, result):
    """Show the table, the CSV download button and the plots for 'digital' or 'counter'."""
    summary, data = result["summary"], result["data"]
    if summary.empty:
        st.info(f"No {kind} signals were detected for this request. "
                "Try other bytes, another CAN ID or a smaller time interval.")
        return

    st.success(f"{len(summary)} {kind} signal(s) detected.")
    st.dataframe(summary, hide_index=True)
    st.download_button(
        f"Download {kind} results (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"{kind}_signals.csv",
        mime="text/csv",
        key=f"download_{kind}",
    )

    if len(summary) > MAX_PLOTS:
        st.caption(f"Plotting the first {MAX_PLOTS} of {len(summary)} signals. The CSV contains all of them.")
    keys = ["can_id", "byte", "bit"] if kind == "digital" else ["can_id", "byte"]
    for number, (key, signal) in enumerate(data.groupby(keys, sort=False)):
        if number >= MAX_PLOTS:
            break
        title = f"ID {key[0]}, {key[1]}" + (f", bit {key[2]}" if kind == "digital" else "")
        plot_signal(signal, title, kind)


def show_output(output):
    """Show everything that was stored by run_analysis()."""
    st.divider()
    st.caption(f"CSV loaded: {output['frames']:,} frames, {output['ids']} CAN ID(s).")

    st.subheader("Interpreted instruction")
    st.caption("Llama 3 turned your sentence into this instruction. Pandas did all of the analysis.")
    st.json(output["request"])

    st.subheader("Detected signals")
    kinds = list(output["results"])                      # "digital" and/or "counter"
    if len(kinds) > 1:
        areas = st.tabs([k.capitalize() for k in kinds])  # separate tabs when both were requested
    else:
        areas = [st.container()]
    for area, kind in zip(areas, kinds):
        with area:
            show_signal_group(kind, output["results"][kind])


def run_analysis(uploaded, user_text):
    """Run all steps. Every problem is shown as a clear message and stops the run."""
    st.session_state.pop("output", None)

    if uploaded is None:
        st.error("Upload a CSV file first.")
        return
    if not user_text.strip():
        st.error("Describe the analysis you want, for example: \"In ID 19 find digital signals\".")
        return

    # 1. Load and check the CSV.
    try:
        processor = SignalProcessor(read_csv(uploaded))
    except ValueError as err:
        st.error(f"CSV problem: {err}")

    try:
        with st.spinner("Llama 3 is reading your request..."):
            raw_request = ask_llama(user_text)
    except RuntimeError as err:
        st.error(f"Ollama problem: {err}")
        return
    except ValueError as err:
        st.error(f"{err} Try rephrasing your request.")
        return

    # 3. Check the JSON against the CSV, then analyse with Pandas.
    try:
        request = processor.validate_request(raw_request)
        results = processor.analyze(request)
    except ValueError as err:
        st.subheader("Interpreted instruction")
        st.json(raw_request)
        st.error(f"The request cannot be run: {err}")
        return

    # Keep the output in session_state so it stays on screen after a download click.
    st.session_state["output"] = {
        "request": request,
        "results": results,
        "frames": len(processor.df),
        "ids": len(processor.get_ids()),
    }


def main():
    st.set_page_config(page_title="CAN signal analyzer", layout="wide")
    st.title("CAN signal analyzer")
    st.caption("Assumes abs_time is in seconds, bytes are hex, and bit 0 is the least-significant bit.")

    uploaded = st.file_uploader("Upload a CAN CSV (columns: abs_time, id, B1 to B8)", type=["csv"])
    user_text = st.text_area(
        "Describe your analysis",
        placeholder="For example: In ID 19 find digital signals with more than 1 second between state changes",
        height=100,
    )

    if st.button("Analyze", type="primary"):
        run_analysis(uploaded, user_text)

    if "output" in st.session_state:
        show_output(st.session_state["output"])


if __name__ == "__main__":
    main()
