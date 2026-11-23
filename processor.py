
import re

import numpy as np
import pandas as pd

BYTE_COLUMNS = [f"B{i}" for i in range(1, 9)]  # B1 ... B8
REQUIRED_COLUMNS = ["abs_time", "id"] + BYTE_COLUMNS

MIN_COUNTER_SAMPLES = 8    # a byte needs at least this many frames to be judged
MIN_INCREMENT_RATIO = 0.5  # at least 50 % of the steps must be "+1"
MIN_VALID_RATIO = 0.9      # at least 90 % of the steps must be "+1" or a wrap/reset

DIGITAL_SUMMARY_COLS = ["can_id", "byte", "bit", "state_changes",
                        "min_gap_s", "first_change_s", "last_change_s"]
DIGITAL_DATA_COLS = ["can_id", "byte", "bit", "abs_time", "value"]
COUNTER_SUMMARY_COLS = ["can_id", "byte", "samples", "min_value", "max_value",
                        "wraps", "increment_pct", "min_gap_s"]
COUNTER_DATA_COLS = ["can_id", "byte", "abs_time", "value"]

# Words the AI may use to say "no restriction" 
_ANY = {"", "all", "any", "*", "none", "null"}


class SignalProcessor:
    def __init__(self, df):
        """`df` is the raw CSV table (all cells as text). Raises ValueError if it is unusable."""
        if df is None or df.empty:
            raise ValueError("The CSV file has no data rows.")
        wanted = {name.lower(): name for name in REQUIRED_COLUMNS}
        df = df.rename(columns=lambda c: wanted.get(str(c).strip().lower(), str(c).strip()))
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing column(s): {', '.join(missing)}. "
                f"The CSV must contain: {', '.join(REQUIRED_COLUMNS)}."
            )
        df = df[REQUIRED_COLUMNS].copy()
        abs_time = pd.to_numeric(df["abs_time"], errors="coerce")
        if abs_time.isna().any():
            raise ValueError(
                "abs_time must be a number on every row. "
                f"Problem at CSV line(s): {self._lines(df.index[abs_time.isna()])}."
            )
        df["abs_time"] = abs_time.astype(float)
        id_text = df["id"].fillna("").astype(str).str.strip().str.replace(r"^0[xX]", "", regex=True)
        bad_id = ~id_text.str.fullmatch(r"[0-9A-Fa-f]{1,8}")
        if bad_id.any():
            examples = ", ".join(repr(x) for x in id_text[bad_id].unique()[:3])
            raise ValueError(
                f"Invalid CAN ID in column 'id': {examples}. "
                f"Problem at CSV line(s): {self._lines(df.index[bad_id])}. IDs must be hex, e.g. 19 or 1A0."
            )
        df["can_id"] = id_text.map(lambda s: int(s, 16)).astype("int64")
        self.df = df[["can_id", "abs_time"] + BYTE_COLUMNS].sort_values("abs_time", kind="stable")

    def get_ids(self):
        """All CAN IDs found in the file, as sorted upper-case hex strings."""
        return [format(int(i), "X") for i in sorted(self.df["can_id"].unique())]

    def validate_request(self, request):
        if not isinstance(request, dict):
            raise ValueError("The instruction must be a JSON object.")
        raw_id = request.get("id")
        items = raw_id if isinstance(raw_id, list) else [raw_id]
        items = [str(i).strip() for i in items if i is not None]
        if not items or any(i.lower() in _ANY for i in items):
            clean_id = "all"
        else:
            known = set(self.df["can_id"].unique())
            names = []
            for item in items:
                number = self._parse_id(item)
                if number not in known:
                    ids = self.get_ids()
                    preview = ", ".join(ids[:20]) + (" ..." if len(ids) > 20 else "")
                    raise ValueError(
                        f"CAN ID {format(number, 'X')} does not exist in this CSV. IDs found: {preview}."
                    )
                name = format(number, "X")
                if name not in names:
                    names.append(name)
            clean_id = names[0] if len(names) == 1 else names

        raw = request.get("analysis")
        text = " ".join(map(str, raw)) if isinstance(raw, list) else str(raw)
        text = text.strip().lower()
        digital = "digital" in text
        counter = "counter" in text
        if text in ("", "none", "null", "all", "both"):
            digital = counter = True
        if not (digital or counter):
            raise ValueError(f"Unknown analysis type {raw!r}. Use 'digital', 'counter' or 'both'.")
        analysis = "both" if (digital and counter) else ("digital" if digital else "counter")
        byte_names = self._clean_bytes(request.get("bytes"))
        raw = request.get("min_interval")
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in _ANY):
            min_interval = None
        else:
            try:
                min_interval = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"min_interval must be a number of seconds, not {raw!r}.") from None
            if not np.isfinite(min_interval) or min_interval < 0:
                raise ValueError("min_interval must be zero or a positive number of seconds.")
            min_interval = min_interval or None  # 0 means "no time condition"

        return {"id": clean_id, "analysis": analysis, "bytes": byte_names, "min_interval": min_interval}
    def analyze(self, request):
        request = self.validate_request(request)
        want_digital = request["analysis"] in ("digital", "both")
        want_counter = request["analysis"] in ("counter", "both")
        byte_names = request["bytes"]
        min_interval = request["min_interval"]
        data = self.df
        if request["id"] != "all":
            wanted = request["id"] if isinstance(request["id"], list) else [request["id"]]
            data = data[data["can_id"].isin([int(i, 16) for i in wanted])]
        numbers = data[["can_id", "abs_time"]].copy()
        for name in byte_names:
            numbers[name] = self._hex_to_number(data[name], name)

        digital_rows, digital_frames, counter_rows, counter_frames = [], [], [], []
        for can_id, group in numbers.groupby("can_id", sort=True):
            id_text = format(int(can_id), "X")
            times = group["abs_time"].to_numpy()
            for byte in byte_names:
                values = group[byte].to_numpy()
                present = ~np.isnan(values)                 # skip frames where this byte is blank
                t, v = times[present], values[present].astype(np.int64)
                if len(v) < 2:
                    continue
                if want_digital:
                    rows, frames = self._find_digital(id_text, byte, t, v, min_interval)
                    digital_rows += rows
                    digital_frames += frames
                if want_counter:
                    rows, frames = self._find_counter(id_text, byte, t, v, min_interval)
                    counter_rows += rows
                    counter_frames += frames

        results = {}
        if want_digital:
            results["digital"] = {
                "summary": pd.DataFrame(digital_rows, columns=DIGITAL_SUMMARY_COLS),
                "data": self._join(digital_frames, DIGITAL_DATA_COLS),
            }
        if want_counter:
            results["counter"] = {
                "summary": pd.DataFrame(counter_rows, columns=COUNTER_SUMMARY_COLS),
                "data": self._join(counter_frames, COUNTER_DATA_COLS),
            }
        return results

    @staticmethod
    def _find_digital(id_text, byte, t, v, min_interval):
        summaries, frames = [], []
        for bit in range(8):
            bits = (v >> bit) & 1                                    # this bit in every frame
            change_idx = np.flatnonzero(bits[1:] != bits[:-1]) + 1   # frames where the bit flipped
            if len(change_idx) == 0:
                continue                                             # constant bit -> ignore
            change_times = t[change_idx]
            gaps = np.diff(change_times)                             # time between state changes
            if min_interval is not None and len(gaps) > 0 and gaps.min() <= min_interval:
                continue                                             # changes too close together

            keep = np.unique(np.concatenate(([0], change_idx, [len(v) - 1])))
            frames.append(pd.DataFrame({
                "can_id": id_text, "byte": byte, "bit": bit,
                "abs_time": t[keep], "value": bits[keep],
            }))
            summaries.append({
                "can_id": id_text, "byte": byte, "bit": bit,
                "state_changes": len(change_idx),
                "min_gap_s": round(float(gaps.min()), 6) if len(gaps) else np.nan,
                "first_change_s": float(change_times[0]),
                "last_change_s": float(change_times[-1]),
            })
        return summaries, frames

    @staticmethod
    def _find_counter(id_text, byte, t, v, min_interval):
        if len(v) < MIN_COUNTER_SAMPLES or len(np.unique(v)) < 3:
            return [], []                                            # too short / (almost) constant
        steps = np.diff(v)
        increments = steps == 1
        wraps = steps < 0
        if increments.mean() < MIN_INCREMENT_RATIO or (increments | wraps).mean() < MIN_VALID_RATIO:
            return [], []

        gaps = np.diff(t)                                            # time between counter updates
        if min_interval is not None and gaps.min() <= min_interval:
            return [], []

        summary = {
            "can_id": id_text, "byte": byte, "samples": len(v),
            "min_value": int(v.min()), "max_value": int(v.max()),
            "wraps": int(wraps.sum()),
            "increment_pct": round(float(increments.mean()) * 100, 1),
            "min_gap_s": round(float(gaps.min()), 6),
        }
        frame = pd.DataFrame({"can_id": id_text, "byte": byte, "abs_time": t, "value": v})
        return [summary], [frame]

    @staticmethod
    def _lines(index, limit=5):
        """Turn row labels into CSV line numbers for error messages (line 1 is the header)."""
        shown = ", ".join(str(int(i) + 2) for i in index[:limit])
        extra = len(index) - limit
        return shown + (f" (and {extra} more)" if extra > 0 else "")

    @staticmethod
    def _parse_id(text):
        """'19', '0x19' or '1a0' -> integer. Raises ValueError if it is not a hex number."""
        cleaned = re.sub(r"^0[xX]", "", str(text).strip())
        if not re.fullmatch(r"[0-9A-Fa-f]{1,8}", cleaned):
            raise ValueError(f"{text!r} is not a valid CAN ID. Use hex, e.g. 19 or 1A0.")
        return int(cleaned, 16)

    @staticmethod
    def _clean_bytes(raw):
        """Turn the AI's 'bytes' value into a sorted list like ['B1', 'B2']."""
        if raw is None or raw == [] or (isinstance(raw, str) and raw.strip().lower() in _ANY):
            return list(BYTE_COLUMNS)                                # nothing given -> all bytes

        parts = []
        for item in (raw if isinstance(raw, list) else [raw]):
            parts += re.split(r"\s*,\s*", str(item).strip().upper())

        names = []
        for part in parts:
            # Understand ranges such as "B1-B4", "B1 TO B4" or "1-4".
            match = re.fullmatch(r"B?(\d)\s*(?:-|TO|THROUGH)\s*B?(\d)", part)
            if match:
                found = [f"B{n}" for n in range(int(match[1]), int(match[2]) + 1)]
            else:
                found = ["B" + part if part.isdigit() else part]     # "4" -> "B4"
            for name in found:
                if name not in BYTE_COLUMNS:
                    raise ValueError(f"{name!r} is not a valid byte name. Use B1 ... B8.")
                if name not in names:
                    names.append(name)
        if not names:
            raise ValueError("No valid byte names found. Use B1 ... B8.")
        return sorted(names)

    @staticmethod
    def _hex_to_number(series, column):
        """Hex text ('FF', '0x1A', '7') -> numbers. Blank -> NaN. Raises ValueError for bad hex."""
        text = series.fillna("").astype(str).str.strip().str.replace(r"^0[xX]", "", regex=True)
        blank = text == ""
        valid = text.str.fullmatch(r"[0-9A-Fa-f]{1,2}")
        bad = ~blank & ~valid
        if bad.any():
            examples = ", ".join(repr(x) for x in text[bad].unique()[:3])
            raise ValueError(
                f"Invalid hex value in column {column}: {examples}. "
                f"Problem at CSV line(s): {SignalProcessor._lines(series.index[bad])}. "
                "Bytes must be hex between 00 and FF."
            )
        numbers = pd.Series(np.nan, index=series.index, dtype="float64")
        filled = ~blank
        numbers.loc[filled] = text[filled].map(lambda s: int(s, 16)).astype("float64")
        return numbers

    @staticmethod
    def _join(frames, columns):
        """Stack many small result tables into one (or return an empty table)."""
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
