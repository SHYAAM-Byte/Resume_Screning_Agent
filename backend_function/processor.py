
import numpy as np
import pandas as pd
import os
import cantools

class SignalProcessor:
    BYTE_COLS = [f"B{i}" for i in range(1, 9)]
    _HEX_LUT = {
        **{f"{v:X}": v for v in range(256)},
        **{f"{v:02X}": v for v in range(256)},
    }

    def __init__(
        self,
        csv_file,
        min_samples: int = 10,
        min_transitions: int = 2,
        min_mean_run: float = 3.0,
        min_change_ratio: float = 0.5,
        min_explained_ratio: float = 0.9,
        min_wraps: int = 1,
    ):
        self.min_samples = max(min_samples, 3)
        self.min_transitions = min_transitions
        self.min_mean_run = min_mean_run
        self.min_change_ratio = min_change_ratio
        self.min_explained_ratio = min_explained_ratio
        self.min_wraps = min_wraps
        self.df, self.skipped_rows = self._load(csv_file)

    # ------------------------------------------------------------------ public

    def process(self, digital: bool = True, counter: bool = True) -> dict:

        digital_out = []
        counter_out = []

        for can_id, g in self.df.groupby("id", sort=True):

            if len(g) < self.min_samples:
                continue

            t = g["abs_time"].to_numpy(dtype=float)

            m = np.ascontiguousarray(
                g[self.BYTE_COLS].to_numpy(dtype=np.uint8)
            )

            counters = self._detect_counters(can_id, t, m)

            if counter:
                counter_out.extend(counters)

            if digital:
                counter_bytes = {c["byte"] for c in counters}

                digital_out.extend(
                    self._detect_digital(
                        can_id, t, m, counter_bytes
                    )
                )

        # Filter digital signals to retain one continuous HIGH or LOW run.
        digital_out = self._filter_digital_signals(digital_out)

        # Generate DBC file.
        dbc_path = self._create_dbc(
            digital_out,
            counter_out
        )

        return {
            "row_count": int(len(self.df)),
            "skipped_rows": int(self.skipped_rows),
            "digital_signals": digital_out,
            "counter_signals": counter_out,
            "dbc_path": dbc_path,
        }

    def _load(self, csv_file) -> tuple[pd.DataFrame, int]:
        try:
            raw = pd.read_csv(csv_file, dtype=str, skipinitialspace=True)
        except Exception as exc:
            raise ValueError(f"Could not read CSV: {exc}") from exc

        canon = {c.lower(): c for c in ["abs_time", "id", *self.BYTE_COLS]}
        raw = raw.rename(
            columns=lambda c: canon.get(str(c).strip().lower(), str(c).strip())
        )
        missing = [c for c in canon.values() if c not in raw.columns]
        if missing:
            raise ValueError(f"CSV is missing column(s): {', '.join(missing)}")

        df = pd.DataFrame(
            {
                "abs_time": pd.to_numeric(raw["abs_time"], errors="coerce"),
                "id": raw["id"].str.strip(),
            }
        )
        for col in self.BYTE_COLS:
            # Empty cells (frames shorter than 8 bytes) count as 0x00.
            # Non-empty cells that are not valid hex bytes make the row invalid.
            s = (
                raw[col]
                .fillna("0")
                .str.strip()
                .str.replace(r"^0[xX]", "", regex=True)
                .str.upper()
                .replace("", "0")
            )
            df[col] = pd.to_numeric(s.map(self._HEX_LUT), errors="coerce")

        total = len(df)
        df = df[df.notna().all(axis=1)].copy()
        if df.empty:
            raise ValueError("CSV contains no valid rows")
        df[self.BYTE_COLS] = df[self.BYTE_COLS].astype(np.uint8)
        df = df.sort_values("abs_time", kind="stable").reset_index(drop=True)
        return df, total - len(df)

    # ----------------------------------------------------------------- counters

    def _detect_counters(self, can_id: str, t: np.ndarray, m: np.ndarray) -> list[dict]:
        found = []
        t_list = t.tolist()
        for b in range(8):
            x = m[:, b].astype(np.int32)
            res = self._analyze_counter(x)
            if res is None:
                continue
            min_val, max_val, step = res
            found.append(
                {
                    "signal_type": "counter",
                    "can_id": can_id,
                    "byte": b + 1,
                    "min": min_val,
                    "max": max_val,
                    "step": step,
                    "abs_time": t_list,
                    "values": x.tolist(),
                }
            )
        return found

    def _analyze_counter(self, x: np.ndarray):
        if np.unique(x).size < 3:
            return None
        d = np.diff(x)
        nonzero = d != 0
        if nonzero.sum() / len(d) < self.min_change_ratio:
            return None

        up = d[d > 0]
        down = d < 0
        if up.size == 0 or not down.any():
            return None

        step = int(np.bincount(up).argmax())  # most common increment

        # Most common (value before drop, value after drop) = (counter max, counter min)
        pairs = x[:-1][down] * 256 + x[1:][down]
        max_val, min_val = divmod(int(np.bincount(pairs).argmax()), 256)

        wrap = down & (x[:-1] == max_val) & (x[1:] == min_val)
        explained = ((d == step) | wrap)[nonzero].mean()
        if explained < self.min_explained_ratio or wrap.sum() < self.min_wraps:
            return None
        return min_val, max_val, step

    # ------------------------------------------------------------------ digital

    def _detect_digital(
        self, can_id: str, t: np.ndarray, m: np.ndarray, skip_bytes: set
    ) -> list[dict]:
        n = len(t)
        bits = np.unpackbits(m, axis=1)  # (n, 64); per byte: column 0 = MSB ... 7 = LSB
        transitions = (bits[1:] != bits[:-1]).sum(axis=0)  # state changes per bit column
        mean_run = n / (transitions + 1)  # average samples a state is held
        ok = (transitions >= self.min_transitions) & (mean_run >= self.min_mean_run)

        found = []
        t_list = t.tolist()
        for col in np.flatnonzero(ok):
            byte, bit = divmod(int(col), 8)
            if byte + 1 in skip_bytes:
                continue
            found.append(
                {
                    "signal_type": "digital",
                    "can_id": can_id,
                    "byte": byte + 1,
                    "bit": bit,
                    "transitions": int(transitions[col]),
                    "abs_time": t_list,
                    "values": bits[:, col].tolist(),
                }
            )
        return found
    def _filter_digital_signals(self, digital_signals):
        filtered = []

        for signal in digital_signals:
            values = signal["values"]

            high_runs = sum(
                1 for i, value in enumerate(values)
                if value == 1 and (i == 0 or values[i - 1] == 0)
            )

            low_runs = sum(
                1 for i, value in enumerate(values)
                if value == 0 and (i == 0 or values[i - 1] == 1)
            )

            if high_runs == 1 or low_runs == 1:
                filtered.append(signal)

        return filtered


    def _create_dbc(self, digital_signals, counter_signals):
        db = cantools.database.Database()

        signals_by_id = {}

        for signal in digital_signals:
            signals_by_id.setdefault(signal["can_id"], []).append({
                "name": (
                    f"B{signal['byte']}_bit{signal['bit']}"
                ),
                "start": (signal["byte"] - 1) * 8 + (7 - signal["bit"]),
                "length": 1,
                "minimum": 0,
                "maximum": 1,
            })

        for signal in counter_signals:
            signals_by_id.setdefault(signal["can_id"], []).append({
                "name": f"B{signal['byte']}_Counter",
                "start": (signal["byte"] - 1) * 8,
                "length": 8,
                "minimum": signal["min"],
                "maximum": signal["max"],
            })

        for can_id, extracted_signals in signals_by_id.items():

            message_signals = []

            for sig in extracted_signals:
                message_signals.append(
                    cantools.database.Signal(
                        name=sig["name"],
                        start=sig["start"],
                        length=sig["length"],
                        byte_order="big_endian" if sig["length"] == 1 else "little_endian",
                        is_signed=False,
                        minimum=sig["minimum"],
                        maximum=sig["maximum"],
                    )
                )

            try:
                frame_id = int(str(can_id), 16)
            except ValueError:
                frame_id = int(can_id)

            message = cantools.database.Message(
                frame_id=frame_id,
                name=f"CAN_{frame_id:X}",
                length=8,
                signals=message_signals,
                is_extended_frame=frame_id > 0x7FF,
            )

            db.messages.append(message)

            import tempfile

            dbc_path = os.path.join(
                tempfile.gettempdir(),
                "extracted_signals.dbc"
            )

        db.refresh()

        cantools.database.dump_file(
            db,
            dbc_path,
            encoding="utf-8"
        )

        return dbc_path


if __name__ == "__main__":
    import sys

    result = SignalProcessor(sys.argv[1]).process()
    print(f"rows used: {result['row_count']}  skipped: {result['skipped_rows']}")
    for s in result["digital_signals"]:
        print(f"digital  id={s['can_id']}  B{s['byte']}.bit{s['bit']}  transitions={s['transitions']}")
    for s in result["counter_signals"]:
        print(f"counter  id={s['can_id']}  B{s['byte']}  {s['min']}..{s['max']} step {s['step']}")
