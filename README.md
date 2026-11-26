# Vehicle Signal Data Processing

Upload a CAN log (CSV), detect **digital signals** (bits that switch) and **counter signals**
(bytes that count and wrap), and view them as time-series plots. Results are stored per project,
so previous reports can be reopened without processing the CSV again.

| Layer | Technology |
|---|---|
| Frontend | Angular 22 (standalone components, signals), ng2-charts / Chart.js |
| Backend | Python, FastAPI, SQLAlchemy |
| Processing | pandas + NumPy, one class: `SignalProcessor` |
| Database | PostgreSQL (users, projects, digital_signals, counter_signals) |

```
vehicle-signal-app/
├── backend/   main.py (API + auth) · processor.py (SignalProcessor) · database.py (models)
├── frontend/  src/app: api · login · project · process · report
├── sample.csv  small test log (4 CAN ids, 905 rows)
└── README.md
```

## Requirements

- Python 3.11+
- PostgreSQL
- Node.js `^22.22.3`, `^24.15.0` or `>=26` (required by the Angular 22 CLI)

## Run it

**1. Database**

```bash
createdb vehicle_signals
```

**2. Backend** (tables are created automatically on startup)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set DATABASE_URL and a long random JWT_SECRET
uvicorn main:app --reload   # http://localhost:8000  (API docs at /docs)
```

**3. Frontend**

```bash
cd frontend
npm install
npm start                   # http://localhost:4200
```

The API address is the `API_URL` constant at the top of `frontend/src/app/api.ts`.
The backend allows `http://localhost:4200` by default (`CORS_ORIGINS` in `.env` to change it).

## Using it

1. Create an account, then sign in.
2. Enter project name, vehicle make, model and year.
3. Choose a CSV, tick **Digital signals** and/or **Counter signals**, press **Generate report**.
   Try `sample.csv` first.
4. The report has a Digital and a Counter tab. Each detected signal gets its own plot
   (x = `abs_time`, y = value) headed by its CAN ID, byte/bit location and signal type.
5. **View history** lists earlier projects; **Open report** loads the stored signals from
   the database. The CSV is not processed again.

## CSV format

```
abs_time,id,B1,B2,B3,B4,B5,B6,B7,B8
0.000,1A0,01,00,FF,F1,A0,00,00,00
```

`B1`–`B8` are hex bytes (`FF`, `0a` and `0x3C` all work; empty cells count as `00`).
`id` is kept as text. Rows with an invalid time, id or byte are skipped and counted.

## Detection rules

Everything runs per CAN id, with rows ordered by `abs_time`. Bits are numbered **0 (MSB) to 7 (LSB)**
inside each byte.

**Counter (per byte).** At least 3 distinct values; the value changes on at least half of the frames;
the most common positive difference is the step; a drop from a common maximum to a common minimum is
a wrap (for example 7 to 1); at least 90 % of the non-zero differences are either the step or a wrap;
and at least one wrap occurs.

**Digital (per bit).** At least 2 state changes (so permanently-0, permanently-1 and one-way flips
are ignored) and the state is held for at least 3 samples on average (which rejects flickering
bits). Bytes detected as counters are excluded so their bits are not reported as digital signals.

The thresholds are arguments of `SignalProcessor(...)` (`min_transitions`, `min_mean_run`,
`min_explained_ratio`, `min_wraps`, `min_samples`). You can also test the processor alone:
`python processor.py ../sample.csv`.

## Notes

- Heuristics, not guarantees: bits of a slowly changing analog value can look digital, a bit that
  toggles on every frame is skipped, and counters are detected on whole bytes (not nibbles).
- Large series are decimated in the browser (Chart.js decimation), and charts are created only when
  they scroll into view.
- Uploads are limited to 20 MB (`MAX_UPLOAD_MB` in `.env`).
- Passwords are stored as bcrypt hashes. The login token is kept in `localStorage`.
- The UI loads the Archivo font from Google Fonts and falls back to the system font when offline.
