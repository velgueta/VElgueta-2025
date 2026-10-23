"""Local three-month DAS TM candidate, pending production-version review.

Preserved without algorithm changes. See README.md in this folder before running.
Requires server data and verified helper-module versions; not validated end to end.
"""

import os, re, glob
from datetime import datetime, timezone, timedelta
from typing import List
from scipy.signal import butter

# your modules (leave them as-is)
from templatematching import *
from getanalisisfiles import *
from template_maker2 import *
# ===== parallel correlations over templates =====
from multiprocessing import Pool, cpu_count
import time

# -------- settings you likely only tweak here --------
# raw layout (50 Hz) in year/julian-day folders:
base_year_dir = "/data/fast0/rainier50Hz/2023/"

# where to write templates and CC results
templates_dir = "/data/data4/veronica-scratch-rainier/swarm_august2023/templates-files/template-5sec-50hz/"
cc_base       = "/data/data2/veronica-rainier/cc_results"
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(cc_base, exist_ok=True)

# AUGUST window for templates (USGS events):
tmpl_start_dt = datetime(2023, 8, 25, 0, 0)
tmpl_end_dt   = datetime(2023, 8, 31, 0, 0)

# MATCHING window (example: full Oct 27 UTC)
#match_start = "2023-08-24_00.00"
#match_end   = "2023-11-24_00.00"
match_start = "2023-08-23_21.00"
match_end   = "2023-11-23_21.00"

# DAS + filter
chan_min, chan_max = 0, 2500
template_size = 5.0
fs = 50
samples_per_file = int(60 * fs)
b, a = butter(2, (2.0, 10.0), 'bp', fs=fs)

# filename pattern and helpers
rx = re.compile(r"rainier_50Hz_(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})_UTC\.h5$")

def yday(dt: datetime) -> int:
    return dt.timetuple().tm_yday

def path_for_time(t_utc: datetime) -> str:
    """Return full path to file that covers t_utc, based on minute-aligned filenames."""
    t_utc = t_utc.replace(tzinfo=timezone.utc)
    # files are 1-minute long and named ..._HH.MM.00_UTC.h5
    t0 = t_utc.replace(second=0, microsecond=0)
    day_dir = os.path.join(base_year_dir, f"{yday(t0):03d}")
    fn = f"rainier_50Hz_{t0.strftime('%Y-%m-%d_%H.%M')}.00_UTC.h5"
    return os.path.join(day_dir, fn)

def build_file_list_julian(start_str: str, end_str: str) -> List[str]:
    """Collect minute files between [start, end)."""
    start = datetime.strptime(start_str, "%Y-%m-%d_%H.%M").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(end_str,   "%Y-%m-%d_%H.%M").replace(tzinfo=timezone.utc)
    if end <= start:
        return []
    files = []
    t = start.replace(second=0, microsecond=0)
    while t < end:
        p = path_for_time(t)
        if os.path.isfile(p):
            files.append(p)
        t += timedelta(minutes=1)
    return files

# ---------------- A) TEMPLATES (AUGUST) ----------------
MANUAL_WORKERS = 8  # cámbialo a 8, 16, 32 para tus tests

# ---------------- A) TEMPLATES (AUGUST) ----------------

events = search(starttime=tmpl_start_dt,
                endtime=tmpl_end_dt,
                latitude=46.879967, longitude=-121.726906,
                maxradius=35/111.32)
event_df = get_summary_data_frame(events).sort_values("time")
print(f"[templates] usgs events in august: {len(event_df)}")

found_files = []
original_dates_fixed = []

miss = 0
hit  = 0
for _, row in event_df.iterrows():
    # event time as naive UTC datetime
    t_evt = row["time"]  # should already be UTC (pandas/obspy)
    if isinstance(t_evt, str):
        t_evt = datetime.strptime(
            t_evt.replace("_"," ").replace(".",":"),
            "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    else:
        t_evt = pd.Timestamp(t_evt).to_pydatetime().replace(tzinfo=timezone.utc)

    p = path_for_time(t_evt)
    if os.path.isfile(p):
        found_files.append(p)
        # template_maker2 expects 'YYYY-MM-DD HH:MM:SS.%f' for the event time
        original_dates_fixed.append(t_evt.strftime("%Y-%m-%d %H:%M:%S.%f"))
        hit += 1
    else:
        print(f"No raw file for event: {t_evt.strftime('%Y-%m-%d_%H.%M.%S')}")
        miss += 1

print(f"[templates] matched raw files: {hit}  | missing: {miss}")

if hit > 0:
    process_files_to_cut(
        found_files, original_dates_fixed,
        base_year_dir, templates_dir,
        chan_min, chan_max, template_size
    )
else:
    print("[templates] no templates cut (no august raw found)")

template_list = glob.glob(os.path.join(templates_dir, "*"))
print(f"[templates] templates found on disk: {len(template_list)}")

# ---------------- B) MATCHING (OCTOBER) ----------------

file_list = build_file_list_julian(match_start, match_end)
print(f"[matching] raw files in window: {len(file_list)}")

cc_out    = os.path.join(cc_base, f"CC_{int(template_size)}sec-tem_{match_start}-{match_end}")
plots_dir = os.path.join(cc_base, f"plot-CC-{int(template_size)}sec-{match_start}-{match_end}")
ts_h5_dir = os.path.join(cc_base, f"h5_timestamps_{match_start}-{match_end}")
os.makedirs(cc_out, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(ts_h5_dir, exist_ok=True)

# run matching (parallel over templates)
if file_list and template_list:
    # evitar oversubscription en el cluster
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # definir nº de workers (núcleos lógicos) a usar
    if MANUAL_WORKERS is None:
        workers = min(cpu_count(), len(template_list)) or 1
    else:
        workers = min(MANUAL_WORKERS, len(template_list)) or 1

    print(f"[matching] cpu_count() = {cpu_count()}")
    print(f"[matching] using {workers} workers over {len(template_list)} templates")

    # split templates into interleaved chunks
    chunks = [template_list[i::workers] for i in range(workers)]

    def _run_chunk(t_sublist):
        if not t_sublist:
            return
        process_files_dos(
            file_list, t_sublist,
            chan_min, chan_max, (chan_max - chan_min),
            samples_per_file, b, a, cc_out
        )

    # medir solo el tiempo del matching paralelo
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_run_chunk, chunks):
            pass
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"[matching] elapsed wall time: {elapsed:.1f} s ({elapsed/60:.2f} min)")

else:
    if not template_list:
        print("[matching] no templates available")
    if not file_list:
        print("[matching] no raw files in the chosen window")

# timestamps + plots (always try after matching attempt)
out_h5   = create_timestamps_h5(file_list, ts_h5_dir)
time_utc = convert_timestamps_to_utc(out_h5)
try:
    process_folders_and_plot(cc_out, fs, time_utc, plots_dir, found_files)
except Exception as e:
    print("[plot] skipped:", e)
