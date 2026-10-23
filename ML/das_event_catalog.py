#!/usr/bin/env python3
"""Create a self-contained DAS event catalog from MORADAS ML S-pick CSVs.

USAGE (no required arguments):

    1. Put this script in the directory that contains the MORADAS CSV files,
       or in a parent directory.
    2. Run:  python3 das_event_catalog.py

The script recursively finds MORADAS_*_AR.csv files, applies a DAS-only event
criterion, performs a physical S-wave moveout grid search using the embedded
cable geometry, and writes DAS_catalog_results/DAS_event_catalog.csv.

It does NOT use ComCat or PyOcto. The resulting catalog is designed to be
matched with a PyOcto station-event catalog in a later step. Absolute DAS time
is never used to judge moveout, so a common interrogator clock error cancels.

Only the Python standard library is required.
"""

from __future__ import print_function

import argparse
import base64
import bisect
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path


ANALYSIS_VERSION = "rainier-das-catalog-v1"

# Study and detection settings. These values are also written into every run's
# configuration file and the main event catalog.
RAINIER_LATITUDE = 46.879967
RAINIER_LONGITUDE = -121.726906
SOURCE_RADIUS_KM = 25.0
GRID_SPACING_KM = 2.5
DEPTHS_KM = tuple(float(value) for value in range(0, 21, 2))
S_VELOCITIES_KM_S = (2.5, 3.0, 3.5, 4.0)
MAX_GRID_PICKS = 80

CHANNEL_STRIDE = 4
MIN_SAMPLED_CHANNELS = 10
STRONG_SUPPORT_CHANNELS = 20
MAX_CHANNEL = 3000
MAX_PICK_SPREAD_S = 1.0
OVERLAPPING_BLOCK_MERGE_S = 2.0
EVENT_GROUPING_GAP_S = 3.0

LOCAL_SMOOTH_WINDOW = 9
MAX_LOCAL_SMOOTH_MAD_S = 0.10
MAX_LOCAL_RESIDUAL_S = 0.20
MIN_LOCAL_INLIER_FRACTION = 0.85

MAX_GRID_MOVEOUT_MAD_S = 0.10
MAX_GRID_MOVEOUT_P95_S = 0.25
MIN_GRID_INLIER_FRACTION_020S = 0.85
RECOMMENDED_PYOCTO_MATCH_HALF_WIDTH_S = 15.0

CHANNEL_RE = re.compile(r"das(\d+)$", re.IGNORECASE)
TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
)

# Cable channels 0-3000 sampled every 10 channels. Values are
# channel,optical_distance_m,latitude,longitude,elevation_m. Geometry for an
# arbitrary channel is obtained by linear interpolation between control points.
# The compressed payload makes this script portable as a single file.
EMBEDDED_GEOMETRY_ZLIB_BASE64 = """
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
""".strip()


def parse_time(text):
    value = str(text).strip()
    for time_format in TIME_FORMATS:
        try:
            return datetime.strptime(value, time_format)
        except ValueError:
            pass
    raise ValueError("Could not parse time: %s" % value)


def iso(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return value


def percentile(values, percent):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def robust_mad(values):
    if not values:
        return float("nan")
    center = statistics.median(values)
    return 1.4826 * statistics.median(abs(value - center) for value in values)


def datetime_median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2


def datetime_to_epoch(value):
    return (value - datetime(1970, 1, 1)).total_seconds()


def epoch_to_datetime(value):
    return datetime(1970, 1, 1) + timedelta(seconds=value)


class EmbeddedGeometry(object):
    def __init__(self):
        packed = base64.b64decode(EMBEDDED_GEOMETRY_ZLIB_BASE64.encode("ascii"))
        text = zlib.decompress(packed).decode("ascii")
        self.channels = []
        self.rows = []
        cosine = math.cos(math.radians(RAINIER_LATITUDE))
        for line in text.splitlines():
            channel, distance, latitude, longitude, elevation = line.split(",")
            row = {
                "channel": int(channel),
                "distance_km": float(distance) / 1000.0,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "elevation_m": float(elevation),
            }
            row["x_km"] = (
                (row["longitude"] - RAINIER_LONGITUDE) * 111.32 * cosine
            )
            row["y_km"] = (row["latitude"] - RAINIER_LATITUDE) * 111.32
            self.channels.append(row["channel"])
            self.rows.append(row)

    def get(self, channel):
        if channel < self.channels[0] or channel > self.channels[-1]:
            return None
        index = bisect.bisect_left(self.channels, channel)
        if index < len(self.channels) and self.channels[index] == channel:
            return dict(self.rows[index])
        lower = self.rows[index - 1]
        upper = self.rows[index]
        fraction = float(channel - lower["channel"]) / (
            upper["channel"] - lower["channel"]
        )
        output = {"channel": channel}
        for key in (
            "distance_km",
            "latitude",
            "longitude",
            "elevation_m",
            "x_km",
            "y_km",
        ):
            output[key] = lower[key] + fraction * (upper[key] - lower[key])
        return output


def read_s_picks(path):
    picks = []
    data_rows = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            return picks, data_rows
        for raw_row in reader:
            data_rows += 1
            row = {str(key).strip(): value for key, value in raw_row.items() if key is not None}
            station_code = str(row.get("station_code", "")).strip()
            match = CHANNEL_RE.match(station_code)
            arrival_text = str(row.get("trace_s_arrival", "")).strip()
            if not match or not arrival_text:
                continue
            try:
                arrival = parse_time(arrival_text)
            except ValueError:
                continue
            picks.append({"channel": int(match.group(1)), "arrival": arrival})
    picks.sort(key=lambda row: row["arrival"])
    return picks, data_rows


def contiguous_runs(channels):
    channels = sorted(set(channels))
    if not channels:
        return []
    runs = [[channels[0]]]
    for channel in channels[1:]:
        if channel - runs[-1][-1] == CHANNEL_STRIDE:
            runs[-1].append(channel)
        else:
            runs.append([channel])
    return runs


def make_block(window, channels):
    allowed = set(channels)
    by_channel = {}
    for pick in window:
        if pick["channel"] in allowed:
            by_channel.setdefault(pick["channel"], []).append(pick["arrival"])
    block_picks = [
        {"channel": channel, "arrival": min(by_channel[channel])}
        for channel in channels
    ]
    arrivals = [row["arrival"] for row in block_picks]
    return {
        "arrival_start": min(arrivals),
        "arrival_end": max(arrivals),
        "channel_start": min(channels),
        "channel_end": max(channels),
        "n_sampled_channels": len(channels),
        "block_picks": block_picks,
    }


def blocks_overlap(first, second):
    close_time = (
        second["arrival_start"]
        <= first["arrival_end"] + timedelta(seconds=OVERLAPPING_BLOCK_MERGE_S)
        and first["arrival_start"]
        <= second["arrival_end"] + timedelta(seconds=OVERLAPPING_BLOCK_MERGE_S)
    )
    close_channel = not (
        second["channel_start"] > first["channel_end"] + CHANNEL_STRIDE
        or first["channel_start"] > second["channel_end"] + CHANNEL_STRIDE
    )
    return close_time and close_channel


def better_block(first, second):
    if first["n_sampled_channels"] != second["n_sampled_channels"]:
        return first if first["n_sampled_channels"] > second["n_sampled_channels"] else second
    first_spread = (first["arrival_end"] - first["arrival_start"]).total_seconds()
    second_spread = (second["arrival_end"] - second["arrival_start"]).total_seconds()
    return first if first_spread <= second_spread else second


def merge_overlapping_blocks(candidates):
    merged = []
    for candidate in sorted(candidates, key=lambda row: row["arrival_start"]):
        matching = [index for index, row in enumerate(merged) if blocks_overlap(row, candidate)]
        if not matching:
            merged.append(candidate)
            continue
        best = candidate
        for index in matching:
            best = better_block(best, merged[index])
        merged = [row for index, row in enumerate(merged) if index not in matching]
        merged.append(best)
    return sorted(merged, key=lambda row: row["arrival_start"])


def detect_blocks(picks):
    candidates = []
    end_index = 0
    for start_index, anchor in enumerate(picks):
        end_index = max(end_index, start_index)
        window_end = anchor["arrival"] + timedelta(seconds=MAX_PICK_SPREAD_S)
        while end_index < len(picks) and picks[end_index]["arrival"] <= window_end:
            end_index += 1
        window = picks[start_index:end_index]
        runs = contiguous_runs(row["channel"] for row in window)
        if not runs:
            continue
        longest = max(runs, key=len)
        if len(longest) >= MIN_SAMPLED_CHANNELS:
            candidate = make_block(window, longest)
            if candidate["channel_end"] <= MAX_CHANNEL:
                candidates.append(candidate)
    return merge_overlapping_blocks(candidates)


def group_event_blocks(blocks):
    groups = []
    for block in sorted(blocks, key=lambda row: row["arrival_start"]):
        if not groups:
            groups.append([block])
            continue
        group_end = max(row["arrival_end"] for row in groups[-1])
        if block["arrival_start"] <= group_end + timedelta(seconds=EVENT_GROUPING_GAP_S):
            groups[-1].append(block)
        else:
            groups.append([block])
    return groups


def rolling_median(values, window):
    radius = window // 2
    output = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        output.append(statistics.median(values[start:end]))
    return output


def longest_contiguous_run(channels):
    ordered = sorted(set(channels))
    if not ordered:
        return 0
    longest = current = 1
    for previous, channel in zip(ordered, ordered[1:]):
        current = current + 1 if channel - previous == CHANNEL_STRIDE else 1
        longest = max(longest, current)
    return longest


def linear_diagnostic(distances, observed):
    mean_x = sum(distances) / len(distances)
    mean_y = sum(observed) / len(observed)
    denominator = sum((value - mean_x) ** 2 for value in distances)
    slope = 0.0 if denominator == 0 else sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(distances, observed)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [
        y_value - (intercept + slope * x_value)
        for x_value, y_value in zip(distances, observed)
    ]
    velocity = float("inf") if abs(slope) < 1e-12 else 1.0 / abs(slope)
    return slope, velocity, robust_mad(residuals)


def classify_block(block, block_number, geometry):
    rows = []
    for pick in block["block_picks"]:
        metadata = geometry.get(pick["channel"])
        if metadata is not None:
            rows.append((pick, metadata))
    rows.sort(key=lambda item: item[1]["distance_km"])
    if len(rows) < MIN_SAMPLED_CHANNELS:
        return None, []

    reference = datetime_median([item[0]["arrival"] for item in rows])
    observed = [(item[0]["arrival"] - reference).total_seconds() for item in rows]
    smooth = rolling_median(observed, min(LOCAL_SMOOTH_WINDOW, len(observed)))
    residuals = [value - trend for value, trend in zip(observed, smooth)]
    residual_center = statistics.median(residuals)
    absolute_residuals = [abs(value - residual_center) for value in residuals]
    inliers = [value <= MAX_LOCAL_RESIDUAL_S for value in absolute_residuals]
    inlier_channels = [
        item[0]["channel"] for item, is_inlier in zip(rows, inliers) if is_inlier
    ]
    n_inliers = sum(inliers)
    inlier_fraction = float(n_inliers) / len(rows)
    longest_run = longest_contiguous_run(inlier_channels)
    smooth_mad = 1.4826 * statistics.median(absolute_residuals)
    smooth_p95 = percentile(absolute_residuals, 95.0)
    adjacent_dt = [
        abs(observed[index] - observed[index - 1])
        for index in range(1, len(observed))
    ]
    distances = [item[1]["distance_km"] for item in rows]
    slope, velocity, linear_mad = linear_diagnostic(distances, observed)

    reasons = []
    if smooth_mad > MAX_LOCAL_SMOOTH_MAD_S:
        reasons.append("local_smooth_mad_above_limit")
    if inlier_fraction < MIN_LOCAL_INLIER_FRACTION:
        reasons.append("too_many_local_outliers")
    if longest_run < MIN_SAMPLED_CHANNELS:
        reasons.append("insufficient_contiguous_local_inliers")

    if reasons:
        status = "rejected"
    elif n_inliers >= STRONG_SUPPORT_CHANNELS:
        status = "coherent"
    else:
        status = "coherent_low_support"

    block_row = {
        "block_number": block_number,
        "block_status": status,
        "block_rejection_reasons": ";".join(reasons),
        "arrival_start_utc": block["arrival_start"],
        "arrival_end_utc": block["arrival_end"],
        "channel_start": block["channel_start"],
        "channel_end": block["channel_end"],
        "n_sampled_channels": len(rows),
        "n_original_channels_equivalent": len(rows) * CHANNEL_STRIDE,
        "n_local_inliers": n_inliers,
        "n_local_outliers": len(rows) - n_inliers,
        "local_inlier_fraction": inlier_fraction,
        "longest_contiguous_local_inlier_run": longest_run,
        "fiber_distance_min_km": min(distances),
        "fiber_distance_max_km": max(distances),
        "fiber_span_km": max(distances) - min(distances),
        "pick_spread_s": (
            block["arrival_end"] - block["arrival_start"]
        ).total_seconds(),
        "adjacent_dt_median_s": statistics.median(adjacent_dt),
        "adjacent_dt_p95_s": percentile(adjacent_dt, 95.0),
        "local_smooth_residual_mad_s": smooth_mad,
        "local_smooth_residual_p95_s": smooth_p95,
        "local_smooth_residual_max_s": max(absolute_residuals),
        "linear_slope_s_per_km_diagnostic": slope,
        "linear_apparent_velocity_km_s_diagnostic": velocity,
        "linear_residual_mad_s_diagnostic": linear_mad,
    }

    pick_rows = []
    for (pick, metadata), relative, trend, residual, absolute, is_inlier in zip(
        rows, observed, smooth, residuals, absolute_residuals, inliers
    ):
        pick_rows.append(
            {
                "block_number": block_number,
                "block_status": status,
                "channel": pick["channel"],
                "arrival_time_utc": pick["arrival"],
                "fiber_distance_km": metadata["distance_km"],
                "channel_latitude": metadata["latitude"],
                "channel_longitude": metadata["longitude"],
                "channel_elevation_m": metadata["elevation_m"],
                "channel_x_from_rainier_km": metadata["x_km"],
                "channel_y_from_rainier_km": metadata["y_km"],
                "block_relative_time_s": relative,
                "local_trend_time_s": trend,
                "local_residual_s": residual,
                "local_residual_abs_centered_s": absolute,
                "is_local_inlier": is_inlier,
            }
        )
    return block_row, pick_rows


def block_rank(row):
    status_rank = {"rejected": 0, "coherent_low_support": 1, "coherent": 2}
    return (
        status_rank[row["block_status"]],
        row["n_local_inliers"],
        -row["local_smooth_residual_mad_s"],
    )


def select_event_picks(block_rows, picks_by_block):
    accepted = [row for row in block_rows if row["block_status"] != "rejected"]
    use_rows = accepted if accepted else [max(block_rows, key=block_rank)]
    use_rows = sorted(use_rows, key=block_rank, reverse=True)
    selected = {}
    for block_row in use_rows:
        for pick in picks_by_block[block_row["block_number"]]:
            if accepted and not pick["is_local_inlier"]:
                continue
            if pick["channel"] not in selected:
                selected[pick["channel"]] = dict(pick)
    return sorted(selected.values(), key=lambda row: row["channel"])


def evenly_spaced_subset(rows, maximum):
    if len(rows) <= maximum:
        return rows
    indices = sorted(set(
        int(round(index * (len(rows) - 1) / float(maximum - 1)))
        for index in range(maximum)
    ))
    return [rows[index] for index in indices]


def hypothesis_score(rows, source_x, source_y, depth_km, velocity_km_s):
    observed = [datetime_to_epoch(row["arrival_time_utc"]) for row in rows]
    predicted = []
    for row in rows:
        horizontal = math.hypot(
            row["channel_x_from_rainier_km"] - source_x,
            row["channel_y_from_rainier_km"] - source_y,
        )
        vertical = depth_km + row["channel_elevation_m"] / 1000.0
        predicted.append(math.hypot(horizontal, vertical) / velocity_km_s)
    raw = [value - travel for value, travel in zip(observed, predicted)]
    center = sum(raw) / len(raw)
    # Clipping prevents one picker outlier from controlling the grid search.
    return sum(min(abs(value - center), 0.50) ** 2 for value in raw) / len(raw)


def source_grid():
    steps = int(math.ceil(SOURCE_RADIUS_KM / GRID_SPACING_KM))
    for x_index in range(-steps, steps + 1):
        source_x = x_index * GRID_SPACING_KM
        for y_index in range(-steps, steps + 1):
            source_y = y_index * GRID_SPACING_KM
            if math.hypot(source_x, source_y) <= SOURCE_RADIUS_KM:
                for depth in DEPTHS_KM:
                    for velocity in S_VELOCITIES_KM_S:
                        yield source_x, source_y, depth, velocity


def refine_grid(best):
    best_x, best_y, best_depth, best_velocity = best
    for x_step in range(-5, 6):
        source_x = best_x + 0.5 * x_step
        for y_step in range(-5, 6):
            source_y = best_y + 0.5 * y_step
            if math.hypot(source_x, source_y) > SOURCE_RADIUS_KM:
                continue
            for depth_step in range(-2, 3):
                depth = min(25.0, max(0.0, best_depth + depth_step))
                for velocity_step in range(-2, 3):
                    velocity = min(4.5, max(2.0, best_velocity + 0.25 * velocity_step))
                    yield source_x, source_y, depth, velocity


def evaluate_best_source(all_rows, hypothesis):
    source_x, source_y, depth, velocity = hypothesis
    raw_residuals = []
    travel_times = []
    origin_estimates = []
    for row in all_rows:
        horizontal = math.hypot(
            row["channel_x_from_rainier_km"] - source_x,
            row["channel_y_from_rainier_km"] - source_y,
        )
        vertical = depth + row["channel_elevation_m"] / 1000.0
        travel = math.hypot(horizontal, vertical) / velocity
        epoch = datetime_to_epoch(row["arrival_time_utc"])
        travel_times.append(travel)
        origin_estimates.append(epoch - travel)
        raw_residuals.append(epoch - travel)
    center = statistics.median(raw_residuals)
    residuals = [value - center for value in raw_residuals]
    absolute = [abs(value) for value in residuals]
    cosine = math.cos(math.radians(RAINIER_LATITUDE))
    latitude = RAINIER_LATITUDE + source_y / 111.32
    longitude = RAINIER_LONGITUDE + source_x / (111.32 * cosine)
    return {
        "best_fit_source_x_from_rainier_km": source_x,
        "best_fit_source_y_from_rainier_km": source_y,
        "best_fit_source_latitude": latitude,
        "best_fit_source_longitude": longitude,
        "best_fit_source_depth_km": depth,
        "best_fit_s_velocity_km_s": velocity,
        "best_fit_das_origin_time_utc": epoch_to_datetime(statistics.median(origin_estimates)),
        "grid_moveout_residual_mad_s": 1.4826 * statistics.median(absolute),
        "grid_moveout_residual_p95_s": percentile(absolute, 95.0),
        "grid_moveout_residual_max_s": max(absolute),
        "grid_moveout_residual_rmse_s": math.sqrt(
            sum(value * value for value in residuals) / len(residuals)
        ),
        "grid_moveout_fraction_within_010s": sum(value <= 0.10 for value in absolute) / float(len(absolute)),
        "grid_moveout_fraction_within_020s": sum(value <= 0.20 for value in absolute) / float(len(absolute)),
        "grid_moveout_fraction_within_030s": sum(value <= 0.30 for value in absolute) / float(len(absolute)),
        "predicted_s_travel_time_min_s": min(travel_times),
        "predicted_s_travel_time_max_s": max(travel_times),
        "grid_search_pick_count": len(all_rows),
    }


def fit_physical_moveout(event_picks):
    search_rows = evenly_spaced_subset(event_picks, MAX_GRID_PICKS)
    best_hypothesis = None
    best_score = float("inf")
    for hypothesis in source_grid():
        score = hypothesis_score(search_rows, *hypothesis)
        if score < best_score:
            best_score = score
            best_hypothesis = hypothesis
    for hypothesis in refine_grid(best_hypothesis):
        score = hypothesis_score(search_rows, *hypothesis)
        if score < best_score:
            best_score = score
            best_hypothesis = hypothesis
    output = evaluate_best_source(event_picks, best_hypothesis)
    output["grid_search_clipped_mse_score"] = best_score
    output["grid_search_subset_pick_count"] = len(search_rows)
    return output


def physical_moveout_rejection_reasons(grid):
    reasons = []
    if grid["grid_moveout_residual_mad_s"] > MAX_GRID_MOVEOUT_MAD_S:
        reasons.append("physical_moveout_mad_above_limit")
    if grid["grid_moveout_residual_p95_s"] > MAX_GRID_MOVEOUT_P95_S:
        reasons.append("physical_moveout_p95_above_limit")
    if grid["grid_moveout_fraction_within_020s"] < MIN_GRID_INLIER_FRACTION_020S:
        reasons.append("physical_moveout_inlier_fraction_below_limit")
    return reasons


def event_id(event_start, event_number):
    return "DAS_%s_%03d" % (
        event_start.strftime("%Y%m%dT%H%M%S%f")[:-3],
        event_number,
    )


def summarize(values):
    values = [float(value) for value in values]
    return min(values), statistics.median(values), max(values)


def serialized(values, formatter=None):
    if formatter is None:
        formatter = lambda value: str(value)
    return ";".join(formatter(value) for value in values)


def analyze_file(path, geometry, source_hash):
    picks, input_rows = read_s_picks(path)
    blocks = detect_blocks(picks)
    block_rows_by_number = {}
    picks_by_block = {}
    valid_blocks = []
    for number, block in enumerate(blocks, 1):
        block_row, block_picks = classify_block(block, number, geometry)
        if block_row is None:
            continue
        block_rows_by_number[number] = block_row
        picks_by_block[number] = block_picks
        block["block_number"] = number
        valid_blocks.append(block)

    event_rows = []
    output_blocks = []
    output_event_picks = []
    for event_number, group in enumerate(group_event_blocks(valid_blocks), 1):
        block_rows = [block_rows_by_number[row["block_number"]] for row in group]
        selected_picks = select_event_picks(block_rows, picks_by_block)
        start = min(row["arrival_start"] for row in group)
        end = max(row["arrival_end"] for row in group)
        identifier = event_id(start, event_number)
        representative = datetime_median(
            [row["arrival_time_utc"] for row in selected_picks]
        )
        combined_grid = fit_physical_moveout(selected_picks)
        combined_grid_reasons = physical_moveout_rejection_reasons(combined_grid)

        # A real arrival may be clean on one cable segment and poorly picked on
        # another. Fit every block independently so one noisy segment cannot
        # reject an otherwise physical event.
        physical_block_fits = []
        for block_row in block_rows:
            block_picks = picks_by_block[block_row["block_number"]]
            inlier_picks = [row for row in block_picks if row["is_local_inlier"]]
            fit_picks = inlier_picks if len(inlier_picks) >= 3 else block_picks
            block_grid = fit_physical_moveout(fit_picks)
            block_grid_reasons = physical_moveout_rejection_reasons(block_grid)
            block_row["physical_moveout_rejection_reasons"] = ";".join(
                block_grid_reasons
            )
            for key, value in block_grid.items():
                block_row["physical_%s" % key] = value
            physical_block_fits.append(
                (block_row, block_grid, block_grid_reasons, len(fit_picks))
            )

        passing_physical_blocks = [
            item
            for item in physical_block_fits
            if item[0]["block_status"] != "rejected" and not item[2]
        ]
        if passing_physical_blocks:
            physical_best = max(
                passing_physical_blocks,
                key=lambda item: (
                    item[3],
                    -item[1]["grid_moveout_residual_mad_s"],
                    -item[1]["grid_moveout_residual_p95_s"],
                ),
            )
        else:
            physical_best = min(
                physical_block_fits,
                key=lambda item: (
                    item[1]["grid_moveout_residual_mad_s"] / MAX_GRID_MOVEOUT_MAD_S
                    + item[1]["grid_moveout_residual_p95_s"] / MAX_GRID_MOVEOUT_P95_S
                    - item[1]["grid_moveout_fraction_within_020s"]
                ),
            )
        physical_best_block, grid, grid_reasons, physical_best_pick_count = physical_best

        local_pass = any(row["block_status"] != "rejected" for row in block_rows)
        rejection_reasons = []
        if not local_pass:
            rejection_reasons.append("no_locally_coherent_block")
        if not passing_physical_blocks:
            rejection_reasons.extend(grid_reasons)
        if rejection_reasons:
            status = "rejected"
        elif (
            physical_best_pick_count >= STRONG_SUPPORT_CHANNELS
            or (not combined_grid_reasons and len(selected_picks) >= STRONG_SUPPORT_CHANNELS)
        ):
            status = "coherent"
        else:
            status = "coherent_low_support"

        best_block = max(block_rows, key=block_rank)
        local_mad = summarize(
            row["local_smooth_residual_mad_s"] for row in block_rows
        )
        local_p95 = summarize(
            row["local_smooth_residual_p95_s"] for row in block_rows
        )
        adjacent_p95 = summarize(row["adjacent_dt_p95_s"] for row in block_rows)
        inlier_fractions = summarize(row["local_inlier_fraction"] for row in block_rows)

        event_row = {
            "das_event_id": identifier,
            "classification": status,
            "rejection_reasons": ";".join(rejection_reasons),
            "source_file": path.name,
            "source_file_sha256": source_hash,
            "analysis_version": ANALYSIS_VERSION,
            "das_time_start_utc": start,
            "das_time_end_utc": end,
            "das_time_representative_utc": representative,
            "best_fit_das_origin_time_utc": grid["best_fit_das_origin_time_utc"],
            "physical_moveout_best_block_number": physical_best_block["block_number"],
            "physical_moveout_best_block_pick_count": physical_best_pick_count,
            "recommended_pyocto_match_start_utc": (
                grid["best_fit_das_origin_time_utc"]
                - timedelta(seconds=RECOMMENDED_PYOCTO_MATCH_HALF_WIDTH_S)
            ),
            "recommended_pyocto_match_end_utc": (
                grid["best_fit_das_origin_time_utc"]
                + timedelta(seconds=RECOMMENDED_PYOCTO_MATCH_HALF_WIDTH_S)
            ),
            "recommended_pyocto_match_half_width_s": RECOMMENDED_PYOCTO_MATCH_HALF_WIDTH_S,
            "event_duration_s": (end - start).total_seconds(),
            "n_blocks": len(block_rows),
            "n_coherent_blocks": sum(row["block_status"] == "coherent" for row in block_rows),
            "n_low_support_blocks": sum(row["block_status"] == "coherent_low_support" for row in block_rows),
            "n_rejected_blocks": sum(row["block_status"] == "rejected" for row in block_rows),
            "block_numbers": serialized(row["block_number"] for row in block_rows),
            "block_classifications": serialized(row["block_status"] for row in block_rows),
            "n_selected_event_picks": len(selected_picks),
            "selected_channel_min": min(row["channel"] for row in selected_picks),
            "selected_channel_max": max(row["channel"] for row in selected_picks),
            "selected_fiber_distance_min_km": min(row["fiber_distance_km"] for row in selected_picks),
            "selected_fiber_distance_max_km": max(row["fiber_distance_km"] for row in selected_picks),
            "selected_fiber_span_km": max(row["fiber_distance_km"] for row in selected_picks) - min(row["fiber_distance_km"] for row in selected_picks),
            "best_block_number": best_block["block_number"],
            "best_block_classification": best_block["block_status"],
            "best_block_n_sampled_channels": best_block["n_sampled_channels"],
            "best_block_n_local_inliers": best_block["n_local_inliers"],
            "best_block_longest_contiguous_inlier_run": best_block["longest_contiguous_local_inlier_run"],
            "best_block_fiber_span_km": best_block["fiber_span_km"],
            "best_block_pick_spread_s": best_block["pick_spread_s"],
            "best_block_local_smooth_mad_s": best_block["local_smooth_residual_mad_s"],
            "best_block_local_smooth_p95_s": best_block["local_smooth_residual_p95_s"],
            "best_block_local_inlier_fraction": best_block["local_inlier_fraction"],
            "best_block_adjacent_dt_p95_s_diagnostic": best_block["adjacent_dt_p95_s"],
            "local_smooth_mad_min_s": local_mad[0],
            "local_smooth_mad_median_s": local_mad[1],
            "local_smooth_mad_max_s": local_mad[2],
            "local_smooth_p95_min_s": local_p95[0],
            "local_smooth_p95_median_s": local_p95[1],
            "local_smooth_p95_max_s": local_p95[2],
            "adjacent_dt_p95_min_s_diagnostic": adjacent_p95[0],
            "adjacent_dt_p95_median_s_diagnostic": adjacent_p95[1],
            "adjacent_dt_p95_max_s_diagnostic": adjacent_p95[2],
            "local_inlier_fraction_min": inlier_fractions[0],
            "local_inlier_fraction_median": inlier_fractions[1],
            "local_inlier_fraction_max": inlier_fractions[2],
            "selected_channels": serialized(row["channel"] for row in selected_picks),
            "selected_pick_times_utc": serialized(
                (row["arrival_time_utc"] for row in selected_picks), iso
            ),
            "selected_fiber_distances_km": serialized(
                (row["fiber_distance_km"] for row in selected_picks),
                lambda value: "%.6f" % value,
            ),
            "selected_channel_latitudes": serialized(
                (row["channel_latitude"] for row in selected_picks),
                lambda value: "%.7f" % value,
            ),
            "selected_channel_longitudes": serialized(
                (row["channel_longitude"] for row in selected_picks),
                lambda value: "%.7f" % value,
            ),
            "selected_channel_elevations_m": serialized(
                (row["channel_elevation_m"] for row in selected_picks),
                lambda value: "%.2f" % value,
            ),
            "selected_local_residuals_s": serialized(
                (row["local_residual_s"] for row in selected_picks),
                lambda value: "%.6f" % value,
            ),
            "channel_stride": CHANNEL_STRIDE,
            "minimum_sampled_channels": MIN_SAMPLED_CHANNELS,
            "strong_support_channels": STRONG_SUPPORT_CHANNELS,
            "maximum_pick_spread_s": MAX_PICK_SPREAD_S,
            "maximum_channel": MAX_CHANNEL,
            "event_grouping_gap_s": EVENT_GROUPING_GAP_S,
            "maximum_local_smooth_mad_s": MAX_LOCAL_SMOOTH_MAD_S,
            "maximum_local_residual_s": MAX_LOCAL_RESIDUAL_S,
            "minimum_local_inlier_fraction": MIN_LOCAL_INLIER_FRACTION,
            "maximum_grid_moveout_mad_s": MAX_GRID_MOVEOUT_MAD_S,
            "maximum_grid_moveout_p95_s": MAX_GRID_MOVEOUT_P95_S,
            "minimum_grid_inlier_fraction_020s": MIN_GRID_INLIER_FRACTION_020S,
            "source_grid_center_latitude": RAINIER_LATITUDE,
            "source_grid_center_longitude": RAINIER_LONGITUDE,
            "source_grid_radius_km": SOURCE_RADIUS_KM,
        }
        event_row.update(grid)
        for key, value in combined_grid.items():
            event_row["combined_%s" % key] = value
        event_row["combined_physical_moveout_rejection_reasons"] = ";".join(
            combined_grid_reasons
        )
        event_rows.append(event_row)

        for block_row in block_rows:
            output = dict(block_row)
            output.update(
                {
                    "das_event_id": identifier,
                    "event_classification": status,
                    "source_file": path.name,
                    "source_file_sha256": source_hash,
                    "analysis_version": ANALYSIS_VERSION,
                }
            )
            output_blocks.append(output)

        for pick_row in selected_picks:
            output = dict(pick_row)
            output.update(
                {
                    "das_event_id": identifier,
                    "event_classification": status,
                    "source_file": path.name,
                    "source_file_sha256": source_hash,
                    "analysis_version": ANALYSIS_VERSION,
                    "event_representative_time_utc": representative,
                    "event_relative_time_s": (
                        pick_row["arrival_time_utc"] - representative
                    ).total_seconds(),
                }
            )
            output_event_picks.append(output)

    file_summary = {
        "source_file": path.name,
        "source_path": str(path.resolve()),
        "source_file_sha256": source_hash,
        "source_size_bytes": path.stat().st_size,
        "input_data_rows": input_rows,
        "analysis_version": ANALYSIS_VERSION,
        "n_s_picks": len(picks),
        "s_pick_time_min_utc": min((row["arrival"] for row in picks), default=""),
        "s_pick_time_max_utc": max((row["arrival"] for row in picks), default=""),
        "n_candidate_blocks": len(valid_blocks),
        "n_grouped_das_events": len(event_rows),
        "n_coherent_events": sum(row["classification"] == "coherent" for row in event_rows),
        "n_low_support_events": sum(row["classification"] == "coherent_low_support" for row in event_rows),
        "n_rejected_events": sum(row["classification"] == "rejected" for row in event_rows),
        "processing_error": "",
    }
    return file_summary, event_rows, output_blocks, output_event_picks


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_fields(rows, leading):
    output = list(leading)
    for row in rows:
        for key in row:
            if key not in output:
                output.append(key)
    return output


def write_csv(path, rows, leading_fields):
    fields = ordered_fields(rows, leading_fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: iso(row.get(key, "")) for key in fields})


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a DAS-only event catalog from MORADAS ML S-pick CSVs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory searched recursively (default: current directory)",
    )
    parser.add_argument(
        "--pattern",
        default="MORADAS_*_AR.csv",
        help="Input filename pattern (default: MORADAS_*_AR.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: INPUT_DIR/DAS_catalog_results)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "DAS_catalog_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(
        path for path in input_dir.rglob(args.pattern) if output_dir not in path.parents
    )
    if not input_files:
        print("ERROR: no files matched %s below %s" % (args.pattern, input_dir))
        return 2

    print("Rainier DAS event catalog")
    print("Input directory: %s" % input_dir)
    print("Files found: %d" % len(input_files))
    print("Output directory: %s" % output_dir)
    print("This can take several minutes when many coherent events require grid search.")

    geometry = EmbeddedGeometry()
    file_rows = []
    event_rows = []
    block_rows = []
    event_pick_rows = []
    for index, path in enumerate(input_files, 1):
        print("[%d/%d] %s" % (index, len(input_files), path))
        source_hash = sha256_file(path)
        try:
            file_row, events, blocks, event_picks = analyze_file(
                path, geometry, source_hash
            )
        except Exception as error:
            file_row = {
                "source_file": path.name,
                "source_path": str(path.resolve()),
                "source_file_sha256": source_hash,
                "source_size_bytes": path.stat().st_size,
                "analysis_version": ANALYSIS_VERSION,
                "processing_error": "%s: %s" % (type(error).__name__, error),
            }
            events, blocks, event_picks = [], [], []
            print("  ERROR: %s" % file_row["processing_error"])
        file_rows.append(file_row)
        event_rows.extend(events)
        block_rows.extend(blocks)
        event_pick_rows.extend(event_picks)
        print(
            "  blocks=%s events=%s coherent=%s low_support=%s rejected=%s"
            % (
                file_row.get("n_candidate_blocks", "NA"),
                file_row.get("n_grouped_das_events", "NA"),
                file_row.get("n_coherent_events", "NA"),
                file_row.get("n_low_support_events", "NA"),
                file_row.get("n_rejected_events", "NA"),
            )
        )

    main_catalog = output_dir / "DAS_event_catalog.csv"
    write_csv(main_catalog, event_rows, ["das_event_id", "classification", "rejection_reasons"])
    write_csv(
        output_dir / "DAS_block_metrics.csv",
        block_rows,
        ["das_event_id", "event_classification", "block_number", "block_status"],
    )
    write_csv(
        output_dir / "DAS_event_picks.csv",
        event_pick_rows,
        ["das_event_id", "event_classification", "channel", "arrival_time_utc"],
    )
    write_csv(
        output_dir / "DAS_file_summary.csv",
        file_rows,
        ["source_file", "processing_error"],
    )

    configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "run_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "python_version": sys.version,
        "input_directory": str(input_dir),
        "input_pattern": args.pattern,
        "input_file_count": len(input_files),
        "event_count": len(event_rows),
        "parameters": {
            "rainier_latitude": RAINIER_LATITUDE,
            "rainier_longitude": RAINIER_LONGITUDE,
            "source_radius_km": SOURCE_RADIUS_KM,
            "grid_spacing_km": GRID_SPACING_KM,
            "depths_km": DEPTHS_KM,
            "s_velocities_km_s": S_VELOCITIES_KM_S,
            "channel_stride": CHANNEL_STRIDE,
            "minimum_sampled_channels": MIN_SAMPLED_CHANNELS,
            "strong_support_channels": STRONG_SUPPORT_CHANNELS,
            "maximum_channel": MAX_CHANNEL,
            "maximum_pick_spread_s": MAX_PICK_SPREAD_S,
            "event_grouping_gap_s": EVENT_GROUPING_GAP_S,
            "maximum_local_smooth_mad_s": MAX_LOCAL_SMOOTH_MAD_S,
            "maximum_local_residual_s": MAX_LOCAL_RESIDUAL_S,
            "minimum_local_inlier_fraction": MIN_LOCAL_INLIER_FRACTION,
            "maximum_grid_moveout_mad_s": MAX_GRID_MOVEOUT_MAD_S,
            "maximum_grid_moveout_p95_s": MAX_GRID_MOVEOUT_P95_S,
            "minimum_grid_inlier_fraction_020s": MIN_GRID_INLIER_FRACTION_020S,
        },
        "classification_definitions": {
            "coherent": "Passes local and physical moveout tests with at least 20 selected channels.",
            "coherent_low_support": "Passes both moveout tests with 10-19 selected channels.",
            "rejected": "Fails local coherence or the physical moveout grid-search test.",
        },
        "important_note": "A coherent DAS candidate is not yet a confirmed earthquake. Compare this catalog with PyOcto next.",
    }
    with (output_dir / "DAS_run_configuration.json").open("w") as handle:
        json.dump(configuration, handle, indent=2, sort_keys=True)

    print("\nFinished.")
    print("Main catalog: %s" % main_catalog)
    print("Events: %d" % len(event_rows))
    print("Coherent: %d" % sum(row["classification"] == "coherent" for row in event_rows))
    print("Coherent low support: %d" % sum(row["classification"] == "coherent_low_support" for row in event_rows))
    print("Rejected: %d" % sum(row["classification"] == "rejected" for row in event_rows))
    print("Files with errors: %d" % sum(bool(row.get("processing_error")) for row in file_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
