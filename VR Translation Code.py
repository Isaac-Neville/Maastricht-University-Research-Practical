import os       # file system operations
import re       # regular expressions for filename parsing
import time     # elapsed-time tracking for digest lines
import traceback
import numpy as np      # numerical operations
import pandas as pd     # DataFrame operations
from scipy.signal import savgol_filter      # Savitzky-Golay smoothing
import tkinter as tk
from tkinter import filedialog, messagebox     # GUI dialogs for file/folder selection
import matplotlib.pyplot as plt     # visualisation
from matplotlib.patches import Patch    # legend patches for fixation timeline


# =============================================================================
# STAGE 0 — CONFIGURATION
# =============================================================================

CONDITION_MAP = {1: "low", 2: "medium", 3: "high"}   # maps condition number to label

SHOPPING_LISTS = {
    1: ["mineraalwater", "orange", "croissant", "pudding vanille",
        "kokosmelk", "soep tomaat", "chips naturel"],
    2: ["brood wit", "cupcakes", "kiwi", "chips paprika",
        "green ice tea", "melk vol", "pudding kokos"],
    3: ["pudding chocolade", "melk halfvol", "koekjes chocolade", "meloen",
        "rijstwafels", "ice tea peach", "tortilla chips"],
}

LIST_LENGTH              = 7      # items per shopping list
FIXATION_MIN_MS          = 100    # minimum fixation duration (ms) ≈ 5 samples at ~16ms
TELEPORT_THRESHOLD       = 1.0    # CAM_POS jump (Unity units) that counts as a teleport
TRACKER_LOSS_MIN_RUNS    = 3      # consecutive zero-gaze samples = tracker loss
SAVGOL_WINDOW            = 7      # Savitzky-Golay window length (must be odd)
SAVGOL_POLY              = 2      # Savitzky-Golay polynomial degree
WRONG_GRAB_PENALTY       = 1      # penalty subtracted per incorrect grab
SAMPLE_INTERVAL_MS       = 16     # nominal ms per eye-tracking sample (~62.5 Hz)


# =============================================================================
# STAGE 1 — FILE INGESTION
# =============================================================================

def find_file_pairs(data_dir):
    """
    Scan data_dir for .txt files. Match task + eye file by participant ID
    and condition number. Warn about any unpaired files.

    Returns list of dicts:
        { participant_id, condition_num, task_path, eye_path }
    """
    # collect all .txt files in the directory
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    # parse each filename and group by (participant_id, condition_num)
    groups = {}
    for filename in all_files:
        result = parse_filename(filename)

        # skip files that don't match the expected naming convention
        if result is None:
            print(f"  Warning: could not parse '{filename}' — skipping.")
            continue

        condition_num, participant_id, is_eye = result
        key = (participant_id, condition_num)   # unique key for this participant × condition

        if key not in groups:
            groups[key] = {"task_path": None, "eye_path": None}   # initialise empty slot

        full_path = os.path.join(data_dir, filename)   # absolute path for later loading
        if is_eye:
            groups[key]["eye_path"] = full_path    # store as eye file
        else:
            groups[key]["task_path"] = full_path   # store as task file

    # build list of complete pairs, warn about incomplete ones
    pairs = []
    for (participant_id, condition_num), paths in groups.items():
        if paths["task_path"] is None:
            print(f"  Warning: no task file for participant {participant_id}, condition {condition_num}.")
        elif paths["eye_path"] is None:
            print(f"  Warning: no eye file for participant {participant_id}, condition {condition_num}.")
        else:
            # both files present — add to the list of processable pairs
            pairs.append({
                "participant_id"  : participant_id,
                "condition_num"   : condition_num,
                "task_path"       : paths["task_path"],
                "eye_path"        : paths["eye_path"]
            })

    return pairs


def parse_filename(filename):
    """
    Extract condition number, participant ID, and file type from filename.

    Patterns:
        Con{c}_ID{id}_VRSuperMarket_{n}_{dt}.txt        → task
        Con{c}_ID{id}_VRSuperMarket_Eye_{n}_{dt}.txt    → eye

    Returns (condition_num: int, participant_id: str, is_eye: bool)
    Returns None if filename doesn't match expected pattern.
    """
    # define the pattern:
    # Con     — literal
    # (\d+)   — one or more digits → condition number
    # _ID     — literal
    # (\d+)   — one or more digits → participant ID
    # rest of filename is ignored
    pattern = r"^Con(\d+)_ID(\d+)_VRSuperMarket"

    match = re.match(pattern, filename)

    # filename doesn't match expected convention
    if not match:
        return None

    condition_num  = int(match.group(1))   # first capture group → condition number (1, 2, or 3)
    participant_id = match.group(2)        # second capture group → participant ID string

    # eye file has 'Eye' anywhere in its name; task file does not
    is_eye = "Eye" in filename

    return condition_num, participant_id, is_eye


def load_task_file(path):
    """
    Load semicolon-delimited task file. Convert OBJECT_ON_LIST to bool.

    Columns: TIMESTAMP (float), PICKED_OBJECT (str), OBJECT_ON_LIST (bool)
    Returns DataFrame.
    """
    # read the semicolon-delimited file — Unity exports use ; as separator
    df = pd.read_csv(path, sep=";")

    # Unity appends a trailing semicolon to each row, which creates an extra empty column
    df = df.dropna(axis=1, how="all")

    # convert OBJECT_ON_LIST from string "True"/"False" to actual Python booleans
    # so .sum() and boolean indexing work correctly downstream
    df["OBJECT_ON_LIST"] = df["OBJECT_ON_LIST"].astype(str).str.strip().str.lower() == "true"

    # lowercase so item names match the keys in SHOPPING_LISTS
    df["PICKED_OBJECT"] = df["PICKED_OBJECT"].astype(str).str.strip().str.lower()

    return df


def load_eye_file(path):
    """
    Load semicolon-delimited eye-tracking file (~9500 rows, ~16ms apart).

    Columns: TIMESTAMP, CAM_POS_X/Y/Z, CAM_ROT_X/Y/Z,
             GAZE_ORIGIN_X/Y/Z, GAZE_DIR_X/Y/Z,
             PUPIL_LEFT, PUPIL_RIGHT, OBJECT
    Returns DataFrame.
    """
    # read the semicolon-delimited file
    df = pd.read_csv(path, sep=";")

    # drop empty trailing column caused by trailing semicolons
    df = df.dropna(axis=1, how="all")

    # lowercase so gaze-object names match the keys in SHOPPING_LISTS
    df["OBJECT"] = df["OBJECT"].astype(str).str.strip().str.lower()

    return df


# =============================================================================
# STAGE 2 — DATA CLEANING
# =============================================================================
# Flag invalid samples with boolean columns. Never delete rows.

def detect_tracker_loss(eye_df):
    """
    Flag samples where GAZE_DIR_X/Y/Z are all exactly 0.0.
    Adds boolean column TRACKER_LOSS.
    Returns eye_df.
    """
    # when the eye tracker loses the pupil it outputs a zero vector for gaze direction
    # all three components being exactly 0 is the Unity SDK's way of signalling dropout
    eye_df["TRACKER_LOSS"] = (
        (eye_df["GAZE_DIR_X"] == 0.0) &
        (eye_df["GAZE_DIR_Y"] == 0.0) &
        (eye_df["GAZE_DIR_Z"] == 0.0)
    )

    return eye_df


def detect_teleportation(eye_df):
    """
    Flag samples around sudden CAM_POS jumps > TELEPORT_THRESHOLD.
    Flags 1 sample before and 2 after each jump.
    Adds boolean column TELEPORT.
    Returns eye_df.
    """
    # extract camera positions as a numpy array for fast vectorised operations
    positions = eye_df[["CAM_POS_X", "CAM_POS_Y", "CAM_POS_Z"]].values   # shape (n, 3)
    deltas    = np.diff(positions, axis=0)      # row[i] = positions[i+1] - positions[i], shape (n-1, 3)
    distances = np.linalg.norm(deltas, axis=1)  # Euclidean step size between consecutive samples

    # np.diff output has length n-1; +1 maps each gap back to the LATER row index
    # so jump_indices[k] is the first sample in the new teleport location
    jump_indices = np.where(distances > TELEPORT_THRESHOLD)[0] + 1

    mask = np.zeros(len(eye_df), dtype=bool)
    for i in jump_indices:
        # flag 1 sample before the jump (last frame before teleport) and
        # 2 samples after (first two frames in the new location — still settling)
        mask[max(0, i - 1) : min(len(eye_df), i + 3)] = True

    eye_df["TELEPORT"] = mask
    return eye_df


def align_timestamps(eye_df, task_df):
    """
    Subtract eye_df's minimum timestamp from both DataFrames.
    Eye file → starts at 0.0.
    Task file → preserves offset from condition onset (first grab may be late).
    Normalises to seconds: if the median inter-sample interval is > 1
    the timestamps are assumed to be in milliseconds and are divided by 1000.
    Returns (eye_df, task_df).
    """
    # use the eye file's earliest timestamp as the shared time origin
    origin = eye_df["TIMESTAMP"].min()

    eye_df["TIMESTAMP"]  = eye_df["TIMESTAMP"]  - origin   # eye file now starts at 0.0
    task_df["TIMESTAMP"] = task_df["TIMESTAMP"] - origin   # task file uses the same origin

    # auto-detect millisecond timestamps — at ~62.5 Hz the median gap is ~16ms (>1),
    # whereas seconds give ~0.016 (<1); dividing converts both files to seconds
    if eye_df["TIMESTAMP"].diff().median() > 1:
        eye_df["TIMESTAMP"]  = eye_df["TIMESTAMP"]  / 1000
        task_df["TIMESTAMP"] = task_df["TIMESTAMP"] / 1000

    return eye_df, task_df


# =============================================================================
# STAGE 3 — GAZE KINEMATICS
# =============================================================================
# Convert raw Euler angles into a smoothed velocity signal for event detection.

def euler_to_unit_vector(x_deg, y_deg):
    """
    Convert pitch/yaw Euler angles (degrees) to 3D unit vectors.
    Roll (z) does not change where the eye points, so it is omitted.
    GAZE_DIR is head-relative, so no CAM_ROT subtraction needed.

    Input:  two arrays (n_samples,)
    Output: array (n_samples, 3)
    """
    # convert degrees to radians — numpy trig functions require radians
    pitch = np.radians(x_deg)      # up/down rotation
    yaw   = np.radians(y_deg)      # left/right rotation

    # convert spherical coordinates to Cartesian unit vector components
    # these formulas give a unit vector pointing in the gaze direction
    x = np.cos(pitch) * np.sin(yaw)    # horizontal component
    y = np.sin(pitch)                   # vertical component
    z = np.cos(pitch) * np.cos(yaw)    # depth component

    # stack into one array of shape (n_samples, 3) — each row is one [x, y, z] unit vector
    return np.column_stack([x, y, z])


def compute_angular_displacement(unit_vectors):
    """
    Angle between consecutive unit vectors via dot product → arccos.
    Clips dot product to [-1, 1] to avoid arccos domain errors.

    Input:  (n_samples, 3)
    Output: (n_samples,) in degrees. First value = 0.
    """
    # vectorised dot product: multiply matching elements then sum across columns
    # unit_vectors[1:] and unit_vectors[:-1] are offset by one row → consecutive pairs
    dots = np.sum(unit_vectors[1:] * unit_vectors[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)   # clamp to valid arccos range to avoid NaN from floating-point drift

    angles = np.zeros(len(unit_vectors))   # first sample has no previous sample, so displacement = 0
    angles[1:] = np.degrees(np.arccos(dots))   # convert radians to degrees for all other samples
    return angles


def compute_velocity(angular_displacement, timestamps):
    """
    velocity[i] = angles[i] / (timestamps[i] - timestamps[i-1])
    Per-sample delta handles dropped frames cleanly.

    Output: (n_samples,) in degrees/second. First value = 0.
    """
    velocity = np.zeros(len(angular_displacement))   # initialise to zero; first sample stays 0

    for i in range(1, len(angular_displacement)):
        time_delta = timestamps[i] - timestamps[i-1]   # time elapsed since previous sample

        # avoid division by zero if two consecutive samples share the same timestamp
        if time_delta > 0:
            velocity[i] = angular_displacement[i] / time_delta   # angular speed in °/s

    return velocity


def compute_acceleration(velocity):
    """
    acceleration[i] = velocity[i] - velocity[i-1]
    Used alongside velocity in saccade onset/offset detection.

    Output: (n_samples,) in °/s². First value = 0.
    """
    acceleration = np.zeros(len(velocity))   # first sample has no previous sample, so acceleration = 0

    for i in range(1, len(velocity)):
        acceleration[i] = velocity[i] - velocity[i-1]   # change in velocity since previous sample

    return acceleration


def smooth_velocity(velocity):
    """
    Savitzky-Golay smoothing (SAVGOL_WINDOW, SAVGOL_POLY).
    Preserves saccade peak shape better than a moving average.

    Output: smoothed velocity array (n_samples,)
    """
    # fit a polynomial of degree SAVGOL_POLY over a sliding window of SAVGOL_WINDOW samples
    # this removes high-frequency noise while keeping the sharp peaks of real saccades intact
    smoothed_velocity = savgol_filter(velocity, SAVGOL_WINDOW, SAVGOL_POLY)

    return smoothed_velocity


# =============================================================================
# STAGE 4 — EVENT DETECTION  (Nyström & Holmqvist, 2010)
# =============================================================================
# Adaptive threshold avoids manual tuning and handles noise differences
# across participants and sessions.

def compute_adaptive_threshold(velocity):
    """
    Iteratively estimate velocity threshold from the signal's noise floor.
    Based on Nyström & Holmqvist (2010).

    Algorithm:
        1. Initial threshold = mean + 3×std of full signal
        2. Keep only samples below threshold ("quiet" = presumed fixation)
        3. Recompute threshold = mean + 6×std of quiet samples
        4. Repeat until convergence

    Returns threshold in °/s.
    """
    # The 6×std multiplier comes directly from Nyström & Holmqvist (2010) —
    # empirically derived from fixation-period velocity noise, so it is citable.
    # Convergence criterion 0.01 means we stop when the threshold shifts by less
    # than 0.01 °/s between iterations — typically reached in 3–5 iterations.

    # start with a generous initial threshold from the full signal
    threshold = np.mean(velocity) + 3 * np.std(velocity)

    while True:
        # isolate samples below the current threshold (presumed fixation)
        quiet_samples = velocity[velocity < threshold]

        # guard: if nothing is below the threshold, the signal is all saccade —
        # keep the current threshold rather than computing mean/std of an empty array
        if len(quiet_samples) == 0:
            break

        # recompute threshold from the noise floor of quiet samples only
        # mean + 6×std is tighter than the initial 3×std because the input is cleaner
        new_threshold = np.mean(quiet_samples) + 6 * np.std(quiet_samples)

        # stop iterating when the threshold has stabilised (change < 0.01 °/s)
        if abs(new_threshold - threshold) < 0.01:
            break

        threshold = new_threshold   # update and iterate again

    return threshold


def classify_samples(velocity, acceleration, threshold, invalid_mask=None):
    """
    Label each sample:
        excluded          : tracker-loss or teleportation sample (from invalid_mask)
        saccade           : velocity > threshold
        glissade          : post-saccade, velocity <= threshold,
                            acceleration still negative (decelerating)
        fixation_candidate: everything else

    Returns array of string labels (n_samples,)
    """
    labels = ["fixation_candidate"] * len(velocity)   # default: assume fixation until proven otherwise

    # mark invalid samples first so the classification loop can never overwrite them
    if invalid_mask is not None:
        for i in range(len(velocity)):
            if invalid_mask[i]:
                labels[i] = "excluded"   # tracker loss or teleport — cannot be classified

    for i in range(len(velocity)):

        if labels[i] == "excluded":
            continue    # never relabel an excluded sample

        if velocity[i] > threshold:
            # eye is moving faster than the noise floor — definite saccade
            labels[i] = "saccade"

        elif i > 0 and labels[i-1] == "saccade" and acceleration[i] < 0:
            # velocity dropped back below threshold but the eye is still decelerating
            # — this post-saccadic oscillation (glissade) is merged into the saccade
            labels[i] = "saccade"

    return labels


def apply_fixation_duration_filter(labels, timestamps):
    """
    Drop fixation_candidate runs shorter than FIXATION_MIN_MS (100ms ≈ 5 samples).
    Surviving candidates -> 'fixation'.
    Returns final labels: 'fixation' | 'saccade' | 'excluded'
    """
    # derive the minimum number of consecutive samples that equals FIXATION_MIN_MS
    # using the actual median sampling interval rather than a hardcoded constant
    dt_ms       = np.median(np.diff(timestamps)) * 1000   # median inter-sample gap in ms
    min_samples = max(1, round(FIXATION_MIN_MS / dt_ms))  # e.g. 100ms / 16ms ≈ 6 samples

    i = 0
    while i < len(labels):

        if labels[i] == "fixation_candidate":

            # advance i to find the end of this consecutive run of candidates
            run_start = i
            while i < len(labels) and labels[i] == "fixation_candidate":
                i += 1
            run_end = i   # exclusive end index

            # promote to real fixation if long enough; demote to saccade if too short
            run_length = run_end - run_start
            new_label = "fixation" if run_length >= min_samples else "saccade"

            # apply the chosen label to every sample in this run
            for j in range(run_start, run_end):
                labels[j] = new_label

        else:
            i += 1   # skip samples already labelled as saccade or excluded

    return labels


# =============================================================================
# STAGE 5 — FIXATION METRICS
# =============================================================================
# Translate labeled samples into attentional DVs for the ANCOVA.

def assign_fixation_objects(eye_df, labels):
    """
    Group consecutive fixation samples into events.
    Assign each event the majority-vote OBJECT label (ignoring 'nothing'
    unless all samples are 'nothing').

    Returns DataFrame:
        [start_time, end_time, duration_ms, object]
    """
    fixations = []
    i = 0

    while i < len(labels):

        if labels[i] == "fixation":

            # advance i to the end of this consecutive fixation run
            run_start = i
            while i < len(labels) and labels[i] == "fixation":
                i += 1
            run_end = i   # exclusive end index

            # slice the eye-tracking data to just this fixation window
            window = eye_df.iloc[run_start:run_end]

            # majority-vote object: use only non-"nothing" rows if any exist,
            # because the eye briefly overshoots onto background between objects
            objects = window["OBJECT"][window["OBJECT"] != "nothing"]
            if len(objects) > 0:
                assigned_object = objects.value_counts().idxmax()   # most frequent named object
            else:
                assigned_object = "nothing"   # entire fixation was on the background

            # record the fixation event with timestamps and duration
            fixations.append({
                "start_time"  : window["TIMESTAMP"].iloc[0],
                "end_time"    : window["TIMESTAMP"].iloc[-1],
                "duration_ms" : (window["TIMESTAMP"].iloc[-1] - window["TIMESTAMP"].iloc[0]) * 1000,
                "object"      : assigned_object
            })

        else:
            i += 1   # non-fixation sample — skip

    return pd.DataFrame(fixations)


def classify_fixation_relevance(fixations_df, condition_num):
    """
    Classify each fixation:
        relevant   : object in SHOPPING_LISTS[condition_num]
        irrelevant : named object not on list
        neither    : object == 'nothing'

    Note from meeting: 'nothing' is included in irrelevant for the
    fixation ratio DV (relevant / all fixation time).

    Adds column 'relevance'. Returns fixations_df.
    """
    shopping_list = SHOPPING_LISTS[condition_num]   # the 7 target items for this condition

    def classify(obj):
        if obj == "nothing":
            return "neither"      # gaze was on the background, not any store item
        elif obj in shopping_list:
            return "relevant"     # the item the participant should be looking for
        else:
            return "irrelevant"   # a real store item but not on this condition's list

    fixations_df["relevance"] = fixations_df["object"].apply(classify)

    return fixations_df


def compute_eye_metrics(fixations_df, labels, timestamps):
    """
    Compute attentional DVs for one participant × condition.

    Returns dict:
        fixation_ratio              relevant / (relevant + irrelevant + neither)
        fixation_time_relevant_ms
        fixation_time_irrelevant_ms
        n_fixations_relevant
        n_fixations_irrelevant
        mean_fixation_dur_relevant_ms
        mean_fixation_dur_irrelevant_ms
        saccade_rate                saccades / second
        n_saccades
    """
    # split the fixation event table into three relevance categories
    relevant   = fixations_df[fixations_df["relevance"] == "relevant"]
    irrelevant = fixations_df[fixations_df["relevance"] == "irrelevant"]
    neither    = fixations_df[fixations_df["relevance"] == "neither"]

    # total fixation time per category in milliseconds
    time_relevant   = relevant["duration_ms"].sum()
    time_irrelevant = irrelevant["duration_ms"].sum()
    time_neither    = neither["duration_ms"].sum()
    time_total      = time_relevant + time_irrelevant + time_neither   # all fixation time combined

    # fixation ratio: proportion of all fixation time spent on relevant items
    # guard against division by zero when no fixations were detected at all
    fixation_ratio = time_relevant / time_total if time_total > 0 else None

    # mean fixation duration per category — None if that category had no fixations
    mean_dur_relevant   = relevant["duration_ms"].mean()   if len(relevant)   > 0 else None
    mean_dur_irrelevant = irrelevant["duration_ms"].mean() if len(irrelevant) > 0 else None

    # count saccade EVENTS (transitions into saccade) rather than saccade samples
    # — this correctly counts one saccade per eye movement regardless of its duration
    n_saccades      = sum(1 for i in range(1, len(labels))
                         if labels[i] == "saccade" and labels[i-1] != "saccade")
    total_duration  = timestamps[-1] - timestamps[0]   # total recording length in seconds
    saccade_rate    = n_saccades / total_duration if total_duration > 0 else None   # saccades per second

    return {
        "fixation_ratio"                  : fixation_ratio,
        "fixation_time_relevant_ms"       : time_relevant,
        "fixation_time_irrelevant_ms"     : time_irrelevant,
        "n_fixations_relevant"            : len(relevant),
        "n_fixations_irrelevant"          : len(irrelevant),
        "mean_fixation_dur_relevant_ms"   : mean_dur_relevant,
        "mean_fixation_dur_irrelevant_ms" : mean_dur_irrelevant,
        "saccade_rate"                    : saccade_rate,
        "n_saccades"                      : n_saccades,
    }


# =============================================================================
# STAGE 6 — TASK PERFORMANCE
# =============================================================================

def objective_performance(object_on_list, picked_object):
    """
    Score a grab sequence with a sliding penalty for repeated wrong grabs.

    +1 for each correct grab.
    -1 for every 2nd wrong grab of the same item (2nd, 4th, 6th wrong pick).
    The 2-wrong-grabs-per-penalty rule prevents over-punishing accidental mis-clicks.

    Inputs: plain Python lists of strings (OBJECT_ON_LIST as "True"/"False", PICKED_OBJECT names).
    Returns: raw integer score (normalised to 0–1 by the caller).
    """
    score = 0
    for i in range(len(object_on_list)):
        if object_on_list[i] == "True":
            score += 1   # correct pick — add one point
        elif object_on_list[i] == "False":
            # count only previous *wrong* picks of the same item (not all picks)
            wrong_prior = sum(
                1 for j in range(i)
                if picked_object[j] == picked_object[i] and object_on_list[j] == "False"
            )
            if wrong_prior % 2 == 1:
                score -= 1    # apply penalty on the 2nd, 4th, 6th, … wrong grab of this item
    return score


def compute_task_performance(task_df):
    """
    Compute performance DVs from the grab log.
    """
    n_correct   = task_df["OBJECT_ON_LIST"].sum()        # number of correct grabs (bool True = 1)
    n_incorrect = (~task_df["OBJECT_ON_LIST"]).sum()     # number of wrong grabs
    n_total     = len(task_df)                           # total number of grabs (correct + wrong)

    # fraction of the 7-item list that was correctly picked (0–1)
    performance_score = n_correct / LIST_LENGTH

    # sliding-penalty score: uses objective_performance() for the penalty logic,
    # then normalises to the same 0–1 range as performance_score
    penalty_score = objective_performance(
        task_df["OBJECT_ON_LIST"].astype(str).tolist(),   # convert bool column to "True"/"False" strings
        task_df["PICKED_OBJECT"].tolist()
    ) / LIST_LENGTH

    # proportion of all grabs that were correct (different from performance_score when n_total > LIST_LENGTH)
    grab_accuracy     = n_correct / n_total if n_total > 0 else None
    # time from the first grab to the last (seconds) — measures how long the shopping took
    completion_time_s = task_df["TIMESTAMP"].iloc[-1] - task_df["TIMESTAMP"].iloc[0]

    return {
        "performance_score" : performance_score,
        "penalty_score"     : penalty_score,
        "grab_accuracy"     : grab_accuracy,
        "n_correct"         : n_correct,
        "n_incorrect"       : n_incorrect,
        "completion_time_s" : completion_time_s
    }


# =============================================================================
# STAGE 7 — OUTPUT
# =============================================================================

def build_output_row(participant_id, low_metrics, medium_metrics, high_metrics):
    """
    Merge all three conditions into one row dict for this participant.
    One row per participant — 15 rows total in results.csv.

    Column order: for each metric, low / medium / high sit next to each other.
    Returns a dict.
    """
    # these metrics are computed during the pipeline but excluded from the JASP CSV:
    # n_fixations are diagnostic only; grab_accuracy and performance_score overlap with penalty_score
    EXCLUDED = {"n_fixations_relevant", "n_fixations_irrelevant", "grab_accuracy", "performance_score"}

    conditions = [("low", low_metrics), ("medium", medium_metrics), ("high", high_metrics)]

    # start the row with participant identity; sps_score comes from the Qualtrics export later
    row = {"participant_id": participant_id, "sps_score": None}

    # iterate by metric so low/medium/high columns sit next to each other in the CSV
    # (e.g. low_fixation_ratio, medium_fixation_ratio, high_fixation_ratio, then next metric)
    for metric in [k for k in low_metrics if k not in EXCLUDED]:
        for label, metrics in conditions:
            row[f"{label}_{metric}"] = metrics.get(metric)   # e.g. "low_fixation_ratio" = 0.42

    # reserve columns for post-condition questionnaire data — filled later from Qualtrics
    for placeholder in ["post_fatigue", "perceived_overload", "perceived_performance"]:
        for label, _ in conditions:
            row[f"{label}_{placeholder}"] = None   # placeholder; will be merged with survey data

    return row


def build_long_rows(participant_id, low_metrics, medium_metrics, high_metrics):
    """
    Convert per-condition metrics into long-format rows.
    Returns a list of 3 dicts — one per condition.

    Column order: participant_id, condition,
        [survey placeholders: sps_score, post_fatigue, perceived_overload, perceived_performance],
        [all metrics without condition prefix]
    """
    EXCLUDED = {"n_fixations_relevant", "n_fixations_irrelevant", "grab_accuracy", "performance_score"}
    metric_keys = [k for k in low_metrics if k not in EXCLUDED]

    long_rows = []
    for cond_label, metrics in [("low", low_metrics), ("medium", medium_metrics), ("high", high_metrics)]:
        row = {
            "participant_id"        : participant_id,
            "condition"             : cond_label,
            # survey placeholders first — filled later from Qualtrics
            "sps_score"             : None,
            "post_fatigue"          : None,
            "perceived_overload"    : None,
            "perceived_performance" : None,
        }
        for key in metric_keys:
            row[key] = metrics.get(key)
        long_rows.append(row)

    return long_rows


def write_results(wide_rows, long_rows, base_path):
    """
    Write wide-format and long-format CSVs from the same pipeline run.

    base_path should NOT include a .csv extension or _wide/_long suffix.
    Writes:
        {base_path}_wide.csv  — one row per participant (JASP / SPSS ready)
        {base_path}_long.csv  — three rows per participant (R / tidy ready)
    Asks before overwriting either file.
    """
    wide_path = base_path + "_wide.csv"
    long_path = base_path + "_long.csv"

    for p in [wide_path, long_path]:
        if os.path.exists(p):
            print(f"  ⚠  Overwriting existing file: {p}")

    pd.DataFrame(wide_rows).to_csv(wide_path, index=False)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)

    print(f"  Wide format  ({len(wide_rows):>2} participant rows)  →  {wide_path}")
    print(f"  Long format  ({len(long_rows):>2} condition rows)    →  {long_path}")


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_velocity(velocity, timestamps, threshold):
    """
    Plot smoothed gaze velocity over time with adaptive threshold marked.
    Useful for visually validating saccade detection.
    """
    plt.figure(figsize=(12, 4))

    # thin line so saccade spikes stay visible without overplotting
    plt.plot(timestamps, velocity, color="steelblue", linewidth=0.8, label="Gaze velocity")

    # horizontal dashed line marks the adaptive threshold — anything above this is a saccade
    plt.axhline(y=threshold, color="red", linewidth=1.2, linestyle="--",
                label=f"Threshold ({threshold:.1f} °/s)")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (°/s)")
    plt.title("Gaze Velocity Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fixation_timeline(fixations_df, total_duration):
    """
    Plot fixation and saccade events as coloured blocks along the time axis.
    Relevant fixations = green, irrelevant = red, neither = grey.
    Useful for validating Stage 4 and Stage 5 output.
    """
    # colour mapping per relevance category
    colour_map = {
        "relevant"   : "green",
        "irrelevant" : "red",
        "neither"    : "lightgrey"
    }

    fig, ax = plt.subplots(figsize=(14, 3))

    for _, fixation in fixations_df.iterrows():
        colour = colour_map[fixation["relevance"]]     # pick bar colour by relevance
        start  = fixation["start_time"]                # left edge of the bar in seconds
        width  = fixation["duration_ms"] / 1000        # bar width in seconds

        # draw one horizontal bar per fixation event; height=0.5 keeps the lane thin
        ax.barh(0, width, left=start, color=colour, edgecolor="none", height=0.5)

        # only label fixations long enough to hold readable text (>300ms)
        if fixation["duration_ms"] > 300:
            ax.text(
                start + width / 2, 0,
                fixation["object"],
                ha="center", va="center",
                fontsize=6, color="white"
            )

    # build a legend manually since barh does not support automatic legend entries
    legend_elements = [
        Patch(facecolor="green",    label="Relevant"),
        Patch(facecolor="red",      label="Irrelevant"),
        Patch(facecolor="lightgrey",label="Neither"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlim(0, total_duration)   # x-axis spans the full recording
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])                # single-lane plot — no y-tick labels needed
    ax.set_title("Fixation Timeline")
    plt.tight_layout()
    plt.show()


# =============================================================================
# PRINT HELPERS
# =============================================================================

_W = 64   # output width for banners and rules


def _banner(title, subtitle=""):
    """Print a boxed banner for major section headings."""
    inner = _W - 4
    print(f"\n╔{'═' * (_W - 2)}╗")
    print(f"║  {title:<{inner}}║")
    if subtitle:
        print(f"║  {subtitle:<{inner}}║")
    print(f"╚{'═' * (_W - 2)}╝")


def _rule(label=""):
    """Print a labelled horizontal rule for stage headings inside inspect."""
    if label:
        rest = max(2, _W - len(label) - 6)
        print(f"\n  ── {label} {'─' * rest}")
    else:
        print("  " + "─" * (_W - 2))


def _kv(key, value, indent=6, key_width=26):
    """Print a key / value pair with consistent indentation."""
    print(f"{' ' * indent}{key:<{key_width}}: {value}")


def _fmt_val(val, decimals=3):
    """Format a metric value for display in summary tables."""
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:,.{decimals}f}" if decimals == 0 else f"{val:.{decimals}f}"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return str(val)


def _print_participant_summary(pid, metrics):
    """
    Print a 3-column comparison table (low | medium | high) for one participant.
    Called after all 3 conditions succeed.
    """
    _rule(label=f"Participant {pid} — summary across conditions")

    lw, vw = 34, 9   # label width, value column width
    print(f"  {'Metric':<{lw}} {'LOW':>{vw}} {'MEDIUM':>{vw}} {'HIGH':>{vw}}")
    print(f"  {'─' * lw} {'─' * vw} {'─' * vw} {'─' * vw}")

    rows = [
        ("fixation_ratio",                  "Fixation ratio",            3),
        ("fixation_time_relevant_ms",        "Time relevant (ms)",        0),
        ("fixation_time_irrelevant_ms",      "Time irrelevant (ms)",      0),
        ("mean_fixation_dur_relevant_ms",    "Mean dur relevant (ms)",    0),
        ("mean_fixation_dur_irrelevant_ms",  "Mean dur irrelevant (ms)",  0),
        ("saccade_rate",                     "Saccade rate (/s)",         2),
        ("n_saccades",                       "N saccades",                0),
        ("penalty_score",                    "Penalty score",             3),
        ("completion_time_s",               "Completion time (s)",       1),
    ]

    for key, label, dec in rows:
        low    = _fmt_val(metrics["low"].get(key),    dec)
        medium = _fmt_val(metrics["medium"].get(key), dec)
        high   = _fmt_val(metrics["high"].get(key),   dec)
        print(f"  {label:<{lw}} {low:>{vw}} {medium:>{vw}} {high:>{vw}}")

    print()


def _print_group_averages(rows):
    """
    Print between-subjects mean ± SD for each metric across all conditions.
    Called at the end of run_all() after the CSV is written.
    """
    df  = pd.DataFrame(rows)
    n   = len(df)

    _banner(f"GROUP AVERAGES  (N = {n} participants)")

    metrics = [
        ("fixation_ratio",                  "Fixation ratio",            3),
        ("fixation_time_relevant_ms",        "Time relevant (ms)",        0),
        ("fixation_time_irrelevant_ms",      "Time irrelevant (ms)",      0),
        ("mean_fixation_dur_relevant_ms",    "Mean dur relevant (ms)",    0),
        ("mean_fixation_dur_irrelevant_ms",  "Mean dur irrelevant (ms)",  0),
        ("saccade_rate",                     "Saccade rate (/s)",         2),
        ("n_saccades",                       "N saccades",                1),
        ("penalty_score",                    "Penalty score",             3),
        ("completion_time_s",               "Completion time (s)",       1),
    ]

    def fmt_cell(col, dec):
        if col not in df.columns:
            return "n/a"
        vals = df[col].dropna()
        if len(vals) == 0:
            return "n/a"
        m  = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        if dec == 0:
            return f"{m:,.0f} ± {sd:,.0f}"
        return f"{m:.{dec}f} ± {sd:.{dec}f}"

    lw, vw = 30, 20
    print(f"\n  {'Metric':<{lw}} {'LOW':<{vw}} {'MEDIUM':<{vw}} {'HIGH':<{vw}}")
    print(f"  {'─' * lw} {'─' * vw} {'─' * vw} {'─' * vw}")
    print(f"  {'(mean ± SD)':<{lw}}")
    print()

    for key, label, dec in metrics:
        low    = fmt_cell(f"low_{key}",    dec)
        medium = fmt_cell(f"medium_{key}", dec)
        high   = fmt_cell(f"high_{key}",   dec)
        print(f"  {label:<{lw}} {low:<{vw}} {medium:<{vw}} {high:<{vw}}")

    print()


# =============================================================================
# STAGE 8 — ORCHESTRATION
# =============================================================================

def process_pair(task_path, eye_path, participant_id, condition_num):
    """
    Run stages 1–6 for one file pair.

    Prints one digest line per condition (streaming — label appears immediately,
    stats fill in after ~1 s of computation):
        Condition LOW   : invalid 1.3%  thresh 124.5°/s  fix_ratio 0.271  score 0.714  [1.2s]

    Returns flat dict of all metrics, or None on failure.
    Full stage-by-stage detail is available via menu option 4 (Inspect).
    """
    label = CONDITION_MAP[condition_num].upper()
    print(f"  Condition {label:<8}", end="", flush=True)

    try:
        t0 = time.time()

        # stage 1 — load
        task_df = load_task_file(task_path)
        eye_df  = load_eye_file(eye_path)
        n       = len(eye_df)

        # stage 2 — clean
        eye_df           = detect_tracker_loss(eye_df)
        eye_df           = detect_teleportation(eye_df)
        eye_df, task_df  = align_timestamps(eye_df, task_df)
        invalid_mask     = (eye_df["TRACKER_LOSS"] | eye_df["TELEPORT"]).values
        n_inv            = int(invalid_mask.sum())

        # stage 3 — kinematics on full signal
        unit_vectors = euler_to_unit_vector(eye_df["GAZE_DIR_X"], eye_df["GAZE_DIR_Y"])
        angles       = compute_angular_displacement(unit_vectors)
        velocity     = compute_velocity(angles, eye_df["TIMESTAMP"].values)
        acceleration = compute_acceleration(velocity)
        velocity     = smooth_velocity(velocity)

        # stage 4 — event detection
        threshold = compute_adaptive_threshold(velocity)
        labels    = classify_samples(velocity, acceleration, threshold, invalid_mask)
        labels    = apply_fixation_duration_filter(labels, eye_df["TIMESTAMP"].values)

        # stage 5 — fixation metrics
        fixations_df = assign_fixation_objects(eye_df, labels)
        fixations_df = classify_fixation_relevance(fixations_df, condition_num)
        eye_metrics  = compute_eye_metrics(fixations_df, labels, eye_df["TIMESTAMP"].values)

        # stage 6 — task performance
        task_metrics = compute_task_performance(task_df)

        # digest line
        pct_inv = 100 * n_inv / n if n > 0 else 0.0
        ratio_s = (f"{eye_metrics['fixation_ratio']:.3f}"
                   if eye_metrics["fixation_ratio"] is not None else "n/a")
        score_s = f"{task_metrics['penalty_score']:.3f}"
        elapsed = time.time() - t0
        print(f": invalid {pct_inv:.1f}%  "
              f"thresh {threshold:.1f}°/s  "
              f"fix_ratio {ratio_s}  "
              f"score {score_s}  "
              f"[{elapsed:.1f}s]")

        return {**eye_metrics, **task_metrics}

    except Exception as e:
        print(f"\n  ✗  ERROR — participant {participant_id}, "
              f"condition {CONDITION_MAP.get(condition_num, condition_num)}: {e}")
        traceback.print_exc()
        return None


def run_all(data_dir, base_path):
    """
    Find all pairs → group by participant → process all 3 conditions per
    participant → build wide and long output rows → write both CSVs.
    """
    pairs = find_file_pairs(data_dir)

    # group file pairs by participant_id
    participants = {}
    for pair in pairs:
        pid = pair["participant_id"]
        if pid not in participants:
            participants[pid] = {}
        participants[pid][pair["condition_num"]] = pair

    n_total = len(participants)
    _banner(
        "VR SUPERMARKET — BATCH PROCESSING",
        f"Found {len(pairs)} file pairs  |  {n_total} participants"
    )

    t_start   = time.time()
    rows      = []
    long_rows = []
    n_skip    = 0

    for i, (pid, conditions) in enumerate(participants.items(), 1):
        print(f"\n{'━' * _W}")
        print(f"  [{i} / {n_total}]  Participant {pid}")
        print(f"{'━' * _W}")

        metrics = {}
        ok = True

        for condition_num in [1, 2, 3]:
            if condition_num not in conditions:
                print(f"  ✗  Missing condition {condition_num} — skipping participant.")
                ok = False
                break
            pair   = conditions[condition_num]
            result = process_pair(pair["task_path"], pair["eye_path"], pid, condition_num)
            if result is None:
                print(f"  ✗  Pipeline error on condition {condition_num} — skipping participant.")
                ok = False
                break
            metrics[CONDITION_MAP[condition_num]] = result

        if ok:
            _print_participant_summary(pid, metrics)
            row = build_output_row(pid, metrics["low"], metrics["medium"], metrics["high"])
            rows.append(row)
            long_rows.extend(build_long_rows(pid, metrics["low"], metrics["medium"], metrics["high"]))
            print(f"  ✓  Participant {pid} — row written.")
        else:
            n_skip += 1

    write_results(rows, long_rows, base_path)

    if rows:
        _print_group_averages(rows)

    elapsed = time.time() - t_start
    _banner(
        "BATCH COMPLETE",
        (f"Processed {len(rows)} / {n_total} participants  |  "
         f"{n_skip} skipped  |  {elapsed:.1f} s total")
    )


def run_one(data_dir, base_path):
    """
    Prompt for participant ID → process that participant's 3 conditions
    → write wide and long CSVs.
    """
    pid   = input("Enter participant ID: ").strip()
    pairs = find_file_pairs(data_dir)

    participant_pairs = {p["condition_num"]: p for p in pairs if p["participant_id"] == pid}

    if not participant_pairs:
        print(f"  No files found for participant {pid}.")
        return

    _banner(f"PROCESSING — Participant {pid}")

    t_start = time.time()
    metrics = {}

    for condition_num in [1, 2, 3]:
        if condition_num not in participant_pairs:
            print(f"  ✗  Missing condition {condition_num} for participant {pid}.")
            return
        pair   = participant_pairs[condition_num]
        result = process_pair(pair["task_path"], pair["eye_path"], pid, condition_num)
        if result is None:
            print(f"  ✗  Failed on condition {condition_num} — aborting.")
            return
        metrics[CONDITION_MAP[condition_num]] = result

    _print_participant_summary(pid, metrics)
    row       = build_output_row(pid, metrics["low"], metrics["medium"], metrics["high"])
    long_rows = build_long_rows(pid,  metrics["low"], metrics["medium"], metrics["high"])
    write_results([row], long_rows, base_path)

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f} s.\n")


def preview_pairs(data_dir):
    """
    Print a table of all found file pairs, grouped by participant.
    No processing. Used to verify data integrity before a full run.
    """
    pairs = find_file_pairs(data_dir)

    if not pairs:
        print("  No file pairs found.")
        return

    # group by participant for a cleaner view
    by_pid = {}
    for p in pairs:
        by_pid.setdefault(p["participant_id"], []).append(p["condition_num"])

    _banner("FILE PAIR PREVIEW", f"{len(pairs)} complete pairs  |  {len(by_pid)} participants")

    print(f"\n  {'Participant':<15} {'Conditions found':<25} {'Status'}")
    print(f"  {'─' * 15} {'─' * 25} {'─' * 20}")

    for pid in sorted(by_pid.keys()):
        conds       = sorted(by_pid[pid])
        cond_labels = ", ".join(CONDITION_MAP[c] for c in conds)
        n_missing   = 3 - len(conds)
        if n_missing:
            missing_nums = [c for c in [1, 2, 3] if c not in conds]
            missing_str  = ", ".join(CONDITION_MAP[c] for c in missing_nums)
            status = f"⚠  missing: {missing_str}"
        else:
            status = "✓  complete"
        print(f"  {pid:<15} {cond_labels:<25} {status}")

    print()


def inspect_one(data_dir):
    """
    Prompt for participant ID + condition. Run full pipeline and print
    all intermediate results stage by stage (for debugging / validation).
    Opens velocity plot and fixation timeline after prompting.
    """
    pid           = input("Enter participant ID: ").strip()
    condition_num = int(input("Enter condition number (1/2/3): ").strip())
    pairs         = find_file_pairs(data_dir)

    match = next((p for p in pairs if p["participant_id"] == pid
                  and p["condition_num"] == condition_num), None)

    if not match:
        print(f"  No files found for participant {pid}, condition {condition_num}.")
        return

    cond_label    = CONDITION_MAP[condition_num].upper()
    shopping_list = SHOPPING_LISTS[condition_num]

    _banner(
        f"INSPECT — Participant {pid}  |  Condition {cond_label}",
        f"Shopping list: {', '.join(shopping_list)}"
    )

    # ── Stage 1 — Load ──────────────────────────────────────────────────────
    _rule(label="Stage 1 — Load")
    task_df = load_task_file(match["task_path"])
    eye_df  = load_eye_file(match["eye_path"])

    _kv("Eye file",   match["eye_path"])
    _kv("Task file",  match["task_path"])
    _kv("Eye rows",   f"{len(eye_df):,}")
    _kv("Task rows",  f"{len(task_df):,}  ({len(task_df)} grab events)")

    print(f"\n  Task grab log ({len(task_df)} events):")
    task_str = task_df.to_string(index=False)
    for line in task_str.splitlines():
        print(f"    {line}")

    # ── Stage 2 — Clean ─────────────────────────────────────────────────────
    _rule(label="Stage 2 — Clean")
    eye_df = detect_tracker_loss(eye_df)
    eye_df = detect_teleportation(eye_df)
    eye_df, task_df = align_timestamps(eye_df, task_df)

    n            = len(eye_df)
    n_tl         = int(eye_df["TRACKER_LOSS"].sum())
    n_tp         = int(eye_df["TELEPORT"].sum())
    invalid_mask = (eye_df["TRACKER_LOSS"] | eye_df["TELEPORT"]).values
    n_inv        = int(invalid_mask.sum())
    n_valid      = n - n_inv

    _kv("Tracker loss",
        f"{n_tl:,} samples ({100 * n_tl / n:.1f}%)  — consecutive-zero gaze vector")
    _kv("Teleports",
        f"{n_tp:,} samples ({100 * n_tp / n:.1f}%)  — CAM_POS jump > {TELEPORT_THRESHOLD} Unity units")
    _kv("Total invalid",
        f"{n_inv:,} samples ({100 * n_inv / n:.1f}%)")
    _kv("Usable samples",
        f"{n_valid:,} ({100 * n_valid / n:.1f}%)  — used for all downstream stats")

    # ── Stage 3 — Kinematics ────────────────────────────────────────────────
    _rule(label="Stage 3 — Kinematics")
    unit_vectors = euler_to_unit_vector(eye_df["GAZE_DIR_X"], eye_df["GAZE_DIR_Y"])
    angles       = compute_angular_displacement(unit_vectors)
    velocity     = compute_velocity(angles, eye_df["TIMESTAMP"].values)
    acceleration = compute_acceleration(velocity)
    velocity     = smooth_velocity(velocity)

    v_valid = velocity[~invalid_mask]
    _kv("Velocity — full signal",
        f"min {velocity.min():.1f}  max {velocity.max():.1f}  mean {velocity.mean():.1f} °/s")
    _kv("Velocity — valid only",
        f"min {v_valid.min():.1f}  max {v_valid.max():.1f}  mean {v_valid.mean():.1f} °/s")
    _kv("Percentiles (p25/p50/p75/p95)",
        (f"{np.percentile(v_valid, 25):.1f} / {np.percentile(v_valid, 50):.1f} / "
         f"{np.percentile(v_valid, 75):.1f} / {np.percentile(v_valid, 95):.1f} °/s"))

    # ── Stage 4 — Event Detection ────────────────────────────────────────────
    _rule(label="Stage 4 — Event Detection  (Nyström & Holmqvist 2010)")
    threshold = compute_adaptive_threshold(velocity)
    labels    = classify_samples(velocity, acceleration, threshold, invalid_mask)
    labels    = apply_fixation_duration_filter(labels, eye_df["TIMESTAMP"].values)

    dt_ms        = np.median(np.diff(eye_df["TIMESTAMP"].values)) * 1000
    min_samples  = max(1, round(FIXATION_MIN_MS / dt_ms))

    n_fix_s = labels.count("fixation")
    n_sac_s = labels.count("saccade")
    n_exc_s = labels.count("excluded")
    n_fix_e = sum(1 for i in range(1, len(labels))
                  if labels[i] == "fixation" and labels[i-1] != "fixation")
    n_sac_e = sum(1 for i in range(1, len(labels))
                  if labels[i] == "saccade"  and labels[i-1] != "saccade")

    _kv("Adaptive threshold",
        f"{threshold:.2f} °/s  (mean + 6×std of quiet-sample noise floor)")
    _kv("Fixation min duration",
        f"{FIXATION_MIN_MS} ms  ≈ {min_samples} samples at {dt_ms:.1f} ms/sample")
    _kv("Fixation samples",
        f"{n_fix_s:,}  ({100 * n_fix_s / n:.1f}%)")
    _kv("Saccade samples",
        f"{n_sac_s:,}  ({100 * n_sac_s / n:.1f}%)")
    _kv("Excluded samples",
        f"{n_exc_s:,}  ({100 * n_exc_s / n:.1f}%)")
    _kv("Fixation EVENTS",  str(n_fix_e))
    _kv("Saccade EVENTS",   str(n_sac_e))

    # ── Stage 5 — Fixation Metrics ───────────────────────────────────────────
    _rule(label="Stage 5 — Fixation Metrics")
    fixations_df = assign_fixation_objects(eye_df, labels)
    fixations_df = classify_fixation_relevance(fixations_df, condition_num)
    eye_metrics  = compute_eye_metrics(fixations_df, labels, eye_df["TIMESTAMP"].values)

    rel = fixations_df[fixations_df["relevance"] == "relevant"]
    irr = fixations_df[fixations_df["relevance"] == "irrelevant"]
    nei = fixations_df[fixations_df["relevance"] == "neither"]

    print(f"\n  Relevance breakdown:")
    hdr = f"  {'Category':<12} {'Events':>7} {'Total ms':>10} {'Mean ms':>9} {'Min ms':>8} {'Max ms':>8}"
    print(hdr)
    print(f"  {'─' * 12} {'─' * 7} {'─' * 10} {'─' * 9} {'─' * 8} {'─' * 8}")
    for cat_label, sub in [("Relevant", rel), ("Irrelevant", irr), ("Neither", nei)]:
        if len(sub) > 0:
            print(f"  {cat_label:<12} {len(sub):>7,} {sub['duration_ms'].sum():>10,.0f} "
                  f"{sub['duration_ms'].mean():>9.0f} {sub['duration_ms'].min():>8.0f} "
                  f"{sub['duration_ms'].max():>8.0f}")
        else:
            print(f"  {cat_label:<12} {'—':>7}")

    print(f"\n  Eye metrics:")
    for k, v in eye_metrics.items():
        _kv(k, _fmt_val(v, 3))

    # top 10 fixated objects by total dwell time
    if len(fixations_df) > 0:
        print(f"\n  Top fixated objects (by total dwell time):")
        print(f"  {'':2} {'Object':<30} {'Total ms':>10}")
        print(f"  {'─' * 2} {'─' * 30} {'─' * 10}")
        top = (fixations_df.groupby("object")["duration_ms"]
               .sum().sort_values(ascending=False).head(10))
        for obj, ms in top.items():
            marker = "✓" if obj in shopping_list else " "
            print(f"  {marker}  {obj:<30} {ms:>10,.0f}")
        print(f"  (✓ = on shopping list for condition {cond_label})")

    # ── Stage 6 — Task Performance ───────────────────────────────────────────
    _rule(label="Stage 6 — Task Performance")
    task_metrics = compute_task_performance(task_df)

    print(f"\n  Task metrics:")
    for k, v in task_metrics.items():
        _kv(k, _fmt_val(v, 3))

    print(f"\n  Shopping list status — Condition {cond_label}:")
    print(f"  {'':2} {'Item':<32} {'Correct grabs':>14} {'Wrong grabs':>12}")
    print(f"  {'─' * 2} {'─' * 32} {'─' * 14} {'─' * 12}")
    for item in shopping_list:
        grabbed  = task_df[task_df["PICKED_OBJECT"] == item]
        n_corr   = int(grabbed[grabbed["OBJECT_ON_LIST"]].shape[0])
        n_wrong  = int(grabbed[~grabbed["OBJECT_ON_LIST"]].shape[0])
        marker   = "✓" if n_corr >= 1 else "✗"
        print(f"  {marker}  {item:<32} {n_corr:>14} {n_wrong:>12}")

    wrong_other = task_df[
        (~task_df["OBJECT_ON_LIST"]) &
        (~task_df["PICKED_OBJECT"].isin(shopping_list))
    ]
    if len(wrong_other) > 0:
        print(f"\n  Wrong grabs of items NOT on the list:")
        for _, row in wrong_other.iterrows():
            print(f"      {row['PICKED_OBJECT']}  (t = {row['TIMESTAMP']:.1f} s)")

    # ── Plots ────────────────────────────────────────────────────────────────
    input("\n  Press Enter to open velocity plot...")
    plot_velocity(velocity, eye_df["TIMESTAMP"].values, threshold)

    input("  Press Enter to open fixation timeline...")
    plot_fixation_timeline(fixations_df, eye_df["TIMESTAMP"].values[-1])


def merge_qualtrics():
    """
    Optional path: merge a Qualtrics survey export into an existing
    results_wide CSV.

    Steps:
        1. GUI: select results_wide CSV
        2. GUI: select Qualtrics CSV
        3. Specify which Qualtrics column holds the participant ID
        4. For each empty placeholder column in results, optionally
           map it to a Qualtrics column
        5. Left-join and write a *_merged_wide.csv

    The researcher should make sure participant IDs are formatted the same
    way in both files (e.g. both "001" or both "1").
    """
    root = tk.Tk()
    root.withdraw()

    _banner("QUALTRICS MERGE")

    # ── Step 1: select results CSV ───────────────────────────────────────────
    print("  Step 1  Select the results_wide CSV (produced by this pipeline)...")
    results_path = filedialog.askopenfilename(
        title="Select results_wide CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not results_path:
        print("  Cancelled.")
        return

    # ── Step 2: select Qualtrics CSV ─────────────────────────────────────────
    print("  Step 2  Select the Qualtrics export CSV...")
    qualtrics_path = filedialog.askopenfilename(
        title="Select Qualtrics CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not qualtrics_path:
        print("  Cancelled.")
        return

    r_df = pd.read_csv(results_path)
    q_df = pd.read_csv(qualtrics_path)

    _rule(label="Files loaded")
    _kv("Results",   f"{len(r_df)} rows  |  {len(r_df.columns)} columns")
    _kv("Qualtrics", f"{len(q_df)} rows  |  {len(q_df.columns)} columns")

    # ── Step 3: identify participant ID column in Qualtrics ──────────────────
    print(f"\n  Qualtrics columns:")
    for i, col in enumerate(q_df.columns, 1):
        n_filled = int(q_df[col].notna().sum())
        print(f"    {i:3}.  {col:<40}  ({n_filled} non-null values)")

    print()
    pid_col = input("  Which Qualtrics column contains the participant ID? ").strip()
    if pid_col not in q_df.columns:
        print(f"  Column '{pid_col}' not found. Aborting.")
        return

    # Normalise IDs to plain strings so "001" and "001" match
    q_df["participant_id"] = q_df[pid_col].astype(str).str.strip()
    r_df["participant_id"] = r_df["participant_id"].astype(str).str.strip()

    # ── Step 4: map empty placeholder columns ────────────────────────────────
    # Only offer columns that are entirely empty in results (the placeholders)
    placeholder_cols = [c for c in r_df.columns
                        if c != "participant_id" and r_df[c].isna().all()]

    if not placeholder_cols:
        print("\n  No empty placeholder columns found in results — nothing to fill.")
        return

    _rule(label="Column mapping")
    print("  Map each empty results column to a Qualtrics column.")
    print("  Press Enter to skip any column.\n")

    mapping = {}   # results_col → qualtrics_col
    for col in placeholder_cols:
        q_col = input(f"    {col:<35} ← Qualtrics column name: ").strip()
        if not q_col:
            continue
        if q_col in q_df.columns:
            mapping[col] = q_col
        else:
            print(f"      ⚠  '{q_col}' not found in Qualtrics CSV — skipped.")

    if not mapping:
        print("  No columns mapped. Nothing to merge.")
        return

    # ── Merge ────────────────────────────────────────────────────────────────
    # Extract only the mapped Qualtrics columns + participant_id
    q_sub = q_df[["participant_id"] + list(mapping.values())].copy()
    q_sub = q_sub.rename(columns={v: k for k, v in mapping.items()})

    # Left join: every results row is kept; unmatched participants get NaN
    merged = r_df.merge(q_sub, on="participant_id", how="left", suffixes=("", "_q"))

    # combine_first fills NaN placeholder cells with Qualtrics values
    for col in mapping:
        q_col = col + "_q"
        if q_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[q_col])
            merged = merged.drop(columns=[q_col])

    # ── Report ───────────────────────────────────────────────────────────────
    matched = int(q_df["participant_id"].isin(r_df["participant_id"]).sum())
    _rule(label="Merge results")
    _kv("Participants matched", f"{matched} / {len(r_df)}")
    for res_col, q_col in mapping.items():
        n_filled = int(merged[res_col].notna().sum())
        _kv(res_col, f"← {q_col}  ({n_filled} values filled)")

    # ── Write ────────────────────────────────────────────────────────────────
    out_path = results_path.replace("_wide.csv", "_merged_wide.csv")
    if out_path == results_path:   # file wasn't named _wide.csv
        out_path = results_path.replace(".csv", "_merged.csv")

    if os.path.exists(out_path):
        if input(f"\n  {out_path} exists. Overwrite? (y/n): ").strip().lower() != "y":
            print("  Cancelled.")
            return

    merged.to_csv(out_path, index=False)
    print(f"\n  Written: {out_path}")
    print(f"  {len(merged)} rows  |  {len(merged.columns)} columns\n")


# =============================================================================
# STAGE 9 — MENU
# =============================================================================

MENU = """
╔══════════════════════════════════════════════════════╗
║       VR SUPERMARKET EYE-TRACKING PIPELINE           ║
║       Seeing Through Sensory Overload                ║
╠══════════════════════════════════════════════════════╣
║  1 — Process all participants  →  write results.csv  ║
║  2 — Process one participant                         ║
║  3 — Preview file pairs        (no processing)       ║
║  4 — Inspect one participant   (detailed output)     ║
║  5 — Merge Qualtrics data      (optional)            ║
║  6 — Exit                                            ║
╚══════════════════════════════════════════════════════╝
"""


def choose_folder():
    """
    Show a welcome message then open a GUI folder picker.
    Returns the selected folder path, or an empty string if cancelled.
    """
    root = tk.Tk()
    root.withdraw()   # hide the blank root window — only the dialogs should be visible

    messagebox.showinfo(
        "Welcome",
        "Welcome to the data storing part.\n\nSelect the folder containing all participant files."
    )

    folder = filedialog.askdirectory(
        title="Select Folder"
    )

    return folder


def get_data_dir():
    """
    Open a GUI folder picker. Repeat until a valid directory is selected.
    """
    while True:
        path = choose_folder()
        if path and os.path.isdir(path):
            return path   # valid directory selected — return it
        print("  No valid directory selected. Please try again.")


def get_output_path(data_dir=None):
    """
    Open a GUI folder picker to select where results should be saved.
    Defaults to the same folder as the data files.
    Writes results_wide.csv and results_long.csv in the chosen folder.
    """
    root = tk.Tk()
    root.withdraw()

    folder = filedialog.askdirectory(
        title="Select folder to save results (results_wide.csv + results_long.csv)",
        initialdir=data_dir or os.getcwd()
    )

    if not folder:
        folder = data_dir or os.getcwd()
        print(f"  No folder selected — saving to: {folder}")

    base_path = os.path.join(folder, "results")
    print(f"  Output: {base_path}_wide.csv  +  {base_path}_long.csv")
    return base_path


def menu():
    """
    Display menu, route input to the correct function. Loop until exit.
    """
    print(MENU)

    while True:
        choice = input("Select an option (1–6): ").strip()

        if choice == "1":
            data_dir  = get_data_dir()
            base_path = get_output_path(data_dir)
            run_all(data_dir, base_path)

        elif choice == "2":
            data_dir  = get_data_dir()
            base_path = get_output_path(data_dir)
            run_one(data_dir, base_path)

        elif choice == "3":
            data_dir = get_data_dir()
            preview_pairs(data_dir)

        elif choice == "4":
            data_dir = get_data_dir()
            inspect_one(data_dir)

        elif choice == "5":
            merge_qualtrics()

        elif choice == "6":
            print("\nGoodbye. Good luck with the analysis!\n")
            break

        else:
            print("  Invalid option. Please enter a number between 1 and 6.")

        # reprint menu after each action so the user always sees their options
        print("\nWhat would you like to do next?")


if __name__ == "__main__":
    menu()
