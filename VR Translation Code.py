import os
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# =============================================================================
# STAGE 0 — CONFIGURATION
# =============================================================================

CONDITION_MAP = {1: "low", 2: "medium", 3: "high"}

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
TELEPORT_THRESHOLD       = 1.0    # CAM_POS jump (Unity units) = teleport
TRACKER_LOSS_MIN_RUNS    = 3      # consecutive zero-samples = tracker loss
SAVGOL_WINDOW            = 7      # Savitzky-Golay window (must be odd)
SAVGOL_POLY              = 2      # Savitzky-Golay polynomial order
WRONG_GRAB_PENALTY       = 1      # penalty subtracted per incorrect grab


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
        key = (participant_id, condition_num)

        if key not in groups:
            groups[key] = {"task_path": None, "eye_path": None}

        full_path = os.path.join(data_dir, filename)
        if is_eye:
            groups[key]["eye_path"] = full_path
        else:
            groups[key]["task_path"] = full_path

    # build list of complete pairs, warn about incomplete ones
    pairs = []
    for (participant_id, condition_num), paths in groups.items():
        if paths["task_path"] is None:
            print(f"  Warning: no task file for participant {participant_id}, condition {condition_num}.")
        elif paths["eye_path"] is None:
            print(f"  Warning: no eye file for participant {participant_id}, condition {condition_num}.")
        else:
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

    condition_num  = int(match.group(1))   # first capture group → condition
    participant_id = match.group(2)        # second capture group → ID

    # eye file has 'Eye' in the filename, task file does not
    is_eye = "Eye" in filename

    return condition_num, participant_id, is_eye


def load_task_file(path):
    """
    Load semicolon-delimited task file. Convert OBJECT_ON_LIST to bool.

    Columns: TIMESTAMP (float), PICKED_OBJECT (str), OBJECT_ON_LIST (bool)
    Returns DataFrame.
    """
    # read the file, semicolon-delimited
    df = pd.read_csv(path, sep=";")

    # drop empty column caused by trailing semicolon on each row
    df = df.dropna(axis=1, how="all")

    # convert OBJECT_ON_LIST from string "True"/"False" to actual boolean
    df["OBJECT_ON_LIST"] = df["OBJECT_ON_LIST"].astype(str).str.strip().str.lower() == "true"

    return df


def load_eye_file(path):
    """
    Load semicolon-delimited eye-tracking file (~9500 rows, ~16ms apart).

    Columns: TIMESTAMP, CAM_POS_X/Y/Z, CAM_ROT_X/Y/Z,
             GAZE_ORIGIN_X/Y/Z, GAZE_DIR_X/Y/Z,
             PUPIL_LEFT, PUPIL_RIGHT, OBJECT
    Returns DataFrame.
    """
    # read the file, semicolon-delimited
    df = pd.read_csv(path, sep=";")

    # drop empty column caused by trailing semicolon on each row
    df = df.dropna(axis=1, how="all")

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
    eye_df["TELEPORT"] = False

    for i in range(1, len(eye_df)):

        # compute distance between this sample and the previous one
        dx = eye_df.iloc[i]["CAM_POS_X"] - eye_df.iloc[i-1]["CAM_POS_X"]
        dy = eye_df.iloc[i]["CAM_POS_Y"] - eye_df.iloc[i-1]["CAM_POS_Y"]
        dz = eye_df.iloc[i]["CAM_POS_Z"] - eye_df.iloc[i-1]["CAM_POS_Z"]
        distance = (dx**2 + dy**2 + dz**2) ** 0.5 # square-root of deltas of each coordinate

        if distance > TELEPORT_THRESHOLD:
            # flag the window around the jump
            for j in range(max(0, i-1), min(len(eye_df), i+3)):
                eye_df.iloc[j, eye_df.columns.get_loc("TELEPORT")] = True

    return eye_df


def align_timestamps(eye_df, task_df):
    """
    Subtract eye_df's minimum timestamp from both DataFrames.
    Eye file → starts at 0.0.
    Task file → preserves offset from condition onset (first grab may be late).
    Returns (eye_df, task_df).
    """
    origin = eye_df["TIMESTAMP"].min()

    eye_df["TIMESTAMP"]  = eye_df["TIMESTAMP"]  - origin
    task_df["TIMESTAMP"] = task_df["TIMESTAMP"] - origin

    return eye_df, task_df


# =============================================================================
# STAGE 3 — GAZE KINEMATICS
# =============================================================================
# Convert raw Euler angles into a smoothed velocity signal for event detection.

def euler_to_unit_vector(x_deg, y_deg, z_deg):
    """
    Convert Euler angles (degrees) to 3D unit vectors.
    GAZE_DIR is head-relative, so no CAM_ROT subtraction needed.

    Input:  three arrays (n_samples,)
    Output: array (n_samples, 3)
    """
    # convert degrees to radians — numpy trig functions require radians
    pitch = np.radians(x_deg)      # up/down rotation
    yaw   = np.radians(y_deg)      # left/right rotation

    # convert spherical coordinates to cartesian unit vector components
    # these formulas give a unit vector pointing in the gaze direction
    x = np.cos(pitch) * np.sin(yaw)    # horizontal component
    y = np.sin(pitch)                   # vertical component
    z = np.cos(pitch) * np.cos(yaw)    # depth component

    # stack into one array of shape (n_samples, 3)
    # each row is one unit vector [x, y, z] for that sample
    return np.column_stack([x, y, z])


def compute_angular_displacement(unit_vectors):
    """
    Angle between consecutive unit vectors via dot product → arccos.
    Clips dot product to [-1, 1] to avoid arccos domain errors.

    Input:  (n_samples, 3)
    Output: (n_samples,) in degrees. First value = 0.
    """
    angles = np.zeros(len(unit_vectors))

    for i in range(1, len(unit_vectors)):
        # dot product of this vector and the previous one
        dot = np.dot(unit_vectors[i], unit_vectors[i-1])

        # clip to [-1, 1] to prevent floating point errors breaking arccos
        dot = np.clip(dot, -1.0, 1.0)

        # arccos gives the angle in radians, convert to degrees
        angles[i] = np.degrees(np.arccos(dot))

    return angles


def compute_velocity(angular_displacement, timestamps):
    """
    velocity[i] = angles[i] / (timestamps[i] - timestamps[i-1])
    Per-sample delta handles dropped frames cleanly.

    Output: (n_samples,) in degrees/second. First value = 0.
    """
    velocity = np.zeros(len(angular_displacement))

    for i in range(1, len(angular_displacement)):
        # time elapsed since previous sample
        time_delta = timestamps[i] - timestamps[i-1]

        # avoid division by zero if two samples share the same timestamp
        if time_delta > 0:
            velocity[i] = angular_displacement[i] / time_delta

    return velocity

def compute_acceleration(velocity):
    """
    acceleration[i] = velocity[i] - velocity[i-1]
    Used alongside velocity in saccade onset/offset detection.

    Output: (n_samples,) in °/s². First value = 0.
    """
    acceleration = np.zeros(len(velocity))

    for i in range(1, len(velocity)):
        # change in velocity since previous sample
        acceleration[i] = velocity[i] - velocity[i-1]

    return acceleration


def smooth_velocity(velocity):
    """
    Savitzky-Golay smoothing (SAVGOL_WINDOW, SAVGOL_POLY).
    Preserves saccade peak shape better than a moving average.

    Output: smoothed velocity array (n_samples,)
    """
    # fit a polynomial of degree SAVGOL_POLY over a sliding window of
    # SAVGOL_WINDOW samples to smooth out noise in the velocity signal
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
# The 6 * std multiplier comes directly from Nyström & Holmqvist (2010) — it's empirically derived from the distribution of fixation-period velocity noise, so it's citable.
# The convergence criterion 0.01 means we stop when the threshold shifts by less than 0.01°/s between iterations — effectively stable. This usually converges in 3-5 iterations.
# start with a generous initial threshold from the full signal

    threshold = np.mean(velocity) + 3 * np.std(velocity)

    while True:
        # isolate samples below the current threshold (presumed fixation)
        quiet_samples = velocity[velocity < threshold]

        # compute a new threshold from the noise floor of quiet samples
        new_threshold = np.mean(quiet_samples) + 6 * np.std(quiet_samples)

        # stop when the threshold has converged (change is negligible)
        if abs(new_threshold - threshold) < 0.01:
            break

        threshold = new_threshold

    return threshold


def classify_samples(velocity, acceleration, threshold):
    """
    Label each sample:
        saccade           : velocity > threshold
        glissade          : post-saccade, velocity <= threshold,
                            acceleration still negative (decelerating)
        fixation_candidate: everything else

    Returns array of string labels (n_samples,)
    """
    labels = ["fixation_candidate"] * len(velocity)

    for i in range(len(velocity)):

        if velocity[i] > threshold:
            # eye is moving fast — definite saccade
            labels[i] = "saccade"

        elif i > 0 and labels[i-1] == "saccade" and acceleration[i] < 0:
            # velocity dropped below threshold but eye is still decelerating
            # — post-saccadic wobble — merged into saccade per glissade policy
            labels[i] = "saccade"

    return labels


def apply_fixation_duration_filter(labels, timestamps):
    """
    Drop fixation_candidate runs shorter than FIXATION_MIN_MS (100ms ≈ 5 samples).
    Surviving candidates -> 'fixation'.
    Returns final labels: 'fixation' | 'saccade'
    """
    i = 0
    while i < len(labels):

        if labels[i] == "fixation_candidate":

            # find where this run ends
            run_start = i
            while i < len(labels) and labels[i] == "fixation_candidate":
                i += 1
            run_end = i

            # long enough → fixation, too short → saccade
            run_length = run_end - run_start
            new_label = "fixation" if run_length >= 5 else "saccade"

            # apply the label to all samples in the run
            for j in range(run_start, run_end):
                labels[j] = new_label

        else:
            i += 1

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

            # find where this fixation run ends
            run_start = i
            while i < len(labels) and labels[i] == "fixation":
                i += 1
            run_end = i

            # get the window of samples for this fixation
            window = eye_df.iloc[run_start:run_end]

            # pick most frequent object, ignoring 'nothing' if possible
            objects = window["OBJECT"][window["OBJECT"] != "nothing"]
            if len(objects) > 0:
                assigned_object = objects.value_counts().idxmax()
            else:
                assigned_object = "nothing"

            # record the fixation event
            fixations.append({
                "start_time"  : window["TIMESTAMP"].iloc[0],
                "end_time"    : window["TIMESTAMP"].iloc[-1],
                "duration_ms" : (window["TIMESTAMP"].iloc[-1] - window["TIMESTAMP"].iloc[0]) * 1000,
                "object"      : assigned_object
            })

        else:
            i += 1

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
    shopping_list = SHOPPING_LISTS[condition_num]

    def classify(obj):
        if obj == "nothing":
            return "neither"
        elif obj in shopping_list:
            return "relevant"
        else:
            return "irrelevant"

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
    # split fixations by relevance
    relevant   = fixations_df[fixations_df["relevance"] == "relevant"]
    irrelevant = fixations_df[fixations_df["relevance"] == "irrelevant"]
    neither    = fixations_df[fixations_df["relevance"] == "neither"]

    # total fixation time per category (ms)
    time_relevant   = relevant["duration_ms"].sum()
    time_irrelevant = irrelevant["duration_ms"].sum()
    time_neither    = neither["duration_ms"].sum()
    time_total      = time_relevant + time_irrelevant + time_neither

    # fixation ratio — relevant time out of all fixation time
    # guard against division by zero if no fixations were detected
    fixation_ratio = time_relevant / time_total if time_total > 0 else None

    # mean fixation duration per category
    mean_dur_relevant   = relevant["duration_ms"].mean()   if len(relevant)   > 0 else None
    mean_dur_irrelevant = irrelevant["duration_ms"].mean() if len(irrelevant) > 0 else None

    # saccade rate — count saccade labels divided by total task duration
    n_saccades      = sum(1 for i in range(1, len(labels))
                         if labels[i] == "saccade" and labels[i-1] != "saccade")
    total_duration  = (timestamps[-1] - timestamps[0])
    saccade_rate    = n_saccades / total_duration if total_duration > 0 else None

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

def compute_task_performance(task_df):
    """
    Compute performance DVs from the grab log.

    Returns dict:
        performance_score   correct / LIST_LENGTH
        penalty_score       (correct - n_wrong_penalty) / LIST_LENGTH                            
        grab_accuracy       correct / total grabs
        n_correct
        n_incorrect
        completion_time_s   last timestamp - first timestamp
    """
    # count correct and incorrect grabs
    n_correct   = task_df["OBJECT_ON_LIST"].sum()
    n_incorrect = (~task_df["OBJECT_ON_LIST"]).sum()
    n_total     = len(task_df)

    # simple performance score — correct picks out of 7
    performance_score = n_correct / LIST_LENGTH

    # penalty score — punishes wrong grabs, wider spread than simple score
    penalty_score = (n_correct - n_incorrect) / LIST_LENGTH

    # grab accuracy — proportion of all grabs that were correct
    grab_accuracy = n_correct / n_total if n_total > 0 else None

    # completion time — duration of active picking phase in seconds
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

    Returns a dict.
    """
    row = {"participant_id": participant_id}

    # add metrics for each condition with condition label as prefix
    for label, metrics in [("low", low_metrics), ("medium", medium_metrics), ("high", high_metrics)]:
        for key, value in metrics.items():
            row[f"{label}_{key}"] = value

    # questionnaire placeholders — filled after merge with Qualtrics data
    row["sps_score"] = None
    for label in ["low", "medium", "high"]:
        row[f"{label}_post_fatigue"]            = None
        row[f"{label}_perceived_overload"]      = None
        row[f"{label}_perceived_performance"]   = None

    return row

def write_results(rows, output_path):
    """
    Write list of row dicts to CSV. 
    Ask user before overwriting existing file.
    """
# warn before overwriting
    if os.path.exists(output_path):
        overwrite = input(f"{output_path} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Cancelled. File not written.")
            return

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Results written to {output_path}")

# =============================================================================
# STAGE 8 — ORCHESTRATION
# =============================================================================

def process_pair(task_path, eye_path, participant_id, condition_num):
    """
    Run stages 1–7 for one file pair. Print progress at each stage.
    Return output row dict of all metrics for this condition, or None on failure (with error message).
    """
    try:
        print(f"  Processing participant {participant_id}, condition {CONDITION_MAP[condition_num]}...")

        # stage 1 — load
        task_df = load_task_file(task_path)
        eye_df  = load_eye_file(eye_path)

        # stage 2 — clean
        eye_df           = detect_tracker_loss(eye_df)
        eye_df           = detect_teleportation(eye_df)
        eye_df, task_df  = align_timestamps(eye_df, task_df)

        # filter out invalid samples before kinematics
        valid = eye_df[~eye_df["TRACKER_LOSS"] & ~eye_df["TELEPORT"]].reset_index(drop=True)

        # stage 3 — kinematics
        unit_vectors  = euler_to_unit_vector(valid["GAZE_DIR_X"], valid["GAZE_DIR_Y"], valid["GAZE_DIR_Z"])
        angles        = compute_angular_displacement(unit_vectors)
        velocity      = compute_velocity(angles, valid["TIMESTAMP"].values)
        acceleration  = compute_acceleration(velocity)
        velocity      = smooth_velocity(velocity)

        # stage 4 — event detection
        threshold     = compute_adaptive_threshold(velocity)
        labels        = classify_samples(velocity, acceleration, threshold)
        labels        = apply_fixation_duration_filter(labels, valid["TIMESTAMP"].values)

        # stage 5 — fixation metrics
        fixations_df  = assign_fixation_objects(valid, labels)
        fixations_df  = classify_fixation_relevance(fixations_df, condition_num)
        eye_metrics   = compute_eye_metrics(fixations_df, labels, valid["TIMESTAMP"].values)

        # stage 6 — task performance
        task_metrics  = compute_task_performance(task_df)

        # merge into one dict
        return {**eye_metrics, **task_metrics}

    except Exception as e:
        print(f"  Error processing {participant_id} condition {condition_num}: {e}")
        return None


def run_all(data_dir, output_path):
    """
    Find all pairs → group by participant → process all 3 conditions per
    participant → build one row per participant → write results.csv.
    
    Print summary: n participants, n rows written, output path.
    """
    pairs = find_file_pairs(data_dir)

    # group pairs by participant
    participants = {}
    for pair in pairs:
        pid = pair["participant_id"]
        if pid not in participants:
            participants[pid] = {}
        participants[pid][pair["condition_num"]] = pair

    rows = []
    for pid, conditions in participants.items():
        print(f"\nParticipant {pid}:")

        # process all three conditions
        metrics = {}
        for condition_num in [1, 2, 3]:
            if condition_num not in conditions:
                print(f"  Missing condition {condition_num} — skipping participant.")
                break
            pair = conditions[condition_num]
            result = process_pair(pair["task_path"], pair["eye_path"], pid, condition_num)
            if result is None:
                print(f"  Failed — skipping participant.")
                break
            metrics[CONDITION_MAP[condition_num]] = result
        else:
            # all three conditions processed successfully
            row = build_output_row(pid, metrics["low"], metrics["medium"], metrics["high"])
            rows.append(row)

    write_results(rows, output_path)
    print(f"\nDone. {len(rows)} participants written to {output_path}.")

def run_one(data_dir, output_path):
    """
    Prompt for participant ID → process that participant's 3 conditions
    → build one output row → write to CSV.
    """
    pid   = input("Enter participant ID: ").strip()
    pairs = find_file_pairs(data_dir)

    # filter to this participant
    participant_pairs = {p["condition_num"]: p for p in pairs if p["participant_id"] == pid}

    if not participant_pairs:
        print(f"No files found for participant {pid}.")
        return

    metrics = {}
    for condition_num in [1, 2, 3]:
        if condition_num not in participant_pairs:
            print(f"Missing condition {condition_num} for participant {pid}.")
            return
        pair   = participant_pairs[condition_num]
        result = process_pair(pair["task_path"], pair["eye_path"], pid, condition_num)
        if result is None:
            print(f"Failed on condition {condition_num}.")
            return
        metrics[CONDITION_MAP[condition_num]] = result

    row = build_output_row(pid, metrics["low"], metrics["medium"], metrics["high"])
    write_results([row], output_path)

def preview_pairs(data_dir):
    """
    Print a table of all found file pairs with pairing status.
    No processing. Used to verify data integrity before a full run.
    """
    pairs = find_file_pairs(data_dir)

    if not pairs:
        print("No file pairs found.")
        return

    print(f"\n{'Participant':<15} {'Condition':<10} {'Task file':<10} {'Eye file':<10}")
    print("-" * 50)
    for p in pairs:
        print(f"{p['participant_id']:<15} {CONDITION_MAP[p['condition_num']]:<10} {'✓':<10} {'✓':<10}")
    print(f"\n{len(pairs)} pairs found.")

def inspect_one(data_dir):
    """
    Prompt for participant ID + condition. Run full pipeline and print
    all intermediate results in detail (for debugging/validation).
    """
    pid           = input("Enter participant ID: ").strip()
    condition_num = int(input("Enter condition number (1/2/3): ").strip())
    pairs         = find_file_pairs(data_dir)

    match = next((p for p in pairs if p["participant_id"] == pid
                  and p["condition_num"] == condition_num), None)

    if not match:
        print(f"No files found for participant {pid}, condition {condition_num}.")
        return

    # load and print intermediate results at each stage
    task_df = load_task_file(match["task_path"])
    eye_df  = load_eye_file(match["eye_path"])
    print(f"\nEye file   : {len(eye_df)} rows")
    print(f"Task file  : {len(task_df)} rows")

    eye_df = detect_tracker_loss(eye_df)
    eye_df = detect_teleportation(eye_df)
    print(f"Tracker loss samples : {eye_df['TRACKER_LOSS'].sum()}")
    print(f"Teleport samples     : {eye_df['TELEPORT'].sum()}")

    eye_df, task_df = align_timestamps(eye_df, task_df)
    valid = eye_df[~eye_df["TRACKER_LOSS"] & ~eye_df["TELEPORT"]].reset_index(drop=True)
    print(f"Valid samples        : {len(valid)}")

    unit_vectors = euler_to_unit_vector(valid["GAZE_DIR_X"], valid["GAZE_DIR_Y"], valid["GAZE_DIR_Z"])
    angles       = compute_angular_displacement(unit_vectors)
    velocity     = compute_velocity(angles, valid["TIMESTAMP"].values)
    acceleration = compute_acceleration(velocity)
    velocity     = smooth_velocity(velocity)
    print(f"\nVelocity — min: {velocity.min():.2f}  max: {velocity.max():.2f}  mean: {velocity.mean():.2f} °/s")

    threshold = compute_adaptive_threshold(velocity)
    print(f"Adaptive threshold   : {threshold:.2f} °/s")

    labels = classify_samples(velocity, acceleration, threshold)
    labels = apply_fixation_duration_filter(labels, valid["TIMESTAMP"].values)
    print(f"\nFixations  : {labels.count('fixation')}")
    print(f"Saccades   : {labels.count('saccade')}")

    fixations_df = assign_fixation_objects(valid, labels)
    fixations_df = classify_fixation_relevance(fixations_df, condition_num)
    print(f"\nFixation events:")
    print(fixations_df.to_string(index=False))

    eye_metrics  = compute_eye_metrics(fixations_df, labels, valid["TIMESTAMP"].values)
    task_metrics = compute_task_performance(task_df)
    print(f"\nEye metrics  : {eye_metrics}")
    print(f"Task metrics : {task_metrics}")


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
║  5 — Exit                                            ║
╚══════════════════════════════════════════════════════╝
"""

def get_data_dir():
    """
    Prompt for data directory path. Strip quotes (macOS drag-in).
    Repeat until a valid directory is entered.
    """
    while True:
        # strip quotes in case user drags folder into terminal on macOS
        path = input("Enter path to data folder: ").strip().strip("'\"")
        if os.path.isdir(path):
            return path
        print(f"  '{path}' is not a valid directory. Please try again.")

def get_output_path():
    """
    Prompt for output CSV path. Default: results.csv in current directory.
    """
    path = input("Enter output file path (press Enter for 'results.csv'): ").strip().strip("'\"")
    if not path:
        return "results.csv"
    if os.path.isdir(path):
        path = os.path.join(path, "results.csv")
        print(f"  Path is a directory — saving to {path}")
    return path

def menu():
    """
    Display menu, route input to the correct function. Loop until exit.
    """
    print(MENU)

    while True:
        choice = input("Select an option (1–5): ").strip()

        if choice == "1":
            data_dir    = get_data_dir()
            output_path = get_output_path()
            run_all(data_dir, output_path)

        elif choice == "2":
            data_dir    = get_data_dir()
            output_path = get_output_path()
            run_one(data_dir, output_path)

        elif choice == "3":
            data_dir = get_data_dir()
            preview_pairs(data_dir)

        elif choice == "4":
            data_dir = get_data_dir()
            inspect_one(data_dir)

        elif choice == "5":
            print("\nGoodbye. Good luck with the analysis!\n")
            break

        else:
            print("  Invalid option. Please enter a number between 1 and 5.")

        # reprint menu after each action
        print("\nWhat would you like to do next?")


if __name__ == "__main__":
    menu()