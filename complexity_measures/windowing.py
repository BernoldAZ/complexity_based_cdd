# Splits each event log into windows and records the window borders.
import os # used for file path handling
import pm4py # core process mining library (reading and filtering event logs)
import pandas as pd # used for dataframe operations
from pm4py.objects.conversion.log import converter as log_converter # needed to convert the event log object into a dataframe
from pm4py.objects.log.util import dataframe_utils

from pm4py.objects.log.importer.xes import importer as xes_importer # module for reading XES files
from pm4py.algo.filtering.log.attributes import attributes_filter # used to filter the log by a specific attribute -- here, to select only "complete" events
from pm4py.objects.log.obj import EventLog

# The functions below are adapted from the Version Clustering code base
# (utilities.py and preprocessing_methods.py).


def import_event_log(path: str, name: str) -> EventLog:
    # Defines the function, taking two parameters (path: str = file path, name: str = file name)
    # -> EventLog: this function returns an EventLog object (type hint)

    # ITERPARSE: fast reading method for large XES files:
    variant = xes_importer.Variants.ITERPARSE
    # load events strictly sorted by timestamp (required for correct windowing)
    parameters = {
        variant.value.Parameters.TIMESTAMP_SORT: True,  # Reads the events in ORDER according to their timestamps.
        variant.value.Parameters.SHOW_PROGRESS_BAR: False,  # don't show a progress bar while reading
    }
    # combine folder + file name, read the log, store the result
    event_log = xes_importer.apply(os.path.join(path, name), variant=variant, parameters=parameters)

    return event_log  # return the read log to the caller
# Source: utilities.py

# import_event_log reads the log with events sorted by timestamp.

# Keeps only "complete" lifecycle events, filtering out unfinished ones.
# Source: version-clustering-cdd-main, utilities.py

def filter_complete_events(log):
    try:
        # select events whose lifecycle:transition is "complete" or "COMPLETE"
        filtered_log = attributes_filter.apply_events(
            log, ["complete", "COMPLETE"], #Complete data is data that contains only the COMPLETE (finished) versions of the steps in a process — we only use the end record to avoid counting both the start and end of the same step separately.
            parameters={
                attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition",
                attributes_filter.Parameters.POSITIVE: True,  # keep matches, don't discard them
            },
        )
    except Exception:
        # if this attribute doesn't exist in the log, don't crash, just use the log as-is
        filtered_log = log
    return filtered_log
# Source: version-clustering-cdd-main, utilities.py

def window_event_log(event_log, n_windows):
    # convert the log into a table (DataFrame) to make slicing easier
    event_log_df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    # convert timestamp columns into real datetime objects (ISO8601 standard format)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(event_log_df, timest_format="ISO8601")

    # get unique trace IDs in chronological order
    unique_traces = pd.unique(event_log_df["case:concept:name"])
    # compute how many traces go into each window (integer division)
    window_size = len(event_log) // n_windows
    windows = []                # will hold the raw EventLog for each window
    borders = []                # will hold the (start, end) trace-index boundaries of each window
    window_information = {}     # will hold (first_trace_id, first_timestamp) per window, for later evaluation

    left_boundary = 0
    right_boundary = window_size

    for i in range(1, n_windows + 1):
        # select the trace IDs that belong to this window
        if i < n_windows:
            w_unique_traces = unique_traces[left_boundary:right_boundary]
        else:
            # last window: take everything remaining, so no traces are left out
            w_unique_traces = unique_traces[left_boundary:]
            right_boundary = len(unique_traces) - 1  # -1 because Python indexing starts at 0

        # filter only the events belonging to a trace in this window
        log_window_df = event_log_df[event_log_df["case:concept:name"].isin(w_unique_traces)]
        log_window = pm4py.convert_to_event_log(log_window_df)
        windows.append(log_window)

        borders.append((left_boundary, right_boundary))
        #pick this window's representative (first) trace
        first_trace = w_unique_traces[0]

        #find the earliest (start) timestamp of that trace
        first_timestamp = min(
            event_log_df.loc[event_log_df["case:concept:name"] == first_trace, "time:timestamp"]
        )
        window_information[i] = (first_trace, first_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        # %Y=year, %m=month, %d=day, %H=hour, %M=minute, %S=second

        # shift the boundaries forward for the next window
        left_boundary = right_boundary
        right_boundary += window_size  # right_boundary = right_boundary + window_size

    return windows, borders, window_information
# window_event_log splits the filtered log into consecutive windows with the same number of
# traces (the last window also takes the remaining traces) and returns the windows, their
# trace-index borders and the first trace/timestamp of each window.


if __name__ == "__main__":
    # read the raw XES log from the "data" folder
    log = import_event_log("data", "Sepsis Cases - Event Log.xes")

    # keep only the finished ("complete") events
    log = filter_complete_events(log)

    #  print total trace count -- lets us manually verify the window_size math (traces // n_windows)
    print("total traces in log:", len(log))

    # split the filtered log into 10 windows
    windows, borders, window_information = window_event_log(log, n_windows=10)

    # sanity check: did we really get 10 windows back?
    print(f"{len(windows)} windows created")

    # inspect window 1's metadata: (first_trace_id, first_timestamp)
    print("window 1 info:", window_information[1])

    # inspect window 1's trace-index boundaries (left, right)
    print("window 1 border (left, right):", borders[0])

    # inspect how many traces actually landed in window 1
    print("window 1 trace count:", len(windows[0]))
    # print a summary of each window

    for i in range(1, 11):
      print(f"window {i}: trace count = {len(windows[i-1])}, info = {window_information[i]}, border = {borders[i-1]}")