"""Change-point evaluation metrics for single-point (1D) drift detections: TP/FP
counts, precision/recall, F1-score, TPR/FPR, and LP-based assignment of detected
change points to ground-truth ones, plus ROC plotting helpers."""

###############################################
############ Evaluation Metrics ###############
###############################################

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD

#from cdrift.utils.helpers import calcAvgDuration, convertToTimedelta


def getTP_FP(detected: List[int], known: List[int], lag: int, count_duplicate_detections: bool = True) -> Tuple[
    int, int]:
    """Returns the number of true and false positives, using assign_changepoints to calculate the assignments of detected change point to actual change point.

    Args:
        detected (List[int]): List of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        count_duplicate_detections (bool, optional): If a detected change point is not assigned to a ground truth change point, but lies within the lag window of some ground truth change point, should it be counted as a false positive (True if yes, False if no). Defaults to True.
    Returns:
        Tuple[int,int]: Tuple of: (true positives, false positives)

    Examples:
    #>>> getTP_FP([1000,1001,2000], [1000,2000], 200, True)
    #>>> (2,1)

    #>>> getTP_FP([1000,1001,2000], [1000,2000], 200, False)
    #>>> (2,0)
    """
    assignments = assign_changepoints(detected, known, lag_window=lag)

    TP = len(assignments)  # Every assignment is a True Positive, and every detected point is assigned at most once
    if count_duplicate_detections:
        FP = len(detected) - TP
    else:
        true_positive_candidates = [d for d in detected if any(
            (k - lag <= d and d <= k + lag) for k in known)]  # Detections that are in range of a
        FP = len(detected) - len(true_positive_candidates)
    return (TP, FP)


def calcPrecisionRecall(detected: List[int], known: List[int], lag: int, zero_division=np.NaN,
                        count_duplicate_detections: bool = True) -> Tuple[float, float]:
    """Calculates the precision and recall, using `get_TP_FP` for True positives and False Negatives, which uses assign_changepoints to calculate the assignments of detected change point to actual change point.

    Args:
        detected (List[int]): A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        zero_division (Any, optional): The value to yield for precision/recall when a zero-division is encountered. Defaults to np.NaN.
        count_duplicate_detections (bool, optional): If a detected change point is not assigned to a ground truth change point, but lies within the lag window of some ground truth change point, should it be counted as a false positive (True if yes, False if no). Defaults to True.

    Returns:
        Tuple[Union[float,np.NaN], Union[float,np.NaN]]: A tuple (precision, recall).
    """

    TP, FP = getTP_FP(detected, known, lag, count_duplicate_detections)
    if (TP + FP > 0):
        precision = TP / (TP + FP)
    else:
        precision = zero_division
    if (len(known) > 0):
        recall = TP / len(known)
    else:
        recall = zero_division
    return (precision, recall)


def F1_Score(detected: List[int], known: List[int], lag: int, zero_division=np.NaN, verbose: bool = False,
             count_duplicate_detections: bool = True) -> float:
    """ Calculates the F1 Score for a Changepoint Detection Result

    - Considering a known changepoint at timepoint t:
        - A True Positive is when a changepoint is detected within [t-`lag`, t+`lag`]
        - A False Negative is when no changepoint is detected in this window around a known changepoint
        - A False Positive is when there is no known changepoint in a window of `lag` around the detected one
        - Note: Only one detected change point can be a TP for a given known changepoint, and vice versa. The assignment of detected change points to actual change points is done using a Linear Program (see assign_changepoints)
    - From this the F1-Score is calculated as (2&middot;precision&middot;recall) / (precision+recall)

    Args:
        detected (List[int]) : A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        zero_division (str, optional): The return value if the calculation of precision/recall/F1 divides by 0.
        verbose (bool, optional): If verbose, warning messages are printed when a zero-division is encountered. Defaults to False.
        count_duplicate_detections (bool, optional): If a detected change point is not assigned to a ground truth change point, but lies within the lag window of some ground truth change point, should it be counted as a false positive (True if yes, False if no). Defaults to True.

    Returns:
        float: The F1-Score corresponding to the given prediction.
    """

    TP, FP = getTP_FP(detected, known, lag, count_duplicate_detections)
    try:
        precision = TP / (TP + FP)
        recall = TP / len(known)

        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score
    except ZeroDivisionError:
        if verbose:
            print("Calculation of F1-Score resulted in division by 0.")
        return zero_division


# Alias for F1_Score
f1 = F1_Score


def calcTPR_FPR(detected: List[int], known: List[int], lag: int, num_possible_negatives: int = None) -> Tuple[
    float, float]:
    """Calculates the True-Positive-Rate and the False-Positive-Rate for a given detection.

    Args:
        detected (List[int]): A list of indices of detected change point locations.
        known (List[int]): The ground truth; List of indices of actual change points.
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        num_possible_negatives (int, optional): The number of possible negatives. In theory, this is `len(log)-len(known)`, however this number is way too large. Defaults to None.

    Returns:
        Tuple[Union[float, np.NaN], Union[float,np.NaN]]: A tuple of: (True-Positive-Rate, False-Positive-Rate)
    """

    TP, FP = getTP_FP(detected, known, lag)
    P = len(known)
    TPR = TP / P
    # So many Negative points it wouldnt make sense....
    FPR = FP / num_possible_negatives if num_possible_negatives is not None else np.NaN
    return (TPR, FPR)


def assign_changepoints(detected_changepoints: List[int], actual_changepoints: List[int], lag_window: int = 200) -> \
List[Tuple[int, int]]:
    """Assigns detected changepoints to actual changepoints using a LP.
    With restrictions:

    - Detected point must be within `lag_window` of actual point.
    - Detected point can only be assigned to one actual point.
    - Every actual point can have at most one detected point assigned.

        This is done by first optimizing for the number of assignments, finding how many detected change points could be assigned, without minimizing the \
        total lag. Then, the LP is solved again, minimizing the sum of squared lags, while keeping the number of assignments as high as possible.

    Args:
        detected_changepoints (List[int]): List of locations of detected changepoints.
        actual_changepoints (List[int]): List of locations of actual changepoints.
        lag_window (int, optional): How close must a detected change point be to an actual changepoint to be a true positive. Defaults to 200.

    Examples:
    >>> detected_changepoints = [1050, 934, 2100]
    >>> actual_changepoints = [1000,1149,2000]
    >>> assign_changepoints(detected_changepoints, actual_changepoints, lag_window=200)
    >>> [(1050, 1149), (934, 1000), (2100, 2000)]
    >>> # Notice how the actual changepoint 1000 gets a further detected changepoint to allow 1149 to also get a changepoint assigned

    Returns:
        List[Tuple[int,int]]: List of tuples of (detected_changepoint, actual_changepoint) assignments
    """

    def buildProb_NoObjective(sense):
        """
            Builds the optimization problem, minus the Objective function. Makes multi-objective optimization simple
        """
        prob = LpProblem("Changepoint_Assignment", sense)

        # Create a variable for each pair of detected and actual changepoints
        vars = LpVariable.dicts("x", (detected_changepoints, actual_changepoints), 0, 1,
                                LpBinary)  # Assign detected changepoint dp to actual changepoint ap?

        # Flatten vars into dict of tuples of keys
        x = {
            (dc, ap): vars[dc][ap] for dc in detected_changepoints for ap in actual_changepoints
        }

        ####### Constraints #########

        # Only assign at most one changepoint to each actual changepoint
        for ap in actual_changepoints:
            prob += (
                lpSum(x[dp, ap] for dp in detected_changepoints) <= 1,
                f"Only_One_Changepoint_Per_Actual_Changepoint : {ap}"
            )
        # Each detected changepoint is assigned to at most one actual changepoint
        for dp in detected_changepoints:
            prob += (
                lpSum(x[dp, ap] for ap in actual_changepoints) <= 1,
                f"Only_One_Actual_Changepoint_Per_Detected_Changepoint : {dp}"
            )
        # Distance between chosen changepoints must be within lag window
        for dp in detected_changepoints:
            for ap in actual_changepoints:
                prob += (
                    x[dp, ap] * abs(dp - ap) <= lag_window,
                    f"Distance_Within_Lag_Window : {dp}_{ap}"
                )
        return prob, x

    solver = PULP_CBC_CMD(msg=0)

    ### Multi-Objective Optimization: First maximize number of assignments to find out the best True Positive number that can be achieved
    # Find the largest number of change points:
    prob1, prob1_vars = buildProb_NoObjective(LpMaximize)
    prob1 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob1_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Maximize number of assignments"
    )
    prob1.solve(solver)
    # Calculate number of TP
    num_tp = len([
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob1_vars[dp, ap].varValue == 1
    ])

    ### Multi-Objective Optimization: Now minimize the squared distance between assigned changepoints, using this maximal number of assignments
    # Use this number as the number of assignments for second optimization
    prob2, prob2_vars = buildProb_NoObjective(LpMinimize)
    prob2 += (
        lpSum(
            # Minimize the squared distance between assigned changepoints
            prob2_vars[dp, ap] * pow(dp - ap, 2)
            for dp in detected_changepoints for ap in actual_changepoints
        ),
        "Squared_Distances"
    )

    # Number of assignments is the number of true positives we found in the first optimization
    prob2 += (
        lpSum(
            prob2_vars[dp, ap]
            for dp in detected_changepoints for ap in actual_changepoints
        ) == num_tp,
        "Maximize Number of Assignments"
    )
    prob2.solve(solver)
    return [
        (dp, ap)
        for dp in detected_changepoints for ap in actual_changepoints
        if prob2_vars[dp, ap].varValue == 1
    ]


def get_avg_lag(detected_changepoints: List[int], actual_changepoints: List[int], lag: int = 200) -> float:
    """Calculates the average lag between detected and actual changepoints (Caution: false positives do not affect this metric!)

    Args:
        detected_changepoints (List[int]): Locations of detected changepoints
        actual_changepoints (List[int]): Locations of actual (known) changepoints
        lag (int, optional): How close must a detected change point be to an actual changepoint to be a true positive. Defaults to 200.

    Examples:
    >>> detected_changepoints = [1050, 934, 2100]
    >>> actual_changepoints = [1000,1149,2000]
    >>> get_avg_lag(detected_changepoints, actual_changepoints, lag=200)
    >>> 88.33333333333333

    Returns:
        float: the average distance between detected changepoints and the actual changepoint they get assigned to
    """
    assignments = assign_changepoints(detected_changepoints, actual_changepoints, lag_window=lag)
    avg_lag = 0
    for (dc, ap) in assignments:
        avg_lag += abs(dc - ap)
    try:
        return avg_lag / len(assignments)
    except ZeroDivisionError:
        return np.nan;


def getROCData(lag: int, df: pd.DataFrame, undefined_equals=0) -> List[Tuple[float, float]]:
    """Returns a list of points, as tuples of Recall (TPR) and Precision (Cannot do FPR because negatives are not really defined for concept drift detection/negatives are practically the entire log (`len(log)-len(detected)`))

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        df (pd.DataFrame): The Dataframe containing the detection results of the approach
        undefined_equals (int, optional): The value to assign to undefined F1-Scores. Defaults to 0.

    Returns:
        List[Tuple[float,float]]: A list of the mean precision and recall values for each Window Size found in the dataframe.
    """

    groups = df.groupby("Window Size")
    points = []
    for win, win_df in groups:
        for idx, row in win_df.iterrows():
            precisions = []
            recalls = []
            prec, rec = calcPrecisionRecall(lag, row["Detected Changepoints"], row["Actual Changepoints for Log"],
                                            zero_division=undefined_equals)
            precisions.append(prec)
            recalls.append(rec)
        # Average precision and recall over all logs for this
        points.append((np.mean(recalls), np.mean(precisions)))
    return points


def plotROC(lag, df: pd.DataFrame, undefined_equals=0) -> None:
    """Plot an ROC Curve (using precision and recall) for the given dataframe and a given lag value for precision and recall evaluation

    Args:
        lag (int): The maximal distance a detected change point can have to an actual change point, whilst still counting as a true positive.
        df (pd.DataFrame): The Dataframe containing the detection results of the approach
        undefined_equals (int, optional): The value to assign to undefined F1-Scores. Defaults to 0.
    """
    dat = getROCData(lag, df, undefined_equals)
    recalls, precisions = list(zip(*dat))
    plt.plot(precisions, recalls)  # x is precisions, y is recalls
    plt.ylim(-0.01, 1.01)
    plt.xlim(-0.01, 1.01)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

