"""
Run all complexity measures with all patched baselines.

Each baseline is tested with every alpha in ALPHAS.
"""

import os

import run_approach as ra
import run_bose_experiments as rb
import run_adwinj_experiments as rad
import run_emd_experiments as rem
import run_cluster_experiments as rvc

from constants import ALL_MEASURES


# Settings

ALPHAS = [0.5, 0.3, 0.1, 0.05]

MEASURES = list(ALL_MEASURES)

DATASETS_PATH = "datasets/cdrift"
OUT_ROOT = "baselines_results"
EVALUATE_NOISY = False

# Separate folders to avoid mixing with previous experiments.
OUTPUT_ROOTS = {
    "bose_j": os.path.join(OUT_ROOT, "patched_bose_all_measures"),
    "bose_wc": os.path.join(OUT_ROOT, "patched_bose_wc_all_measures"),
    "adwinj": os.path.join(OUT_ROOT, "patched_adwinj_all_measures"),
    "emd": os.path.join(OUT_ROOT, "patched_emd_all_measures"),
    "vc": os.path.join(OUT_ROOT, "patched_vc_all_measures"),
}


def make_run_name(baseline, alpha):
    alpha_label = str(alpha).replace(".", "_")
    return f"{baseline}_combined_all_alpha_{alpha_label}"


# Bose J and Bose WC

def run_bose():

    # The Bose runner reads these values from module-level variables.
    rb.DATASETS_PATH = DATASETS_PATH
    rb.N_WINDOWS = ra.DEFAULT_N_WINDOWS
    rb.LAG_WINDOW = 200
    rb.EVALUATE_NOISY = EVALUATE_NOISY

    for bose_measure, output_key in [
        ("J", "bose_j"),
        ("WC", "bose_wc"),
    ]:

        print(f"\n BOSE {bose_measure}", flush=True)

        for alpha in ALPHAS:

            rb.ALPHA = alpha
            rb.BOSE_MEASURE = bose_measure

            run_name = make_run_name(f"bose_{bose_measure.lower()}", alpha)

            print(f"\nRunning {run_name}", flush=True)

            rb.run(
                run_name=run_name,
                mode="combined",
                measures=MEASURES,
                out_root=OUTPUT_ROOTS[output_key],
                evaluate_noisy=EVALUATE_NOISY,
            )


# ADWIN/J

def run_adwinj():

    print("\n ADWIN/J", flush=True)

    rad.DATASETS_PATH = DATASETS_PATH
    rad.N_WINDOWS = ra.DEFAULT_N_WINDOWS
    rad.LAG_WINDOW = 200
    rad.EVALUATE_NOISY = EVALUATE_NOISY
    rad.CACHE_DIR = os.path.join(OUT_ROOT, "_cache")

    for alpha in ALPHAS:

        rad.ALPHA = alpha

        run_name = make_run_name("adwinj", alpha)

        print(f"\nRunning {run_name}", flush=True)

        rad.run(
            run_name=run_name,
            mode="combined",
            measures=MEASURES,
            out_root=OUTPUT_ROOTS["adwinj"],
            evaluate_noisy=EVALUATE_NOISY,
        )


# EMD

def run_emd():

    print("\n EMD", flush=True)

    for alpha in ALPHAS:

        run_name = make_run_name("emd", alpha)

        print(f"\nRunning {run_name}", flush=True)

        rem.run(
            run_name=run_name,
            feature_mode="combined",
            fusion="weighted_concat",
            alpha=alpha,
            measures=MEASURES,
            datasets_path=DATASETS_PATH,
            out_root=OUTPUT_ROOTS["emd"],
            evaluate_noisy=EVALUATE_NOISY,
        )


# Version Clustering

def run_vc():

    print("\n VC", flush=True)

    for alpha in ALPHAS:

        run_name = make_run_name("vc", alpha)

        print(f"\nRunning {run_name}", flush=True)

        rvc.run(
            run_name=run_name,
            feature_mode="combined",
            fusion="weighted_concat",
            alpha=alpha,
            measures=MEASURES,
            pen=ra.DEFAULT_PEN,
            n_windows=ra.DEFAULT_N_WINDOWS,
            min_cluster_size=None,
            datasets_path=DATASETS_PATH,
            out_root=OUTPUT_ROOTS["vc"],
            evaluate_noisy=EVALUATE_NOISY,
        )


# Main

if __name__ == "__main__":

    print("Starting all-measures patched experiments", flush=True)
    print(f"Number of measures: {len(MEASURES)}", flush=True)
    print(f"Alphas: {ALPHAS}", flush=True)

    for output_root in OUTPUT_ROOTS.values():
        os.makedirs(output_root, exist_ok=True)

    run_bose()
    run_adwinj()
    run_emd()
    run_vc()

    print("\nAll experiments completed.", flush=True)
