## Version Clustering: A Top-Down Approach for Process Concept Drift Detection

<sub>
Written by <a href="mailto:bernold.rodrigo.abarca.zuniga@univie.ac.at">Bernold Rodrigo Abarca Zúñiga</a><br />
</sub>

---

## About 

This repository contains the **implementation, evaluation scripts, and results** as described in the manuscript:

***Version Clustering: A Top-Down Approach for Process Concept Drift Detection*** by B. Abarca, A. Yeshchenko, and H. van der Aa, accepted at the main track of the CAiSE 2026 conference.

### Abstract

Business processes constantly evolve as organizations adapt to new regulations, technologies, and market conditions. Consequently, models discovered through process mining quickly lose validity unless process concept drifts are detected and managed. The newest and best drift detection approaches rely on supervised learning or complex statistical models, demanding labeled data and high computational effort.

This paper introduces Version Clustering (VC), a novel top-down unsupervised approach for process concept drift detection that is both robust and interpretable. 
VC's novelty lies in the inversion of the drift detection logic in comparison to existing approaches. Rather than pinpointing drift points outright, VC first clusters windows into coherent process versions via hierarchical density-based clustering, and then marks drift points where execution transitions between versions. This inversion  yields three advantages:
(i) improved robustness to noise and local behavioral variability,
(ii) more accurate predictions, and
(iii) a version-level representation that naturally supports visual and analytical interpretation.

Evaluated within a unified experimental framework and tested on established benchmarks, VC achieves high robustness to noise, strong accuracy, practical scalability, coverage of most drift types, and clear interpretability. Under the considered evaluation metrics, it matched or exceeded baseline approaches, offering an efficient and interpretable solution for tracking process evolution in noisy, dynamic environments.

---

## Setup

To run the Version Clustering approach, follow these steps.

### Get Datasets

Download the **CDLG** and **CDRIFT** datasets used for the evaluation of Version Clustering. Save the datasets in separate folders and note the paths for use during execution.

> A small subset of the CDLG dataset (`datasets/data_cdlg_small`) and the full CDRIFT dataset (`datasets/cdrift`) are already included in this repository for quick usage. A real-life event log without ground-truth drift points (`datasets/real/Road_Traffic_Fine_Management_Process.xes`) is also included as an example for the "Other Datasets" use case described below.

1.  **CDLG Dataset:**
    * Download and unzip all the files from the **"data/cdlg/test folder"** at the following source: [https://huggingface.co/datasets/pm-science/cv4cdd_4d](https://huggingface.co/datasets/pm-science/cv4cdd_4d)
2.  **CDRIFT Dataset:**
    * It is already downloaded and unzipped under `datasets/cdrift`.
    * Original source: [https://github.com/cpitsch/cdrift-evaluation/tree/main/EvaluationLogs](https://github.com/cpitsch/cdrift-evaluation/tree/main/EvaluationLogs)

### Installation and Environment

The project is designed to run within a **Dockerized Jupyter Lab** environment.

1.  **Clone this repository:**
    ```bash
    git clone https://gitlab.cs.univie.ac.at/bernolda00cs/version-clustering-cdd.git
    ```
2.  **Navigate to the project root directory:**
    ```bash
    cd version-clustering-cdd
    ```
3.  **Build the Docker image:**
    ```bash
    docker build -t vc-jupyter-image .
    ```
4.  **Run the Docker container:**
    * **IMPORTANT:** Replace `/path/to/your/local/folder` with the **absolute path** to your local project root directory to mount it into the container.
    * The image is built on a CUDA base (`nvidia/cuda:11.2.2-cudnn8`). If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, add `--gpus all` to use it; otherwise omit it to run on CPU only.
    ```bash
    # With GPU
    docker run -it --rm -p 127.0.0.1:8887:8887 -v "/path/to/your/local/folder:/app" --gpus all vc-jupyter-image

    # CPU only
    docker run -it --rm -p 127.0.0.1:8887:8887 -v "/path/to/your/local/folder:/app" vc-jupyter-image
    ```
    * This command will set up a **Jupyter Lab** environment accessible at **`http://localhost:8887`**.
    * Open the URL in your browser and copy the provided access token to log in.

---

## Usage Examples

The core execution is performed using the provided Jupyter notebooks.

### 1. Process Concept Drift Detection using Version Clustering (VC)

Open the **`VC_execution.ipynb`** notebook in your Jupyter Lab environment.

#### **A. Setup Parameters**

Modify the following variables to configure the VC run:

| Parameter | Description |
| :--- | :--- |
| **`log_dir`** | Path where your unzipped event logs are located. |
| **`n_windows`** | Number of windows to divide the event logs into (sets the granularity). |
| **`min_cluster_size`** | The smallest number of windows needed to form a process version (a key parameter for **Hierarchical Clustering**). |
| **`pen`** | Controls the number of detected change points (**penalty for change**). Lower penalties detect more local fluctuations; higher penalties focus on major shifts. |
| **`threshold`** | Affects **visualization only**, not drift detection. It allows analysts to choose the confidence level needed to visualize a window as a stable version. |

#### **B. Run Predict Cell**

This cell executes VC to detect drift points for all event logs found at the `log_dir` path.

* After execution, **two new folders** will be created within the `log_dir` path:
    * **`Annotations`**: Contains general information about each event log (timestamps, traces, window info). The `algorithm_results` file is a summary of detected/predicted change points (**not** created for CDLG dataset).
    * **`outputs`**: Stores the outputs for each event log, including:
        * `log_df_info` and `log_general` (used for visualization).
        * `log_output` (a simple visualization of the process evolution, showing detected and real drifts).
        * `log.txt` (a list of windows with a detected drift).

#### **C. Run Evaluate Cell**

This cell writes multiple evaluation files to the specified `eval_dir` path. The most important results are found in:

* `evaluation_results_aggregated_level_cp_lag.csv`
* `evaluation_results_aggregated_level_cp_lag_noise_level.csv`

#### **D. Run Sensitivity Analysis Cell**

This cell runs VC for all defined parameter combinations.

* Results will be written to the output path, which will be used later in the `VC_sensitivity_analysis.ipynb` notebook.

#### **E. Special Use Cases**

* **CDRIFT Dataset:** Before running the **Predict** step, and in case you have to dowload the dataset from the source, you need to create the `drift_info.csv` file with the ground truth drift points. For the **Bose** log, preprocessing is also required.
* **Other Datasets:** You can run the **Predict** step on any other dataset without needing ground truth drift points.

### 2. Sensitivity Analysis of Version Clustering

Open the **`VC_sensitivity_analysis.ipynb`** notebook.

1.  Ensure you have completed the **Sensitivity analysis** step in `VC_execution.ipynb` for both CDLG and CDRIFT (Bose, Ostovar, Ceravolo).
2.  Fill in the variables under the **Path cells** with the resulting path to the folder of each sensitivity analysis run.
3.  Run all cells.
4.  The notebook will display **heatmaps** showing the F1-scores for all different combinations of the key parameters, allowing for detailed parameter optimization. 

### 3. Evaluation by Unified Framework (Adams et al.)

The file **`algorithm_results_cdrift.csv`** already includes the results from the VC approach and CV4CDD-4D, along with the results for other techniques evaluated in the unified framework.

If you wish to use the results you obtained during your own execution, you must process them and append them to the `algorithm_results_cdrift_original.csv` file.

* Run the cells under **"Process VC results to use in unified evaluation framework"** in the `VC_execution.ipynb`, or use the dedicated **`process_cdrift_results_for_evaluation.ipynb`** notebook, to format your results correctly. This notebook also lets you append CV4CDD-4D results (read from `res_cvdd_cdrift.csv`) and any other technique's results.
* Once `algorithm_results_cdrift.csv` is up to date, open **`evaluation_unified_framework.ipynb`** and run all cells to compute the unified-framework metrics (Accuracy, Latency, Robustness, Scalability, Versatility, Parameter Sensitivity) and generate the corresponding plots and CSVs under `evaluation_results/Unified_framework_results`.

---

## Repository Structure

* **`utilities/`** — Core implementation of Version Clustering (clustering, evaluation, preprocessing, and configuration) called from the notebooks.
* **`cdrift/`** — Baseline drift-detection approaches (Bose, Maaradji, Martjushev, Zheng, Earthmover, LCDD, Process Graph Metrics) used for comparison; results from these are what populate `algorithm_results_cdrift.csv`.
* **`results_Bose/`, `results_Ceravolo/`, `results_Ostovar/`, `results_sensitivity/`, `evaluation_results/`** — Ship with precomputed outputs from the paper's runs; re-running the notebooks with the same parameters will overwrite these.
* **`bose_log.xes`** — Preprocessed version of the Bose log produced by the preprocessing step mentioned above; used in place of the raw log under `datasets/cdrift/Bose/`.

---

## License

Distributed under the **MIT License**.