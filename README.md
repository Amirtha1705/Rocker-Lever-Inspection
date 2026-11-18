# Rocker Lever Inspection - Automated Quality Analytics

An automated quality-control analytics system for rocker lever inspection data, built with Python and Flask. The user uploads an inspection CSV and the system automatically runs statistical process control, anomaly detection and a data-grounded AI analyst, and shows everything on a live dashboard. The whole application (backend and frontend) lives in a single file, `app.py`.

## Problem

- Rocker levers are critical engine components; small dimensional deviations cause valve-train wear and failures.
- Inspection produces hundreds of CSV records per shift, and reviewing them manually in Excel is slow and error-prone.
- Problems are usually noticed hours late, after many bad parts have already been produced.

## Objectives

- Automate analysis of inspection CSV data with no manual calculation.
- Detect out-of-control processes and abnormal parts early.
- Let engineers ask questions in plain language and get answers computed from their own data.
- Keep the tool lightweight and offline so it can run on any factory workstation.

## Features

- CSV upload: drag and drop any inspection CSV; numeric parameters are detected automatically.
- SPC (Statistical Process Control):
  - X-bar control charts with mean, UCL and LCL (plus or minus 3 sigma)
  - Western Electric rules 1 to 3 (point beyond 3 sigma, 9 points on one side, 6-point trend)
  - Cp and Cpk indices
- Anomaly detection: Isolation Forest over all numeric parameters together, to catch unusual combinations a single-parameter limit would miss.
- RAG-based AI analyst:
  - Every inspection row and every column summary is indexed as a text chunk (TF-IDF vectors).
  - A question retrieves the most similar chunks by cosine similarity, shown as evidence.
  - The answer itself is computed from the data: reject rate by machine, shift, operator and lot, out-of-spec parameters, drift, SPC status and anomalies.
- Dashboard:
  - Pareto chart of SPC violations
  - Process capability (Cpk) bar chart
  - Control chart with UCL, LCL and mean
  - Anomaly doughnut chart
  - Parameter radar chart
  - Raw data table and anomaly table
- Report export: download a JSON inspection report.

## Tech Stack

- Language: Python 3.10+
- Backend: Flask
- Data processing: pandas, NumPy
- Machine learning: scikit-learn (Isolation Forest, PCA, StandardScaler, TF-IDF)
- Frontend: HTML, CSS, JavaScript, Chart.js (inside app.py)
- Optional upgrades: sentence-transformers and FAISS for semantic search

## Project Structure

- app.py - complete application (Flask backend, analytics engines, HTML/JS frontend)
- generate_sample_data.py - creates simulated rocker lever inspection data
- rocker_lever_inspection_sample.csv - sample dataset with 500 records
- .gitignore

## How to Run

1. Install the dependencies:

   pip install flask pandas numpy scikit-learn

2. (Optional) Regenerate the sample data:

   python generate_sample_data.py

3. Start the app:

   python app.py

4. Open http://127.0.0.1:5000 in a browser and drag in the CSV file.

## How It Works

- Step 1: The CSV is uploaded and cleaned; dates are parsed and numeric columns are scaled.
- Step 2: Isolation Forest flags abnormal parts (5 percent contamination setting).
- Step 3: The SPC engine computes control limits, rule violations and Cp/Cpk for each parameter.
- Step 4: The RAG engine indexes all rows and column statistics for retrieval.
- Step 5: The dashboard shows the results, and the AI analyst answers questions using retrieved evidence plus computed statistics.

## Sample Questions for the AI Analyst

- Which machine has the most rejects?
- Which shift has the most defects?
- What are the top defect causes?
- Detect trends and drift
- Show Cpk capability summary
- Which parameters are out of control?
- Root cause of anomalies
- Full inspection report

## CSV Format

- Any CSV with numeric measurement columns works.
- The sample file uses: Inspection_DateTime, Part_ID, Shift, Operator_ID, Machine_ID, Material_Lot, Rocker_Arm_Length, Pivot_Bore_Diameter, Valve_Tip_Radius, Push_Rod_Socket_Depth, Lash_Pad_Thickness, Surface_Roughness_Ra, Bore_Roundness, Pivot_Runout, Hardness_HRC, Weight_grams, Visual_Defects, Inspection_Result.
- Text columns such as Shift, Machine_ID and Operator_ID (up to 12 distinct values) enable the group-wise reject analysis.
- A column named like Inspection_Result with values such as PASS or REJECT is used to count rejects.

## API Endpoints

- GET / - dashboard
- POST /upload - upload CSV and get full analytics
- POST /api/chat - ask the AI analyst a question
- GET /api/spc/<column> - SPC detail for one parameter
- GET /api/anomalies - parts flagged as anomalies
- GET /api/export/report - JSON inspection report

## Results on the Sample Data (500 simulated records)

- The sample contains 15 deliberately injected outliers; Isolation Forest found all 15 (100 percent recall) and flagged 10 extra parts (60 percent precision).
- 35 parts are labelled REJECT by specification check.
- The AI analyst correctly identifies the Night shift as the worst (11.7 percent reject rate vs 4.3 percent for Morning), matching the hardness drop injected into the simulation.
- These results come from simulated data and do not represent real production performance.

## Data Note

The dataset is simulated, because real inspection data from the company is confidential. Parameter ranges, tolerances and failure patterns (tool wear drift, night-shift hardness dip, random outliers) were designed to resemble real rocker lever inspection. The system accepts any CSV, so real CMM or gauge exports can be used directly.

## Limitations

- Data is stored in memory for one session only; there is no database.
- The anomaly rate is fixed at 5 percent.
- The AI analyst is rule-based and retrieval-grounded; it does not use a large language model.
- Cp/Cpk assume roughly normal data and currently use the process control limits; entering engineering specification limits per parameter is a planned improvement.
- The Flask development server is used; production use needs a WSGI server and debug mode turned off.

## Future Work

- Automatic ingestion of CMM or gauge CSV files from a watch folder
- Engineering specification limits (LSL and USL) per parameter for exact Cpk
- Semantic embeddings (sentence-transformers and FAISS) and an LLM-generated report
- Database storage and per-machine, per-shift dashboards

