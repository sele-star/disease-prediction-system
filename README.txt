DISEASE PREDICTION APPLICATION — GOOGLE COLAB

Contents:
- app.py: Streamlit symptom-based prediction interface
- model_output/: trained Random Forest model and metadata/reports
- requirements.txt: application dependencies

GOOGLE COLAB SETUP
1. Upload this ZIP file to a new or existing Google Colab notebook.
2. Run:

import zipfile, os
zipfile.ZipFile('/content/disease_prediction_app_colab.zip').extractall('/content/disease_prediction_app')
%cd /content/disease_prediction_app
!pip install -r requirements.txt
!npm install -g localtunnel

3. Start Streamlit in the background:

!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

4. In another cell, start a public temporary tunnel:

!lt --port 8501

5. Open the https URL printed by localtunnel. If it asks for a tunnel password,
run this in another cell to display the Colab runtime public IP:

!curl -s https://api.ipify.org

Use that IP as the password if requested.

NOTES
- This app uses the exact 132 feature names stored in model_metadata.json.
- Select symptoms and click Predict disease.
- This is a research prototype, not a clinical diagnostic tool.
- It does not independently detect or forecast population-level outbreaks.
- The dataset is small after exact duplicate removal (304 unique rows), so reported
  benchmark performance must not be interpreted as proof of clinical validity.
