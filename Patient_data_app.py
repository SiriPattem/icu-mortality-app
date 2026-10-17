
import json, os

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="ICU Mortality Risk", page_icon="🏥", layout="centered")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(HERE, "models", "icu_model.joblib"))
    with open(os.path.join(HERE, "models", "metadata.json")) as f:
        meta = json.load(f)
    return model, meta


model, meta = load_model()
features, info, cutoff = meta["app_features"], meta["info"], meta["threshold"]

st.title("🏥 ICU In-Hospital Mortality Risk")
st.warning("**Educational project only.** This is not a medical device and must not be used "
           "for real clinical decisions.")

if meta.get("sklearn_version") != sklearn.__version__:
    st.warning(f"The model was saved with scikit-learn {meta.get('sklearn_version')}, but this app "
               f"is running {sklearn.__version__}. Predictions may fail or differ.")

st.write("Enter the patient's values. The boxes start at typical values from the dataset. "
         "If a value was never measured, select it in the list below the boxes.")


values = {}
cols = st.columns(2)
for i, f in enumerate(features):
    d = info[f]
    if f.startswith("GCS"):               
        lo, hi = 3.0, 15.0
    else:
        lo, hi = 0.0, round(d["max"] * 1.5, 1)
    default = float(min(max(round(d["median"], 1), lo), hi))
    values[f] = cols[i % 2].number_input(d["label"], min_value=lo, max_value=hi,
                                         value=default, key=f)

not_measured = st.multiselect("Values that were not measured",
                              [info[f]["label"] for f in features])
for f in features:
    if info[f]["label"] in not_measured:
        values[f] = np.nan                  


if st.button("Calculate risk", type="primary"):
    X_new = pd.DataFrame([values])[features]
    risk = float(model.predict_proba(X_new)[0, 1])

    st.metric("Estimated risk of in-hospital death", f"{risk:.1%}")
    st.progress(min(risk, 1.0))

    if risk >= cutoff:
        st.error(f"**High risk.** The estimate is above the cut-off of {cutoff:.0%} chosen for this model.")
    elif risk >= 0.10:
        st.warning("**Medium risk.** Above 10%, but below the model's cut-off.")
    else:
        st.success("**Low risk.** Below 10%.")
    st.caption("This is a statistical estimate from a small dataset, not a diagnosis.")

with st.expander("About this model"):
    st.write(f"- **Model:** Random Forest trained on {meta['n_patients']:,} ICU patients "
             f"({meta['death_rate']:.0%} died in hospital).")
    st.write(f"- **Test ROC-AUC:** {meta['test_roc_auc']} on patients the model had never seen.")
    st.write(f"- **Cut-off for 'High risk':** {cutoff:.0%}.")
    st.write("- **Inputs:** the 12 measurements that mattered most in the permutation-importance step.")
    st.write("- **Limits:** associations in a small dataset, not causes. Not for clinical use.")