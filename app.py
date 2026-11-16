"""Price estimate form. Run `python pipeline.py` first, then `streamlit run app.py`."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import to_X

art = joblib.load(Path(__file__).parent / "data" / "model.joblib")
TYPES = {"Detached": "D", "Semi-detached": "S", "Terraced": "T", "Flat": "F"}

st.title("West Midlands house price estimate")
outcode = st.selectbox("Postcode district", art["lookup"].index)
kind = st.selectbox("Property type", list(TYPES))
new_build = st.checkbox("New build")
leasehold = st.checkbox("Leasehold", value=kind == "Flat")

place = art["lookup"].loc[outcode]
row = pd.DataFrame([{"type": TYPES[kind], "new_build": "Y" if new_build else "N",
                     "tenure": "L" if leasehold else "F", "district": place.district, "outcode": outcode,
                     "lat": place.lat, "lon": place.lon,
                     "neighbour_logp": place.neighbour_logp, "month_idx": art["latest_month_idx"]}])
price = float(np.exp(art["model"].predict(to_X(row, art["cats"])))[0])
st.metric("Estimated price", f"£{price:,.0f}")
st.caption("Typical error is large because the model has no floor area or condition data. See the README.")
