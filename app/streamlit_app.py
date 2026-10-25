"""Run from the repo root: python -m streamlit run app/streamlit_app.py."""

import streamlit as st

from churn_retention.demo import load_demo_customers

st.set_page_config(page_title="Churn Retention AI", layout="centered")
st.title("Churn Retention AI")
st.warning("Synthetic demo only. No trained model, computed SHAP values or LLM calls.")
customers = load_demo_customers()
selected = st.selectbox("Customer", [row["customer_id"] for row in customers])
customer = next(row for row in customers if row["customer_id"] == selected)
st.metric("Illustrative churn probability", f"{customer['churn_probability']:.0%}")
st.caption(f"Snapshot: {customer['prediction_date']} | {customer['model_version']}")
st.subheader("Illustrative prediction drivers")
st.dataframe(customer["drivers"], hide_index=True)
st.subheader("Retention assistant placeholder")
st.info("Static example: consider an approved music-discovery message; human review required.")
st.write("Draft template: Explore something new with your next listening session.")
with st.expander("Shared prediction record"):
    st.json(customer)
