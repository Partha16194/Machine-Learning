import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Indian Gold Price Predictor", layout="centered")

# ---------------- CUSTOM FULL-BLACK THEME CSS ----------------
st.markdown("""
<style>
/* Global Page */
html, body, [class*="stApp"] {
    background-color: #000000 !important;
    color: #d4af37 !important;
    font-family: 'Poppins', sans-serif;
}

/* Remove default Streamlit white padding */
.main {
    background-color: #000000 !important;
    color: #d4af37 !important;
    padding: 2rem 4rem;
}

/* Title and Headers */
h1, h2, h3, h4 {
    color: #f5ce42 !important;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 1px;
}
h1 {
    font-weight: 700;
    border-bottom: 1px solid #d4af37;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Section containers */
.block-container {
    background-color: #000000 !important;
}

/* Input fields */
input, textarea, select, .stNumberInput input {
    background-color: #1a1a1a !important;
    color: #f0d77b !important;
    border: 1px solid #d4af37 !important;
    border-radius: 5px;
    padding: 0.4rem 0.8rem;
    font-weight: 500;
}

/* Labels */
label, .stMarkdown, .stText {
    color: #d4af37 !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(to right, #d4af37, #f5ce42);
}
.stSlider label {
    color: #d4af37 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(145deg, #d4af37, #b58b19);
    color: #000000;
    font-weight: 600;
    font-size: 15px;
    border-radius: 6px;
    border: none;
    padding: 0.6rem 1.8rem;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(145deg, #f5ce42, #d4af37);
    transform: scale(1.04);
}

/* Info and success boxes */
.stAlert {
    background-color: #0d0d0d !important;
    border: 1px solid #d4af37 !important;
    color: #f0d77b !important;
    border-radius: 6px;
}

/* Help tooltips */
.stTooltip {
    background-color: #1a1a1a !important;
    color: #f0d77b !important;
}

/* Footer text */
.footer {
    text-align: center;
    color: #bfa84a;
    font-size: 13px;
    margin-top: 40px;
    border-top: 1px solid #d4af37;
    padding-top: 10px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #d4af37;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>Indian Gold Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:16px; color:#d4af37; max-width:700px; margin:auto;'>
This interactive tool estimates the <b>approximate Indian retail gold price</b> based on global financial indicators. 
The model predicts global gold price (USD/oz), converts it to INR, and adjusts for import duties and market margin.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid #d4af37;'>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("gold_price_model.pkl", "rb"))

# ---------------- USER INPUTS ----------------
st.markdown("<h2>Market Indicators</h2>", unsafe_allow_html=True)

spx = st.number_input(
    "Global Stock Market Index (S&P 500)",
    min_value=0.0, max_value=6000.0, value=3500.0,
    help="Overall U.S. stock market performance indicator."
)

uso = st.number_input(
    "Crude Oil Price Index (Oil ETF Value)",
    min_value=0.0, max_value=200.0, value=50.0,
    help="Approximate global crude oil price."
)

slv = st.number_input(
    "Silver Market Value (SLV ETF Price)",
    min_value=0.0, max_value=100.0, value=25.0,
    help="Represents global silver price trends."
)

eurusd = st.number_input(
    "Euro to U.S. Dollar Exchange Rate",
    min_value=0.5, max_value=2.0, value=1.1,
    help="Indicates strength of Euro against USD."
)

usd_inr = st.number_input(
    "Current USD to INR Exchange Rate",
    min_value=60.0, max_value=100.0, value=83.0,
    help="Current exchange rate between U.S. Dollar and Indian Rupee."
)

markup_percent = st.slider(
    "Add Import Duty & Retail Margin (%)",
    min_value=0, max_value=15, value=5,
    help="Additional cost for Indian market (import duty, GST, jeweller margin)."
)

st.markdown("<hr style='border-top: 1px solid #d4af37;'>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    input_data = np.array([[spx, uso, slv, eurusd]])
    global_price_usd = model.predict(input_data)[0]

    # Convert USD → INR and approximate Indian gold price
    indian_price_inr_per_gram = (global_price_usd * usd_inr / 31.1035) * (1 + markup_percent / 100)
    indian_price_inr_per_10g = indian_price_inr_per_gram * 10

    st.markdown(
    f"""
    <div style='background-color:#0d0d0d; border:1px solid #d4af37; border-radius:8px;
                padding:15px; margin-bottom:10px;'>
        <p style='color:#f5ce42; font-size:18px; font-weight:600; text-align:center;'>
            Predicted Global Gold Price: ${global_price_usd:.2f} per ounce
        </p>
    </div>
    """, unsafe_allow_html=True
    )

    st.markdown(
    f"""
    <div style='background-color:#141414; border:1px solid #d4af37; border-radius:8px;
                padding:15px;'>
        <p style='color:#ffd700; font-size:20px; font-weight:700; text-align:center;'>
            Approximate Indian Gold Price: ₹{indian_price_inr_per_10g:,.2f} per 10 grams
        </p>
    </div>
    """, unsafe_allow_html=True
    )

st.markdown("<hr style='border-top: 1px solid #d4af37;'>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
This model employs Random Forest Regression trained on global market data.  
Use results as indicative research estimates — not as financial advice.
</div>
""", unsafe_allow_html=True)
