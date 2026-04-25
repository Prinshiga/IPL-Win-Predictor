import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f8fa;
}

.main { background-color: #f7f8fa; }

.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 40px 36px 32px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
    color: white;
}
.hero p {
    font-size: 1rem;
    opacity: 0.75;
    margin: 0;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin-top: 14px;
    letter-spacing: 0.5px;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8a94a6;
    margin-bottom: 12px;
    margin-top: 28px;
}

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #e8eaed;
    text-align: center;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8a94a6;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 600;
    color: #1a1f2e;
    font-family: 'DM Serif Display', serif;
}

.predict-btn > button {
    background: linear-gradient(135deg, #0f2027, #2c5364) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 14px !important;
    border: none !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}
.predict-btn > button:hover { opacity: 0.88 !important; }

.result-win {
    background: #f0faf4;
    border: 1px solid #b7e4c7;
    border-radius: 12px;
    padding: 16px 20px;
    color: #1e7e44;
    font-weight: 500;
    font-size: 1rem;
    margin-top: 8px;
}
.result-loss {
    background: #fff5f5;
    border: 1px solid #fca5a5;
    border-radius: 12px;
    padding: 16px 20px;
    color: #b91c1c;
    font-weight: 500;
    font-size: 1rem;
    margin-top: 8px;
}
.result-neutral {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 12px;
    padding: 16px 20px;
    color: #92400e;
    font-weight: 500;
    font-size: 1rem;
    margin-top: 8px;
}

.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #b0b8c4;
    margin-top: 40px;
    padding-bottom: 20px;
    letter-spacing: 0.3px;
}

div[data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
    border: 1px solid #e0e3e8 !important;
    background: white !important;
}
div[data-testid="stNumberInput"] > div {
    border-radius: 10px !important;
    border: 1px solid #e0e3e8 !important;
    background: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    best = os.path.join(BASE_DIR, 'best_model.pkl')
    fallback = os.path.join(BASE_DIR, 'model.pkl')
    le_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    model = joblib.load(best if os.path.exists(best) else fallback)
    le = joblib.load(le_path)
    return model, le

try:
    model, le = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>IPL Win Predictor</h1>
    <p>Real-time match win probability powered by machine learning</p>
    <div class="badge">XGBoost &nbsp;|&nbsp; 91.74% Accuracy &nbsp;|&nbsp; 260,920 ball records</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Model files not found. Please run notebook 04_model_comparison.ipynb first.")
    st.stop()

# ── Teams ─────────────────────────────────────────────────────────────────────
TEAMS = sorted([
    'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
    'Lucknow Super Giants', 'Gujarat Titans'
])

# ── Team Selection ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Select Teams</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting Team (Chasing)", TEAMS, index=3)
with col2:
    bowling_team = st.selectbox("Bowling Team (Defending)",
                                 [t for t in TEAMS if t != batting_team], index=0)

# ── Match Situation ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Match Situation</div>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    target = st.number_input("Target Score", min_value=50, max_value=300, value=180)
with col4:
    runs_so_far = st.number_input("Runs Scored", min_value=0, max_value=int(target)-1, value=80)
with col5:
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=9, value=2)

overs_done = st.slider("Overs Completed", min_value=0.1, max_value=19.5, value=10.0, step=0.1)

# ── Live Calculations ─────────────────────────────────────────────────────────
balls_bowled  = int(overs_done) * 6 + round((overs_done % 1) * 10)
balls_remaining = max(120 - balls_bowled, 1)
runs_needed   = max(target - runs_so_far, 1)
crr = round((runs_so_far / max(balls_bowled, 1)) * 6, 2)
rrr = round((runs_needed / balls_remaining) * 6, 2)

# ── Live Stats Cards ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Live Statistics</div>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
s1.markdown(f'<div class="stat-card"><div class="stat-label">Runs Needed</div><div class="stat-value">{runs_needed}</div></div>', unsafe_allow_html=True)
s2.markdown(f'<div class="stat-card"><div class="stat-label">Balls Left</div><div class="stat-value">{balls_remaining}</div></div>', unsafe_allow_html=True)
s3.markdown(f'<div class="stat-card"><div class="stat-label">CRR</div><div class="stat-value">{crr}</div></div>', unsafe_allow_html=True)
s4.markdown(f'<div class="stat-card"><div class="stat-label">RRR</div><div class="stat-value">{rrr}</div></div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("Predict Win Probability", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict:
    try:
        all_teams = list(le.classes_)
        bat_enc  = le.transform([batting_team])[0]  if batting_team  in all_teams else 0
        bowl_enc = le.transform([bowling_team])[0]  if bowling_team  in all_teams else 1

        features = np.array([[bat_enc, bowl_enc, target, runs_so_far,
                               wickets, balls_remaining, runs_needed, crr, rrr]])
        proba = model.predict_proba(features)[0]
        win_prob  = round(float(proba[1]) * 100, 1)
        loss_prob = round(float(proba[0]) * 100, 1)

        # Gauge
        st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)
        gauge_color = "#16a34a" if win_prob >= 50 else "#dc2626"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_prob,
            title={'text': f"{batting_team}", 'font': {'size': 17, 'family': 'DM Sans'}},
            number={'suffix': "%", 'font': {'size': 44, 'family': 'DM Serif Display'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ccc"},
                'bar': {'color': gauge_color, 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0,  40], 'color': '#fef2f2'},
                    {'range': [40, 60], 'color': '#fefce8'},
                    {'range': [60, 100],'color': '#f0fdf4'},
                ],
                'threshold': {
                    'line': {'color': '#64748b', 'width': 3},
                    'thickness': 0.75, 'value': 50
                }
            }
        ))
        fig.update_layout(
            height=320,
            margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font={'family': 'DM Sans'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Both teams
        r1, r2 = st.columns(2)
        r1.markdown(f'<div class="stat-card"><div class="stat-label">{batting_team}</div><div class="stat-value">{win_prob:.1f}%</div></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="stat-card"><div class="stat-label">{bowling_team}</div><div class="stat-value">{loss_prob:.1f}%</div></div>', unsafe_allow_html=True)

        # Verdict
        st.markdown("<br>", unsafe_allow_html=True)
        if win_prob >= 70:
            st.markdown(f'<div class="result-win">{batting_team} are in a commanding position and are strong favourites to win this match.</div>', unsafe_allow_html=True)
        elif win_prob >= 50:
            st.markdown(f'<div class="result-neutral">It is a close contest. {batting_team} have a slight edge with {win_prob:.1f}% win probability.</div>', unsafe_allow_html=True)
        elif win_prob >= 30:
            st.markdown(f'<div class="result-neutral">{bowling_team} are currently favourites to defend. {batting_team} need to accelerate.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-loss">{bowling_team} are in a very strong position. {batting_team} require a dramatic turnaround.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    IPL Win Predictor &nbsp;|&nbsp; Data Science Module Project &nbsp;|&nbsp;
    XGBoost Model &nbsp;|&nbsp; Trained on 260,920 IPL deliveries (2008–2024)
</div>
""", unsafe_allow_html=True)
