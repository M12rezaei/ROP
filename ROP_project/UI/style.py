import streamlit as st

def apply_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
 
    /* ── page background ── */
    .stApp {
        background-color: #F0F4F9;
    }
 
    /* ── hide default streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; max-width: 1100px; }
 
    /* ── top nav bar ── */
    .top-nav {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 60%, #1976D2 100%);
        padding: 16px 28px;
        border-radius: 0 0 16px 16px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 20px rgba(13,71,161,0.18);
    }
    .top-nav h1 {
        color: white !important;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .top-nav p {
        color: rgba(255,255,255,0.75);
        font-size: 13px;
        margin: 2px 0 0 0;
    }
    .nav-icon {
        font-size: 32px;
        line-height: 1;
    }
 
    /* ── cards ── */
    .card {
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        border: 1px solid #E1E9F5;
        box-shadow: 0 2px 8px rgba(13,71,161,0.05);
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        color: #1565C0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 7px;
    }
 
    /* ── result summary card ── */
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        border: 1px solid #E1E9F5;
        box-shadow: 0 2px 8px rgba(13,71,161,0.05);
        margin-bottom: 16px;
    }
    .result-label {
        font-size: 28px;
        font-weight: 700;
        margin: 4px 0 10px 0;
        letter-spacing: -0.5px;
    }
 
    /* ── referral badges ── */
    .badge {
        display: inline-block;
        padding: 7px 18px;
        border-radius: 24px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .badge-severe  { background:#FEE2E2; color:#B91C1C; border:1.5px solid #FCA5A5; }
    .badge-mild    { background:#FEF3C7; color:#92400E; border:1.5px solid #FCD34D; }
    .badge-normal  { background:#DCFCE7; color:#15803D; border:1.5px solid #86EFAC; }
 
    /* ── confidence bar ── */
    .conf-row { display:flex; align-items:center; gap:12px; margin:10px 0; }
    .conf-bar-track {
        flex:1; height:10px;
        background:#EEF2F8; border-radius:6px; overflow:hidden;
    }
    .conf-bar-fill {
        height:10px; border-radius:6px;
        transition: width 0.5s ease;
    }
    .conf-pct { font-size:14px; font-weight:700; min-width:44px; text-align:right; }
 
    /* ── prob breakdown ── */
    .prob-row {
        display:flex; align-items:center;
        gap:10px; margin:5px 0; font-size:13px;
    }
    .prob-label { width:60px; color:#475569; font-weight:500; }
    .prob-track {
        flex:1; height:8px;
        background:#EEF2F8; border-radius:4px; overflow:hidden;
    }
    .prob-fill { height:8px; border-radius:4px; }
    .prob-pct  { width:40px; text-align:right; color:#475569; font-size:12px; }
 
    /* ── info grid ── */
    .info-grid {
        display:grid; grid-template-columns:1fr 1fr;
        gap:10px; margin-top:14px;
    }
    .info-item {
        background:#F8FAFD; border-radius:8px;
        padding:10px 14px; border:1px solid #E8EEF8;
    }
    .info-item .label { font-size:11px; color:#94A3B8; font-weight:600;
        text-transform:uppercase; letter-spacing:0.5px; }
    .info-item .value { font-size:16px; font-weight:700; color:#1E293B; margin-top:2px; }
 
    /* ── disclaimer ── */
    .disclaimer {
        background:#F8FAFD; border:1px solid #E1E9F5;
        border-left:4px solid #1565C0;
        border-radius:0 8px 8px 0;
        padding:12px 16px;
        font-size:12px; color:#64748B;
        margin-top:10px; line-height:1.6;
    }
 
    /* ── buttons ── */
    .stButton > button {
        border-radius: 10px;
        background: #1565C0;
        color: white;
        font-weight: 600;
        font-size: 14px;
        border: none;
        padding: 10px 22px;
        font-family: 'DM Sans', sans-serif;
        transition: background 0.2s, transform 0.1s;
        box-shadow: 0 2px 8px rgba(21,101,192,0.20);
    }
    .stButton > button:hover {
        background: #0D47A1;
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }
 
    /* ── back button (secondary) ── */
    .stButton.back > button {
        background: white;
        color: #1565C0;
        border: 1.5px solid #C7D8F0;
        box-shadow: none;
    }
 
    /* ── export hamburger popover trigger ── */
    [data-testid="stPopover"] > button {
        border-radius: 10px;
        background: white !important;
        color: #1565C0 !important;
        font-weight: 600;
        font-size: 14px;
        border: 1.5px solid #C7D8F0 !important;
        padding: 8px 16px;
        box-shadow: 0 1px 4px rgba(13,71,161,0.08);
        transition: background 0.15s, border-color 0.15s;
    }
    [data-testid="stPopover"] > button:hover {
        background: #EFF6FF !important;
        border-color: #1565C0 !important;
    }
 
    /* ── popover panel ── */
    [data-testid="stPopoverBody"] {
        border-radius: 12px !important;
        border: 1px solid #E1E9F5 !important;
        box-shadow: 0 8px 24px rgba(13,71,161,0.12) !important;
        padding: 14px !important;
        min-width: 210px !important;
    }
    [data-testid="stPopoverBody"] .stDownloadButton > button {
        background: #F8FAFD !important;
        color: #1E293B !important;
        border: 1px solid #E1E9F5 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        margin-bottom: 6px;
    }
    [data-testid="stPopoverBody"] .stDownloadButton > button:hover {
        background: #EFF6FF !important;
        border-color: #93C5FD !important;
        color: #1565C0 !important;
    }
 
    /* ── file uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #93C5FD;
        border-radius: 12px;
        background: #EFF6FF;
        padding: 6px;
    }
 
    /* ── number inputs ── */
    .stNumberInput input {
        border-radius: 8px !important;
        border: 1.5px solid #CBD5E1 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        padding: 8px 12px !important;
    }
    .stNumberInput input:focus {
        border-color: #1565C0 !important;
        box-shadow: 0 0 0 3px rgba(21,101,192,0.12) !important;
        outline: none !important;
    }
 
    /* ── slider ── */
    .stSlider [data-baseweb="slider"] { padding: 6px 0; }
 
    /* ── radio ── */
    .stRadio [data-testid="stMarkdownContainer"] p { font-size: 14px; }
 
    /* ── divider ── */
    hr { border:none; border-top:1px solid #E8EEF8; margin:16px 0; }
 
    /* ── image captions ── */
    .img-label {
        font-size:12px; font-weight:600; color:#64748B;
        text-transform:uppercase; letter-spacing:0.5px;
        margin-bottom:6px;
    }
 
    /* ── step indicator ── */
    .step-bar {
        display:flex; gap:8px; margin-bottom:20px; align-items:center;
    }
    .step {
        display:flex; align-items:center; gap:6px;
        font-size:13px; color:#94A3B8; font-weight:500;
    }
    .step.active { color:#1565C0; }
    .step-dot {
        width:8px; height:8px; border-radius:50%;
        background:#CBD5E1;
    }
    .step.active .step-dot { background:#1565C0; }
    .step-sep { color:#CBD5E1; font-size:16px; }
 
    </style>
    """, unsafe_allow_html=True)
 
 
def render_header():
    st.markdown("""
    <div class="top-nav">
        <span class="nav-icon">🩺</span>
        <div>
            <h1>Retinopathy of Prematurity Screening</h1>
            <p>AI-assisted clinical decision support — for research use only</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
 
def render_step_bar(active_step):
    steps = [("1", "Upload & Input", 1), ("2", "Analysis", 2), ("3", "Results", 3)]
    html = '<div class="step-bar">'
    for i, (num, label, idx) in enumerate(steps):
        cls = "step active" if idx == active_step else "step"
        html += f'<div class="{cls}"><span class="step-dot"></span>{num}. {label}</div>'
        if i < len(steps) - 1:
            html += '<span class="step-sep">›</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
 
 
def render_confidence_bar(confidence, color):
    pct = confidence * 100
    st.markdown(f"""
    <div style="margin:4px 0 10px 0;">
        <div style="font-size:12px;color:#94A3B8;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
            Model Confidence
        </div>
        <div class="conf-row">
            <div class="conf-bar-track">
                <div class="conf-bar-fill"
                     style="width:{pct:.1f}%;background:{color};"></div>
            </div>
            <span class="conf-pct" style="color:{color};">{pct:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
 
def render_prob_breakdown(probs, class_names):
    bar_colors = ["#16A34A", "#F59E0B", "#DC2626"]
    st.markdown("""
    <div style="font-size:12px;color:#94A3B8;font-weight:600;
                text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">
        Class Probabilities
    </div>
    """, unsafe_allow_html=True)
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        pct = prob * 100
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-label">{name}</span>
            <div class="prob-track">
                <div class="prob-fill"
                     style="width:{pct:.1f}%;background:{bar_colors[i]};"></div>
            </div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
 
 
def render_referral_badge(ref_text, label_name):
    cls_map = {"Severe": "badge-severe", "Mild": "badge-mild", "Normal": "badge-normal"}
    cls = cls_map.get(label_name, "badge-normal")
    st.markdown(f'<span class="badge {cls}">{ref_text}</span>', unsafe_allow_html=True)
 
 
def render_footer():
    st.markdown("""
    <hr>
    <div style="text-align:center;color:#94A3B8;font-size:12px;padding:8px 0 16px 0;">
        This tool is for research and assistive purposes only and does not replace clinical diagnosis.
        All predictions must be reviewed by a qualified clinician.
    </div>
    """, unsafe_allow_html=True)
