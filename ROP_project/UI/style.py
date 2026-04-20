# style.py
import streamlit as st

def header(title="🩺 Retinopathy of Prematurity Detection System", subtitle="AI-assisted clinical decision support"):
    st.markdown(f"""
        <div style="
            background-color:#1565C0;
            padding:15px 20px;
            border-radius:10px;
            margin-bottom:20px;
            color:white;
        ">
            <h2 style="margin:0;">{title}</h2>
            <p style="margin:0; font-size:14px;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def subheader(text):
    st.markdown(f"""
        <div style="
            font-size:18px;
            font-weight:600;
            color:#1565C0;
            margin-bottom:10px;
        ">
            {text}
        </div>
    """, unsafe_allow_html=True)


def apply_style():
    st.markdown("""
    <style>
    /* Overall App Background */
    .stApp {
        background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(200,220,255,0.1));
        backdrop-filter: blur(10px);
        min-height: 100vh;
    }

    /* Glassy Cards */
    .card {
        background: rgba(255, 255, 255, 0.25);  /* semi-transparent */
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Headers & Subheaders */
    h1, h2, h3 {
        color: #1565C0;
        text-shadow: 1px 1px 4px rgba(255,255,255,0.6);
    }

    .stButton>button {
        border-radius: 10px;
        background: #1565C0;   /* solid blue */
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        border: 2px solid #0D47A1;  /* visible border */
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: #0D47A1;
        border: 2px solid #1565C0;
        color: white;
    }

    /* Progress bars with glassy tint */
    .stProgress > div > div > div {
        background-color: rgba(21, 101, 192, 0.7);
    }

    </style>
    """, unsafe_allow_html=True)        
def footer():
    st.markdown("""
        <hr style="margin-top:40px;">
        <div style='text-align: center; color: #Black; font-size: 14px; font-weight: 500;'>
            This tool is for research/assistive purposes only and does not replace clinical diagnosis.
        </div>
    """, unsafe_allow_html=True)