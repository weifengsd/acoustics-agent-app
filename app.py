import streamlit as st
import numpy as np
import pandas as pd
import yaml
import sys
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# 1. Setup Robust Path Resolution for pyacoustics
app_dir = Path(__file__).resolve().parent
possible_project_roots = [
    app_dir.parent / "acoustics-agent",
]

acoustics_agent_path = None
for p in possible_project_roots:
    if p.exists():
        acoustics_agent_path = p
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

# 2. Try Importing pyacoustics with Graceful Fallback
import_success = False
try:
    from pyacoustics.simulation import Simulation
    from pyacoustics.schema import SimulationConfig
    from pyacoustics.plot import get_auto_clim
    import_success = True
except ImportError as e:
    import_error_msg = str(e)

# 3. Configure Streamlit Page
st.set_page_config(
    page_title="Acoustics Agent App - AI-Native Simulation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphism / Dark Theme CSS
st.markdown("""
<style>
    /* Main App Custom Styling removed to let native config.toml dark theme handle the global background */
    
    /* Title and Subtitle Header CSS */
    .header-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(to right, #10B981, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Cards and Glass containers */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
    }
    .glass-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #10B981;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.4rem;
    }
    
    /* Badges & Glowing indicators */
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    /* Cleaned up aggressive CSS text overrides to allow native Streamlit Dark Mode (from config.toml) to render perfectly */
    div[data-testid="stExpander"] summary p {
        font-weight: bold !important;
        color: #10B981 !important;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-error {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Customize Streamlit widget borders */
    div[data-baseweb="input"] {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    div[data-baseweb="select"] {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. Standard Environmental Presets Definition
PRESETS = {
    "Munk Sound Channel (Deep Sea)": {
        "project": "Munk Channel Deep Sea",
        "frequency": 50.0,
        "depth": 5000.0,
        "source_depth": 1000.0,
        "ssp_type": "c-linear",
        "ssp_data": [
            {"depth": 0.0, "c": 1548.5},
            {"depth": 200.0, "c": 1530.0},
            {"depth": 500.0, "c": 1508.0},
            {"depth": 1000.0, "c": 1490.0},
            {"depth": 1200.0, "c": 1488.0},  # Channel Axis
            {"depth": 1500.0, "c": 1492.0},
            {"depth": 2000.0, "c": 1500.0},
            {"depth": 3000.0, "c": 1515.0},
            {"depth": 4000.0, "c": 1530.0},
            {"depth": 5000.0, "c": 1550.0},
        ],
        "bottom_type": "acousto-elastic",
        "bottom_c_p": 1600.0,
        "bottom_density": 1.8,
        "bottom_attenuation": 0.5,
        "receivers_max_range": 50000.0,
        "solver_type": "bellhop",
        "angles": [-20.0, 20.0],
        "num_beams": 400,
        "step_size": 10.0,
    },
    "Pekeris Waveguide (Shallow Sea)": {
        "project": "Pekeris Shallow Water",
        "frequency": 200.0,
        "depth": 200.0,
        "source_depth": 300.0, # Adjusted lower relative to depth
        "ssp_type": "c-linear",
        "ssp_data": [
            {"depth": 0.0, "c": 1500.0},
            {"depth": 200.0, "c": 1500.0},
        ],
        "bottom_type": "acousto-elastic",
        "bottom_c_p": 1650.0,
        "bottom_density": 1.9,
        "bottom_attenuation": 0.8,
        "receivers_max_range": 10000.0,
        "solver_type": "bellhop",
        "angles": [-45.0, 45.0],
        "num_beams": 200,
        "step_size": 1.0,
    },
    "Surface Duct (Mixed Layer)": {
        "project": "Surface Duct Waveguide",
        "frequency": 300.0,
        "depth": 1000.0,
        "source_depth": 20.0,
        "ssp_type": "c-linear",
        "ssp_data": [
            {"depth": 0.0, "c": 1525.0},
            {"depth": 50.0, "c": 1526.0},
            {"depth": 100.0, "c": 1523.0},
            {"depth": 150.0, "c": 1495.0},
            {"depth": 250.0, "c": 1485.0},
            {"depth": 1000.0, "c": 1490.0},
        ],
        "bottom_type": "acousto-elastic",
        "bottom_c_p": 1700.0,
        "bottom_density": 2.0,
        "bottom_attenuation": 0.3,
        "receivers_max_range": 25000.0,
        "solver_type": "bellhop",
        "angles": [-15.0, 15.0],
        "num_beams": 300,
        "step_size": 2.0,
    },
}

# Fix preset source depths to avoid out-of-bounds at initialization
for preset_name, preset_vals in PRESETS.items():
    if preset_vals["source_depth"] >= preset_vals["depth"]:
        preset_vals["source_depth"] = preset_vals["depth"] * 0.15

# Initialize Session State
if "preset" not in st.session_state:
    st.session_state.preset = "Munk Sound Channel (Deep Sea)"
if "project_name" not in st.session_state:
    st.session_state.project_name = "Munk Channel Deep Sea"
if "frequency" not in st.session_state:
    st.session_state.frequency = 50.0
if "depth" not in st.session_state:
    st.session_state.depth = 5000.0
if "source_depth" not in st.session_state:
    st.session_state.source_depth = 1000.0
if "ssp_type" not in st.session_state:
    st.session_state.ssp_type = "c-linear"
if "ssp_df" not in st.session_state:
    st.session_state.ssp_df = pd.DataFrame(PRESETS["Munk Sound Channel (Deep Sea)"]["ssp_data"])
if "bottom_c_p" not in st.session_state:
    st.session_state.bottom_c_p = 1600.0
if "bottom_density" not in st.session_state:
    st.session_state.bottom_density = 1.8
if "bottom_attenuation" not in st.session_state:
    st.session_state.bottom_attenuation = 0.5
if "receivers_max_range" not in st.session_state:
    st.session_state.receivers_max_range = 50000.0
if "solver_type" not in st.session_state:
    st.session_state.solver_type = "bellhop"
if "angle_min" not in st.session_state:
    st.session_state.angle_min = -20.0
if "angle_max" not in st.session_state:
    st.session_state.angle_max = 20.0
if "num_beams" not in st.session_state:
    st.session_state.num_beams = 400
if "step_size" not in st.session_state:
    st.session_state.step_size = 10.0
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "simulation_stats" not in st.session_state:
    st.session_state.simulation_stats = None
if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = []
if "shd_computed_tl" not in st.session_state:
    st.session_state.shd_computed_tl = None

# Function to Load Presets into Session State
def load_preset(name):
    if name in PRESETS:
        p = PRESETS[name]
        st.session_state.project_name = p["project"]
        st.session_state.frequency = p["frequency"]
        st.session_state.depth = p["depth"]
        st.session_state.source_depth = p["source_depth"]
        st.session_state.ssp_type = p["ssp_type"]
        st.session_state.ssp_df = pd.DataFrame(p["ssp_data"])
        st.session_state.bottom_c_p = p["bottom_c_p"]
        st.session_state.bottom_density = p["bottom_density"]
        st.session_state.bottom_attenuation = p["bottom_attenuation"]
        st.session_state.receivers_max_range = p["receivers_max_range"]
        st.session_state.solver_type = p["solver_type"]
        st.session_state.angle_min = p["angles"][0]
        st.session_state.angle_max = p["angles"][1]
        st.session_state.num_beams = p["num_beams"]
        st.session_state.step_size = p["step_size"]
        st.session_state.simulation_results = None
        st.session_state.simulation_stats = None
        st.session_state.shd_computed_tl = None

# Header HTML
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 15px;">
        <svg width="45" height="45" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z" fill="#10B981"/>
            <path d="M12 6C11.45 6 11 6.45 11 7V17C11 17.55 11.45 18 12 18C12.55 18 13 17.55 13 17V7C13 6.45 12.55 6 12 6Z" fill="#3B82F6"/>
            <path d="M15.5 8.5C14.95 8.5 14.5 8.95 14.5 9.5V14.5C14.5 15.05 14.95 15.5 15.5 15.5C16.05 15.5 16.5 15.05 16.5 14.5V9.5C16.5 8.95 16.05 8.5 15.5 8.5Z" fill="#3B82F6"/>
            <path d="M8.5 9C7.95 9 7.5 9.45 7.5 10V14C7.5 14.55 7.95 15 8.5 15C9.05 15 9.5 14.55 9.5 14V10C9.5 9.45 9.05 9 8.5 9Z" fill="#3B82F6"/>
            <path d="M19 10C18.45 10 18 10.45 18 11V13C18 13.55 18.45 14 19 14C19.55 14 20 13.55 20 13V11C20 10.45 19.55 10 19 10Z" fill="#10B981"/>
            <path d="M5 10.5C4.45 10.5 4 10.95 4 11.5V12.5C4 13.05 4.45 13.5 5 13.5C5.55 13.5 6 13.05 6 12.5V11.5C6 10.95 5.55 10.5 5 10.5Z" fill="#10B981"/>
        </svg>
        <div>
            <h1 class="main-title">Acoustics Agent Studio</h1>
            <p class="subtitle">AI-Native Underwater Acoustics Simulation & Decision Studio</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 5. Side Control Panel
with st.sidebar:
    st.markdown('<div class="glass-header">🌊 Environment & Presets</div>', unsafe_allow_html=True)
    
    # Preset Selector
    current_preset = st.selectbox(
        "Load Preset Scenario",
        options=list(PRESETS.keys()) + ["Custom Design"],
        index=0 if st.session_state.preset not in PRESETS else list(PRESETS.keys()).index(st.session_state.preset)
    )
    
    if current_preset != "Custom Design" and current_preset != st.session_state.preset:
        st.session_state.preset = current_preset
        load_preset(current_preset)
        st.rerun()
    elif current_preset == "Custom Design":
        st.session_state.preset = "Custom Design"

    st.markdown("---")
    st.markdown('<div class="glass-header">⚙️ Solver & Mode</div>', unsafe_allow_html=True)
    
    # Solver Selector
    st.session_state.solver_type = st.radio(
        "Acoustic Solver Engine",
        options=["bellhop", "normal_modes"],
        format_func=lambda x: "Ray Tracing (Bellhop)" if x == "bellhop" else "Normal Modes (Kraken)",
        index=0 if st.session_state.solver_type == "bellhop" else 1
    )
    
    # Execution Mode Selection
    exec_mode = st.radio(
        "Solver Binary Mode",
        options=["native", "legacy"],
        format_func=lambda x: "🟢 Pure Python (Native)" if x == "native" else "🛡️ Fortran Binary (Legacy)",
        index=0
    )
    
    # Show Environment Library Details
    st.markdown("---")
    st.markdown('<div class="glass-header">ℹ️ System Status</div>', unsafe_allow_html=True)
    if import_success:
        st.markdown(f"**pyacoustics**: <span class='badge-success'>Loaded</span>", unsafe_allow_html=True)
        st.markdown(f"**Path**: `{acoustics_agent_path.name if acoustics_agent_path else 'Global'}`")
    else:
        st.markdown(f"**pyacoustics**: <span class='badge-error'>Failed</span>", unsafe_allow_html=True)
        st.markdown(f"**Error**: `{import_error_msg}`")
        st.error("Please ensure acoustics-agent is installed or path is correct.")

# 6. Main Dashboard Design - Two Column Layout
col_input, col_viz = st.columns([2, 3])

with col_input:

    st.subheader("🛠️ Configuration Studio")
    
    # Project Title
    st.session_state.project_name = st.text_input("Project Name", value=st.session_state.project_name)
    
    # Interactive Configuration Tabs
    tab_ssp, tab_waveguide = st.tabs(["📉 Sound Speed Profile", "📐 Waveguide & Solver"])
    
    with tab_ssp:
        st.write("Modify the table below to configure depth (m) and sound speed (m/s). The profile chart on the right will update in real-time.")
        
        # SSP Interpolation Selection
        st.session_state.ssp_type = st.selectbox(
            "SSP Interpolation Type",
            options=["c-linear", "spline", "n2-linear"],
            index=["c-linear", "spline", "n2-linear"].index(st.session_state.ssp_type)
        )
        
        # Data Editor for SSP
        edited_df = st.data_editor(
            st.session_state.ssp_df,
            num_rows="dynamic",
            column_config={
                "depth": st.column_config.NumberColumn("Depth (meters)", min_value=0.0, format="%.1f m"),
                "c": st.column_config.NumberColumn("Sound Speed (m/s)", min_value=1400.0, max_value=1700.0, format="%.2f m/s"),
            },
            key="ssp_editor"
        )
        # Check and enforce limits relative to max bottom depth
        max_ssp_depth = float(edited_df["depth"].max()) if not edited_df.empty else 0.0
        if max_ssp_depth > st.session_state.depth:
            st.session_state.depth = max_ssp_depth + 10.0 # Force water depth below SSP
            
        st.session_state.ssp_df = edited_df
        
        # Render dynamic Plotly SSP chart
        if not edited_df.empty:
            ssp_sorted = edited_df.sort_values(by="depth")
            fig_ssp_live = go.Figure()
            fig_ssp_live.add_trace(go.Scatter(
                x=ssp_sorted["c"],
                y=ssp_sorted["depth"],
                mode="lines+markers",
                line=dict(color="#10B981", width=2.5),
                marker=dict(size=6, color="#3B82F6"),
                name="SSP"
            ))
            fig_ssp_live.update_layout(
                title=dict(text="Sound Speed Profile (SSP)", font=dict(color="#94a3b8")),
                xaxis=dict(title=dict(text="Sound Speed (m/s)", font=dict(color="#94a3b8")), gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title=dict(text="Depth (meters)", font=dict(color="#94a3b8")), range=[st.session_state.depth, 0], gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_ssp_live, use_container_width=True)
            
    with tab_waveguide:
        # Physical Attributes
        st.markdown("**🌊 Water & Bottom Environment**")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.depth = st.number_input(
                "Bottom Sea Depth (meters)", 
                value=float(st.session_state.depth), 
                min_value=max_ssp_depth, 
                step=10.0
            )
            st.session_state.source_depth = st.number_input(
                "Acoustic Source Depth (meters)", 
                value=float(min(st.session_state.source_depth, st.session_state.depth - 1.0)), 
                min_value=0.0, 
                max_value=float(st.session_state.depth - 0.5), 
                step=5.0
            )
            st.session_state.frequency = st.number_input(
                "Source Frequency (Hz)", 
                value=float(st.session_state.frequency), 
                min_value=1.0, 
                step=10.0
            )
        with c2:
            st.session_state.bottom_c_p = st.number_input("Bottom Sound Speed (m/s)", value=float(st.session_state.bottom_c_p), step=50.0)
            st.session_state.bottom_density = st.number_input("Bottom Density (g/cm³)", value=float(st.session_state.bottom_density), step=0.1)
            st.session_state.bottom_attenuation = st.number_input("Bottom Attenuation (dB/λ)", value=float(st.session_state.bottom_attenuation), step=0.05)
            
        # Receiver / Range Settings
        st.markdown("**📐 Simulation Range Geometry**")
        st.session_state.receivers_max_range = st.slider(
            "Maximum Horizontal Range (meters)", 
            min_value=1000.0, 
            max_value=100000.0, 
            value=float(st.session_state.receivers_max_range), 
            step=1000.0
        )
        
        # Solver Beams & Step settings
        st.markdown("**🎯 Solver Execution Parameters**")
        c3, c4 = st.columns(2)
        with c3:
            st.session_state.angle_min = st.number_input("Min Launch Angle (deg)", value=float(st.session_state.angle_min), step=1.0)
            st.session_state.angle_max = st.number_input("Max Launch Angle (deg)", value=float(st.session_state.angle_max), step=1.0)
        with c4:
            st.session_state.num_beams = st.number_input("Number of Beams/Rays", value=int(st.session_state.num_beams), min_value=2, step=10)
            st.session_state.step_size = st.number_input("Step Size (meters, 0 = auto)", value=float(st.session_state.step_size), min_value=0.0, step=1.0)

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Copilot Panel Section

    st.subheader("🤖 AI Acoustics Copilot")
    st.write("Command the studio with natural language! Describe waveguide parameters (e.g. *'Set source depth to 150m, frequency to 150Hz, bottom speed to 1750m/s'*) and click Send.")
    
    user_prompt = st.text_input("Instruct Co-pilot...", placeholder="Type prompt here... (e.g. 'Load Pekeris waveguide', 'Change launch angles to -30 and 30')")
    
    if st.button("🚀 Send to Copilot", use_container_width=True):
        if user_prompt:
            p_lower = user_prompt.lower()
            actions = []
            
            # 1. Preset loader rules
            if "pekeris" in p_lower:
                load_preset("Pekeris Waveguide (Shallow Sea)")
                st.session_state.preset = "Pekeris Waveguide (Shallow Sea)"
                actions.append("Loaded Pekeris Waveguide Preset")
            elif "munk" in p_lower:
                load_preset("Munk Sound Channel (Deep Sea)")
                st.session_state.preset = "Munk Sound Channel (Deep Sea)"
                actions.append("Loaded Munk Sound Channel Preset")
            elif "duct" in p_lower or "surface" in p_lower:
                load_preset("Surface Duct (Mixed Layer)")
                st.session_state.preset = "Surface Duct (Mixed Layer)"
                actions.append("Loaded Surface Duct Preset")
                
            # 2. Key value parsers
            import re
            
            # Water Depth
            match_depth = re.search(r'(?:water|bottom)?\s*depth\s*(?:to|is|=)?\s*(\d+(?:\.\d+)?)', p_lower)
            if match_depth:
                st.session_state.depth = float(match_depth.group(1))
                actions.append(f"Set Bottom Sea Depth to {st.session_state.depth} m")
                
            # Source Depth
            match_src = re.search(r'source\s*(?:depth)?\s*(?:to|is|=)?\s*(\d+(?:\.\d+)?)', p_lower)
            if match_src:
                st.session_state.source_depth = float(match_src.group(1))
                actions.append(f"Set Source Depth to {st.session_state.source_depth} m")
                
            # Frequency
            match_freq = re.search(r'freq(?:uency)?\s*(?:to|is|=)?\s*(\d+(?:\.\d+)?)', p_lower)
            if match_freq:
                st.session_state.frequency = float(match_freq.group(1))
                actions.append(f"Set Frequency to {st.session_state.frequency} Hz")
                
            # Bottom speed
            match_cs = re.search(r'bottom\s*(?:sound)?\s*(?:speed)?\s*(?:to|is|=)?\s*(\d+(?:\.\d+)?)', p_lower)
            if match_cs:
                st.session_state.bottom_c_p = float(match_cs.group(1))
                actions.append(f"Set Bottom Sound Speed to {st.session_state.bottom_c_p} m/s")
                
            # Launch Angles
            match_angles = re.findall(r'angle[s]?\s*(?:to|between)?\s*(-?\d+)\s*(?:and|to|,)?\s*(-?\d+)', p_lower)
            if match_angles:
                st.session_state.angle_min = float(match_angles[0][0])
                st.session_state.angle_max = float(match_angles[0][1])
                actions.append(f"Set launch angles to [{st.session_state.angle_min}°, {st.session_state.angle_max}°]")
                
            if actions:
                status_msg = "🎯 **Copilot Action Executed Successfully:**\n" + "\n".join([f"- {act}" for act in actions])
                st.session_state.copilot_history.append({"prompt": user_prompt, "response": status_msg, "success": True})
                st.success("Copilot instruction processed! Page updated.")
                st.rerun()
            else:
                sim_response = f"🤖 I processed your instruction: *\"{user_prompt}\"*. However, I couldn't map it directly to a system parameter. Try using commands like:\n- *'Set source depth to 150m'* \n- *'Change frequency to 150Hz'*\n- *'Load Pekeris Waveguide'*."
                st.session_state.copilot_history.append({"prompt": user_prompt, "response": sim_response, "success": False})
                st.info("No matching environment changes found. Copilot gave response.")

    if st.session_state.copilot_history:
        with st.expander("💬 Copilot Conversation Logs"):
            for msg in reversed(st.session_state.copilot_history):
                st.markdown(f"**You**: {msg['prompt']}")
                st.markdown(f"**Copilot**: {msg['response']}")
                st.markdown("---")
                
    st.markdown("</div>", unsafe_allow_html=True)

with col_viz:

    st.subheader("🚀 Simulation Operations Center")
    
    # Assembly config dictionary dynamically
    ssp_formatted = [{"depth": float(row["depth"]), "c": float(row["c"])} for _, row in st.session_state.ssp_df.sort_values(by="depth").iterrows()]
    
    sim_config_dict = {
        "project": st.session_state.project_name,
        "frequency": float(st.session_state.frequency),
        "environment": {
            "ssp": {
                "type": st.session_state.ssp_type,
                "data": ssp_formatted
            },
            "surface": {"type": "vacuum"},
            "bottom": {
                "type": "acousto-elastic", 
                "depth": float(st.session_state.depth),
                "c_p": float(st.session_state.bottom_c_p),
                "density": float(st.session_state.bottom_density),
                "attenuation_p": float(st.session_state.bottom_attenuation)
            }
        },
        "geometry": {
            "source": {"depths": [float(st.session_state.source_depth)]},
            "receivers": {
                "ranges": [0.0, float(st.session_state.receivers_max_range)],
                "depths": [0.0, float(st.session_state.depth)]
            }
        },
        "solver": {
            "type": st.session_state.solver_type,
            "angles": [float(st.session_state.angle_min), float(st.session_state.angle_max)],
            "num_beams": int(st.session_state.num_beams),
            "step_size": float(st.session_state.step_size)
        }
    }
    
    # Run Simulation Trigger Buttons
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        run_ray = st.button("✨ Run Ray Tracing Simulation", type="primary", use_container_width=True)
    with c_btn2:
        run_coherent = st.button("🌈 Compute Coherent TL Field", use_container_width=True)
        
    # Execution Logic for Ray Tracing
    if run_ray:
        if not import_success:
            st.error("Cannot run simulation: `pyacoustics` module is not loaded correctly.")
        else:
            with st.spinner("Executing acoustic solvers... Please wait..."):
                try:
                    # Initialize simulation with dict config
                    sim = Simulation(sim_config_dict, mode=exec_mode)
                    
                    start_time = time.time()
                    results = sim.run(mode=exec_mode)
                    exec_time = time.time() - start_time
                    
                    st.session_state.simulation_results = results
                    
                    # Compute stats
                    if exec_mode == "native":
                        total_rays = len(results)
                        points_per_ray = [len(r[0]) for r in results]
                        avg_points = sum(points_per_ray) / total_rays if total_rays > 0 else 0
                        
                        st.session_state.simulation_stats = {
                            "status": "Success",
                            "execution_time": exec_time,
                            "rays_traced": total_rays,
                            "avg_steps": avg_points,
                            "solver": st.session_state.solver_type,
                            "mode": exec_mode
                        }
                    else:
                        st.session_state.simulation_stats = {
                            "status": "Success",
                            "execution_time": exec_time,
                            "message": "Legacy simulation executed successfully.",
                            "solver": st.session_state.solver_type,
                            "mode": exec_mode
                        }
                    st.success(f"Simulation completed in {exec_time:.4f} seconds!")
                except Exception as ex:
                    st.error(f"Error during simulation: {str(ex)}")
                    st.session_state.simulation_results = None
                    st.session_state.simulation_stats = None

    # Execution Logic for Coherent TL
    if run_coherent:
        if not import_success:
            st.error("Cannot compute TL: `pyacoustics` module is not loaded correctly.")
        else:
            with st.spinner("Computing full phase-coherent acoustic field (Gaussian Beam Summation)..."):
                try:
                    sim = Simulation(sim_config_dict, mode=exec_mode)
                    start_time = time.time()
                    # compute_coherent_tl populates self._coherent_tl_cache
                    TL = sim.compute_coherent_tl(num_r=300, num_z=150)
                    exec_time = time.time() - start_time
                    
                    st.session_state.shd_computed_tl = {
                        "TL": TL,
                        "r_grid": sim._coherent_tl_cache[1],
                        "z_grid": sim._coherent_tl_cache[2],
                        "exec_time": exec_time
                    }
                    st.success(f"Coherent TL calculated in {exec_time:.3f} seconds!")
                except Exception as ex:
                    st.error(f"Error computing coherent TL: {str(ex)}")

    # Show Simulation Stats Dashboard Card
    if st.session_state.simulation_stats:
        stats = st.session_state.simulation_stats
        st.markdown(f"""
        <div style="background-color: rgba(59, 130, 246, 0.05); padding: 0.8rem; border-radius: 8px; border: 1px dashed rgba(59, 130, 246, 0.2); margin-bottom: 1rem;">
            <strong>📈 Execution Statistics:</strong> &nbsp;&nbsp;
            <span class="badge-success">Completed</span> &nbsp;|&nbsp;
            <b>Time:</b> {stats['execution_time']:.4f}s &nbsp;|&nbsp;
            <b>Solver:</b> {stats['solver']} ({stats['mode']}) &nbsp;|&nbsp;
            <b>Rays Traced:</b> {stats.get('rays_traced', 'N/A')} &nbsp;|&nbsp;
            <b>Avg steps/ray:</b> {stats.get('avg_steps', 0.0):.1f}
        </div>
        """, unsafe_allow_html=True)

    # Visualization Output Workspace Tab System
    st.write("### 📊 Visualization Workspace")
    tab_rays, tab_tl_grid, tab_tl_coherent, tab_arrivals, tab_yaml = st.tabs([
        "📈 Ray Paths", 
        "🌋 Transmission Loss (Ray Density)", 
        "🌈 Coherent TL", 
        "🔔 Arrivals Analysis",
        "📄 YAML Configuration"
    ])

    with tab_rays:
        if st.session_state.simulation_results is not None and st.session_state.solver_type == "bellhop":
            # Dynamic Plotly Ray Plot
            st.markdown("#### Interactive Ray Paths (Plotly)")
            fig_rays_plotly = go.Figure()
            
            # Surface Boundary Line
            fig_rays_plotly.add_trace(go.Scatter(
                x=[0, st.session_state.receivers_max_range / 1000.0],
                y=[0, 0],
                mode="lines",
                line=dict(color="#3B82F6", width=2, dash="dash"),
                name="Sea Surface"
            ))
            
            # Bottom Boundary Line
            fig_rays_plotly.add_trace(go.Scatter(
                x=[0, st.session_state.receivers_max_range / 1000.0],
                y=[st.session_state.depth, st.session_state.depth],
                mode="lines",
                line=dict(color="#B45309", width=3),
                fill="toself",
                fillcolor="rgba(180, 83, 9, 0.15)",
                name="Sea Bottom"
            ))
            
            # Plot individual rays
            ray_data = st.session_state.simulation_results
            # Subsample rays if there are too many to keep Plotly performant
            max_plot_rays = 150
            step = max(1, len(ray_data) // max_plot_rays)
            
            for idx, ray in enumerate(ray_data[::step]):
                r_path, z_path, amp_path = ray
                fig_rays_plotly.add_trace(go.Scatter(
                    x=r_path / 1000.0,
                    y=z_path,
                    mode="lines",
                    line=dict(color="#cbd5e1", width=0.5),
                    opacity=0.5,
                    hoverinfo="none",
                    showlegend=False
                ))
            
            # Plot Source location
            fig_rays_plotly.add_trace(go.Scatter(
                x=[0],
                y=[st.session_state.source_depth],
                mode="markers",
                marker=dict(color="#EF4444", size=10, symbol="star"),
                name="Acoustic Source"
            ))
            
            fig_rays_plotly.update_layout(
                xaxis=dict(title=dict(text="Range (km)", font=dict(color="#94a3b8")), gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title=dict(text="Depth (meters)", font=dict(color="#94a3b8")), range=[st.session_state.depth * 1.05, 0], gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#e2e8f0")),
                height=500,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_rays_plotly, use_container_width=True)
            
            # Static High Fidelity Matplotlib Ray Plot
            st.markdown("#### High-Fidelity Static Plot (Matplotlib)")
            try:
                sim = Simulation(sim_config_dict, mode=exec_mode)
                sim.ray_paths = st.session_state.simulation_results
                fig_static = sim.plot_rays()
                st.pyplot(fig_static)
                plt.close(fig_static)
            except Exception as plot_ex:
                st.error(f"Error drawing static plot: {plot_ex}")
        else:
            st.info("💡 Run a **Ray Tracing Simulation** first to generate and view ray paths.")

    with tab_tl_grid:
        if st.session_state.simulation_results is not None and st.session_state.solver_type == "bellhop":
            st.markdown("#### Transmission Loss Field from Ray Density")
            st.write("This map is generated by binning and smoothing ray density across range-depth bins, weighting by boundary reflection losses.")
            try:
                sim = Simulation(sim_config_dict, mode=exec_mode)
                sim.ray_paths = st.session_state.simulation_results
                fig_tl = sim.plot_tl()
                st.pyplot(fig_tl)
                plt.close(fig_tl)
            except Exception as plot_ex:
                st.error(f"Error drawing TL plot: {plot_ex}")
        else:
            st.info("💡 Run a **Ray Tracing Simulation** first to compute and view the Transmission Loss field.")

    with tab_tl_coherent:
        if st.session_state.shd_computed_tl is not None:
            st.markdown("#### Coherent Transmission Loss (Gaussian Beam Summation)")
            st.write("This represents the phase-coherent acoustic field computed using physically accurate Gaussian Beam summation. It shows fine acoustic striations and interference patterns.")
            
            c_data = st.session_state.shd_computed_tl
            TL_grid = c_data["TL"]
            r_grid = c_data["r_grid"]
            z_grid = c_data["z_grid"]
            
            # Matplotlib Imshow
            fig, ax = plt.subplots(figsize=(12, 6))
            vmin, vmax = get_auto_clim(TL_grid)
            im = ax.imshow(
                TL_grid.T, 
                extent=[r_grid[0] / 1000.0, r_grid[-1] / 1000.0, st.session_state.depth, 0],
                aspect='auto', 
                cmap='jet_r',
                vmin=vmin, 
                vmax=vmax
            )
            plt.colorbar(im, label='Transmission Loss (dB)')
            ax.set_xlabel('Range (km)')
            ax.set_ylabel('Depth (m)')
            ax.set_title(f"Coherent TL Field ({st.session_state.project_name})")
            
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("💡 Click **'Compute Coherent TL Field'** above to generate the full physical Gaussian Beam wave interference field.")

    with tab_arrivals:
        st.markdown("#### Multipath Arrivals & Impulse Response")
        st.write("Analyze individual acoustic ray paths, travel times, and amplitudes reaching a specific receiver coordinate.")
        
        c_rx1, c_rx2 = st.columns(2)
        with c_rx1:
            rx_range = st.slider(
                "Receiver Range Location (meters)", 
                min_value=500.0, 
                max_value=float(st.session_state.receivers_max_range), 
                value=float(st.session_state.receivers_max_range * 0.75),
                step=500.0
            )
        with c_rx2:
            rx_depth = st.slider(
                "Receiver Depth Location (meters)", 
                min_value=1.0, 
                max_value=float(st.session_state.depth - 2.0), 
                value=float(st.session_state.depth * 0.5),
                step=10.0
            )
            
        run_arr = st.button("🔔 Analyze Arrivals at Receiver Location", type="primary", use_container_width=True)
        
        if run_arr:
            if not import_success:
                st.error("Cannot compute arrivals: `pyacoustics` module is not loaded correctly.")
            else:
                with st.spinner("Calculating multipath arrivals..."):
                    try:
                        sim = Simulation(sim_config_dict, mode=exec_mode)
                        # We must run the ray tracing first
                        sim.run()
                        arrivals = sim.run_arrivals(range_m=rx_range, depth_m=rx_depth)
                        
                        if not arrivals:
                            st.warning("No ray paths intersected the receiver volume. Try widening the launch angles or increasing the number of beams.")
                        else:
                            st.success(f"Successfully calculated {len(arrivals)} multipath arrivals!")
                            
                            # Preprocess complex values to avoid serialization errors
                            processed_arrivals = []
                            for arr in arrivals:
                                amp = complex(arr['amplitude'])
                                processed_arrivals.append({
                                    'tau': float(arr['tau']),
                                    'amplitude_magnitude': abs(amp),
                                    'amplitude_phase_deg': np.degrees(np.angle(amp)),
                                    'amplitude_str': f"{amp.real:.3e} + {amp.imag:.3e}j",
                                    'launch_angle': float(arr['launch_angle']),
                                    'arrival_angle': float(arr['arrival_angle'])
                                })
                            
                            # Convert arrivals to DataFrame
                            arr_df = pd.DataFrame(processed_arrivals)
                            
                            # Interactive Scatter Plot of arrivals
                            st.markdown("##### Arrival Impulse Response (Plotly)")
                            fig_arr_plotly = go.Figure()
                            
                            # Stem lines for arrivals
                            for idx, row in arr_df.iterrows():
                                fig_arr_plotly.add_trace(go.Scatter(
                                    x=[row['tau'], row['tau']],
                                    y=[0, row['amplitude_magnitude']],
                                    mode="lines",
                                    line=dict(color="#10B981", width=2),
                                    hoverinfo='none',
                                    showlegend=False
                                ))
                            
                            # Nodes for arrivals
                            fig_arr_plotly.add_trace(go.Scatter(
                                x=arr_df['tau'],
                                y=arr_df['amplitude_magnitude'],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color="#3B82F6",
                                ),
                                text=[f"Launch Angle: {r['launch_angle']:.2f}°<br>Arrival Angle: {r['arrival_angle']:.2f}°<br>Phase: {r['amplitude_phase_deg']:.1f}°" for _, r in arr_df.iterrows()],
                                hoverinfo="x+y+text",
                                name="Arrival Paths"
                            ))
                            
                            fig_arr_plotly.update_layout(
                                xaxis=dict(title=dict(text="Travel Time / Delay (seconds)", font=dict(color="#94a3b8")), gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                                yaxis=dict(title=dict(text="Arrival Amplitude", font=dict(color="#94a3b8")), gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=400,
                                margin=dict(l=10, r=10, t=10, b=10)
                            )
                            st.plotly_chart(fig_arr_plotly, use_container_width=True)
                            
                            # Table of top arrivals
                            st.markdown("##### Tabular Path Analysis")
                            arr_df_sorted = arr_df.sort_values(by="tau")
                            st.dataframe(
                                arr_df_sorted[['tau', 'amplitude_magnitude', 'amplitude_phase_deg', 'amplitude_str', 'launch_angle', 'arrival_angle']].style.format({
                                    'tau': '{:.5f} s',
                                    'amplitude_magnitude': '{:.3e}',
                                    'amplitude_phase_deg': '{:.1f}°',
                                    'launch_angle': '{:.2f}°',
                                    'arrival_angle': '{:.2f}°',
                                }),
                                use_container_width=True
                            )
                    except Exception as arr_ex:
                        st.error(f"Error computing arrivals: {arr_ex}")
        else:
            st.info("💡 Set receiver range and depth above, then click **'Analyze Arrivals'** to compute.")

    with tab_yaml:
        st.markdown("#### Exported YAML Configuration Schema")
        st.write("This YAML schema matches the standard YAML configuration used by `acoustics-agent`.")
        
        yaml_str = yaml.dump(sim_config_dict, sort_keys=False)
        st.code(yaml_str, language="yaml")
        
        st.download_button(
            label="💾 Download Configuration File (YAML)",
            data=yaml_str,
            file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}.yaml",
            mime="text/yaml",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
