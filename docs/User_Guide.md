# Acoustics Agent Studio User Guide

## 1. Introduction
Acoustics Agent Studio (`acoustics-agent-app`) is an interactive underwater acoustics simulation and visualization web platform based on the `acoustics-agent` library. It provides a modern, "glassmorphism" dark-themed GUI that allows oceanographers and engineers to intuitively design physical waveguides, configure environments, and visualize ray trajectories and coherent transmission loss fields in real-time.

## 2. Interface Overview
![Acoustics Studio Main Interface - Munk Channel Ray Paths](./munk_ray_paths.png)

The interface is divided into two main areas:
1. **Sidebar**: Load classic presets, select the underlying calculation engine (e.g., pure Python ray tracing), and monitor system status.
2. **Main Workspace**: Fine-tune environmental parameters, run simulations, send natural language commands to the AI Copilot, and analyze calculation results.

---

## 3. Core Features & Operations

### 3.1 Environment & Sound Speed Profile (SSP)
- **Load Presets**: Under the `Load Preset Scenario` dropdown in the sidebar, you can instantly configure standard benchmark scenarios (such as the Munk Deep-Sea Channel, Pekeris Shallow-Water Waveguide, or Surface Duct). The workspace parameters will reset automatically to match.
- **Dynamic SSP Designer**: Double-click cells in the SSP data table to modify the depth and sound speed (c). The corresponding profile chart will update dynamically.

### 3.2 Waveguide & Geometry Configuration
In the `🌊 Water & Bottom Environment` section, you can precisely adjust:
- **Bottom Sea Depth**: The maximum depth of the water column.
- **Source Parameters**: Set the acoustic source depth and frequency (Hz).
- **Bottom Acoustics Properties**: Expand the advanced settings to customize compressional speed, density, and attenuation for an elastic bottom model.
- **Transmitter/Receiver Array**: Define launch angle bounds and ray density under the geometry settings.

### 3.3 Running the Simulation
Once configured, click the green **✨ Run Ray Tracing Simulation** button. The system will instantly invoke the backend Python solver and display a success notification.

### 3.4 Result Analytics & Visualization
After simulation, utilize the tabs below to dive deep into the results:
- **📈 Ray Paths**:
  - Interactive Plotly charts (pan, zoom, hover) and high-fidelity Matplotlib plots depicting the refraction and reflection trajectories of sound rays.
- **🌈 Coherent TL (Transmission Loss)**:
  - Click **🌈 Compute Coherent TL Field** to initiate a Gaussian Beam Summation algorithm, rendering a coherent sound field heatmap with detailed interference striations.
- **🔔 Arrivals Analysis**:
  - Enter a specific range and depth for the receiver and click **Analyze Arrivals at Receiver Location**.
  - The system will calculate and visualize the multipath impulse response, displaying travel time delays, amplitudes, launch angles, and arrival angles in both stem plots and data tables.

### 3.5 AI Acoustics Copilot
The **🤖 AI Acoustics Copilot** at the bottom of the page provides a revolutionary control method. Type natural language commands such as:
> *"Load Pekeris waveguide"*
> *"Set source depth to 150m and frequency to 300Hz"*
> *"Change bottom sound speed to 1750 m/s"*

Click send, and the Copilot will automatically interpret your intent and update the dashboard settings instantly!

---

## 4. Configuration Export
In the **📝 YAML Schema** tab, the system continuously translates your environment design into the standard `acoustics-agent` YAML configuration format. You can copy this YAML snippet for use in offline command-line simulations or batch scripts.
