# KinePose: Kinesiological Motion Characterization System

Welcome to the KinePose repository! This guide provides step-by-step instructions to set up your local development environment and run the complete kinematic analysis pipeline.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:
*   **Python 3.10** (Recommended for MediaPipe stability on Windows)
*   **Git**

---

## ⚙️ Installation & Environment Setup

Clone the repository to your local machine and set up an isolated Python environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ManuelDeLaHoz/Caracterizacion_de_Movimientos_en_Rehabilitacion_Kinesiologica.git
   cd Caracterizacion_de_Movimientos_en_Rehabilitacion_Kinesiologica
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   *   **Windows (PowerShell):**
       ```powershell
       .\venv\Scripts\Activate
       ```
   *   **macOS / Linux:**
       ```bash
       source venv/bin/activate
       ```

4. **Upgrade pip and install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🧪 Step 1: Calibrate PDI Threshold (Optional)

If you are using a new custom video and want to ensure proper subject isolation:
1. Open Visual Studio Code and make sure your interpreter is set to the `venv` you just created.
2. Open the `calibracion_pdi.ipynb` Jupyter Notebook.
3. Run the cells step-by-step to inspect your video's first frame and adjust the `UMBRAL` variable until the silhouette is cleanly binarized.

---

## ▶️ Step 2: Run the Pipeline

To execute the entire system (Acquisition $\rightarrow$ Segmentation $\rightarrow$ Pose Estimation $\rightarrow$ Tracking $\rightarrow$ Comparison):

1. Ensure your test video files (e.g., `sentadilla1.mp4`, `sentadilla2.mp4`) are located in the **root folder** of the project.
2. Open `main.py` and verify that the video file names in the `if __name__ == "__main__":` block match your files.
3. Run the orchestrator from your activated terminal:
   ```bash
   python main.py
   ```

*   **During execution:** A GUI window titled **"KinePose - Visor"** will pop up, rendering the digital skeleton and real-time knee flexion angles over the video frames. You can press the **`q`** key at any time to abort playback.
*   **After execution:** The terminal will display the comparative clinical report, showing the peak flexions and flagging any deviation exceeding 10%.