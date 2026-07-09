# Hand Tracking 3D VFX System

A professional, real-time, interactive hand-tracking system that overlays premium 3D models on a user's hand, integrated with a GPU particle simulation engine, high-quality post-processing glow (Bloom), and dynamic lighting.

This repository includes 5 distinct 3D VFX overlays:
1.  **Orchid Plant**: A blooming pink/violet plant that sprouts neon green magic sparks when appearing.
2.  **Fantasy Wing**: A crimson flame wing that leaves a trailing ember trail.
3.  **Butterfly Swarm**: A neon blue butterfly that flutters its wings and is orbited by secondary tiny butterflies.
4.  **Phoenix Bird**: A glowing golden-orange-crimson legendary bird with flapping wings and fiery trails.
5.  **Enchanted Dark Dragon**: An enchanted deep violet serpentine dragon with webbed bat wings and purple magic sparks.

---

## 🛠️ Technology Stack

*   **Logic & Runtime:** Python 3.10+
*   **Video Processing:** OpenCV (`opencv-python`)
*   **AI Hand Tracking:** MediaPipe Hands (21 Landmarks with Depth Estimation)
*   **Rendering:** ModernGL (OpenGL 3.3+ Core Profile)
*   **Mathematics:** PyGLM (OpenGL Mathematics) and NumPy
*   **User Interface:** Dear PyGui (GPU-accelerated desktop UI)
*   **Configuration:** YAML

---

## 🚀 Installation & Setup

### 1. Prerequisites
Make sure Python 3.10+ is installed on your Windows machine. Verify by running:
```powershell
python --version
```

### 2. Install Dependencies
Install all package dependencies via `pip`:
```powershell
pip install -r requirements.txt
```

### 3. Setup Project & Generate 3D Models
Run the setup command to check system dependencies, verify project directories, and **procedurally generate all 3D model OBJ files** (including the new Phoenix and Dragon models):
```powershell
python run.py --setup
```

### 4. Run the System
Start the application and open the webcam feed:
```powershell
python run.py --run
```

---

## 🎮 How to Select and Interact with 3D Models

The application overlays the selected 3D model directly on your hand when your palm is visible to the webcam.

### 1. Switching Between the 5 Models
There are two ways to cycle through the models:
*   **Via UI Control Panel:** Open the **3D ASSETS SELECTION** section on the Dear PyGui controller window, click the **Active Asset** combo box, and select your model:
    *   `Orchid Plant`
    *   `Fantasy Wing`
    *   `Butterfly Swarm`
    *   `Phoenix Bird`
    *   `Dark Dragon`
*   **Via Swipe Gestures (Hand Tracking):** Swipe your hand quickly to the **Left** or **Right** in front of the camera to trigger an edge-swipe model transition.

### 2. Triggering Gesture Actions
The particle engine and model colors adapt dynamically to your hand gestures:
*   **Open Palm:** Spawns gentle ambient sparkles around your hand.
*   **Pinch Gesture (Thumb + Index Tip):** Spawns dense, high-energy red sparks at the pinch point and intensifies the model's emissive glow.
*   **Pointing (Index Up):** Spawns fire sparks from your index fingertip.
*   **Peace Sign (Index + Middle Up):** Emits dual magic particle trails from both fingertips.
*   **Closed Fist:** Smoothly shrinks and hides the 3D model overlay.

### 3. Adjusting VFX in Real-Time
Use the GUI Control Panel on the side to customize the experience:
*   Adjust **Bloom Intensity**, **Exposure (HDR)**, and **Gamma Correction** sliders to control the glow strength.
*   Toggle **Animate Wings** to enable or disable flapping on the Butterfly, Phoenix, and Dragon models.
*   Toggle **One Euro Smoothing** to stabilize hand tracking and reduce model jitter.

---

## 🧪 Testing

To run the automated verification suite (covering assets, shaders, UI state, and particle engines):
```powershell
python -m unittest discover -s tests
```
