![Build Status](https://img.shields.io/badge/Security-AI%20Powered-red?style=for-the-badge&logo=shield)
![Static Badge](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Static Badge](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)
![Static Badge](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia)
  
# Multi camera intruder detection with alerts #

**A high-performance, multi-camera security monitoring system with real-time AI-powered person detection, boundary crossing alerts, and optimized streaming architecture**

<div align="center">
<img src="images/flowchart.png" alt="Flow diagram" width="300"/>
</div>

🎯 **Key Features**
<table><tr><td width="50%">
🔥 Core Capabilities

- ✅ Multi-Camera Support - Monitor 16+ cameras simultaneously
- ✅ AI Person Detection - YOLOv8 with GPU acceleration
- ✅ Interactive Boundaries - Mouse-drawable security zones
- ✅ Real-Time Alerts - Instant popup notifications
- ✅ RTSP & USB Support - IP cameras and local devices
- ✅ Smart Grid Layout - Automatic camera arrangement
</td><td width="50%">
  
⚡ **Advanced Features**

- 🚀 Async Processing - Non-blocking AI inference
- 🎯 Auto Window Management - Smart alert focusing
- 🔊 Audio Alarms - Customizable sound alerts
- 📈 Performance Monitoring - Real-time statistics
- 🔄 Auto Reconnection - Handles dropped connections
- 💾 Memory Efficient - Smart caching system
</td></tr></table>

System Architecture Comparison
<table>
<tr>
<th>📊 Metric</th>
<th>🔴 Before (Synchronous)</th>
<th>🟢 After (Asynchronous)</th>
<th>📈 Improvement</th>
</tr>
<tr>
<td><strong>YOLO Calls/Second</strong></td>
<td>360 (12 cams × 30 FPS)</td>
<td>60 (12 cams × 5 FPS)</td>
<td><code>6x Reduction</code></td>
</tr>
<tr>
<td><strong>Stream Lag</strong></td>
<td>500-2000ms delay</td>
<td><50ms delay</td>
<td><code>20x Faster</code></td>
</tr>
<tr>
<td><strong>CPU Usage</strong></td>
<td>85-95% (bottleneck)</td>
<td>30-45% (optimal)</td>
<td><code>50% Reduction</code></td>
</tr>
<tr>
<td><strong>Memory Usage</strong></td>
<td>4-6GB (leaks)</td>
<td>2-3GB (stable)</td>
<td><code>50% Reduction</code></td>
</tr>
<tr>
<td><strong>Alert Response</strong></td>
<td>1-3 seconds</td>
<td><500ms</td>
<td><code>6x Faster</code></td>
</tr>
</table>

## 🏗️ System Architecture ##
**Thread Architecture Diagram**
<div align="center">
<img src="images/Thread architecture.png" alt="Flow diagram" width="500"/>
</div>

**Data Flow Architecture**
<div align="center">
<img src="images/Data flow.png" alt="Flow diagram" width="500"/>
</div>

### 🚀 Quick Start ###
**1️⃣ Installation**
```
# Clone repository
git clone https://github.com/yourusername/multi-camera-security-system.git
cd multi-camera-security-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**2️⃣ Configuration**
```
# Edit camera_sources in main()
camera_sources = [
    {
        "name": "Front Door",
        "url": "rtsp://admin:password@192.168.1.100:554/stream1"
    },
    {
        "name": "Back Yard", 
        "url": "rtsp://admin:password@192.168.1.101:554/stream1"
    },
    {
        "name": "USB Camera",
        "url": 0
    }
]
```

**3️⃣ Run System**
```
python multi-camera-stream-v4.py
```

**🐋 Docker Installation**
```
# Use official Python image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD ["python", "multi-camera-stream-v4.py"]
```

```
# Build and run with Docker
docker build -t security-system .
docker run --gpus all -it --rm security-system
```

### ⚡ Performance Tuning ###
<details>
<summary>🚀 Optimization Settings</summary>
**Detection Parameters**
  
```
# In AsyncYOLOProcessor.__init__()
self.detection_intervals = {
    'detection_interval': 0.2,     # Detect every 200ms (5 FPS)
    'cache_timeout': 0.5,          # Cache results for 500ms
    'max_workers': 2,              # YOLO thread pool size
}

# Frame processing settings
YOLO_RESOLUTION = 640              # Downscale to 640px for detection
DISPLAY_FPS = 40                   # Target display framerate
DETECTION_CONFIDENCE = 0.5         # YOLO confidence threshold
```

**Memory Management**
```
# OpenCV buffer settings
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # Minimal buffering
cap.set(cv2.CAP_PROP_FPS, 30)          # Target FPS

# Queue sizes
detection_queue_size = 50               # Max pending detections
frame_cache_size = 100                  # Frame history size
```

**Network Optimization**
```
# RTSP settings
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|'           # Use TCP for stability
    'reorder_queue_size;0|'         # Disable reordering  
    'max_delay;500000'              # 500ms max delay
)
```
</details>


### 📊 Usage Guide ###

**🎮 Control Interface**

**⌨️ Keyboard Shortcuts**
  
| Key     | Action           | Description                        |
|---------|------------------|------------------------------------|
| 0       | 🏠 Main Grid     | Focus on main grid window          |
| 1-9     | 🎯 Camera Select | Focus on camera alert windows      |
| `       | 📋 Camera List   | Show all cameras and status        |
| S       | 🛑 Stop All      | Reset all alarms system-wide       |
| R       | 🔄 Reset Camera  | Reset selected camera alarm        |
| C       | 🧹 Clear Boundaries | Remove boundary lines           |
| P       | 👁️ Toggle Detection | Enable/disable person detection |
| Q       | 🚪 Quit          | Exit application                   |

**🖱️ Mouse Control**

| Action         | Function                       |
|----------------|--------------------------------|
| Click Camera   | Select for boundary drawing    |
| Click & Drag   | Draw security boundary lines   |
| Multiple Drags | Create complex boundary shapes |
