# BioMechScanner — Context-Aware Computer Vision Engine

Upload any structural asset. Predict target taxonomy. Procedurally reconstruct the geometric footprint.
* **Live Production Space:** [BioMechScanner](https://paras1512-biomechscanner.hf.space/)

---

## What is this?

**BioMechScanner** is a full-stack, context-aware Deep Learning and Computer Vision terminal that bridges semantic image classification with structural geometric rendering. 

Instead of just labeling an image, the system executes real-time neural inference using a convolutional backbone, extracts raw spatial contours, maps localized RGB pixel distributions, and completely rebuilds the asset's structural shape from absolute scratch on a digital laboratory canvas.

---

## The Mathematical Spatial Re-Projection
Unlike standard AI apps that just call an API and display text, **BioMechScanner** intercepts the image data matrix. It utilizes custom **OpenCV vector parsing** to calculate exact geometric contours, determines local pixel density, and feeds a clean matrix loop back to an isolated frontend coordinate pipeline to draw a raw digital canvas match of the original object live.

---

## System Architecture

### Operational Data Flow
* **Data Ingestion:** The client interface streams multipart image arrays asynchronously to the backend runtime.
* **Neural Inference:** The classification pipeline processes the array through a convolutional model to calculate taxonomy probabilities.
* **Vector Processing:** The vision processor isolates edge matrices, samples spatial coordinate maps, and streams the geometry tokens back to the frontend.
* **Canvas Rendering:** The interface intercepts the data matrix to dynamically paint structural contours on a digital canvas.

### Core Architecture Matrix

| Layer | System Component | Operational Mechanics |
| :--- | :--- | :--- |
| **API Gateway** | FastAPI / Uvicorn | Manages low-latency asynchronous routing and binary image stream data transfer. |
| **Neural Backbone** | PyTorch MobileNetV2 | Executes deep convolutional inference across pre-trained weight tracks to predict object classification. |
| **Vision Processor** | Headless OpenCV Array | Parses incoming binary channels to map localized RGB distribution scales and vector boundaries. |
| **Container Layer** | Docker Engine (Debian) | Isolates core graphics dependencies and environment wheels inside a single cloud container. |

---

## Tech Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend UI** | HTML5, Vanilla JavaScript, CSS3 Core (Theme: **Carbon Frost**) |
| **Backend API** | Python, FastAPI, Uvicorn, Python-Multipart |
| **AI / Computer Vision** | PyTorch (`torch`, `torchvision`), OpenCV (`opencv-python-headless`), NumPy, Pillow |
| **Infrastructure** | Docker Engine, Headless Debian Linux layers, Hugging Face Spaces Cloud |

---


**Developer Profiles**
**Paras Panchal** — BTech Student in Artificial Intelligence & Machine Learning

**GitHub Profile:** [Github](https://github.com/paras151206)

**Hugging Face Space:** [Hugging Space](https://huggingface.co/paras1512)

**Linkedin Profile:** [Linkedin](www.linkedin.com/in/paras-panchal-0b76a7320)
