# Off-Road Semantic Scene Segmentation

## Idea Title
AI-Based Off-Road Semantic Scene Segmentation for Autonomous Terrain Understanding

---

# Idea Description

Autonomous vehicles operating in off-road environments must interpret complex and unstructured terrain in real time. Unlike urban roads with lane markings and predictable obstacles, off-road environments contain irregular terrain elements such as rocks, bushes, dry grass, logs, and uneven ground surfaces.

Traditional machine learning models rely on large volumes of real-world annotated data, which is expensive and time-consuming to collect, especially in remote desert environments.

This project presents an AI-based semantic segmentation system trained using **synthetic data generated from Duality AI’s Falcon digital twin simulation platform**. The system analyzes off-road terrain images and predicts pixel-level terrain classes such as landscape, rocks, bushes, and sky.

By enabling machines to understand terrain structures automatically, the system can assist autonomous vehicles and robotic systems in safe navigation and obstacle awareness.

---

# Technical Details

## Technologies Used

### Frontend / UI
- Streamlit (Interactive web interface)

### Backend
- Python

### AI / Machine Learning
- PyTorch
- DINOv2 backbone model
- Semantic segmentation architecture

### Data Processing
- NumPy
- OpenCV
- Pillow

### Visualization
- Matplotlib

### Development Tools
- Anaconda
- VS Code

---

# Architecture Overview

The system follows a deep learning pipeline for semantic segmentation.

### Workflow

Input Image  
↓  
Image Preprocessing  
↓  
DINOv2 Backbone (Feature Extraction)  
↓  
Segmentation Head (Pixel Classification)  
↓  
Segmentation Mask Generation  
↓  
Visualization of Terrain Classes

The **DINOv2 backbone** extracts meaningful features from the image, while the **segmentation head predicts terrain classes for each pixel**.

---

# Dataset Used

The dataset used in this project was generated using **Duality AI’s Falcon digital twin simulation platform**.

Synthetic environments replicate real-world desert terrains and allow generation of labeled datasets with controlled variations.

### Dataset contains

Training Images  
Validation Images  
Test Images

### Terrain Classes

Trees  
Lush Bushes  
Dry Grass  
Dry Bushes  
Ground Clutter  
Flowers  
Logs  
Rocks  
Landscape  
Sky

Synthetic datasets offer several advantages:

• Large volumes of labeled data can be generated quickly  
• Edge cases such as lighting variations can be simulated  
• Annotation errors are minimized  

Note: The full training dataset is not included in this repository due to size limitations.

---

# Evaluation Metrics

The model performance was evaluated using **Intersection over Union (IoU)**.

### Mean IoU Achieved

0.2149

### Validation Accuracy

69.3%

Training performance was monitored using:

• Loss curves  
• Dice score  
• IoU score  
• Pixel accuracy  

Training and evaluation graphs are included in the **train_stats folder**.

---

# Prototype UI

A Streamlit-based prototype interface was developed to demonstrate the model.

### Features

• Upload off-road terrain images  
• Generate segmentation predictions  
• Visualize colored segmentation masks  
• Display model performance metrics  
• Show sample prediction comparisons

Run the prototype locally:
streamlit run app.py

---

# Applications

Semantic segmentation for off-road environments has several real-world applications:

• Autonomous off-road vehicles  
• Agricultural robotics  
• Military and defense navigation systems  
• Disaster rescue robots  
• Environmental monitoring systems  

These applications require reliable perception systems to understand terrain structure and detect obstacles.

---

# Future Improvements

• Real-time terrain segmentation for vehicle camera feeds  
• Integration with robotic navigation systems  
• Hybrid training using synthetic + real-world datasets  
• Improved model accuracy with GPU-based training  
• Deployment as a scalable cloud API

---

# Database Used

This project does **not require a traditional database** because it processes image data directly through the machine learning pipeline.

All model outputs such as predictions, metrics, and training logs are stored as files within the project directories.

---

# Third-Party Integrations

Duality AI Falcon Digital Twin Platform – synthetic dataset generation  

DINOv2 – feature extraction backbone  

Streamlit – interactive UI for prototype demonstration

---

# Project Structure:

- offroad-semantic-segmentation

- app.py
- train_segmentation.py
- test_segmentation.py
- visualize.py
- segmentation_head.pth

- train_stats/
- predictions/

- README.md
- requirements.txt


---

# Team

Team **TRINOVA**

Shubhreet Kaur  
Varsha Rani  
Urvashi Sharma
