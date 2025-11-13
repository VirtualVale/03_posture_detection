# 🏋️‍♂️ 03_posture_detection

This repository is a small experimental project exploring **posture detection** using **Google’s MediaPipe Pose** model.  
The main goal is to **analyze posture during front squats and deadlifts** — a practical and interesting use case for computer vision–based human pose estimation.

---

![annotated image](https://github.com/VirtualVale/03_posture_detection/blob/main/annotated_image.jpg)

## 🎯 Project Overview

This project demonstrates:
- How to perform **frame-by-frame posture detection** with MediaPipe.  
- How to visualize **landmarks and skeletal joints** in real time.  
- How to experiment with pose tracking to better understand **form and alignment** during exercises.

Most of the code is adapted from **Google’s official MediaPipe Pose example**, and the **landmark model** itself is a **fully trained model provided by Google**.

---

## 📁 Repository Structure

The project contains **two main files**:

- **`pose-detection.ipynb`** – The initial notebook used for testing and developing the **frame detection logic**.  
  It served as the **starting point** to understand how MediaPipe Pose detects landmarks on single frames.  

- **`detector.py`** – A Python script that builds upon the logic from the notebook and implements **video-based posture detection**.  
  It processes full workout videos frame by frame to visualize and analyze your posture dynamically.

---

## 🧠 Insights & Learnings

While testing the model on real workout footage, a few interesting points came up:

- **Occlusions are a big challenge** – when weights, bars, or other objects cover parts of the body, the pose estimation tends to fail or produce inaccurate results.  
- **Full-body visibility is essential** – for reliable tracking, the entire body (head to feet) should remain visible to the camera.  
- **Lighting and camera angle** also play a significant role in detection quality.

---

## ✍️ README Information

This README was **structured and formatted with assistance from GPT-5** to ensure clarity, consistency, and readability.
