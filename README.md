#  VisualGuard — Hybrid Deepfake Detection System  
### *Spatiotemporal Deepfake Detection using CNNs, LSTMs & Explainable AI*

VisualGuard is a complete end-to-end deepfake detection system capable of analyzing **images** and **videos** using a hybrid approach:

- **ResNet-18 CNN** for image-level deepfake detection  
- **LSTM spatiotemporal model** for video-level deepfake classification  
- **Grad-CAM explainability** for both images **and** videos  
- **Interactive results with heatmaps, overlays, and contact sheets**

This project also includes a planned **React UI** for a user-friendly web interface.

---

##  Features

###  1. Image Deepfake Detection  
- Trained using **Celeb-DF** dataset  
- Custom PyTorch pipeline (dataset loader, augmentations, training loop)  
- Achieves strong frame-level accuracy  
- Inference script included (`infer_image.py`)

---

###  2. Video Deepfake Detection  
- Uses **FaceForensics++ (FF++)** dataset  
- Extracts frames from videos  
- Extracts deep features using the trained CNN  
- LSTM model trained on frame sequences  
- Final prediction via frame-level aggregation  
- Inference script included (`infer_video.py`)

---

###  3. Explainable AI (XAI)  
Grad-CAM implemented for:

- **Images** → highlights manipulated facial regions  
- **Videos** → generates heatmaps for sampled frames  
- Contact sheet generation  
- Saves overlay results automatically  

Scripts:
- `infer_gradcam.py`
- `infer_video_gradcam.py`

---


Outputs are saved under `/results/`.

---

## 4. Tech Stack

- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy / Pandas**
- **Grad-CAM**
- **FF++ / Celeb-DF datasets**

Planned:
- **React + Tailwind UI**
- **Flask/FastAPI backend**

---

## 5. Planned Future Enhancements

- Face detection + alignment before inference  
- UI dashboard for uploading images/videos  
- PDF report generator for academic submission  
- Deployment on cloud (AWS/GCP)  
- Mobile-friendly interface  
- Model quantization + optimization  

---

## 5 Author

**Ashtami**  
Engineering Student • AI/ML Learner  
Deep learning, computer vision, and web development enthusiast.

---





