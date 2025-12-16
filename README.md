# 🪸 Coral Health Assessment

A Streamlit web application that uses deep learning to classify coral reef images as **Bleached** or **Healthy (Unbleached)** with Grad-CAM visualizations to explain model predictions.

## 🚀 Live Demo

**Try the app now**: [https://coral-health.streamlit.app/](https://coral-health.streamlit.app/)

Upload coral images or paste image URLs to get instant health assessments with AI explanations!

## Features

- **Image Classification**: Upload images or paste URLs to classify coral health
- **Grad-CAM Visualization**: See where the model focuses its attention
- **Clean UI**: Sidebar controls, confidence metrics, and responsive layout
- **Error Handling**: Safe image loading and inference with user feedback

## Setup

### Option 1: Use the Live App (Recommended)
Simply visit [https://coral-health.streamlit.app/](https://coral-health.streamlit.app/) - no setup required!

### Option 2: Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashutosh8021/Coral-Health.git
   cd Coral-Health
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**:
   - The trained model `coral_health_resnet50.pth` is included via Git LFS
   - If running locally and Git LFS isn't set up, the model will be downloaded automatically

## Usage

### Online (Recommended)
1. **Visit**: [https://coral-health.streamlit.app/](https://coral-health.streamlit.app/)
2. **Upload an image** or **paste an image URL** in the sidebar
3. **View the prediction** with confidence score  
4. **Toggle Grad-CAM** to see model attention

### Local Development

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open in browser**: Navigate to `http://localhost:8501`

3. **Classify images**:
   - Upload an image using the sidebar file uploader
   - Or paste an image URL in the text input
   - View the prediction with confidence score
   - Toggle Grad-CAM to see model attention

## Model Details

- **Architecture**: ResNet-50 with custom classification head
- **Classes**: Bleached (0), Healthy (1)
- **Input**: 224x224 RGB images
- **Preprocessing**: ImageNet normalization

## File Structure

```
├── app.py              # Main Streamlit application
├── model.py            # Model loading utilities
├── gradcam_utils.py    # Grad-CAM generation
├── coral_health_resnet50.pth  # Trained model weights
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Development

The application includes:
- Helper functions for image loading (upload/URL)
- Error handling for network requests and model inference
- Responsive layout with columns and sidebar
- Progress indicators and status messages

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- See `requirements.txt` for complete list