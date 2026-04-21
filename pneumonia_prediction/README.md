# 🫁 Pneumonia Prediction

A deep learning web application for detecting pneumonia in chest X-ray images using ResNet18. Upload an X-ray image and get instant predictions with confidence scores.

## Features

- **Fast Detection**: ResNet18-based model for quick inference
- **High Accuracy**: Binary classification (NORMAL / PNEUMONIA)
- **Web Interface**: Clean, modern UI for easy image uploads
- **Confidence Scores**: Detailed probability predictions for both classes
- **Cross-Origin Support**: CORS enabled for flexible deployment
- **Health Check**: Built-in endpoint to verify model status

## Project Structure

```
pneumonia_prediction/
├── server.py                 # Flask backend server
├── pneumonia_best.pth        # Pre-trained model weights
├── static/
│   └── index.html            # Web UI frontend
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch
- Flask & Flask-CORS
- Pillow (PIL)
- torchvision

## Installation

1. **Clone/Download the project**:
   ```bash
   cd pneumonia_prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision flask flask-cors pillow
   ```

3. **Ensure the model file exists**:
   - Place `pneumonia_best.pth` in the project root directory
   - The file is required to run the application

## Usage

1. **Start the server**:
   ```bash
   python server.py
   ```

2. **Access the web interface**:
   - Open your browser and go to `http://localhost:5000`

3. **Upload an X-ray image**:
   - Click or drag-and-drop a chest X-ray image
   - The model will process the image and return results

## API Endpoints

### `GET /`
- Serves the web UI

### `POST /predict`
- **Description**: Predicts pneumonia in an X-ray image
- **Input**: Form data with `file` field containing the image
- **Response**: JSON with prediction results
- **Example Response**:
  ```json
  {
    "label": "PNEUMONIA",
    "confidence": 95.42,
    "probabilities": {
      "NORMAL": 4.58,
      "PNEUMONIA": 95.42
    }
  }
  ```

### `GET /health`
- **Description**: Health check endpoint
- **Response**: JSON with server status and model state
- **Example Response**:
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

## Model Details

- **Architecture**: ResNet18
- **Input Size**: 224×224 pixels
- **Output Classes**: 2 (NORMAL, PNEUMONIA)
- **Preprocessing**: 
  - Resize to 224×224
  - Normalize using ImageNet statistics
  - Dropout (0.3) for regularization

## Troubleshooting

### "Model not loaded" error
- Ensure `pneumonia_best.pth` exists in the project root
- Check file permissions
- Restart the server

### Port already in use
- Change the port in `server.py` (line 83: `app.run(..., port=5001, ...)`)
- Or kill the process using port 5000

### Image upload fails
- Ensure the image is a valid format (JPEG, PNG, etc.)
- Check file size and image dimensions
- Verify the file is an actual image

## System Requirements

- **CPU**: Any modern processor (model runs on CPU by default)
- **RAM**: Minimum 2GB, recommended 4GB+
- **Disk**: ~200MB for model + dependencies

## Deployment

For production deployment, consider using:
- **Gunicorn**: `gunicorn -w 4 -b 0.0.0.0:5000 server.py`
- **Docker**: Containerize for consistent environments
- **NGINX**: Reverse proxy for load balancing

## Performance

- Inference time: ~100-200ms per image (CPU)
- Model accuracy: Varies based on training data and validation set
- Batch processing: Currently processes one image at a time

## Future Enhancements

- [ ] Batch prediction support
- [ ] GPU acceleration option
- [ ] Model ensemble for improved accuracy
- [ ] Result history/logging
- [ ] Admin dashboard with statistics
- [ ] Dark mode UI

## Notes

- This tool is for educational purposes and should not be used as a medical diagnostic device without proper validation and regulatory approval
- Always consult medical professionals for actual diagnostics

## License

[Specify your license here]

## Contact

For issues or questions, please reach out or open an issue in the repository.
