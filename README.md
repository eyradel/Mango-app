# Mango Leaf Disease Classification API

A FastAPI-based web service that uses deep learning to classify mango leaf diseases from uploaded images. The system uses a fine-tuned ResNet50 model to identify 8 different conditions including various diseases and healthy leaves.

## Features

- Real-time image classification
- Top 3 prediction probabilities
- Web interface for easy testing
- RESTful API endpoints
- Support for multiple image formats
- Confidence scores for predictions

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Pillow
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mango.git
cd mango
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install fastapi[all] tensorflow pillow numpy python-multipart
```

4. Download the model weights:
- Place the `mango_leaf_classification_model_weights_omdena_resnet50.hdf5` file in the `model` directory

## Project Structure

```
MANGO/
├── fastapi_app/
│   ├── model/
│   ├── templates/
│   ├── __init__.py
│   └── main.py
├── venv/
├── build_estiloai_sh.sh
├── build_run.py
├── build.bat
├── Dockerfile
└── requirements.txt
```

## Running the Application

1. Start the FastAPI server:
```bash
python build_run.py
```

2. Access the web interface at `http://localhost:8000`

## API Endpoints

### GET /
- Serves the web interface
- Returns: HTML page with image upload form

### POST /predict/
- Accepts image file for classification
- Request body: form-data with 'file' field containing the image
- Returns: JSON response with predictions

Example response:
```json
{
    "prediction": "Healthy",
    "confidence": 95.67,
    "top_3_predictions": [
        {
            "disease": "Healthy",
            "confidence": 95.67,
            "is_primary": "true"
        },
        {
            "disease": "Anthracnose",
            "confidence": 3.21,
            "is_primary": "false"
        },
        {
            "disease": "Bacterial Canker",
            "confidence": 1.12,
            "is_primary": "false"
        }
    ],
    "is_healthy": "true"
}
```

## Supported Disease Classes

1. Anthracnose
2. Bacterial Canker
3. Cutting Weevil
4. Die Back
5. Gall Midge
6. Healthy
7. Powdery Mildew
8. Sooty Mould

## Model Architecture

The classification model uses a transfer learning approach with the following architecture:
- Base model: ResNet50 (pre-trained on ImageNet)
- Additional layers:
  - Global Max Pooling
  - Dense layer (128 neurons, ReLU activation)
  - Dropout layer (0.5)
  - Output layer (8 neurons)

## Error Handling

The API includes comprehensive error handling:
- Invalid image format
- Image processing errors
- Model prediction errors
- Server-side exceptions

All errors return appropriate HTTP status codes and detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Acknowledgments

- ResNet50 model: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- FastAPI framework: [FastAPI](https://fastapi.tiangolo.com/)