from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
import io

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Model configuration
img_height = 224
img_width = 224

# Initialize and compile model
model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.layers[0].trainable = False
model.compile(
    Adam(learning_rate=0.001), 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)

# Load model weights
model.load_weights('model/mango_leaf_classification_model_weights_omdena_resnet50.hdf5')

# Define classes
classes = [
    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew',
    'Sooty Mould'
]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "classes": classes
    })

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.resize((img_height, img_width))
        img = img.convert("RGB")
        
        # Prepare image for prediction
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Convert prediction to probabilities
        softmax_output = tf.nn.softmax(prediction[0]).numpy()
        predicted_class_index = np.argmax(softmax_output)
        predicted_class = classes[predicted_class_index]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(softmax_output)[-3:][::-1]
        
        # Create top 3 predictions list with proper JSON serialization
        top_3_predictions = []
        for idx in top_3_indices:
            confidence = float(softmax_output[idx] * 100)  # Convert to float
            prediction_dict = {
                "disease": classes[idx],
                "confidence": round(confidence, 2),  # Round to 2 decimal places
                "is_primary": str(idx == predicted_class_index)  # Convert boolean to string
            }
            top_3_predictions.append(prediction_dict)

        # Prepare response
        response_data = {
            "prediction": predicted_class,
            "confidence": round(float(softmax_output[predicted_class_index] * 100), 2),
            "top_3_predictions": top_3_predictions,
            "is_healthy": str(predicted_class == "Healthy")  # Convert boolean to string
        }

        return JSONResponse(content=response_data)
    
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500, 
            content={"detail": f"Error processing image: {str(e)}\n{traceback.format_exc()}"}
        )