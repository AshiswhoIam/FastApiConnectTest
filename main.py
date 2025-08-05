from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model("Dexter_PokemonGen1Ai27.h5")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Model is ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224)) 
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    predictions = model.predict(image)
    return {"prediction": predictions.tolist()}
