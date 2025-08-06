from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
from typing import Dict, Optional
import cv2
import requests

app = FastAPI(title="Pokemon Classifier API", version="1.0.0", description="Gen 1 Pokemon Classification API")

#Add CORS middleware for Next.js frontend
#Basically fast api attaches middleware to hand cross origin resousrce sharing since hosted differently
#Corsm intercepts HTTP req and decides if allowed dep on origin methods,headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  #Next.js default port
    #like cookies more to it tho...
    allow_credentials=True,
    #Below is for GET POST PUT DELETE so all of em etc
    allow_methods=["*"],
    #Content-type, auth if using user autho from front end requ
    #Maybe Change this to specifics later
    allow_headers=["*"],
)

#Global variables
model = None
pokemon_classes = []

#FastAPI decorator to run specific funtciotn  auto start when apps starts before api req
@app.on_event("startup")
#async for await later and other functions
async def load_model():
    """Load the model and Pokemon classes when the API starts"""
    global model, pokemon_classes
    try:
        #Loading the model
        model_path = "Dexter_PokemonGen1Ai27.h5"
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        
        #Load Pokemon class names from JSON file
        with open("PokeData.json", "r") as f:
            pokemon_classes = json.load(f)
        
        print(f"Loaded {len(pokemon_classes)} Pokemon classes")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

#Fectching data from poke api like the old project.
def fetch_pokemon_info(pokemon_name: str) -> Optional[Dict]:
    """
    Fetch Pokemon information from PokeAPI
    """
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            types = ", ".join([t['type']['name'].capitalize() for t in data['types']])
            height = data['height'] / 10 
            weight = data['weight'] / 10 
            sprite_url = data['sprites']['front_default']
            
            #Fetch species info for habitat and flavor text
            species_url = data['species']['url']
            species_response = requests.get(species_url)
            
            habitat = "Unknown"
            flavor_text = "No description available"
            
            if species_response.status_code == 200:
                species_data = species_response.json()
                
                #Get habitat
                if 'habitat' in species_data and species_data['habitat']:
                    habitat = species_data['habitat']['name'].capitalize()
                
                #Get flavor text
                for entry in species_data['flavor_text_entries']:
                    if entry['language']['name'] == 'en':
                        flavor_text = entry['flavor_text']
                        break
                
                #Clean up flavor text
                flavor_text = flavor_text.replace("\n", " ")
                flavor_text = ' '.join(flavor_text.split())
            
            return {
                "name": pokemon_name.capitalize(),
                "types": types,
                "height": f"{height} m",
                "weight": f"{weight} kg",
                "sprite_url": sprite_url,
                "habitat": habitat,
                "flavor_text": flavor_text
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching Pokemon info: {e}")
        return None

#Model was trained specifically using CV2 changes and specific image dets, just prerpossing images here.
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image using CV2 method to match your training
    """
    try:
        #Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        #Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #Resize to match your model's expected input (224x224)
        image = cv2.resize(image, (224, 224))
        #Normalize (0-1 range)
        image = image / 255.0
        #Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Pokemon Classifier API is running"}

@app.get("/classes")
async def get_classes():
    return {"classes": pokemon_classes}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make prediction on uploaded Pokemon image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        #Read the uploaded file
        contents = await file.read()
        
        #Preprocess the image
        processed_image = preprocess_image(contents)
        
        #Make prediction
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_idx])
        
        #Get predicted Pokemon name
        predicted_pokemon = pokemon_classes[predicted_idx]
        
        #Fetch Pokemon information from PokeAPI
        pokemon_info = fetch_pokemon_info(predicted_pokemon)
        
        response = {
            "predicted_pokemon": predicted_pokemon,
            "confidence": confidence,
            "pokemon_info": pokemon_info
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)