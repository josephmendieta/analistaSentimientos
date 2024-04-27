from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Cargar el modelo de Hugging Face
clasificador = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.get("/analizar-sentimiento")
def analizar_sentimiento(texto: str):
    # Llamar a la función de análisis de sentimientos del modelo de Hugging Face
    resultado = analizar_sentimiento_hf(texto)
    return {"resultado": resultado[0]['label']}

def analizar_sentimiento_hf(texto):
    resultado = clasificador(texto)
    return resultado

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
