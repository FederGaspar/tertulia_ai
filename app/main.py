from fastapi import FastAPI
from image_generation import ImageGenerator
import uvicorn
from pydantic import BaseModel


app = FastAPI()
image_generator = ImageGenerator()


class InputParams(BaseModel):
    ean: str
    prompt: str

@app.post('/generate_image/')
def generate_image(params: InputParams):
    return image_generator.generate_image(params.ean, params.prompt)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)