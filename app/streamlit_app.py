import streamlit as st
from PIL import Image
import io
import base64
from image_generation import ImageGenerator

def main():
    st.title("Generador de Imágenes por EAN y Prompt")
    
    # Crear una instancia del generador de imágenes
    image_generator = ImageGenerator()
    
    # Crear dos columnas para los inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Input para el EAN
        ean = st.text_input("Ingrese el EAN:", "")
    
    with col2:
        # Input para el prompt
        prompt = st.text_input("Ingrese el prompt:", "")
    
    # Botón para generar la imagen
    if st.button("Generar Imagen"):
        if ean and prompt:  # Verificar que ambos campos estén llenos
            try:
                # Llamar a la función generate_image con ambos parámetros
                result = image_generator.generate_image(ean, prompt)
                
                # Decodificar el base64 a bytes
                image_bytes = base64.b64decode(result['generated_image'])
                
                # Convertir los bytes a imagen
                image = Image.open(io.BytesIO(image_bytes))
                
                # Mostrar la imagen
                st.image(image, caption=f'Imagen generada para EAN: {ean}\nPrompt: {prompt}')
                    
            except Exception as e:
                st.error(f"Error al generar la imagen: {str(e)}")
                # Para debug
                st.write("Tipo de resultado:", type(result['generated_image']))
        else:
            st.warning("Por favor, complete tanto el EAN como el prompt")

if __name__ == "__main__":
    main()