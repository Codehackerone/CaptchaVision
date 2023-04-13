from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageDraw, ImageFont
import random
import torch
from model import LACC

app = FastAPI()

# Load the saved LACC model
model = LACC()
checkpoint = torch.load('Checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define a function for generating random captchas
def generate_captcha(size=4):
    captcha_text = ''.join(random.choices(string.ascii_lowercase, k=size))
    captcha_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(captcha_image)
    font = ImageFont.truetype('arial.ttf', size=32)
    draw.text((32, 80), captcha_text, font=font, fill=(0, 0, 0))
    return captcha_text, captcha_image
  
@app.get("/health")
async def health_check():
  return {"status": "ok"}
  
@app.get("/captcha")
async def generate_new_captcha():
    # Generate a new captcha and return its image as a binary response
    captcha_text, captcha_image = generate_captcha()
    captcha_image.save('captcha.png', format='PNG')
    with open('captcha.png', 'rb') as f:
        captcha_data = f.read()
    return captcha_data

@app.post("/predict")
async def predict_captcha(file: UploadFile = File(...)):
    # Open the uploaded image file and convert it to grayscale
    image = Image.open(file.file).convert('L')
    
    # Resize the image to the expected input size of the LACC model
    image = image.resize((224, 224))
    
    # Convert the image to a PyTorch tensor and normalize it
    image_tensor = torch.tensor(image).unsqueeze(0).float() / 255.0
    
    # Use the LACC model to predict the captcha text
    with torch.no_grad():
        output = model(image_tensor)
        captcha_text = ''
        for i in range(output.shape[1]):
            captcha_text += chr(torch.argmax(output[:,i,:]) + ord('a'))
    
    # Return the predicted captcha text as a JSON response
    return {"captcha_text": captcha_text}
