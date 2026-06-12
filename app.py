import gradio as gr
import numpy as np
import pathlib
import platform
import torch
from PIL import Image
from fastai.vision.all import load_learner

if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('skin_model.pkl', cpu=True)

# Get vocab from model
vocab = learn.dls.vocab

labels = {
    'mel':   '🚨 Melanoma (Skin Cancer)',
    'nv':    '✅ Melanocytic Nevi (Mole)',
    'bcc':   '🚨 Basal Cell Carcinoma',
    'akiec': '⚠️ Actinic Keratosis (Pre-cancer)',
    'bkl':   '✅ Benign Keratosis',
    'df':    '✅ Dermatofibroma',
    'vasc':  '✅ Vascular Lesion'
}

def predict(img):
    # Manually preprocess — same as fast.ai internally
    pil_img = Image.fromarray(img).convert('RGB').resize((224, 224))
    
    # Convert to tensor
    x = torch.tensor(np.array(pil_img)).permute(2,0,1).float() / 255.0
    
    # fast.ai default normalization (same as ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std
    x = x.unsqueeze(0)
    
    # Run through trained model weights directly
    learn.model.eval()
    with torch.no_grad():
        output = learn.model(x)
        probs  = torch.softmax(output[0], dim=0)
    
    return {labels[k]: float(v) for k, v in zip(vocab, probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='numpy'),
    outputs=gr.Label(num_top_classes=3),
    title="🔬 Skin Disease Classifier",
    description="""Upload a photo of a skin lesion.
    The AI will predict the condition with confidence scores.
    ⚠️ DISCLAIMER: This is a learning project only.
    Not a medical diagnostic tool. Always consult a doctor."""
)

demo.launch()