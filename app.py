import os
import pickle
import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import base64
from io import BytesIO
from PIL import Image, ImageDraw


import torchvision.models as models
import torchvision.transforms as transforms

app = FastAPI(title="Deep Generative Color Reconstruction Engine v8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("Loading Deep Learning Image Classification weights into RAM...")

weights = models.MobileNet_V2_Weights.DEFAULT
classifier_model = models.mobilenet_v2(weights=weights)
classifier_model.eval()


categories_labels = weights.meta["categories"]


img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/decode-thought/")
async def decode_thought(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        
        raw_pil_img = Image.open(BytesIO(contents)).convert('RGB')
        uploaded_image = raw_pil_img.resize((128, 128))
        
        
        input_tensor = img_transform(raw_pil_img).unsqueeze(0)
        with torch.no_grad():
            output_logits = classifier_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output_logits[0], dim=0)
            top_prob, top_cat_idx = torch.topk(probabilities, 1)
            
        predicted_class_name = categories_labels[top_cat_idx.item()]
        confidence_score = float(top_prob.item()) * 100
        
        
        predicted_class_lower = predicted_class_name.lower()
        if any(w in predicted_class_lower for w in ["dog", "cat", "tiger", "lion", "leopard", "bear", "wolf", "animal"]):
            comp_text = "Biological Tissue, Fur Matrix, Organic Cellular Keratin"
            detail_text = f"Active mammalian entity recognized as a [{predicted_class_name}]. High visual cortex activation matching standard animal profile shapes."
        elif any(w in predicted_class_lower for w in ["car", "vehicle", "truck", "wheel", "engine", "plane"]):
            comp_text = "Structural Alloys, Synthetic Polymers, Carbon Composites"
            detail_text = f"Mechanical system identified as a [{predicted_class_name}]. Rigid geometric engineering borders mapped along hardware lines."
        else:
            comp_text = "Surface Texture Matrix / Mixed Substrate Material"
            detail_text = f"Identified object entity classified as a [{predicted_class_name}] with {confidence_score:.1f}% AI model confidence."

        prediction_payload = {
            "entity": predicted_class_name.upper(),
            "composition": comp_text,
            "details": detail_text
        }

        
        open_cv_image = np.array(uploaded_image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=20, threshold2=80)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15] 
        
        
        
        avg_bg_color = open_cv_image.mean(axis=(0,1)).astype(np.uint8).tolist()
        blank_matrix = np.zeros((128, 128, 3), dtype=np.uint8)
        blank_matrix[:] = [20, 20, 25]
        
        generative_canvas = Image.fromarray(blank_matrix)
        draw = ImageDraw.Draw(generative_canvas, "RGBA")
        
        
        for contour in reversed(contours):
            points = [tuple(p[0]) for p in contour]
            if len(points) < 3: continue
            
            
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(open_cv_image, mask=mask)[:3]
            
            
            r_fill, g_fill, b_fill = int(mean_val[0]), int(mean_val[1]), int(mean_val[2])
            
            
            
            draw.polygon(points, fill=(r_fill, g_fill, b_fill, 230))
            
            
            if any(w in predicted_class_lower for w in ["dog", "cat", "tiger", "animal"]):
                draw.line(points, fill=(255, 69, 58, 100), width=1) 
            else:
                draw.line(points, fill=(48, 209, 88, 100), width=1) 

        
        buffered = BytesIO()
        generative_canvas.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        simulated_metrics = np.random.randn(50).tolist()
        
        return JSONResponse({
            "status": "success",
            "prediction": prediction_payload,
            "image_data": f"data:image/png;base64,{img_str}",
            "telemetry": simulated_metrics
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"status": "error", "message": str(e)}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)