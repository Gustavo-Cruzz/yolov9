import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from models.common import DetectMultiBackend

device = "cpu"

weights_path = hf_hub_download(
    repo_id="Gustavo-Cruzz/yolov9-fruit-detector",
    filename="best.pt"
)

model = DetectMultiBackend(weights=weights_path, device=device)
model.eval()

def detect(image):
    # inferência aqui
    return image

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLOv9 Fruit Detector"
)

demo.launch()

