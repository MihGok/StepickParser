from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ClipService:
    def __init__(self):
        model_id = "openai/clip-vit-base-patch32"
        self.device = "cuda"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def get_image_features(self, image_path: str):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features.tolist() # Возвращаем список float
