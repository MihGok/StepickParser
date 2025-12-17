import torch
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
import os

class LlavaService:
    def __init__(self):

        self.model_id = r"C:\Users\user\Desktop\models\llama3-llava"
        
        print(f"Loading LLaVA from: {self.model_id}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto" 
        )

    def analyze(self, image_path: str, prompt: str):
        with Image.open(image_path) as img_file:
            img_file.load()
            image = img_file.convert("RGB") 

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt_str = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt_str, 
            images=image, 
            padding=True, 
            return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(**inputs, max_new_tokens=250)
        
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)

        if "assistant" in generated_text:
             generated_text = generated_text.split("assistant")[-1].strip()
             
        return generated_text
