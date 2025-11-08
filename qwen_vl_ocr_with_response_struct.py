import time
from openai import OpenAI
import json


jason_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NutritionFacts",
  "type": "object",
  "properties": {
    "product_name": { "type": ["string", "null"] },
    "brand": { "type": ["string", "null"] },
    "source_url": { "type": "string", "format": "uri" },
    "servings_per_container": { "type": ["number", "string", "null"] },
    "serving_size": {
      "type": "object",
      "properties": {
        "quantity": { "type": ["number", "null"] },
        "unit": { "type": ["string", "null"] },
        "description": { "type": ["string", "null"] }
      },
      "required": ["quantity", "unit"]
    },
    "calories": { "type": ["number", "null"] },
    "macros": {
      "type": "object",
      "properties": {
        "total_fat_g": { "type": ["number", "null"] },
        "total_fat_dv_percent": { "type": ["number", "null"] },
        "saturated_fat_g": { "type": ["number", "null"] },
        "saturated_fat_dv_percent": { "type": ["number", "null"] },
        "trans_fat_g": { "type": ["number", "null", "string"] },
        "cholesterol_mg": { "type": ["number", "null", "string"] },
        "cholesterol_dv_percent": { "type": ["number", "null"] },
        "sodium_mg": { "type": ["number", "null"] },
        "sodium_dv_percent": { "type": ["number", "null"] },
        "total_carbohydrate_g": { "type": ["number", "null"] },
        "total_carbohydrate_dv_percent": { "type": ["number", "null"] },
        "dietary_fiber_g": { "type": ["number", "null", "string"] },
        "dietary_fiber_dv_percent": { "type": ["number", "null"] },
        "total_sugars_g": { "type": ["number", "null"] },
        "added_sugars_g": { "type": ["number", "null"] },
        "added_sugars_dv_percent": { "type": ["number", "null"] },
        "protein_g": { "type": ["number", "null"] }
      },
      "required": ["total_fat_g","saturated_fat_g","trans_fat_g","sodium_mg","total_carbohydrate_g","protein_g"]
    },
    "micronutrients_dv_percent": {
      "type": "object",
      "properties": {
        "vitamin_d": { "type": ["number", "null"] },
        "calcium": { "type": ["number", "null"] },
        "iron": { "type": ["number", "null"] },
        "potassium": { "type": ["number", "null"] }
      }
    },
    "notes": { "type": ["string", "null"] }
  },
  "required": ["source_url", "serving_size", "macros"]
}


client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

instruction = f"Extract all the nutrition information in JSON format according to the following schema:\n{json.dumps(jason_schema, indent=2)}"
image_link = "https://s7d1.scene7.com/is/image/hersheyprodcloud/0_34000_00246_7_701_24600_073_Item_Back_B?fmt=webp-alpha&hei=908&qlt=75"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_link
                }
            },
            {"type": "text", 
            "text": instruction}
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-4B-Instruct",
    messages=messages,
    max_tokens=1024*8 # 8k tokens
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")

