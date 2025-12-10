# 🤖 LLM + Gradio Chatbot

A simple and customizable chatbot built using **Local LLMs** and **Gradio**, running inside **Google Colab** with model storage in Google Drive.

![Banner](assets/banner.png)

---

## 🚀 Features

- Local LLM chatbot running in Google Colab  
- Gradio-powered UI  
- Google Drive model persistence  
- Works with Mistral, Llama, Gemma, and any HF model  
- Clean, modular notebook  
- Easy to extend for production use  

---

## 📂 Project Structure

├── gradio_llm.ipynb # Main Colab notebook




---

## 🛠️ Setup (Google Colab)

### 1️⃣ Clone the repo (optional)

```bash
!git clone https://github.com/Akshay-rs22/Gradio-LLM.git

2️⃣ Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')

3️⃣ Create storage folders

custom_package_path = '/content/drive/MyDrive/my_colab_packages'
model_storage_path = '/content/drive/MyDrive/colab_models'

import sys, os
sys.path.append(custom_package_path)
os.makedirs(custom_package_path, exist_ok=True)
os.makedirs(model_storage_path, exist_ok=True)

4️⃣ Install requirements

!pip install -r requirements.txt
🤖 Load the LLM Model

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
💬 Chat Function

def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


🖥️ Gradio UI

import gradio as gr

chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Ask something..."),
    outputs=gr.Textbox(label="Output"),
    title="LLM.ai Chatbot"
)

chat_application.launch(server_name="127.0.0.1", server_port=7868)
▶️ How to Use
Open the notebook in Google Colab

Run all cells

A Gradio link will appear

Start chatting with your LLM 🎉

📘 Customization Options
Change the LLM to Llama, Gemma, Falcon, or custom models

Use quantized models for speed

Add chat history support

Deploy on HuggingFace Spaces

Convert to a Flask/FastAPI backend

