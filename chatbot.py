import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

modelo_id = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(modelo_id)
modelo = AutoModelForCausalLM.from_pretrained(modelo_id)

# Historial para mantener la conversación
chat_history_ids = None

def responder(mensaje):
    global chat_history_ids

    new_input_ids = tokenizer.encode(mensaje + tokenizer.eos_token, return_tensors='pt')

    # Concatenar el nuevo mensaje al historial (si existe)
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generar respuesta
    chat_history_ids = modelo.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    # Extraer solo la parte nueva (respuesta del bot)
    respuesta = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return respuesta

gr.Interface(fn=responder, inputs="text", outputs="text", title="Chatbot DialoGPT").launch()
