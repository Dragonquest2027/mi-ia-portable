import sys
import os
import gradio as gr
from rag_engine import BaseConocimiento
from llm_engine import MotorLogicoIA

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

db = BaseConocimiento(os.path.join(BASE_DIR, "datos", "memoria_ia"))
ia = MotorLogicoIA(os.path.join(BASE_DIR, "modelos", "modelo.gguf"))

def procesar_archivo(archivo):
    if archivo is None:
        return "Selecciona un archivo PDF."
    chunks = db.indexar_pdf(archivo.name)
    return f"Indexado correctamente: {chunks} fragmentos añadidos a la base de datos."

def responder(pregunta, modo):
    contexto = db.buscar_contexto(pregunta, limite=5)
    return ia.generar_respuesta(pregunta, contexto, modo)

with gr.Blocks(title="IA Offline Portable") as demo:
    gr.Markdown("## Asistente IA Offline — Análisis Lógico y Redacción de Informes")
    with gr.Row():
        archivo_input = gr.File(label="Alimentar Base de Datos (PDF)", file_types=[".pdf"])
        estado_db = gr.Textbox(label="Estado de Indexación", interactive=False)
    archivo_input.change(procesar_archivo, inputs=archivo_input, outputs=estado_db)
    
    modo_radio = gr.Radio(["consulta", "informe"], value="consulta", label="Modo de Operación")
    pregunta_box = gr.Textbox(lines=3, label="Consulta o tema del informe")
    btn = gr.Button("Ejecutar Análisis", variant="primary")
    salida_box = gr.Markdown(label="Resultado")
    
    btn.click(responder, inputs=[pregunta_box, modo_radio], outputs=salida_box)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1")
