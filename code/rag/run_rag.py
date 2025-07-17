# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os
from uuid import uuid4

import gradio as gr
from src.rag_pipeline import RagPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo.png")

rag_pipeline = RagPipeline(chunk_size=500, chunk_overlap=50, use_tools=False)


def get_endpoint_config():
    config = rag_pipeline.get_endpoint_config()
    return [
        config.get("llm_model_name", ""),
        config.get("llm_api_key", ""),
        config.get("llm_url", ""),
        config.get("embeddings_model_name", ""),
        config.get("embeddings_api_key", ""),
        config.get("embeddings_url", ""),
    ]


def set_endpoint_config(
    llm_model_name,
    llm_api_key,
    llm_url,
    embeddings_model_name,
    embeddings_api_key,
    embeddings_url,
):
    config = {
        "llm_model_name": llm_model_name,
        "llm_api_key": llm_api_key,
        "llm_url": llm_url,
        "embeddings_model_name": embeddings_model_name,
        "embeddings_api_key": embeddings_api_key,
        "embeddings_url": embeddings_url,
    }
    try:
        rag_pipeline.set_endpoint_config(config)
        error_msg = ""
        error_visible = False
    except Exception as e:
        error_msg = f"<span style='color:red; font-weight:bold;'>Error: {e}</span>"
        error_visible = True
    config_values = get_endpoint_config()
    doc_list, file_table = clear_document_list()
    return (
        *config_values,
        doc_list,
        file_table,
        gr.update(value=error_msg, visible=error_visible),
    )


def toggle_api_key_visibility(visible, value):
    type = "text" if visible else "password"
    return gr.Textbox(label="API Key", type=type, value=value)


def clear_history():
    new_id = uuid4()
    print(f"New thread_id: {new_id}")
    return str(new_id), None


def clear_document_list():
    rag_pipeline.clear_vector_store()
    return [], gr.Dataset(sample_labels=[], type="values")


def add_message(history, message, thread_id):
    if message:
        history.append({"role": "user", "content": message})
        try:
            response = rag_pipeline.query(message, thread_id=thread_id)
            bot_message = response["messages"][-1].content
        except Exception as e:
            bot_message = (
                f"<span style='color:red; font-weight:bold;'>Error: {e}</span>"
            )
        history.append({"role": "assistant", "content": bot_message})

    return history, ""


def add_document(new_docs, doc_list):
    rag_pipeline.add_documents(new_docs)
    doc_list.extend([doc.split("/")[-1] for doc in new_docs])
    return doc_list, gr.Dataset(sample_labels=doc_list, type="values")


css = """
footer { visibility: hidden; }
.gradio-container { background: #333333; background: linear-gradient(90deg,rgba(51, 51, 51, 1) 0%, rgba(122, 72, 72, 1) 50%, rgba(24, 47, 133, 1) 100%); }
h1,h2 { color: white; }
#filetable { button { color: black; background-color: white; } }
"""

with gr.Blocks(
    title="FlexBot: Ask me anything!",
    fill_height=True,
    css=css,
    theme=gr.themes.Default(primary_hue=gr.themes.colors.red),
) as demo:
    uuid_state = gr.State(value=str(uuid4()))
    doc_list = gr.State(value=[])
    title = f"""
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1>🤖 FlexBot: Ask me anything! 🤖</h1>
            <h2 style="margin: 0;">FlexBot can answer questions based on the provided documents.</h2>
        </div>
        <img src="/gradio_api/file={logo_path}" alt="Logo" style="height: 80px; margin-left: 20px;">
    </div>
    """
    gr.HTML(title)
    chatbot = gr.Chatbot(elem_id="chatbot", type="messages", scale=1)
    chat_input = gr.Textbox(
        interactive=True,
        placeholder="Ask me anything!",
        show_label=False,
        submit_btn=True,
    )

    with gr.Row():
        file_table = gr.Examples(
            [],
            doc_list,
            label="Knowledge base content (RAG documents)",
            elem_id="filetable",
        )
    with gr.Row():
        with gr.Column(scale=2):
            doc_input = gr.UploadButton(
                label="Upload file documents",
                file_types=[
                    ".txt",
                    ".pdf",
                    ".csv",
                    ".doc",
                    ".docx",
                    ".html",
                    ".htm",
                ],
                file_count="multiple",
            )
        with gr.Column(scale=1):
            clear_doc_button = gr.ClearButton(value="Clear all documents")

    with gr.Accordion("Endpoint Configuration", open=False):
        llm_model_name = gr.Textbox(label="LLM Model Name")
        llm_url = gr.Textbox(label="LLM URL")
        with gr.Row():
            llm_api_key = gr.Textbox(
                placeholder="LLM API Key",
                type="password",
                scale=4,
                show_label=False,
            )
            llm_api_key_visible = gr.Checkbox(
                label="Show LLM API Key",
                value=False,
                scale=1,
            )
        embeddings_model_name = gr.Textbox(label="Embeddings Model Name")
        embeddings_url = gr.Textbox(label="Embeddings URL")
        with gr.Row():
            embeddings_api_key = gr.Textbox(
                label="Embeddings API Key",
                type="password",
                scale=4,
                show_label=False,
            )
            embeddings_api_key_visible = gr.Checkbox(
                label="Show Embeddings API Key",
                value=False,
                scale=1,
            )
        save_btn = gr.Button("Save")
        config_error = gr.Markdown(value="", visible=False)

    # Prefill on load
    demo.load(
        get_endpoint_config,
        inputs=None,
        outputs=[
            llm_model_name,
            llm_api_key,
            llm_url,
            embeddings_model_name,
            embeddings_api_key,
            embeddings_url,
        ],
    )
    demo.load(
        clear_document_list,
        inputs=None,
        outputs=[doc_list, file_table.dataset],
    )
    demo.load(
        clear_history,
        inputs=None,
        outputs=[uuid_state, chatbot],
    )

    save_btn.click(
        set_endpoint_config,
        inputs=[
            llm_model_name,
            llm_api_key,
            llm_url,
            embeddings_model_name,
            embeddings_api_key,
            embeddings_url,
        ],
        outputs=[
            llm_model_name,
            llm_api_key,
            llm_url,
            embeddings_model_name,
            embeddings_api_key,
            embeddings_url,
            doc_list,
            file_table.dataset,
            config_error,
        ],
    )
    llm_api_key_visible.change(
        toggle_api_key_visibility,
        inputs=[llm_api_key_visible, llm_api_key],
        outputs=llm_api_key,
    )
    embeddings_api_key_visible.change(
        toggle_api_key_visibility,
        inputs=[embeddings_api_key_visible, embeddings_api_key],
        outputs=embeddings_api_key,
    )
    chatbot.clear(clear_history, outputs=[uuid_state, chatbot])

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input, uuid_state], [chatbot, chat_input]
    ).then(lambda: gr.Textbox(interactive=True), None, [chat_input])

    doc_input.upload(
        add_document, [doc_input, doc_list], [doc_list, file_table.dataset]
    )
    clear_doc_button.click(clear_document_list, outputs=[doc_list, file_table.dataset])

if __name__ == "__main__":
    demo.launch(allowed_paths=[logo_path], favicon_path=logo_path)
