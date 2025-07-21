# Rag application

The RAG (Retrieval-Augmented Generation) application is designed to provide an interactive interface for users to ask questions based on provided documents.

## Start the FlexAI endpoints

Create the FlexAI secret that contains your HF token in order to access the inference models:

```bash
# Enter your HF token value when prompted
flexai secret create hf-token
```

Start the FlexAI endpoint of the LLM:

```bash
LLM_INFERENCE_NAME=qwen-llm
export LLM_MODEL_NAME=Qwen/Qwen2.5-32B-Instruct
flexai inference serve $LLM_INFERENCE_NAME --hf-token-secret hf-token -- --model=$LLM_MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384
# store the returned information
export LLM_API_KEY=<store the given API key>
export LLM_URL=$(flexai inference inspect $LLM_INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

Start the FlexAI endpoint of the embedder:

```bash
EMBED_INFERENCE_NAME=e5-embed
export EMBEDDINGS_MODEL_NAME=intfloat/multilingual-e5-large
flexai inference serve $EMBED_INFERENCE_NAME --hf-token-secret hf-token -- --model=$EMBEDDINGS_MODEL_NAME --task=embed --trust-remote-code --dtype=float32
# store the returned information
export EMBEDDINGS_API_KEY=<store the given API key>
export EMBEDDINGS_URL=$(flexai inference inspect $EMBED_INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

##  LangSmith (Optional)

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT=rag-with-flexai
```

## Setup

The code of this experiment is located at `code/rag` and the following commands should be run from this location.

### Using Docker

1. Build the Docker image:

   ```bash
   docker build -t rag-application .
   ```

2. Run the Docker container:

   ```bash
   docker run -p 7860:7860 \
    -e LLM_MODEL_NAME=$LLM_MODEL_NAME \
    -e LLM_API_KEY=$LLM_API_KEY \
    -e LLM_URL=$LLM_URL \
    -e EMBEDDINGS_MODEL_NAME=$EMBEDDINGS_MODEL_NAME \
    -e EMBEDDINGS_API_KEY=$EMBEDDINGS_API_KEY \
    -e EMBEDDINGS_URL=$EMBEDDINGS_URL \
    -e LANGSMITH_TRACING=$LANGSMITH_TRACING \
    -e LANGSMITH_API_KEY=$LANGSMITH_API_KEY \
    -e LANGSMITH_PROJECT=$LANGSMITH_PROJECT \
    rag-application
   ```

### Local Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:

   ```bash
   python ./run_rag.py
   ```

## Usage

Once the application is running, you can access the Gradio interface in your web browser at `http://localhost:7860`. You can upload documents and ask questions based on the content of those documents.

For examples, you can upload documents located at `code/rag/data` and ask questions such as

- What is this demo about?
- For which workflows LLM agents are useful?
- Where can I find bioluminescent fungis?
