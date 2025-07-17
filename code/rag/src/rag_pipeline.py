# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os
import uuid
from typing import List

from langchain.schema import BaseMessage
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from src.reader import MixedFileTypeLoader
from transformers import AutoConfig


class RagPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        use_tools: bool = False,
    ):
        """
        Initializes the pipeline and sets up models and vector store.

        Args:
            chunk_size (int, optional): The size of each chunk for processing. Defaults to 500.
            chunk_overlap (int, optional): The overlap size between consecutive chunks. Defaults to 50.
            top_k (int, optional): The number of top scored documents to retrieve. Defaults to 5.
            use_tool_calling (bool, optional): Flag to indicate whether to use LLM tool calling.
                If True, the LLM will decide whether it needs to use tools to retrieve information or directly
                respond to the query without invoking the document search.
                If False, document search will always be performed and the LLM will always receive the
                corresponding context to respond to the query. Defaults to False.
        """
        self.top_k: int = top_k
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.use_tools: bool = use_tools

        self.intro_prompt: str = (
            "You are FlexBot, an assistant for question-answering tasks. "
        )
        self.rag_prompt: str = (
            "You are FlexBot, an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
        )

        self.llm: ChatOpenAI
        self.embeddings: OpenAIEmbeddings
        self.vector_store: InMemoryVectorStore
        self.prompt: PromptTemplate
        self.graph: StateGraph
        self.llm_model_name: str
        self.llm_api_key: str
        self.llm_url: str
        self.embeddings_model_name: str
        self.embeddings_api_key: str
        self.embeddings_url: str

        self._set_endpoint_config()
        self._set_models()
        self._set_vector_store()
        self._set_graph()

    def add_documents(self, file_paths: List[str]) -> None:
        # parse documents into chunks
        loader = MixedFileTypeLoader(file_paths)
        docs = loader.load()
        for doc in docs:
            # only keep document filename
            doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)

        # index chunked documents
        _ = self.vector_store.add_documents(documents=all_splits)

    def _check_env(self) -> None:
        if "LLM_API_KEY" not in os.environ:
            raise ValueError("Please set the LLM_API_KEY environment variable.")
        if "LLM_URL" not in os.environ:
            raise ValueError("Please set the LLM_URL environment variable.")
        if "LLM_MODEL_NAME" not in os.environ:
            raise ValueError("Please set the LLM_MODEL_NAME environment variable.")
        if "EMBEDDINGS_API_KEY" not in os.environ:
            raise ValueError("Please set the EMBEDDINGS_API_KEY environment variable.")
        if "EMBEDDINGS_URL" not in os.environ:
            raise ValueError("Please set the EMBEDDINGS_URL environment variable.")
        if "EMBEDDINGS_MODEL_NAME" not in os.environ:
            raise ValueError(
                "Please set the EMBEDDINGS_MODEL_NAME environment variable."
            )

    def get_endpoint_config(self) -> dict:
        return {
            "llm_model_name": self.llm_model_name,
            "llm_api_key": self.llm_api_key,
            "llm_url": self.llm_url,
            "embeddings_model_name": self.embeddings_model_name,
            "embeddings_api_key": self.embeddings_api_key,
            "embeddings_url": self.embeddings_url,
        }

    def set_endpoint_config(self, config: dict) -> None:
        for key, value in config.items():
            if key not in [
                "llm_model_name",
                "llm_api_key",
                "llm_url",
                "embeddings_model_name",
                "embeddings_api_key",
                "embeddings_url",
            ]:
                raise ValueError(f"Invalid config key: {key}")
            setattr(self, key, value)
        self._set_models()
        self._set_vector_store()

    def _set_endpoint_config(
        self,
    ) -> None:
        self._check_env()
        self.llm_model_name = os.getenv("LLM_MODEL_NAME")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_url = os.getenv("LLM_URL")
        self.embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
        self.embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY")
        self.embeddings_url = os.getenv("EMBEDDINGS_URL")

    def _set_models(self) -> None:
        config = AutoConfig.from_pretrained(self.embeddings_model_name)
        assert self.chunk_size <= config.max_position_embeddings

        llm = ChatOpenAI(
            model_name=self.llm_model_name,
            openai_api_key=self.llm_api_key,
            openai_api_base=self.llm_url + "/v1",
        )

        embeddings = OpenAIEmbeddings(
            model=self.embeddings_model_name,
            deployment=self.embeddings_model_name,
            openai_api_key=self.embeddings_api_key,
            openai_api_base=self.embeddings_url + "/v1",
            tiktoken_enabled=False,
        )
        self.llm = llm
        self.embeddings = embeddings

    def _set_vector_store(self) -> None:
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def _get_generate_prompt(
        self, state: MessagesState, docs_content: str
    ) -> List[BaseMessage]:
        # Format into prompt
        system_message_content = f"{self.rag_prompt}" "\n\n" f"{docs_content}"
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        return prompt

    def _serialize_docs(self, docs: List[Document]) -> str:
        """Serialize documents."""
        serialized_docs = []
        for doc in docs:
            if "source" in doc.metadata:
                # remove "source" key from metadata and display it first
                src_string = doc.metadata.pop("source")
                if doc.metadata:
                    src_string += f" {doc.metadata}"
            else:
                src_string = str(doc.metadata)
            serialized_docs.append(
                f"Source: {src_string}\n" f"Content: {doc.page_content}"
            )

        return "\n\n".join(serialized_docs)

    def _set_graph(self) -> None:
        if self.use_tools:
            self._set_graph_with_tool_calling()
        else:
            self._set_graph_without_tool_calling()

    def _set_graph_with_tool_calling(self) -> None:
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Use this tool to retrieve information from documents stored in the knowledge base."""
            retrieved_docs = self.vector_store.similarity_search(query, k=self.top_k)
            serialized = self._serialize_docs(retrieved_docs)
            # return content, artifact
            return serialized, retrieved_docs

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = self.llm.bind_tools([retrieve])
            prompt = [SystemMessage(self.intro_prompt)] + state["messages"]
            response = llm_with_tools.invoke(prompt)
            # MessagesState appends messages to state instead of overwriting
            return {"messages": [response]}

        def generate(state: MessagesState):
            """Generate answer."""
            # Get context from generated ToolMessages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            docs_content = "\n\n".join(doc.content for doc in tool_messages)

            prompt = self._get_generate_prompt(state, docs_content)

            # Run
            response = self.llm.invoke(prompt)
            return {"messages": [response]}

        graph_builder = StateGraph(MessagesState)
        memory = MemorySaver()
        tools = ToolNode([retrieve])

        # Node 1: Generate an AIMessage that may include a tool-call to be sent.
        graph_builder.add_node(query_or_respond)
        # Node 2: Execute the retrieval tool.
        graph_builder.add_node(tools)
        # Node 3: Generate a response using the retrieved content.
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        self.graph = graph_builder.compile(checkpointer=memory)

    def _set_graph_without_tool_calling(self) -> None:
        class ContextState(MessagesState):
            context: List[Document]

        def retrieve(state: ContextState):
            # Get last HumanMessage, which is the question
            for message in reversed(state["messages"]):
                if message.type == "human":
                    question = message.content
                    break
            else:
                raise ValueError("No human message found in the state.")
            retrieved_docs = self.vector_store.similarity_search(question)
            return {"context": retrieved_docs}

        def generate(state: ContextState):
            """Generate answer."""
            docs_content = self._serialize_docs(state["context"])
            prompt = self._get_generate_prompt(state, docs_content)

            # Run
            response = self.llm.invoke(prompt)
            return {"messages": [response]}

        graph_builder = StateGraph(ContextState).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)

    def query(
        self, input_message: str, thread_id: str = None
    ) -> dict[str, List[BaseMessage]]:
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        res = self.graph.invoke(
            {"messages": [{"role": "user", "content": input_message}]},
            config=config,
        )
        return res

    def clear_vector_store(self) -> None:
        """Clear the vector store."""
        self._set_vector_store()
