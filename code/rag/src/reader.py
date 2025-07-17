# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os

from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader


class MixedFileTypeLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if isinstance(self.file_path, list):
            documents = []
            for file in self.file_path:
                docs = self._load_file(file)
                if isinstance(docs, list):
                    documents.extend(docs)
                else:
                    documents.append(docs)
            return documents

        return self._load_file(self.file_path)

    def _load_file(self, file):
        if file.startswith("http://") or file.startswith("https://"):
            is_url = True
        else:
            is_url = False
            if not os.path.exists(file):
                raise FileNotFoundError(f"File not found: {file}")
            file_extension = os.path.splitext(file)[1].lower()

        if is_url:
            return self._load_web(file)
        elif file_extension == ".pdf":
            return self._load_pdf(file)
        elif file_extension in [".doc", ".docx"]:
            return self._load_word(file)
        elif file_extension == ".txt":
            return self._load_txt(file)
        elif file_extension in [".html", ".htm"]:
            return self._load_html(file)
        elif file_extension == ".csv":
            return self._load_csv(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_word(self, file_path):
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def _load_txt(self, file_path):
        loader = TextLoader(file_path)
        return loader.load()

    def _load_html(self, file_path):
        loader = BSHTMLLoader(file_path)
        return loader.load()

    def _load_csv(self, file_path):
        loader = CSVLoader(file_path)
        return loader.load()

    def _load_web(self, file_path):
        loader = WebBaseLoader(file_path)
        return loader.load()
