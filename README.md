# 🧠 Retrieval-Augmented Generation (RAG) for LIC Insurance Policies

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain** and **ChromaDB** to answer queries from **LIC (Life Insurance Corporation of India)** policy documents.  
It loads LIC policy PDFs, extracts section-based chunks, generates embeddings, stores them in a vector database, and uses an LLM to answer user queries **strictly based on the policy context**.

---

## 📘 Project Overview

### 🎯 Objective
Enable users to ask **natural-language questions** about Indian LIC insurance policies and get **accurate, context-grounded answers** from policy PDFs — without hallucinations.

### ⚙️ Core Components
| Step | Description |
|------|--------------|
| **1. Document Loading** | Load LIC policy PDFs using `PyMuPDFLoader` |
| **2. Section-based Chunking** | Extract text by major sections (e.g., *Introduction*, *Benefits*, *Surrender Value*, etc.) |
| **3. Cleaning** | Remove extra newlines, spaces, and merge cross-page text |
| **4. Token-free Character Splitting** | Split long sections into overlapping chunks (700 chars, 100 overlap) |
| **5. Metadata Tagging** | Attach metadata (plan name, UIN, section, page, etc.) to every chunk |
| **6. Embedding Generation** | Generate OpenAI text embeddings (`text-embedding-3-small`) |
| **7. Vector Storage** | Store embeddings in ChromaDB for fast similarity search |
| **8. Retrieval** | Retrieve top-matching chunks for a given query |
| **9. LLM Answer Generation** | Use GPT model (`gpt-3.5-turbo`) with custom prompt template for concise and accurate responses |

---

## 🗂️ Project Structure

```
RAG_Insurance/
│
├── data/
│   └── raw/
│       └── LIC_Bima_Shree.pdf             # Sample LIC policy PDF
│
├── output/
│   └── bima_shree_chunks.jsonl            # Cleaned & section-tagged JSONL chunks
│
├── chroma_db/                             # Chroma persistent database
│
├── .env                                   # Store your OpenAI API key
│
├── LIC_RAG.ipynb                          # Main notebook (this project)
│
└── README.md                              # Project documentation
```

---

## 🔑 Environment Setup

### 1️⃣ Create Virtual Environment
```bash
uv venv .venv
source .venv/bin/activate
```

### 2️⃣ Install Dependencies
```bash
uv add langchain langchain-openai langchain-community chromadb PyMuPDF python-dotenv
```

### 3️⃣ Add OpenAI API Key
Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key_here
```

---

## 🧩 Code Flow Summary

### **1. Load PDF**
```python
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("../data/raw/LIC_Bima_Shree.pdf")
docs = loader.load()
```

### **2. Extract & Clean Sections**
```python
sectional_chunks = extract_section_chunks_cleaned(docs)
```

Each chunk includes:
```python
{
  "policy_name": "Money Back Plans",
  "product_name": "LIC’s Bima Shree",
  "plan_no": "748",
  "uin_no": "512N316V03",
  "section": "2. Benefits",
  "page": 3
}
```

### **3. Chunk by Character Limit**
```python
split_chunks = split_by_char_length(text, max_len=700, overlap=100)
```

### **4. Save Processed Chunks**
```python
with open("../output/bima_shree_chunks.jsonl", "w") as f:
    ...
```

### **5. Create Chroma Vector Store**
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
    collection_name="rag_collection"
)
```

### **6. Search**
```python
_ = search("What is the minimum entry age and maximum maturity age?")
_ = search_with_score("Explain the Survival Benefit schedule.")
```

### **7. Retriever + LLM**
```python
retriever = vectorstore.as_retriever(search_type="mmr", search_kwarg={"k":3})
llm = init_chat_model("openai:gpt-3.5-turbo")
```

### **8. Prompt + Chain**
```python
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer strictly using the provided LIC policy context..."),
    ("user", "Question:\n{question}\n\nContext:\n{context}")
])
```

### **9. Invoke RAG Chain**
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Explain the Survival Benefit schedule.")
```

---

## 🧠 Sample Output


**Q**: What is payable at the end of the policy term if all premiums are paid and what are there percentages?
**A**: At the end of the policy term, if all premiums are paid, the following percentages of the Basic Sum Assured will be payable:
- 40% for a policy term of 14 years
- 30% for a policy term of 16 years
- 20% for a policy term of 18 years
- 10% for policy terms of 20, 24, and 28 years.

---

## 📊 Features

- ✅ **Section-aware chunking** — preserves context within LIC sections  
- ✅ **Policy-specific metadata** — ensures correct plan retrieval  
- ✅ **700-char overlapping chunks** — fine-grained similarity matches  
- ✅ **Persistent vector DB (Chroma)** — scalable multi-policy support  
- ✅ **Strict grounding rules** — no hallucinated answers  
- ✅ **Flexible filtering** — query by UIN, product, or section  

---

## 🚀 Next Steps

- [ ] Add more LIC policy PDFs (e.g., *Jeevan Anand*, *Tech-Term*, *Jeevan Labh*)  
- [ ] Extend metadata filtering for **multi-policy retrieval**  
- [ ] Integrate with a **Streamlit or FastAPI UI**  
- [ ] Add **evaluation pipeline** for accuracy testing  

---

## 🧩 Example Multi-Policy Search

```python
_ = search(
    "Explain the survival benefits schedule.",
    where={"product_name": "LIC’s Bima Shree"}
)
```

---

## 🧑‍💻 Author
**Jivesh Dhakate**  
MSc Computer Science (Negotiated Learning), University College Dublin  
📧 [LinkedIn](https://www.linkedin.com/in/jivesh-dhakate/) | 🧠 LangChain | 🧩 ChromaDB | ☁️ Cloud & ML

---

## 🏷️ License
MIT License © 2025  
This repository is for educational and research purposes.
