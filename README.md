# HealthCare ChatBot RAG

**Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS**

A healthcare question-answering chatbot built with **Retrieval-Augmented Generation (RAG)**. It uses medical PDF documents as knowledge sources, embeds them in a vector store (Pinecone), and answers user questions using OpenAI's GPT-4 with retrieved context.

## Features

- **RAG pipeline**: Retrieves relevant document chunks from Pinecone, then generates answers using GPT-4
- **Medical knowledge base**: Indexes PDF documents (e.g., medical textbooks) for context-aware responses
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) for semantic search
- **Two app variants**: FastAPI (recommended) and Flask
- **AWS CI/CD**: Deploy via GitHub Actions to EC2 + ECR

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI (app2) / Flask (app) |
| LLM | OpenAI GPT-4o |
| Vector DB | Pinecone (serverless, AWS) |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| RAG / Chains | LangChain (LCEL in app2) |
| PDF Loader | PyPDF, LangChain DocumentLoader |
| Deployment | Docker, AWS ECR, EC2, GitHub Actions |

## Project Structure

```
HealthCare-ChatBot-Rag/
├── app2.py              # FastAPI app (recommended) - RAG chat API
├── app.py               # Flask app - alternative implementation
├── store_index.py       # Script to load PDFs, chunk, embed, and upsert to Pinecone
├── setup.py             # Package setup (healthcare_chatbot_rag)
├── requirements.txt     # Python dependencies
├── template.sh          # Project scaffold script
│
├── data/                # Source documents
│   └── Medical_book.pdf
│
├── src/                 # Core logic
│   ├── helper.py        # PDF loading, text splitting, embeddings
│   └── prompt.py        # System prompt for the LLM
│
├── templates/           # HTML templates
│   └── chat.html        # Chat UI
│
├── static/              # Static assets
│   └── style.css
│
├── research/            # Experimentation
│   └── trails.ipynb
│
└── .env                 # API keys (create from .env.example, not in repo)
```

## Prerequisites

- Python 3.10+
- [Pinecone](https://www.pinecone.io/) account
- [OpenAI](https://platform.openai.com/) API key

---

## How to Run

### STEP 1 — Clone the repository

```bash
git clone https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS
```

### STEP 2 — Create environment (choose one)

**Option A: Conda**

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

**Option B: venv**

```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux/macOS: source venv/bin/activate
```

### STEP 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### STEP 4 — Configure environment

Create a `.env` file in the project root:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### STEP 5 — Index documents (one-time)

```bash
python store_index.py
```

### STEP 6 — Run the application

```bash
# Flask (port 8080)
python app.py

# Or FastAPI (port 8085)
uvicorn app2:app --host 0.0.0.0 --port 8085 --reload
```

Then open **http://localhost:8080** (Flask) or **http://localhost:8085** (FastAPI).

---

## Indexing Documents

`store_index.py` will:

- Load PDFs from `data/`
- Split them into chunks (500 chars, 20 overlap)
- Generate embeddings with Hugging Face
- Create a Pinecone index `medical-chatbot` (if it does not exist)
- Upsert embeddings to the index

---

## AWS CI/CD Deployment with GitHub Actions

Deploy the chatbot to AWS using Docker, ECR, and EC2.

### 1. Login to AWS Console

### 2. Create IAM user for deployment

Create an IAM user with these permissions:

| Service | Purpose |
|---------|---------|
| **EC2** | Virtual machine to run the app |
| **ECR** | Elastic Container Registry to store Docker images |

**Attach policies:**

- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

**Deployment flow:**

1. Build Docker image of the source code  
2. Push image to ECR  
3. Launch EC2 instance  
4. Pull image from ECR on EC2  
5. Run the Docker container on EC2  

### 3. Create ECR repository

Create a repository to store the Docker image (e.g. `medicalbot`).

- Save the URI, e.g.: `315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot`

### 4. Create EC2 instance (Ubuntu)

### 5. Install Docker on EC2

```bash
# Optional
sudo apt-get update -y
sudo apt-get upgrade -y

# Required
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 6. Configure EC2 as self-hosted runner

1. Go to **GitHub repo → Settings → Actions → Runners**
2. Click **New self-hosted runner**
3. Select OS (Linux)
4. Run the provided commands one by one on your EC2 instance

### 7. Add GitHub Secrets

Add these secrets in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | ECR repository URI |
| `PINECONE_API_KEY` | Pinecone API key |
| `OPENAI_API_KEY` | OpenAI API key |

> **Note:** You need a `Dockerfile` and a `.github/workflows/deploy.yml` workflow that builds the image, pushes to ECR, and deploys to the EC2 self-hosted runner.

---

## API Endpoints (FastAPI)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the chat UI |
| POST | `/get` | Chat endpoint; expects form field `msg` and returns `{"answer": "..."}` |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `OPENAI_API_KEY` | Yes | OpenAI API key |

## License

See [LICENSE](LICENSE).
