# NPS Analysis and Recommendation Chatbot

**kb-chat** is a FastAPI-based web service that allows to upload PDF documents to a knowledge base, interact with it through natural language queries, and perform Excel-based NPS analysis. The backend is powered by Azure OpenAI for semantic search and chat capabilities.

---

## 📁 Project Structure
```
kb-chat/ 
├── api/
│ └── routes.py # Main API logic and route definitions 
├── ui/ 
│ └── ui.py # Chatbot frontend
├── services/
│ ├── chat_service.py # Logic for handling chat requests and OpenAI interactions 
│ ├── knowledge_base_service.py # Handles PDF ingestion, embedding, and search 
│ ├── azure_service.py #  Access to Azure services
│ ├── config_manager.py #  Configuration files and environment variables management
│ └── analysis_service.py # Handles Excel-based NPS analysis 
├── ui/ # Chatbot frontend (ui.py)
├── public/ # Publicly served assets 
├── Dockerfile # Docker image configuration 
├── docker-compose.yml # Docker Compose setup 
├── requirements.txt # Python package dependencies 
└── .gitignore # Files and folders ignored by git
```

## 🛠️ Installation
Prerequisites
Ensure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Python 3.8+](https://www.python.org/downloads/) (to run without Docker)

#### Get the code
Clone the repository:
```
git clone https://gitlab.devlits.com/circles/rltw/relationwise.git
cd kb-chat
```

#### Configuration
Before running the application, create a .env file in the root directory with the necessary environment variables:

```env
# Example .env file
# Azure Form Recognizer
AZURE_FORM_RECOGNIZER_ENDPOINT=""
AZURE_FORM_RECOGNIZER_KEY=""

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=""
AZURE_OPENAI_GPT_DEPLOYMENT=""

# Azure AI Search
AZURE_SEARCH_ENDPOINT=""
# AZURE_SEARCH_ENDPOINT=""
AZURE_AI_SEARCH_API_KEY=""
AZURE_SEARCH_INDEX_NAME=""
AZURE_AI_SEARCH_SERVICE_NAME=""

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=""
AZURE_STORAGE_CONTAINER_NAME=""

# Azure Speech 
AZURE_SPEECH_KEY=""
AZURE_SPEECH_REGION=""

# MongoDB
MONGODB_URI=""
```
Replace the placeholder values with your actual configuration details.

#### Running the Application
You have two options to run the application:

**Option 1**: Using Docker
```
docker-compose up --build
```
run ```docker-compose down``` to delete the containars

**Option 2**: Without Docker

1. Set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the backend:
```
uvicorn api.routes:app --host 0.0.0.0 --port 8000
```
or with python
```
python -m api.routes --host 0.0.0.0 --port 8000
```
4. Run the frontend (in a new terminal):
```
chainlit run ui/ui.py --host 0.0.0.0 --port 8501
```
## 📄 Access the chat application
Visit http://localhost:8501 in your browser to use the chat.

After accessing the application, you can start interacting with the chatbot. The chatbot is designed to respond to queries based on the integrated knowledge base and provide NPS analysis and recommendations using uploaded Excel-files. 

## 🧠 Knowledge Base Enrichment
The application provides an endpoint to enrich the chatbot's knowledge base by uploading documents.​

#### Uploading Documents
1. Get URL to your document
You need a public URL pointing to the document you want to upload. You can obtain this URL in one of the following ways:

**Option 1**: Upload your file to Azure Blob Storage and generate a public (or shared access) link to the uploaded file.

**Option 2**: Use a URL from another accessible source, such as a YouTube video link, a publicly shared Google Drive or Dropbox file, or a direct link to a document hosted on a website.

2. Access the Swagger UI
Navigate to http://localhost:8000/docs to access the Swagger UI.

3. Locate the /kb-upload Endpoint
In the Swagger UI, find the POST /kb-upload endpoint.

4. Upload a Document
Click on the endpoint to expand it, then click "Try it out". Use the file upload field to select and upload your document and url field to provide the corresponding URL to your file. Supported formats may include, .pdf, .pptx, .mp4 etc.

5. Submit the Request
Click "Execute" to send the request. The document will be processed and added to the knowledge base.

## 🧰 Services Overview
The services/ directory contains modular components that encapsulate the core functionalities of the kb-chat application.

#### 📊 analysis_service.py
This service is responsible for analyzing user Excel inputs and. It includes functionalities such as comments categorization and analysis, keyword extraction, intent recognition to provide NPS analysis and generate recomendations.​

#### ☁️ azure_service.py
Handles interactions with Microsoft Azure services. This involves connection to Azure-based resources.​

#### 💬 chat_service.py
Manages the core chat functionalities, including message handling, session management. It ensures that user messages are correctly processed and appropriate responses are generated.​

#### ⚙️ config_manager.py
Provides configuration management capabilities. This service likely handles the loading, parsing, and validation of configuration files and environment variables.​

#### 📚 knowledge_base_service.py
Manages the knowledge base that the chatbot uses to generate responses. This includes functionalities for querying the knowledge base, updating content, handling document uploads.


## Azure Architecture Resources
| Resource Name | Resource Type | Purpose |
|----------------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **DocumentDataExtractor**  | Azure AI Document Intelligence (Cognitive Service) | Extracts structured text/tables from documents (OCR) for downstream processing | 
| **kb-speech-to-text**  | Azure Cog | Transcrib put into text (e.g. for voice queries or audio data) | 
| **OpenAIDataProcessor** | Azure Cognitive Services (Azure OpenAI Service) | Runs OpenAI model inference (e.g. summarization or generation) on data. | 
| **pip-vw-front** | Azure Public IP Address | Public IP for inbound traffic to the VMSS and other resources. |
| **cv-vnet** | Virtual Network | Private network for all resources (isolates traffic, enables secure comms)| 
| **rv-bastion** | Azure Bastion | Provides secure RDP/SSH access to VMs over TLS without public IPs.|
| **nw-nsg** | Network Security Group| Applies allow/deny rules to filter network traffic in the VNet. | 
| **rv-net-ip** | Azure Public IP Address | Public IP for Azure Bastion (and NSG) to enable RDP/SSH access. |
| **rw-chat-dev-db** | Azure Cosmos DB (MongoDB API) | Globally distributed NoSQL DB storing chat and nps data |
| **index-search-demo** | Azure AI Search (Cognitive Search) | Search index for testing and development (enables text and vector search) |
| **kb-index-search** | Azure AI Search (Cognitive Search) | Basic search index (production) for knowledge-base queries. | 

![Azure diagram](./img/Azure_resourse_scheme.jpg)