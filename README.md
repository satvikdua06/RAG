# Enhanced RAG and Pandas Agent System

This project is an advanced Retrieval-Augmented Generation (RAG) system with a built-in agent that can perform data analysis using the Pandas library. It features a web-based chat interface allowing users to upload documents (PDFs, Excel/CSV, and text files) or ingest web pages, and then ask questions about them.

The system intelligently routes user queries to either a standard RAG pipeline for information retrieval or a Pandas agent for data aggregation and analysis, providing a versatile tool for interacting with various data sources.

## ✨ Features

* **Multi-Source Ingestion**: Supports uploading and processing of PDF, Excel (.xlsx, .xls), CSV, and plain text files, as well as content ingestion from public URLs.
* **Advanced PDF Processing**: Automatically extracts both plain text and tables from PDF documents, treating tables as separate, queryable data sources.
* **Intelligent Agent Routing**: Dynamically analyzes user queries to decide whether to use the RAG pipeline (for informational questions) or the Pandas agent (for data analysis, calculations, and aggregations).
* **Data-Aware Pandas Agent**: Generates and safely executes Python Pandas code to answer complex questions about tabular data.
* **Interactive Chat UI**: A clean, user-friendly web interface for uploading files, managing sessions, and chatting with your data.
* **Session Management**: Manages user sessions with tagged data sources, allowing for conversation context and organized data handling.
* **Powered by Gemini**: Utilizes Google's Gemini models for high-quality text embeddings and powerful generative responses.

## ⚙️ How It Works

The application follows a systematic process to handle user interactions and data processing:

1.  **Session Start**: A new session is created when a user first visits the web interface.
2.  **Data Ingestion**: The user uploads a file or provides a URL and assigns it a "tag".
3.  **Processing & Indexing**:
    * The system chunks the content (text, tables, web content) into manageable pieces.
    * It uses a Google Gemini embedding model to convert these chunks into vector embeddings.
    * These embeddings are stored in a FAISS vector index for efficient similarity searching. For data files (Excel/CSV), the raw DataFrame is also cached.
4.  **Chat Interaction**: The user asks a question through the chat interface.
5.  **Source Selection**: The system identifies the relevant data source based on the tag mentioned in the query.
6.  **Intelligent Routing**: The query is analyzed to determine the user's intent.
    * If the query involves calculations, aggregations, or specific data filtering (e.g., "what is the sum of...", "count the number of..."), it is routed to the **Pandas Agent**.
    * If the query is informational (e.g., "what is...","describe..."), it is routed to the **RAG Pipeline**.
7.  **Response Generation**:
    * **RAG Pipeline**: Retrieves the most relevant text chunks from the vector index and feeds them, along with the query, into a Gemini generative model to synthesize an answer.
    * **Pandas Agent**: Constructs a detailed prompt including the query, dataset schema, and sample data. It uses the Gemini model to generate Pandas code, which is then executed safely to get the result.
8.  **Display**: The final answer is displayed to the user in the chat window.

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

* Python 3.9+
* Access to the Google Gemini API

1. Clone the Repository

git clone <your-repository-url>
cd <repository-directory>
2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install Dependencies
Install all the required packages from the requirements.txt file.

Bash

pip install -r requirements.txt
4. Set Up Environment Variables
Create a file named .env in the root directory of the project. Copy the contents of .env.example into it and add your credentials.

.env.example
Code snippet:
  SECRET_KEY='generate-a-strong-random-secret-key-yourself'
  GOOGLE_API_KEY='YOUR_GEMINI_API_KEY'
  
SECRET_KEY: A secret key for Flask session management.

GOOGLE_API_KEY: Your API key for the Google Gemini service.

5. Run the Application
Start the Flask server with the following command:


python app.py
The application will be accessible at http://127.0.0.1:5000. Open this URL in your web browser to start using the chat assistant.

📂 Project Structure
.
├── app.py              # Main Flask application, backend logic, RAG pipeline
├── requirements.txt    # Python package dependencies
├── templates
│   └── index.html      # Frontend chat interface
├── uploads/            # Directory for storing uploaded files temporarily
├── .env                # File for environment variables (API keys)
└── rag_system.log      # Log file for debugging and tracking
📜 API Endpoints
The Flask application exposes the following RESTful API endpoints:

Method	Endpoint	Description
POST	/api/session/create	Initializes a new user session.
POST	/api/ingest	Ingests an uploaded file (PDF, Excel, CSV, etc.).
POST	/api/ingest/url	Ingests content from a public URL.
POST	/api/chat	Handles user chat queries and returns a response.
POST	/api/session/clear	Clears all data associated with the current session.
GET	/api/session/info	Retrieves information about the current session.
GET	/api/health	Health check endpoint for monitoring.
GET	/api/stats	Provides system-wide statistics.

