# RAG Chat Assistant

A Flask-based Retrieval-Augmented Generation (RAG) system that allows users to upload documents and chat with their data using Google's Gemini AI.

**Still a work in progress the Tables are still not perfect**

## Features

- 📄 **Multi-format Support**: PDF, Excel, CSV, and text files
- 🌐 **URL Ingestion**: Extract content from web pages
- 🤖 **Intelligent Routing**: Automatic selection between pandas analysis and RAG retrieval
- 💬 **Session Management**: Persistent chat sessions with history
- 🏷️ **Document Tagging**: Organize documents with custom tags
- 🔍 **Vector Search**: FAISS-powered similarity search
- 📊 **Data Analysis**: Built-in pandas agent for data queries

## Project Structure

```
rag-chat-assistant/
├── app.py                 # Your existing main application
├── run.py                 # New alternative entry point
├── requirements.txt       # Your existing dependencies
├── README.md             # New complete documentation
├── LICENSE               # New MIT license
├── .env.example          # New environment template
├── .gitignore           # New gitignore
├── Dockerfile           # New Docker support
├── docker-compose.yml   # New Docker Compose
├── templates/
│   └── index.html       # Your existing frontend
├── uploads/             # Created automatically
├── .env                 # User creates from .env.example
└── rag_system.log      # Created automatically
```

## API Endpoints

The Flask application exposes the following RESTful API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/session/create` | Initializes a new user session |
| `POST` | `/api/ingest` | Ingests an uploaded file (PDF, Excel, CSV, etc.) |
| `POST` | `/api/ingest/url` | Ingests content from a public URL |
| `POST` | `/api/chat` | Handles user chat queries and returns a response |
| `POST` | `/api/session/clear` | Clears all data associated with the current session |
| `GET` | `/api/session/info` | Retrieves information about the current session |
| `GET` | `/api/health` | Health check endpoint for monitoring |
| `GET` | `/api/stats` | Provides system-wide statistics |

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-chat-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `SECRET_KEY`: Flask session secret key (optional, auto-generated if not provided)

### API Key Setup

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Usage

1. **Start a Session**: The application automatically creates a session when you visit the page
2. **Upload Documents**: Use the upload area to add PDF, Excel, CSV, or text files
3. **Add Tags**: Organize your documents with descriptive tags
4. **Chat**: Ask questions about your uploaded documents
5. **URL Ingestion**: Alternatively, ingest content from web URLs

### Example Queries

- "What is the total revenue in the sales data?"
- "Summarize the main points from the research paper"
- "Show me the top 10 customers by sales volume"
- "What are the key findings in the report?"

## Architecture

- **Frontend**: Vanilla JavaScript with modern CSS
- **Backend**: Flask with CORS support
- **AI**: Google Gemini for embeddings and text generation
- **Vector Store**: FAISS for similarity search
- **Data Processing**: pandas for structured data analysis
- **Document Processing**: PyMuPDF for PDFs, BeautifulSoup for web scraping

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Google API key is valid and has access to Gemini API
2. **File Upload Issues**: Check that uploaded files are not corrupted and are in supported formats
3. **Memory Issues**: For large files, consider increasing system memory or processing smaller chunks

### Logs

Check `rag_system.log` for detailed application logs and error messages.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
