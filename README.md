# PDF Document Comparison Tool

A Flask-based web application that compares two PDF documents by extracting text and images, analyzing visual content using AI, and providing comprehensive comparison reports.

## Features

1. **PDF Text Extraction** - Extracts text from PDF documents using PyMuPDF
2. **Image Processing** - Extracts and analyzes images from PDFs
3. **AI-Powered Image Analysis** - Uses Moondream model for image captioning and OCR
4. **Document Comparison** - Compares two documents with detailed analysis using Llama3.2
5. **Comprehensive Reports** - Generates structured comparison reports with similarities, differences, and summaries
6. **Web Interface** - Easy-to-use web interface for uploading and comparing documents

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) running locally at http://localhost:11434

### Required Ollama Models

```bash
ollama pull moondream
ollama pull llama3.2
```

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/shiuli-19/PDFcomparison
cd pdf-document-comparison
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install flask pymupdf pillow requests werkzeug base64 io tempfile shutil json os
```


### 4. Start Ollama Service

Ensure Ollama is running with required models:
```bash
ollama serve
```

### 5. Run the Application

```bash
python app.py
```

## API Endpoints

### Upload Files
- **POST** `/upload`
- Upload two PDF files for comparison
- Returns: File upload confirmation

### Compare Documents
- **POST** `/compare`
- Compare two uploaded PDF documents
- Returns: Comprehensive comparison report

### Cleanup
- **POST** `/cleanup`
- Remove all uploaded files
- Returns: Cleanup confirmation

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open web browser**
   Navigate to `http://localhost:5000`

3. **Upload PDF files**
   - Select two PDF files to compare
   - Click upload to process files

4. **Compare documents**
   - Click compare to analyze differences
   - View comprehensive comparison report

5. **Clean up**
   - Remove uploaded files when done

## Comparison Report Structure

1. **Similarities** - Identical elements between documents
2. **Text Differences** - Changes in textual content
3. **Image Differences** - Changes in visual content
4. **Structural Differences** - Layout and formatting changes
5. **Tabular Summary** - Key changes in table format
6. **Conclusion** - Summary of most impactful changes

## Configuration

### File Size Limits
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
```

### Allowed File Types
```python
ALLOWED_EXTENSIONS = {'pdf'}
```

### Ollama URLs
```python
moondream_url = "http://localhost:11434"
ollama_url = "http://localhost:11434"
```

## Dependencies

### Python Packages
- **Flask==2.3.3** - Web framework
- **PyMuPDF==1.23.5** - PDF processing (imported as `fitz`)
- **Pillow==10.0.0** - Image processing
- **requests==2.31.0** - HTTP requests to Ollama
- **Werkzeug==2.3.7** - File upload utilities

### Built-in Python Modules (No installation needed)
- **base64** - Base64 encoding/decoding
- **io** - Input/output operations
- **os** - Operating system interface
- **json** - JSON data handling
- **tempfile** - Temporary file handling
- **shutil** - High-level file operations

## Error Handling

The application includes comprehensive error handling for:
- File upload failures
- PDF processing errors
- AI model connectivity issues
- Document comparison failures

## Troubleshooting

### Check Ollama Status
```bash
ollama list
```

### Verify Models are Available
```bash
ollama show moondream
ollama show llama3.2
```

### Test Ollama Connection
```bash
curl http://localhost:11434/api/tags
```
