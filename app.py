from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import fitz  # PyMuPDF
import base64
import io
from PIL import Image
import requests
import json
from werkzeug.utils import secure_filename
import tempfile
import shutil

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
class DocumentProcessor:
    def __init__(self, moondream_url="http://localhost:11434", ollama_url="http://localhost:11434"):
        self.moondream_url = moondream_url
        self.ollama_url = ollama_url
    
    def extract_text_and_images(self, pdf_path):
        """Extract text and images from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text_sections = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            text = page.get_text()
            if text.strip():
                text_sections.append({
                    'page': page_num + 1,
                    'text': text.strip()
                })
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        images.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'data': base64.b64encode(img_data).decode(),
                            'format': 'png'
                        })
                    pix = None
                except Exception as e:
                    print(f"Error extracting image: {e}")
                    continue
        
        doc.close()
        return text_sections, images
    
    def caption_image(self, image_data):
        """Generate caption for image using Moondream via Ollama"""
        try:
            payload = {
                "model": "moondream",
                "prompt": "Describe this image in detail.",
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(f"{self.moondream_url}/api/generate", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json().get('response', 'No caption generated')
            else:
                return f"Error generating caption: {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Moondream: {str(e)}"
    
    def ocr_image(self, image_data):
        """Extract text from image using Moondream OCR"""
        try:
            payload = {
                "model": "moondream",
                "prompt": "Extract all text from this image. If there's no text, respond with 'No text found'.",
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(f"{self.moondream_url}/api/generate", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json().get('response', 'No text extracted')
            else:
                return f"Error extracting text: {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Moondream: {str(e)}"
    
    def process_pdf(self, pdf_path):
        """Process PDF to extract and analyze all content"""
        text_sections, images = self.extract_text_and_images(pdf_path)
        
        # Process images with Moondream
        image_data = []
        for img in images:
            print(f"Processing image from page {img['page']}")
            caption = self.caption_image(img['data'])
            ocr_text = self.ocr_image(img['data'])
            
            image_data.append({
                'page': img['page'],
                'index': img['index'],
                'caption': caption,
                'ocr_text': ocr_text,
                'data': img['data']  # Keep for display purposes
            })
        
        return {'text': text_sections, 'images': image_data}
    
    def combine_content(self, doc):
        """Combine text and image information into a single structure"""
        combined = []
        
        # Group content by page
        pages = {}
        
        # Add text sections
        for section in doc['text']:
            page_num = section['page']
            if page_num not in pages:
                pages[page_num] = {'text': [], 'images': []}
            pages[page_num]['text'].append(section['text'])
        
        # Add image information
        for img in doc['images']:
            page_num = img['page']
            if page_num not in pages:
                pages[page_num] = {'text': [], 'images': []}
            
            img_info = f"[IMAGE] Caption: {img['caption']}\n[IMAGE] OCR Text: {img['ocr_text']}"
            pages[page_num]['images'].append(img_info)
        
        # Combine all content in page order
        for page_num in sorted(pages.keys()):
            page_content = f"\n--- Page {page_num} ---\n"
            
            # Add text content
            if pages[page_num]['text']:
                page_content += "\n".join(pages[page_num]['text'])
            
            # Add image information
            if pages[page_num]['images']:
                page_content += "\n" + "\n".join(pages[page_num]['images'])
            
            combined.append(page_content)
        
        return "\n\n".join(combined)
    
    def compare_documents(self, content1, content2):
        """Compare two documents using Ollama LLM"""
        try:
            prompt = f"""Perform a comprehensive comparison of the following two documents. Structure the output into the following sections:

1. **Similarities**: 
   - Clearly list all identical or unchanged elements between Document 1 and Document 2. This includes matching text, identical image descriptions, and any structural elements that remain the same. 
   - If there are multiple rules or sections, mention which ones are identical in both documents.

2. **Text Differences**:
   - Highlight and explain all differences in the text between the two documents. This includes additions, deletions, modifications, or any wording changes. 
   - For each difference, show the version from both Document 1 and Document 2 to facilitate comparison.

3. **Image Differences**:
   - Identify any changes related to images between the two documents, including any images that have been added, removed, or altered.
   - Compare captions and OCR-extracted text where applicable and highlight any significant differences.

4. **Structural Differences**:
   - Mention any reordering of sections or rules, as well as added or removed sections (e.g., new rules or changes in the order of content).
   - Highlight any structural or formatting changes that impact the layout or overall flow of the documents.

5. **Tabular Summary (if applicable)**:
   - Provide a table summarizing key changes or rule comparisons if patterns exist. The table should have the following columns: `Section/Rule`, `Document 1`, `Document 2`, `Change Type (e.g., Modified, Added, Removed)`. Use emojis like ✅ for unchanged, 🔄 for modified, 🆕 for added, and ❌ for removed.

6. **Conclusion**:
   - Summarize the most impactful changes between the two documents. Consider how the differences affect the overall meaning, intent, or guidelines within the documents.
   - Highlight which sections or rules are most affected by the differences.

---
Document 1:
{content1}

---
Document 2:
{content2}

Provide a detailed comparison report."""

            payload = {
                "model": "llama3.2",  # or your preferred model
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, timeout=120)
            
            if response.status_code == 200:
                return response.json().get('response', 'No comparison generated')
            else:
                return f"Error comparing documents: {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"

# Initialize processor
processor = DocumentProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Both files must be selected'}), 400
    
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        return jsonify({
            'success': True,
            'file1': filename1,
            'file2': filename2,
            'message': 'Files uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_documents():
    data = request.get_json()
    file1 = data.get('file1')
    file2 = data.get('file2')
    
    if not file1 or not file2:
        return jsonify({'error': 'File names are required'}), 400
    
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], file1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], file2)
    
    if not (os.path.exists(filepath1) and os.path.exists(filepath2)):
        return jsonify({'error': 'Files not found'}), 404
    
    try:
        # Process both documents
        print("Processing document 1...")
        doc1 = processor.process_pdf(filepath1)
        
        print("Processing document 2...")
        doc2 = processor.process_pdf(filepath2)
        
        # Combine content
        content1 = processor.combine_content(doc1)
        content2 = processor.combine_content(doc2)
        
        # Compare documents
        print("Comparing documents...")
        comparison_result = processor.compare_documents(content1, content2)
        
        return jsonify({
            'success': True,
            'comparison': comparison_result,
            'doc1_summary': {
                'text_sections': len(doc1['text']),
                'images': len(doc1['images'])
            },
            'doc2_summary': {
                'text_sections': len(doc2['text']),
                'images': len(doc2['images'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up uploaded files"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({'success': True, 'message': 'Files cleaned up'})
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
     app.run(debug=True)
     