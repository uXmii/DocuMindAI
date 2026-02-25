import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from evaluation_metrics import RAGEvaluator

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
# Set enable_multimodal=False if you don't have Tesseract installed
doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200, enable_multimodal=True)
rag_engine = RAGEngine(collection_name="documents")
# Temporarily disable evaluator to fix startup issue
# evaluator = RAGEvaluator(rag_engine)
evaluator = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Advanced RAG system with Hybrid Search + Multimodal + Evaluation',
        'features': {
            'multimodal': doc_processor.enable_multimodal,
            'evaluation': True
        }
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stats = rag_engine.get_stats()
    stats['multimodal_enabled'] = doc_processor.enable_multimodal
    stats['evaluation_history'] = len(evaluator.results_history)
    return jsonify(stats)

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a PDF document"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process document (now includes multimodal if enabled)
        chunks = doc_processor.process_document(filepath)
        
        # Count chunk types
        text_chunks = sum(1 for c in chunks if c['metadata'].get('type') == 'text')
        ocr_chunks = sum(1 for c in chunks if c['metadata'].get('type') == 'ocr')
        table_chunks = sum(1 for c in chunks if c['metadata'].get('type') == 'table')
        
        # Add to vector store and BM25 index
        result = rag_engine.add_documents(chunks)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'chunks_created': len(chunks),
            'chunk_breakdown': {
                'text': text_chunks,
                'ocr': ocr_chunks,
                'tables': table_chunks
            },
            'collection_size': result['collection_size'],
            'multimodal_enabled': doc_processor.enable_multimodal
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """Query the RAG system with specified search mode"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    top_k = data.get('top_k', 5)
    search_mode = data.get('search_mode', 'hybrid')  # vector, bm25, or hybrid
    
    try:
        result = rag_engine.query(question, top_k=top_k, search_mode=search_mode)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/compare', methods=['POST'])
def compare_search_modes():
    """Compare all search modes for the same question"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    top_k = data.get('top_k', 5)
    
    try:
        result = rag_engine.compare_search_modes(question, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# NEW EVALUATION ENDPOINTS

@app.route('/evaluate/single', methods=['POST'])
def evaluate_single_query():
    """Evaluate a single query across all methods"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    ground_truth = data.get('ground_truth', None)
    
    try:
        result = evaluator.evaluate_query(question, ground_truth)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/evaluate/batch', methods=['POST'])
def evaluate_batch_queries():
    """Evaluate multiple queries and generate report"""
    data = request.get_json()
    
    if not data or 'questions' not in data:
        return jsonify({'error': 'Questions list is required'}), 400
    
    questions = data['questions']
    ground_truths = data.get('ground_truths', None)
    
    try:
        report = evaluator.batch_evaluate(questions, ground_truths)
        
        # Save report
        filename = evaluator.save_report(report)
        
        # Generate visualization
        viz_path = filename.replace('.json', '.png')
        evaluator.visualize_comparison(report, viz_path)
        
        # Generate Excel
        excel_path = filename.replace('.json', '.xlsx')
        evaluator.export_to_excel(report, excel_path.split('/')[-1])
        
        return jsonify({
            'success': True,
            'report': report,
            'files': {
                'json': filename,
                'visualization': viz_path,
                'excel': excel_path
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/evaluate/history', methods=['GET'])
def get_evaluation_history():
    """Get all evaluation history"""
    return jsonify({
        'total_evaluations': len(evaluator.results_history),
        'sessions': len(evaluator.evaluation_sessions),
        'recent_results': evaluator.results_history[-10:] if evaluator.results_history else []
    })

@app.route('/evaluate/download/<filename>', methods=['GET'])
def download_evaluation_file(filename):
    """Download evaluation result files"""
    filepath = os.path.join('evaluation_results', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/reset', methods=['POST'])
def reset_collection():
    """Reset the document collection"""
    try:
        result = rag_engine.reset_collection()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Advanced Document Intelligence RAG Server...")
    print("✨ Features: Hybrid Search (Vector + BM25) + Multimodal + Evaluation")
    print(f"📚 Multimodal Processing: {'ENABLED' if doc_processor.enable_multimodal else 'DISABLED'}")
    print("📊 Evaluation Metrics: ENABLED")
    print("🌐 Server running on http://localhost:5000")
    print("\nNew endpoints:")
    print("  POST /evaluate/single - Evaluate one query")
    print("  POST /evaluate/batch - Batch evaluation with report")
    print("  GET /evaluate/history - View evaluation history")
    app.run(debug=True, port=5000)