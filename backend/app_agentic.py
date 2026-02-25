"""
Flask App with Agentic RAG Support
PRODUCTION VERSION — agent and dashboard share one evaluator instance
so winner selection is always consistent.
"""

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
from agentic_rag import AgenticRAG

app = Flask(__name__)
CORS(app)

# ── Configuration ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# ── Shared components ─────────────────────────────────────────────────────────
doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200, enable_multimodal=True)
rag_engine    = RAGEngine(collection_name="documents")

# ONE evaluator shared by both the dashboard endpoints and the agent
evaluator     = RAGEvaluator(rag_engine)

# Pass the SAME evaluator into AgenticRAG so scoring is identical
agentic_rag   = AgenticRAG(rag_engine, evaluator=evaluator)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Health / stats ────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Advanced RAG system with Agentic capabilities',
        'features': {
            'multimodal':    doc_processor.enable_multimodal,
            'evaluation':    True,
            'agentic':       True,
            'langgraph':     True,
            'caching':       True,
            'unified_scorer': True   # agent + dashboard use same scorer
        }
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    stats = rag_engine.get_stats()
    stats['multimodal_enabled']  = doc_processor.enable_multimodal
    stats['evaluation_history']  = len(evaluator.results_history)
    stats['agentic_enabled']     = True
    cache_stats = agentic_rag.get_cache_stats()
    stats['cache'] = {
        'total_cached_queries': cache_stats['total_cached_queries'],
        'cache_enabled': True
    }
    return jsonify(stats)


# ── Upload ────────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        chunks = doc_processor.process_document(filepath)

        text_chunks  = sum(1 for c in chunks if c['metadata'].get('type') == 'text')
        ocr_chunks   = sum(1 for c in chunks if c['metadata'].get('type') == 'ocr')
        table_chunks = sum(1 for c in chunks if c['metadata'].get('type') == 'table')

        result = rag_engine.add_documents(chunks)
        os.remove(filepath)

        return jsonify({
            'success': True,
            'filename': filename,
            'chunks_created': len(chunks),
            'chunk_breakdown': {'text': text_chunks, 'ocr': ocr_chunks, 'tables': table_chunks},
            'collection_size': result['collection_size'],
            'multimodal_enabled': doc_processor.enable_multimodal
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Standard query ────────────────────────────────────────────────────────────

@app.route('/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400

    try:
        result = rag_engine.query(
            data['question'],
            top_k=data.get('top_k', 5),
            search_mode=data.get('search_mode', 'hybrid')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Agentic endpoints ─────────────────────────────────────────────────────────

@app.route('/query/agentic', methods=['POST'])
def query_agentic():
    """
    Intelligent agent query.
    Winner is selected using the SAME RAGEvaluator scoring as the dashboard,
    so the two will always agree.
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400

    try:
        result = agentic_rag.query(
            data['question'],
            max_iterations=data.get('max_iterations', 2)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/agentic/workflow-graph', methods=['GET'])
def get_workflow_graph():
    try:
        output_path = "evaluation_results/agentic_workflow.png"
        agentic_rag.visualize_workflow(output_path)
        if os.path.exists(output_path):
            return send_file(output_path, mimetype='image/png')
        return jsonify({'success': False, 'message': 'Could not generate visualization'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/agentic/cache/clear', methods=['POST'])
def clear_cache():
    try:
        agentic_rag.clear_cache()
        return jsonify({'success': True, 'message': 'Cache cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/agentic/cache/stats', methods=['GET'])
def get_cache_stats():
    try:
        return jsonify({'success': True, 'stats': agentic_rag.get_cache_stats()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Evaluation / compare endpoints ───────────────────────────────────────────

@app.route('/compare', methods=['POST'])
def compare_search_modes():
    """
    Compare all search modes.
    Uses the agent's evaluation cache so results are always consistent
    with what the agent would choose.
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400

    question = data['question']
    top_k    = data.get('top_k', 5)

    try:
        cache_key = agentic_rag._get_cache_key(question)

        if cache_key in agentic_rag.evaluation_cache:
            print(f"💾 Cache HIT for compare: {question[:50]}...")
            comparison_results = agentic_rag.evaluation_cache[cache_key]
            cache_hit = True
        else:
            print(f"🔬 Cache MISS – running fresh compare: {question[:50]}...")
            comparison_results = rag_engine.compare_search_modes(question, top_k=top_k)
            agentic_rag.evaluation_cache[cache_key] = comparison_results
            cache_hit = False

        comparison_results['cache_hit']  = cache_hit
        comparison_results['cache_key']  = cache_key[:8] + '...'
        return jsonify(comparison_results)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/evaluate/single', methods=['POST'])
def evaluate_single_query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400

    try:
        result = evaluator.evaluate_query(data['question'], data.get('ground_truth'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/evaluate/batch', methods=['POST'])
def evaluate_batch_queries():
    data = request.get_json()
    if not data or 'questions' not in data:
        return jsonify({'error': 'Questions list is required'}), 400

    try:
        report   = evaluator.batch_evaluate(
            data['questions'],
            data.get('ground_truths'),
            test_consistency=data.get('test_consistency', False)
        )
        filename = evaluator.save_report(report)
        viz_path = filename.replace('.json', '.png')
        evaluator.visualize_comparison(report, viz_path)
        excel_path = filename.replace('.json', '.xlsx')
        evaluator.export_to_excel(report, excel_path.split('/')[-1])

        return jsonify({
            'success': True,
            'report': report,
            'files': {'json': filename, 'visualization': viz_path, 'excel': excel_path}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/evaluate/consistency', methods=['POST'])
def evaluate_consistency():
    """
    Test how consistently the RAG system answers the same question.
    POST body: { "question": "...", "n_runs": 3, "mode": "hybrid" }
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'question is required'}), 400
    try:
        result = evaluator.evaluate_consistency(
            data['question'],
            n_runs=data.get('n_runs', 3),
            mode=data.get('mode', 'hybrid')
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/evaluate/history', methods=['GET'])
def get_evaluation_history():
    return jsonify({
        'total_evaluations': len(evaluator.results_history),
        'sessions':          len(evaluator.evaluation_sessions),
        'recent_results':    evaluator.results_history[-10:] if evaluator.results_history else []
    })


@app.route('/evaluate/download/<filename>', methods=['GET'])
def download_evaluation_file(filename):
    filepath = os.path.join('evaluation_results', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/reset', methods=['POST'])
def reset_collection():
    try:
        result = rag_engine.reset_collection()
        agentic_rag.clear_cache()
        result['cache_cleared'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 INTELLIGENT AGENTIC RAG SYSTEM")
    print("="*80)
    print(f"   Multimodal : {'ENABLED' if doc_processor.enable_multimodal else 'DISABLED'}")
    print( "   Agentic    : ENABLED")
    print( "   Caching    : ENABLED")
    print( "   Scorer     : UNIFIED (agent + dashboard share one RAGEvaluator)")
    print("\n   Endpoints:")
    print("     POST /query              – standard query")
    print("     POST /query/agentic      – agent query (consistent winner)")
    print("     POST /compare            – compare modes (cached)")
    print("     POST /evaluate/single    – evaluate one query")
    print("     POST /evaluate/batch     – batch evaluation")
    print("     GET  /agentic/cache/stats")
    print("     POST /agentic/cache/clear")
    print("\n🌐 http://localhost:5000")
    print("="*80 + "\n")
    app.run(debug=True, port=5000)