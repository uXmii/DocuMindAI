import React, { useState, useEffect } from 'react';
import EvaluationDashboard from './components/EvaluationDashboard';

const API_BASE = 'http://localhost:5000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [stats, setStats] = useState(null);
  const [showSources, setShowSources] = useState(false);
  const [currentSources, setCurrentSources] = useState([]);
  const [searchMode, setSearchMode] = useState('hybrid');
  const [showComparison, setShowComparison] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'evaluation'

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      
      if (data.success) {
        const multimodalInfo = data.multimodal_enabled && data.chunk_breakdown 
          ? ` (${data.chunk_breakdown.text} text, ${data.chunk_breakdown.ocr} OCR, ${data.chunk_breakdown.tables} tables)`
          : '';
        
        setMessages(prev => [...prev, {
          type: 'system',
          content: `Document "${data.filename}" processed successfully. ${data.chunks_created} chunks indexed${multimodalInfo}. Total collection: ${data.collection_size} chunks.`
        }]);
        fetchStats();
      } else {
        setMessages(prev => [...prev, {
          type: 'error',
          content: `Upload failed: ${data.error}`
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Error uploading file: ${error.message}`
      }]);
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleQuery = async () => {
    if (!input.trim() || querying) return;

    const question = input.trim();
    setInput('');
    setQuerying(true);

    setMessages(prev => [...prev, {
      type: 'user',
      content: question
    }]);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5, search_mode: searchMode })
      });
      const data = await response.json();

      if (data.success) {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: data.answer,
          sources: data.sources,
          contextUsed: data.context_used,
          searchMode: data.search_mode,
          searchTime: data.search_time,
          totalTime: data.total_time
        }]);
      } else {
        setMessages(prev => [...prev, {
          type: 'error',
          content: `Query failed: ${data.error}`
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Error querying: ${error.message}`
      }]);
    } finally {
      setQuerying(false);
    }
  };

  const handleCompare = async () => {
    if (!input.trim()) return;

    const question = input.trim();
    setQuerying(true);

    try {
      const response = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5 })
      });
      const data = await response.json();
      
      setComparisonData(data);
      setShowComparison(true);
    } catch (error) {
      console.error('Error comparing:', error);
    } finally {
      setQuerying(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Reset all documents? This action cannot be undone.')) return;

    try {
      const response = await fetch(`${API_BASE}/reset`, { method: 'POST' });
      const data = await response.json();
      
      if (data.success) {
        setMessages([{
          type: 'system',
          content: 'Collection reset successfully.'
        }]);
        fetchStats();
      }
    } catch (error) {
      console.error('Error resetting:', error);
    }
  };

  const showSourcesModal = (sources) => {
    setCurrentSources(sources);
    setShowSources(true);
  };

  const getChunkTypeIcon = (type) => {
    switch(type) {
      case 'ocr': return '📸';
      case 'table': return '📊';
      default: return '📄';
    }
  };

  const getChunkTypeBadge = (type) => {
    const badges = {
      ocr: { label: 'OCR', color: 'bg-purple-100 text-purple-700 border-purple-300' },
      table: { label: 'Table', color: 'bg-blue-100 text-blue-700 border-blue-300' },
      text: { label: 'Text', color: 'bg-slate-100 text-slate-700 border-slate-300' }
    };
    
    const badge = badges[type] || badges.text;
    
    return (
      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${badge.color}`}>
        {getChunkTypeIcon(type)} {badge.label}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-gradient-to-br from-slate-700 to-slate-900 rounded-lg flex items-center justify-center shadow-md">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
                  DocuMind AI
                </h1>
                <p className="text-sm text-slate-500 font-medium">
                  Advanced Multimodal RAG with Evaluation
                </p>
              </div>
            </div>
            {stats && (
              <div className="flex gap-3">
                <div className="px-4 py-2 bg-slate-100 rounded-lg border border-slate-200">
                  <div className="text-xs text-slate-500 font-medium">Documents</div>
                  <div className="text-lg font-semibold text-slate-900">{stats.total_chunks}</div>
                </div>
                {stats.multimodal_enabled && (
                  <div className="px-4 py-2 bg-purple-50 rounded-lg border border-purple-200">
                    <div className="text-xs text-purple-600 font-medium">Multimodal</div>
                    <div className="text-lg font-semibold text-purple-700">ON</div>
                  </div>
                )}
                {stats.bm25_indexed && (
                  <div className="px-4 py-2 bg-emerald-50 rounded-lg border border-emerald-200">
                    <div className="text-xs text-emerald-600 font-medium">Status</div>
                    <div className="text-lg font-semibold text-emerald-700">Active</div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 mt-4">
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                activeTab === 'chat'
                  ? 'bg-slate-900 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200'
              }`}
            >
              💬 Chat
            </button>
            <button
              onClick={() => setActiveTab('evaluation')}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-colors ${
                activeTab === 'evaluation'
                  ? 'bg-slate-900 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200'
              }`}
            >
              📊 Evaluation
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3">
            {activeTab === 'chat' ? (
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="border-b border-slate-200 bg-slate-50 px-6 py-4">
                  <div className="flex gap-2">
                    {[
                      { mode: 'hybrid', label: 'Hybrid', icon: '⚡' },
                      { mode: 'vector', label: 'Semantic', icon: '🔍' },
                      { mode: 'bm25', label: 'Keyword', icon: '📝' }
                    ].map(({ mode, label, icon }) => (
                      <button
                        key={mode}
                        onClick={() => setSearchMode(mode)}
                        className={`flex-1 px-4 py-2.5 rounded-lg font-medium text-sm transition-all ${
                          searchMode === mode
                            ? 'bg-slate-900 text-white shadow-md'
                            : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                        }`}
                      >
                        <span className="mr-1.5">{icon}</span>
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="h-[600px] flex flex-col">
                  <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.length === 0 ? (
                      <div className="h-full flex items-center justify-center">
                        <div className="text-center max-w-2xl">
                          <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                            <svg className="w-10 h-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                            </svg>
                          </div>
                          <h2 className="text-2xl font-semibold text-slate-900 mb-3">
                            Welcome to DocuMind AI
                          </h2>
                          <p className="text-slate-600 mb-8 leading-relaxed">
                            Upload your documents and leverage advanced hybrid search with OCR and table extraction.
                          </p>
                          <div className="grid grid-cols-3 gap-4 text-left">
                            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                              <div className="text-slate-900 font-semibold mb-1">Multimodal</div>
                              <div className="text-sm text-slate-500">Extract text from images & tables</div>
                            </div>
                            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                              <div className="text-slate-900 font-semibold mb-1">Hybrid Search</div>
                              <div className="text-sm text-slate-500">Semantic + Keyword fusion</div>
                            </div>
                            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                              <div className="text-slate-900 font-semibold mb-1">Evaluation</div>
                              <div className="text-sm text-slate-500">Compare method performance</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      messages.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[85%] rounded-lg p-4 ${
                            msg.type === 'user' 
                              ? 'bg-slate-900 text-white' 
                              : msg.type === 'assistant'
                              ? 'bg-slate-50 text-slate-900 border border-slate-200'
                              : msg.type === 'system'
                              ? 'bg-blue-50 text-blue-900 border border-blue-200'
                              : 'bg-red-50 text-red-900 border border-red-200'
                          }`}>
                            <div className="leading-relaxed">{msg.content}</div>
                            {msg.searchMode && (
                              <div className="mt-3 flex gap-2 items-center flex-wrap">
                                <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-white text-slate-700 border border-slate-200">
                                  {msg.searchMode === 'hybrid' ? '⚡ Hybrid' : msg.searchMode === 'vector' ? '🔍 Semantic' : '📝 Keyword'}
                                </span>
                                {msg.searchTime && (
                                  <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-white text-slate-700 border border-slate-200">
                                    {(msg.searchTime * 1000).toFixed(0)}ms
                                  </span>
                                )}
                                {msg.contextUsed && (
                                  <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-white text-slate-700 border border-slate-200">
                                    {msg.contextUsed} sources
                                  </span>
                                )}
                              </div>
                            )}
                            {msg.sources && msg.sources.length > 0 && (
                              <button 
                                onClick={() => showSourcesModal(msg.sources)}
                                className="mt-3 text-sm font-medium text-slate-900 hover:text-slate-700 underline underline-offset-2"
                              >
                                View {msg.sources.length} sources →
                              </button>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                    {querying && (
                      <div className="flex justify-start">
                        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                          <div className="flex space-x-1.5">
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                            <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="border-t border-slate-200 bg-white p-6 space-y-3">
                    <div className="flex gap-3">
                      <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask a question about your documents..."
                        disabled={querying || stats?.total_chunks === 0}
                        className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:border-slate-500 focus:ring-2 focus:ring-slate-200 disabled:bg-slate-50 disabled:text-slate-400 text-slate-900"
                      />
                      <button 
                        onClick={handleQuery}
                        disabled={querying || !input.trim() || stats?.total_chunks === 0}
                        className="px-6 py-3 bg-slate-900 text-white rounded-lg hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed font-medium transition-colors"
                      >
                        Send
                      </button>
                    </div>
                    <button
                      onClick={handleCompare}
                      disabled={querying || !input.trim() || stats?.total_chunks === 0}
                      className="w-full px-4 py-2.5 bg-white text-slate-700 rounded-lg hover:bg-slate-50 disabled:bg-slate-100 disabled:text-slate-400 font-medium border border-slate-300 transition-colors text-sm"
                    >
                      Compare Search Methods
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <EvaluationDashboard apiBase={API_BASE} />
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h3 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">Upload Documents</h3>
              <label className="block">
                <div className="w-full px-4 py-3 bg-slate-900 text-white rounded-lg hover:bg-slate-800 cursor-pointer text-center font-medium transition-colors text-sm">
                  {uploading ? 'Processing...' : 'Choose File'}
                </div>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileUpload}
                  disabled={uploading}
                  className="hidden"
                />
              </label>
              <p className="text-xs text-slate-500 mt-3">
                PDF files only, up to 16MB
                {stats?.multimodal_enabled && <span className="block mt-1 text-purple-600">✓ Multimodal processing enabled</span>}
              </p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h3 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">System Actions</h3>
              <div className="space-y-2">
                <button 
                  onClick={handleReset}
                  className="w-full px-4 py-2.5 bg-white text-red-600 rounded-lg hover:bg-red-50 font-medium border border-red-300 transition-colors text-sm"
                >
                  Reset Collection
                </button>
                <button 
                  onClick={fetchStats}
                  className="w-full px-4 py-2.5 bg-white text-slate-700 rounded-lg hover:bg-slate-50 font-medium border border-slate-300 transition-colors text-sm"
                >
                  Refresh Statistics
                </button>
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <h3 className="text-sm font-semibold text-slate-900 mb-4 uppercase tracking-wide">Features</h3>
              <div className="space-y-3 text-sm">
                <div className="pb-3 border-b border-slate-100">
                  <div className="font-medium text-slate-900 mb-1 flex items-center gap-2">
                    📸 Multimodal Processing
                  </div>
                  <div className="text-slate-500 text-xs leading-relaxed">
                    OCR text extraction from images
                  </div>
                </div>
                <div className="pb-3 border-b border-slate-100">
                  <div className="font-medium text-slate-900 mb-1 flex items-center gap-2">
                    📊 Table Extraction
                  </div>
                  <div className="text-slate-500 text-xs leading-relaxed">
                    Automatic table detection and parsing
                  </div>
                </div>
                <div className="pb-3 border-b border-slate-100">
                  <div className="font-medium text-slate-900 mb-1 flex items-center gap-2">
                    ⚡ Hybrid Search
                  </div>
                  <div className="text-slate-500 text-xs leading-relaxed">
                    Vector + BM25 with rank fusion
                  </div>
                </div>
                <div className="pb-3">
                  <div className="font-medium text-slate-900 mb-1 flex items-center gap-2">
                    📈 Evaluation Metrics
                  </div>
                  <div className="text-slate-500 text-xs leading-relaxed">
                    Compare search method performance
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {showComparison && comparisonData && (
        <div className="fixed inset-0 bg-slate-900 bg-opacity-50 backdrop-blur-sm flex items-center justify-center p-4 z-50" onClick={() => setShowComparison(false)}>
          <div className="bg-white rounded-xl max-w-7xl w-full max-h-[90vh] overflow-hidden shadow-2xl border border-slate-200" onClick={(e) => e.stopPropagation()}>
            <div className="p-6 border-b border-slate-200 bg-slate-50">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-xl font-semibold text-slate-900">Search Method Comparison</h2>
                  <p className="text-sm text-slate-600 mt-1">{comparisonData.question}</p>
                </div>
                <button onClick={() => setShowComparison(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
              </div>
            </div>
            <div className="p-6 overflow-y-auto max-h-[75vh]">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  { mode: 'vector', label: 'Semantic', icon: '🔍' },
                  { mode: 'bm25', label: 'Keyword', icon: '📝' },
                  { mode: 'hybrid', label: 'Hybrid', icon: '⚡' }
                ].map(({ mode, label, icon }) => (
                  <div key={mode} className="border border-slate-200 rounded-lg overflow-hidden bg-slate-50">
                    <div className="bg-slate-900 text-white px-4 py-3 text-center font-medium">
                      <span className="mr-2">{icon}</span>
                      {label}
                    </div>
                    <div className="p-4 space-y-3">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-500">Processing Time</span>
                        <span className="font-semibold text-slate-900">{(comparisonData[mode].search_time * 1000).toFixed(0)}ms</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-500">Sources Found</span>
                        <span className="font-semibold text-slate-900">{comparisonData[mode].sources?.length || 0}</span>
                      </div>
                      <div className="pt-3 border-t border-slate-200">
                        <div className="text-sm text-slate-700 leading-relaxed max-h-64 overflow-y-auto">
                          {comparisonData[mode].answer}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {showSources && (
        <div className="fixed inset-0 bg-slate-900 bg-opacity-50 backdrop-blur-sm flex items-center justify-center p-4 z-50" onClick={() => setShowSources(false)}>
          <div className="bg-white rounded-xl max-w-4xl w-full max-h-[85vh] overflow-hidden shadow-2xl border border-slate-200" onClick={(e) => e.stopPropagation()}>
            <div className="p-6 border-b border-slate-200 bg-slate-50 flex justify-between items-center">
              <h2 className="text-xl font-semibold text-slate-900">Source Documents</h2>
              <button onClick={() => setShowSources(false)} className="text-slate-400 hover:text-slate-600 text-2xl">×</button>
            </div>
            <div className="p-6 overflow-y-auto max-h-[70vh] space-y-4">
              {currentSources.map((source, idx) => (
                <div key={idx} className="border border-slate-200 rounded-lg p-5 bg-slate-50">
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-2">
                      <span className="px-3 py-1 bg-slate-900 text-white text-xs font-medium rounded-md">
                        Source {idx + 1}
                      </span>
                      {source.metadata?.type && getChunkTypeBadge(source.metadata.type)}
                    </div>
                    <span className="text-xs text-slate-600">
                      {source.metadata.source} • Page {source.metadata.page}
                    </span>
                  </div>
                  <div className="text-sm text-slate-700 leading-relaxed bg-white p-4 rounded border border-slate-200">
                    {source.text}
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-slate-900 rounded-full"
                        style={{width: `${(source.score || 0.5) * 100}%`}}
                      />
                    </div>
                    <span className="text-xs font-medium text-slate-600">
                      {((source.score || 0.5) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
