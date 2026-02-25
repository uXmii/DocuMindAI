import React, { useState } from 'react';

const EvaluationDashboard = ({ apiBase }) => {
  const [questions, setQuestions] = useState('');
  const [evaluating, setEvaluating] = useState(false);
  const [report, setReport] = useState(null);
  const [showReport, setShowReport] = useState(false);

  const testQuestions = [
    "What percentage of commodity crops in the United States aren't harvested?",
    "Who is the author of American Wasteland?",
    "What is Pinker's Law of Conservation of Moralization?",
    "Why do people reject eating garbage?",
    "What did Parker say about cucumbers?"
  ];

  const loadSampleQuestions = () => {
    setQuestions(testQuestions.join('\n'));
  };

  const runEvaluation = async () => {
    const questionList = questions.split('\n').filter(q => q.trim());
    
    if (questionList.length === 0) {
      alert('Please enter at least one question');
      return;
    }

    setEvaluating(true);

    try {
      const response = await fetch(`${apiBase}/evaluate/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ questions: questionList })
      });

      const data = await response.json();
      
      if (data.success) {
        setReport(data.report);
        setShowReport(true);
      } else {
        alert(`Evaluation failed: ${data.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setEvaluating(false);
    }
  };

  const MetricCard = ({ title, value, subtitle, color = 'blue' }) => {
    const colorClasses = {
      blue: 'bg-blue-50 border-blue-200 text-blue-900',
      green: 'bg-green-50 border-green-200 text-green-900',
      purple: 'bg-purple-50 border-purple-200 text-purple-900',
      orange: 'bg-orange-50 border-orange-200 text-orange-900'
    };

    return (
      <div className={`p-4 rounded-lg border ${colorClasses[color]}`}>
        <div className="text-xs font-medium opacity-80 mb-1">{title}</div>
        <div className="text-2xl font-bold">{value}</div>
        {subtitle && <div className="text-xs opacity-70 mt-1">{subtitle}</div>}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
          <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <div>
          <h2 className="text-xl font-semibold text-slate-900">Evaluation Dashboard</h2>
          <p className="text-sm text-slate-600">Compare search method performance</p>
        </div>
      </div>

      {!showReport ? (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Test Questions (one per line)
            </label>
            <textarea
              value={questions}
              onChange={(e) => setQuestions(e.target.value)}
              placeholder="Enter questions to evaluate...&#10;One question per line"
              rows={10}
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-200 text-slate-900 resize-none"
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={loadSampleQuestions}
              className="flex-1 px-4 py-2.5 bg-white text-slate-700 rounded-lg hover:bg-slate-50 font-medium border border-slate-300 transition-colors"
            >
              Load Sample Questions
            </button>
            <button
              onClick={runEvaluation}
              disabled={evaluating || !questions.trim()}
              className="flex-1 px-4 py-2.5 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:from-slate-300 disabled:to-slate-300 disabled:cursor-not-allowed font-medium transition-colors"
            >
              {evaluating ? 'Evaluating...' : 'Run Evaluation'}
            </button>
          </div>

          {evaluating && (
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                </div>
                <span className="text-sm text-purple-900 font-medium">
                  Running evaluation across all search methods...
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-slate-900">Evaluation Results</h3>
            <button
              onClick={() => setShowReport(false)}
              className="px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 font-medium"
            >
              New Evaluation
            </button>
          </div>

          {/* Winner Banner */}
          <div className="p-6 bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-300 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl">🏆</span>
              <div>
                <div className="text-sm font-medium text-orange-700">Overall Winner</div>
                <div className="text-2xl font-bold text-orange-900 uppercase">
                  {report.winner?.best_overall_method || 'N/A'}
                </div>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div>
                <div className="text-xs text-orange-700">Win Rate</div>
                <div className="text-lg font-bold text-orange-900">
                  {((report.winner?.win_rate || 0) * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-orange-700">Fastest</div>
                <div className="text-lg font-bold text-orange-900 uppercase">
                  {report.winner?.fastest_method || 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-xs text-orange-700">Most Relevant</div>
                <div className="text-lg font-bold text-orange-900 uppercase">
                  {report.winner?.most_relevant_method || 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4">
            <MetricCard
              title="Total Queries"
              value={report.summary?.total_queries || 0}
              color="blue"
            />
            <MetricCard
              title="Successful Queries"
              value={report.summary?.successful_queries || 0}
              color="green"
            />
            <MetricCard
              title="Total Time"
              value={`${(report.summary?.total_time || 0).toFixed(1)}s`}
              color="purple"
            />
          </div>

          {/* Method Comparison */}
          <div className="space-y-4">
            <h4 className="font-semibold text-slate-900">Method Performance</h4>
            {report.method_comparison && Object.entries(report.method_comparison).map(([method, data]) => (
              <div key={method} className="border border-slate-200 rounded-lg p-4 bg-slate-50">
                <div className="flex justify-between items-center mb-3">
                  <h5 className="font-semibold text-slate-900 uppercase">{method}</h5>
                  <span className="text-xs text-slate-600">
                    Avg: {(data.avg_response_time * 1000).toFixed(0)}ms
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  {Object.entries(data.metrics || {}).map(([metric, values]) => (
                    <div key={metric} className="flex justify-between">
                      <span className="text-slate-600 capitalize">
                        {metric.replace(/_/g, ' ')}:
                      </span>
                      <span className="font-semibold text-slate-900">
                        {values.mean?.toFixed(3) || 'N/A'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Download Links */}
          {report.session_id && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-sm font-medium text-blue-900 mb-2">
                📊 Detailed reports generated
              </div>
              <div className="text-xs text-blue-700">
                JSON, Excel, and visualization files saved to evaluation_results/
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default EvaluationDashboard;
