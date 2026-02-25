"""
Agent State Definition for Agentic RAG
Defines the state that flows through the LangGraph workflow
"""

from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict):
    """State that flows through the agent workflow"""
    
    # Input
    question: str
    original_question: str
    
    # Search strategy
    search_mode: str  # 'vector', 'bm25', 'hybrid'
    query_type: str  # 'factual', 'conceptual', 'comparative', 'complex'
    
    # Retrieved information
    retrieved_docs: List[Dict]
    search_results: Dict
    
    # Generated answer
    answer: str
    answer_attempts: List[str]  # REMOVED operator.add
    
    # Quality metrics
    quality_score: float
    has_sufficient_sources: bool
    answer_is_complete: bool
    
    # Agent reasoning
    agent_thoughts: List[str]  # REMOVED operator.add
    tools_used: List[str]  # REMOVED operator.add
    
    # Control flow
    iteration: int
    max_iterations: int
    needs_refinement: bool
    should_use_web: bool
    
    # Results
    final_answer: str
    confidence: float
    workflow_path: List[str]
    
    # Metadata
    total_time: float
    search_times: Dict[str, float]


class QueryAnalysis(TypedDict):
    """Result of query analysis"""
    query_type: str
    recommended_mode: str
    complexity: str  # 'simple', 'medium', 'complex'
    requires_comparison: bool
    requires_calculation: bool
    key_terms: List[str]
    reasoning: str


class QualityAssessment(TypedDict):
    """Result of answer quality assessment"""
    score: float
    has_sources: bool
    answer_length_ok: bool
    has_citations: bool
    is_relevant: bool
    is_complete: bool
    issues: List[str]
    suggestions: List[str]