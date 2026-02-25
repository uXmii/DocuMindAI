"""
Standalone script to run RAG evaluations
Upload your PDF first, then run this script
"""

import requests
import json
from datetime import datetime

API_BASE = "http://localhost:5000"

# TEST QUESTIONS FOR FOOD WASTE PDF
test_questions = [
    "What percentage of commodity crops in the United States aren't harvested?",
    "Who is the author of American Wasteland?",
    "What is Pinker's Law of Conservation of Moralization?",
    "Why do people reject eating garbage?",
    "What did Parker say about cucumbers?",
    "What is dumpster diving?",
    "How much food could be recovered to feed people?",
    "What are the ethical concerns about food waste?",
    "What is the Fourth Amendment?",
    "What is core disgust according to Rozin?",
    "How does the framing effect relate to food waste?",
    "What happens to blemished fruits and vegetables?",
    "What is the Tragedy of the Commons?",
    "How do expiration dates contribute to food waste?",
    "What is Modified Atmosphere Packaging?",
    "What are food taboos?",
    "How does food waste at farms compare to consumer waste?",
    "What is moral rationalization?",
    "What did the USDA say about food recovery?",
    "What is interpersonal disgust?"
]

# Optional: Add ground truth answers for more accurate evaluation
ground_truths = [
    "9 percent of commodity crops aren't harvested",
    "Jonathan Bloom",
    "As old behaviors are removed from the moralized column, new ones are added",
    "People reject eating garbage due to disgust, purity concerns, and moral rationalization",
    "Cucumbers can be too curved, which hinders box packing and supermarket stacking",
    # Add more ground truths as needed...
]

def run_single_evaluation(question):
    """Test single question evaluation"""
    print(f"\n🔍 Evaluating: {question}")
    
    response = requests.post(
        f"{API_BASE}/evaluate/single",
        json={'question': question}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        
        # Show quick summary
        for method in ['vector', 'bm25', 'hybrid']:
            data = result['methods'][method]
            if data.get('success'):
                print(f"  {method.upper()}: {data['response_time']*1000:.0f}ms, "
                      f"{data['sources_count']} sources, "
                      f"relevance: {data['avg_relevance_score']:.3f}")
        
        return result
    else:
        print(f"❌ Error: {response.text}")
        return None

def run_batch_evaluation(questions, ground_truths=None):
    """Test batch evaluation"""
    print(f"\n📊 Running batch evaluation of {len(questions)} questions...")
    print("This will generate a full report with visualizations.\n")
    
    response = requests.post(
        f"{API_BASE}/evaluate/batch",
        json={
            'questions': questions,
            'ground_truths': ground_truths
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Batch evaluation complete!")
        
        report = result['report']
        
        # Print summary
        print(f"\n📋 EVALUATION SUMMARY:")
        print(f"Total queries: {report['summary']['total_queries']}")
        print(f"Total time: {report['summary']['total_time']:.2f}s")
        
        print(f"\n🏆 WINNERS:")
        winners = report['winner']
        print(f"  Fastest: {winners['fastest']['method'].upper()} ({winners['fastest']['time']*1000:.0f}ms)")
        print(f"  Most Relevant: {winners['most_relevant']['method'].upper()} (score: {winners['most_relevant']['score']:.3f})")
        print(f"  🎯 OVERALL: {winners['overall']['method'].upper()}")
        
        print(f"\n📁 Files generated:")
        for file_type, filepath in result['files'].items():
            print(f"  {file_type}: {filepath}")
        
        return result
    else:
        print(f"❌ Error: {response.text}")
        return None

def check_system_status():
    """Check if system is ready"""
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health = response.json()
            print("✓ System is healthy")
            print(f"  Multimodal: {health['features'].get('multimodal', False)}")
            print(f"  Evaluation: {health['features'].get('evaluation', False)}")
            return True
        else:
            print("❌ System health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on http://localhost:5000?")
        return False

def check_documents_loaded():
    """Check if documents are loaded"""
    try:
        response = requests.get(f"{API_BASE}/stats")
        if response.status_code == 200:
            stats = response.json()
            chunk_count = stats.get('total_chunks', 0)
            print(f"✓ Documents loaded: {chunk_count} chunks")
            if chunk_count == 0:
                print("⚠ WARNING: No documents loaded! Upload a PDF first.")
                return False
            return True
        return False
    except:
        return False

def main():
    """Main evaluation runner"""
    print("="*80)
    print(" 🎯 RAG EVALUATION SYSTEM")
    print("="*80)
    
    # Check system
    if not check_system_status():
        return
    
    if not check_documents_loaded():
        print("\n❌ Please upload a PDF document first!")
        print("   1. Go to http://localhost:3000")
        print("   2. Upload your PDF")
        print("   3. Run this script again")
        return
    
    print("\n" + "="*80)
    print(" Choose evaluation type:")
    print("="*80)
    print(" 1. Single question (quick test)")
    print(" 2. Batch evaluation (20 questions, full report)")
    print(" 3. Custom questions")
    print("="*80)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Single question test
        question = input("\nEnter your question: ").strip()
        if question:
            run_single_evaluation(question)
        else:
            print("Using default question...")
            run_single_evaluation(test_questions[0])
    
    elif choice == '2':
        # Batch evaluation
        confirm = input(f"\nThis will evaluate {len(test_questions)} questions. Continue? (y/n): ")
        if confirm.lower() == 'y':
            run_batch_evaluation(test_questions, ground_truths)
    
    elif choice == '3':
        # Custom questions
        print("\nEnter questions (one per line, empty line to finish):")
        custom_questions = []
        while True:
            q = input(f"Question {len(custom_questions)+1}: ").strip()
            if not q:
                break
            custom_questions.append(q)
        
        if custom_questions:
            run_batch_evaluation(custom_questions)
        else:
            print("No questions entered.")
    
    else:
        print("Invalid choice.")
    
    print("\n" + "="*80)
    print(" Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()