import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np
import time

# ── Optional Gemini for generation ───────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Optional CLIP for multimodal retrieval ────────────────────────────────────
try:
    from multimodal_processor import CLIP_ENCODER
    CLIP_AVAILABLE = CLIP_ENCODER.available
except Exception:
    CLIP_ENCODER   = None
    CLIP_AVAILABLE = False

load_dotenv()

def _init_gemini_model(lib=None):
    """
    Find a working Gemini model WITHOUT burning quota.
    Uses list_models() which is free, then picks the best available flash model.
    Falls back to trying known names without probing if list_models fails.
    """
    import google.generativeai as _g
    if lib is None:
        lib = _g

    PREFERRED = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    # Try list_models() first — zero quota cost
    try:
        available = []
        for m in lib.list_models():
            if "generateContent" in m.supported_generation_methods:
                available.append(m.name.replace("models/", ""))
        # Pick best preferred model that's available
        for name in PREFERRED:
            if name in available:
                model = lib.GenerativeModel(name)
                print(f"   ✔  Gemini model selected: {name}")
                return model
        # If none of our preferred list matched, use first available
        if available:
            name = available[0]
            model = lib.GenerativeModel(name)
            print(f"   ✔  Gemini model selected (first available): {name}")
            return model
    except Exception as e:
        print(f"   ⚠  list_models() failed ({e}), trying known names without probe...")

    # Fallback: try names without live probe (no quota used)
    for name in PREFERRED:
        try:
            model = lib.GenerativeModel(name)
            print(f"   ✔  Gemini model assumed: {name} (unverified)")
            return model
        except Exception:
            continue

    print("   ✗  No Gemini model could be initialised")
    return None




class RAGEngine:
    """Advanced RAG engine with Hybrid Search (Vector + BM25)"""
    
    def __init__(self, collection_name: str = "documents"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.chroma_client.get_collection(name=collection_name)

        # Separate collection for CLIP vision embeddings (512-dim)
        # This enables true multimodal retrieval — query text → CLIP space → image chunks
        self._clip_collection = None
        if CLIP_AVAILABLE:
            clip_name = f"{collection_name}_clip"
            try:
                self._clip_collection = self.chroma_client.create_collection(
                    name=clip_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print("✓ CLIP vision collection created")
            except:
                try:
                    self._clip_collection = self.chroma_client.get_collection(clip_name)
                    print("✓ CLIP vision collection loaded")
                except:
                    self._clip_collection = None
        
        # BM25 index (will be built when documents are added)
        self.bm25_index = None
        self.documents_cache = []
        self.metadata_cache = []
        self.ids_cache = []
        
        # Hugging Face setup - Using more reliable model
        self.hf_api_key = os.getenv('HF_API_KEY')
        self.hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        self.hf_api_base = os.getenv("HF_API_BASE", "https://router.huggingface.co")
        self.hf_api_url  = f"{self.hf_api_base}/models/{self.hf_model}"

        # Gemini setup (preferred over HF — free tier, much better quality)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        if self.gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = _init_gemini_model()
            if self.gemini_model:
                print(f"✓ RAG generation: {self.gemini_model.model_name}")
        elif self.hf_api_key:
            print("✓ RAG generation: Hugging Face API")
        else:
            print("⚠  RAG generation: extractive fallback (set GEMINI_API_KEY for better answers)")
    
    def add_documents(self, chunks: List[Dict]) -> Dict:
        """Add chunks — deduplicates by ID so re-uploading the same PDF is safe."""
        try:
            # Deduplicate: skip IDs already in ChromaDB
            existing_ids = set(self.collection.get(include=[])['ids'])
            new_chunks   = [c for c in chunks if c['id'] not in existing_ids]

            skipped = len(chunks) - len(new_chunks)
            if skipped:
                print(f"i  Skipping {skipped} duplicate chunks (already indexed)")

            if new_chunks:
                texts      = [c['text']     for c in new_chunks]
                ids        = [c['id']       for c in new_chunks]
                metadatas  = [c['metadata'] for c in new_chunks]
                embeddings = self.embedding_model.encode(texts).tolist()
                self.collection.add(embeddings=embeddings, documents=texts,
                                    metadatas=metadatas, ids=ids)

                # CLIP embeddings for vision chunks
                if self._clip_collection and CLIP_AVAILABLE:
                    clip_ex = set(self._clip_collection.get(include=[])['ids'])
                    clip_ch = [c for c in new_chunks
                               if c.get('clip_embedding') and c['id'] not in clip_ex]
                    if clip_ch:
                        self._clip_collection.add(
                            embeddings=[c['clip_embedding'] for c in clip_ch],
                            documents =[c['text']           for c in clip_ch],
                            metadatas =[c['metadata']       for c in clip_ch],
                            ids       =[c['id']             for c in clip_ch],
                        )
                        print(f"v  Stored {len(clip_ch)} CLIP vision embeddings")

            self._rebuild_bm25_index()
            return {
                'success':            True,
                'chunks_added':       len(new_chunks),
                'collection_size':    self.collection.count(),
                'clip_indexed':       len([c for c in new_chunks if c.get('clip_embedding')]),
                'skipped_duplicates': skipped,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from all documents in collection"""
        try:
            # Get all documents from ChromaDB
            all_data = self.collection.get()
            
            if all_data['documents']:
                # Cache documents, metadata, and IDs
                self.documents_cache = all_data['documents']
                self.metadata_cache = all_data['metadatas']
                self.ids_cache = all_data['ids']
                
                # Tokenize documents for BM25
                tokenized_docs = [doc.lower().split() for doc in self.documents_cache]
                self.bm25_index = BM25Okapi(tokenized_docs)
                
                print(f"✓ BM25 index built with {len(self.documents_cache)} documents")
        except Exception as e:
            print(f"Error building BM25 index: {e}")
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean up garbled OCR text for better readability - AGGRESSIVE VERSION"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # CRITICAL: Remove repetitive single letters (I I I, etc.)
        text = re.sub(r'\b([A-Z])\s+\1(\s+\1)+\b', r'\1', text)  # "I I I" -> "I"
        text = re.sub(r'\b([A-Z])\s+\1\b', r'\1', text)  # "I I" -> "I"
        
        # Fix common OCR errors
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l to I
            r'\b0\b': 'O',  # zero to O in words
            r'[|\\]': 'I',  # pipes/slashes to I
            r'[\[\]{}]': '',  # remove brackets
            r'[~`]': "'",  # tildes to quotes
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove repeated words (word word -> word)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        # Remove lines that are mostly single letters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Count single letter words
            words = line.split()
            single_letters = sum(1 for w in words if len(w) == 1 and w.isalpha())
            # Keep line only if less than 50% are single letters
            if len(words) == 0 or single_letters / len(words) < 0.5:
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Remove non-printable characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        
        # Final cleanup: remove multiple spaces again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _generate_with_gemini(self, question: str, context: str) -> str:
        """Generate a synthesized answer using Gemini."""
        prompt = f"""You are a precise document assistant. Answer the question using ONLY the provided context.

Rules:
1. Give a clear, direct answer in 2-4 sentences.
2. If the context contains data from charts, diagrams or tables, extract and present the specific values.
3. Always cite: "Source: <filename>, Page <n>" at the end.
4. If the answer is not in the context, say "This information is not available in the document."

Context:
{context}

Question: {question}

Answer:"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            print(f"Gemini generation error: {err}")
            # Model name issue — try to reinitialise with a working model
            if "not found" in err.lower() or "404" in err:
                import google.generativeai as genai
                new_model = _init_gemini_model(genai)
                if new_model:
                    self.gemini_model = new_model
                    try:
                        return self.gemini_model.generate_content(prompt).text.strip()
                    except Exception as e2:
                        print(f"Gemini retry failed: {e2}")
            return None

    def _generate_with_hf(self, prompt: str, context_length: int = 500) -> str:
        """Generate text using Hugging Face Inference API with improved prompting"""
        try:
            if not self.hf_api_key:
                return None
            
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            # Enhanced prompt for better synthesis and relevance
            formatted_prompt = f"""You are a helpful assistant that provides clear, accurate answers based on provided context.

Context:
{prompt}

Instructions:
1. Answer the question using ONLY information from the context
2. Write in clear, professional language
3. Always cite the source and page number
4. If the context doesn't contain the answer, say so clearly
5. Use complete sentences and proper grammar

Provide a detailed, well-structured answer:"""
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 400,
                    "temperature": 0.5,
                    "top_p": 0.85,
                    "do_sample": True,
                    "repetition_penalty": 1.2
                }
            }
            
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    # Clean up the response
                    answer = generated.replace(formatted_prompt, "").strip()
                    if answer:
                        return answer
                elif isinstance(result, dict) and 'generated_text' in result:
                    return result['generated_text'].replace(formatted_prompt, "").strip()
                
                return None
            else:
                print(f"HF API error: {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"Error calling Hugging Face API: {str(e)}")
            return None

    def _build_extractive_answer(self, question: str, chunks: List[Dict]) -> str:
        """Build a clean extractive answer — used only when no LLM is available."""
        if not chunks:
            return "I couldn't find relevant information in the uploaded documents."

        import re

        question_words = set(
            w.lower().strip('.,!?;:()[]"\'') for w in question.split() if len(w) > 3
        )

        best_sentences = []
        sources_used   = []

        for chunk in chunks[:3]:
            text     = self._clean_ocr_text(chunk.get('text', ''))
            metadata = chunk.get('metadata', {})
            source   = metadata.get('source', 'document')
            page     = metadata.get('page', '?')

            if (source, page) not in sources_used:
                sources_used.append((source, page))

            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 40]

            for sent in sentences:
                sent_words = set(w.lower().strip('.,!?;:()[]"\'') for w in sent.split())
                overlap    = len(question_words & sent_words)
                if overlap > 0:
                    best_sentences.append((overlap, sent))

        best_sentences.sort(reverse=True, key=lambda x: x[0])
        top = [s for _, s in best_sentences[:4]]

        if not top:
            # Fall back to first 300 chars of best chunk
            excerpt = self._clean_ocr_text(chunks[0].get('text', ''))[:300].strip()
            top = [excerpt + ('…' if len(excerpt) == 300 else '')]

        answer = ' '.join(top)
        if not answer.endswith(('.', '!', '?')):
            answer += '.'

        # Citation
        primary = sources_used[0] if sources_used else ('document', '?')
        citation = f"\n\n**Source:** {primary[0]}, Page {primary[1]}"
        if len(sources_used) > 1:
            extra = ', '.join(f"p.{p}" for _, p in sources_used[1:3])
            citation += f" (also see {extra})"

        return answer + citation

    def _clip_search(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        True multimodal retrieval: encode the text question with CLIP,
        then search the CLIP image-embedding collection.
        This finds visually relevant pages (charts, diagrams, tables)
        that a text-only search would miss.
        """
        if not self._clip_collection or not CLIP_AVAILABLE:
            return []
        try:
            if self._clip_collection.count() == 0:
                return []
            clip_query = CLIP_ENCODER.encode_text(question)
            if not clip_query:
                return []
            actual_k = min(top_k, self._clip_collection.count())
            results  = self._clip_collection.query(
                query_embeddings=[clip_query],
                n_results=actual_k
            )
            chunks = []
            for i in range(len(results['ids'][0])):
                chunks.append({
                    'id':       results['ids'][0][i],
                    'text':     results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score':    1 - results['distances'][0][i],
                    'search_type': 'clip_multimodal',
                })
            return chunks
        except Exception as e:
            print(f"CLIP search error: {e}")
            return []

    def _vector_search(self, question: str, top_k: int = 10) -> List[Dict]:
        """Perform semantic vector search"""
        try:
            if self.collection.count() == 0:
                return []
            
            query_embedding = self.embedding_model.encode([question]).tolist()[0]
            
            actual_k = min(top_k, self.collection.count())
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k
            )
            
            chunks = []
            for i in range(len(results['ids'][0])):
                chunks.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i] if 'distances' in results else 0.5,
                    'search_type': 'vector'
                })
            
            return chunks
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def _bm25_search(self, question: str, top_k: int = 10) -> List[Dict]:
        """Perform keyword-based BM25 search - FIXED VERSION"""
        if not self.bm25_index or not self.documents_cache:
            print("BM25 index not available")
            return []
        
        try:
            # Tokenize query with minimal stopword removal
            tokens = question.lower().split()
            # Only remove true stopwords, keep question words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'by', 'from', 'at', 'as', 'it', 'that', 'this'}
            tokenized_query = [t for t in tokens if t and t not in stopwords]
            
            if not tokenized_query:
                tokenized_query = tokens  # Use all tokens if all are stopwords
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k indices
            n_docs = len(self.documents_cache)
            actual_top_k = min(top_k, n_docs)
            
            top_indices = np.argsort(scores)[::-1][:actual_top_k]
            
            chunks = []
            # FIXED: Only include documents with meaningful scores
            score_threshold = 0.1  # Minimum relevance threshold
            
            for idx in top_indices:
                if idx < n_docs:
                    score = float(scores[idx])
                    # Only add if score is above threshold OR it's in top 3 and score > 0
                    if score > score_threshold or (len(chunks) < 3 and score > 0):
                        chunks.append({
                            'id': self.ids_cache[idx],
                            'text': self.documents_cache[idx],
                            'metadata': self.metadata_cache[idx],
                            'score': score,
                            'search_type': 'bm25'
                        })
            
            print(f"BM25 found {len(chunks)} relevant results")
            return chunks
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion"""
        # Create score dictionaries
        rrf_scores = {}
        doc_map = {}
        
        # Add vector search scores
        for rank, result in enumerate(vector_results):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final ranked list
        combined_results = []
        for doc_id in sorted_ids:
            result = doc_map[doc_id].copy()
            result['score'] = rrf_scores[doc_id]
            result['search_type'] = 'hybrid'
            combined_results.append(result)
        
        return combined_results
    
    def query(self, question: str, top_k: int = 5, search_mode: str = 'hybrid') -> Dict:
        """Query the RAG system with specified search mode - FIXED VERSION"""
        try:
            start_time = time.time()
            
            # FIXED: Check if collection is empty upfront
            if self.collection.count() == 0:
                return {
                    'success': True,
                    'answer': 'No documents have been uploaded yet. Please upload a PDF document to get started.',
                    'sources': [],
                    'context_used': 0,
                    'search_mode': search_mode,
                    'search_time': 0,
                    'total_time': 0
                }
            
            # Perform searches based on mode
            if search_mode == 'vector':
                relevant_chunks = self._vector_search(question, top_k)
                # Always add CLIP results to augment with visual chunks
                clip_results = self._clip_search(question, top_k // 2)
                if clip_results:
                    relevant_chunks = self._reciprocal_rank_fusion(
                        relevant_chunks, clip_results
                    )[:top_k]
                search_time = time.time() - start_time
            elif search_mode == 'bm25':
                relevant_chunks = self._bm25_search(question, top_k)
                search_time = time.time() - start_time
            else:  # hybrid
                vector_results = self._vector_search(question, top_k * 2)
                bm25_results   = self._bm25_search(question, top_k * 2)
                clip_results   = self._clip_search(question, top_k)
                # Merge all three: text vector + keyword + visual CLIP
                if clip_results:
                    combined = self._reciprocal_rank_fusion(vector_results, bm25_results)
                    relevant_chunks = self._reciprocal_rank_fusion(
                        combined, clip_results
                    )[:top_k]
                else:
                    relevant_chunks = self._reciprocal_rank_fusion(
                        vector_results, bm25_results
                    )[:top_k]
                search_time = time.time() - start_time

            clip_used = any(
                c.get('search_type') == 'clip_multimodal' for c in relevant_chunks
            )
            
            # FIXED: Better handling when no relevant chunks found
            if not relevant_chunks:
                search_mode_names = {
                    'vector': 'semantic search',
                    'bm25': 'keyword search',
                    'hybrid': 'hybrid search'
                }
                mode_name = search_mode_names.get(search_mode, search_mode)
                
                return {
                    'success': True,
                    'answer': f'No relevant information found using {mode_name}. The uploaded documents may not contain information about "{question}". Try rephrasing your question or uploading additional documents.',
                    'sources': [],
                    'context_used': 0,
                    'search_mode': search_mode,
                    'search_time': search_time,
                    'total_time': time.time() - start_time
                }
            
            # Build context with cleaned text
            context = "\n\n".join([
                f"[Source: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}]\n{self._clean_ocr_text(chunk['text'])}"
                for chunk in relevant_chunks
            ])
            
            # Generate response — priority: Gemini > HF > extractive
            answer = None

            if self.gemini_model:
                answer = self._generate_with_gemini(question, context)
                if answer:
                    answer = self._clean_ocr_text(answer)

            if not answer and self.hf_api_key:
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nPlease provide a detailed answer based on the context above. Always cite the source and page number when referencing information."
                answer = self._generate_with_hf(prompt)
                if answer:
                    answer = self._clean_ocr_text(answer)

            # FIXED: Better extractive fallback
            if not answer:
                answer = self._build_extractive_answer(question, relevant_chunks)
            
            total_time = time.time() - start_time
            
            return {
                'success':      True,
                'answer':       answer,
                'sources':      relevant_chunks,
                'context_used': len(relevant_chunks),
                'search_mode':  search_mode,
                'search_time':  search_time,
                'total_time':   total_time,
                'clip_used':    clip_used,
            }
            
        except Exception as e:
            print(f"Query error: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': f'An error occurred while processing your question: {str(e)}',
                'search_mode': search_mode,
                'sources': [],
                'context_used': 0,
                'search_time': 0,
                'total_time': 0
            }
    
    def compare_search_modes(self, question: str, top_k: int = 5) -> Dict:
        """Compare all three search modes side by side"""
        vector_result = self.query(question, top_k, 'vector')
        bm25_result = self.query(question, top_k, 'bm25')
        hybrid_result = self.query(question, top_k, 'hybrid')
        
        return {
            'vector': vector_result,
            'bm25': bm25_result,
            'hybrid': hybrid_result,
            'question': question
        }
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            'total_chunks': self.collection.count(),
            'collection_name': self.collection.name,
            'bm25_indexed': self.bm25_index is not None
        }
    
    def reset_collection(self):
        """Reset the collection"""
        try:
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            if self._clip_collection:
                clip_name = self._clip_collection.name
                self.chroma_client.delete_collection(clip_name)
                self._clip_collection = self.chroma_client.create_collection(
                    name=clip_name,
                    metadata={"hnsw:space": "cosine"}
                )
            self.bm25_index      = None
            self.documents_cache = []
            self.metadata_cache  = []
            self.ids_cache       = []
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}