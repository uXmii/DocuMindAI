import PyPDF2
from typing import List, Dict
import hashlib
import re
from multimodal_processor import MultimodalProcessor

class DocumentProcessor:
    """Advanced document processor with multimodal support"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, enable_multimodal: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_multimodal = enable_multimodal
        
        if self.enable_multimodal:
            self.multimodal_processor = MultimodalProcessor()
            print("✓ Multimodal processing ENABLED (OCR + Table extraction)")
        else:
            self.multimodal_processor = None
            print("ℹ Text-only processing enabled")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text with page-level granularity"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_data = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'char_count': len(text)
                    })
                
                return {
                    'filename': pdf_path.split('\\')[-1].split('/')[-1],
                    'total_pages': len(pdf_reader.pages),
                    'pages': pages_data
                }
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def create_smart_chunks(self, document_data: Dict) -> List[Dict]:
        """Create overlapping chunks with rich metadata"""
        chunks = []
        chunk_id = 0
        
        for page_data in document_data['pages']:
            text = page_data['text']
            page_num = page_data['page_number']
            
            # Skip empty pages
            if not text.strip():
                continue
            
            # Split by sentences for better semantic chunks
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            current_sentences = []
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.chunk_size:
                    current_chunk += sentence + " "
                    current_sentences.append(sentence)
                else:
                    if current_chunk:
                        # Create chunk with metadata
                        chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                        chunks.append({
                            'id': f"text_chunk_{chunk_id}_{chunk_hash}",
                            'text': current_chunk.strip(),
                            'metadata': {
                                'source': document_data['filename'],
                                'page': page_num,
                                'chunk_index': chunk_id,
                                'sentence_count': len(current_sentences),
                                'char_count': len(current_chunk),
                                'type': 'text'
                            }
                        })
                        chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = " ".join(current_sentences[-2:]) if len(current_sentences) > 2 else ""
                    current_chunk = overlap_text + " " + sentence + " "
                    current_sentences = current_sentences[-2:] + [sentence] if len(current_sentences) > 2 else [sentence]
            
            # Add final chunk
            if current_chunk.strip():
                chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                chunks.append({
                    'id': f"text_chunk_{chunk_id}_{chunk_hash}",
                    'text': current_chunk.strip(),
                    'metadata': {
                        'source': document_data['filename'],
                        'page': page_num,
                        'chunk_index': chunk_id,
                        'sentence_count': len(current_sentences),
                        'char_count': len(current_chunk),
                        'type': 'text'
                    }
                })
                chunk_id += 1
        
        return chunks
    
    def process_document(self, pdf_path: str) -> List[Dict]:
        """
        Main processing pipeline
        Now includes multimodal processing if enabled
        """
        # Extract text (always)
        print("📄 Extracting text from PDF...")
        document_data = self.extract_text_from_pdf(pdf_path)
        text_chunks = self.create_smart_chunks(document_data)
        print(f"✓ Created {len(text_chunks)} text chunks")
        
        all_chunks = text_chunks
        
        # Process multimodal content if enabled
        if self.enable_multimodal and self.multimodal_processor:
            try:
                print("\n🎨 Starting multimodal processing...")
                multimodal_data = self.multimodal_processor.process_multimodal_document(pdf_path)
                
                # Add OCR chunks
                all_chunks.extend(multimodal_data['ocr_chunks'])
                
                # Add table chunks
                all_chunks.extend(multimodal_data['table_chunks'])
                
                print(f"✓ Total chunks (text + OCR + tables): {len(all_chunks)}")
                
            except Exception as e:
                print(f"⚠ Multimodal processing error: {e}")
                print("  Continuing with text-only chunks...")
        
        return all_chunks