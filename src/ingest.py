import os
import pytesseract
from pdfminer.high_level import extract_text
from PIL import Image
from typing import List, Dict, Any
import logging
from src.exceptions import IngestionError
from src.utils import chunk_text_with_overlap
from src.config import Config

logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self, config: Config):
        self.config = config
        self.supported_code_extensions = {".py", ".js", ".java", ".cpp", ".html", ".md", ".txt", ".json", ".yaml", ".yml"}
        self.supported_image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def _safe_read_file(self, filepath: str, encoding: str = 'utf-8') -> str:
        """Safely read file with fallback encodings."""
        encodings = [encoding, 'utf-8', 'latin-1', 'cp1252']
        
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise IngestionError(f"Could not decode file {filepath} with any supported encoding")

    def read_code(self) -> List[Dict[str, Any]]:
        """Read and chunk code files."""
        texts = []
        if not os.path.exists(self.config.CODE_DIR):
            logger.warning(f"Code directory {self.config.CODE_DIR} does not exist")
            return texts

        for file in os.listdir(self.config.CODE_DIR):
            file_path = os.path.join(self.config.CODE_DIR, file)
            if not os.path.isfile(file_path):
                continue
                
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in self.supported_code_extensions:
                try:
                    content = self._safe_read_file(file_path)
                    chunks = chunk_text_with_overlap(
                        content, 
                        self.config.CHUNK_SIZE, 
                        self.config.CHUNK_OVERLAP
                    )
                    
                    for idx, chunk in enumerate(chunks):
                        texts.append({
                            "text": chunk,
                            "doc_id": file,
                            "chunk_id": f"code_{idx}",
                            "doc_type": "code",
                            "file_extension": file_ext
                        })
                except Exception as e:
                    logger.error(f"Error reading code file {file}: {e}")
                    
        return texts

    def read_policies(self) -> List[Dict[str, Any]]:
        """Read and chunk policy PDF files."""
        texts = []
        if not os.path.exists(self.config.POLICY_DIR):
            logger.warning(f"Policy directory {self.config.POLICY_DIR} does not exist")
            return texts

        for file in os.listdir(self.config.POLICY_DIR):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.config.POLICY_DIR, file)
                try:
                    policy_text = extract_text(file_path)
                    if not policy_text.strip():
                        logger.warning(f"No text extracted from {file}")
                        continue
                        
                    chunks = chunk_text_with_overlap(
                        policy_text,
                        self.config.CHUNK_SIZE,
                        self.config.CHUNK_OVERLAP
                    )
                    
                    for idx, chunk in enumerate(chunks):
                        texts.append({
                            "text": chunk,
                            "doc_id": file,
                            "chunk_id": f"policy_{idx}",
                            "doc_type": "policy"
                        })
                except Exception as e:
                    logger.error(f"Error reading policy file {file}: {e}")
                    
        return texts

    def read_product_specs(self) -> List[Dict[str, Any]]:
        """Read and chunk product specification files."""
        texts = []
        if not os.path.exists(self.config.SPECS_DIR):
            logger.warning(f"Specs directory {self.config.SPECS_DIR} does not exist")
            return texts

        for file in os.listdir(self.config.SPECS_DIR):
            file_path = os.path.join(self.config.SPECS_DIR, file)
            if not os.path.isfile(file_path):
                continue
                
            try:
                content = self._safe_read_file(file_path)
                chunks = chunk_text_with_overlap(
                    content,
                    self.config.CHUNK_SIZE,
                    self.config.CHUNK_OVERLAP
                )
                
                for idx, chunk in enumerate(chunks):
                    texts.append({
                        "text": chunk,
                        "doc_id": file,
                        "chunk_id": f"spec_{idx}",
                        "doc_type": "product_spec"
                    })
            except Exception as e:
                logger.error(f"Error reading spec file {file}: {e}")
                
        return texts

    def read_screenshots(self) -> List[Dict[str, Any]]:
        """Read and process screenshot images with OCR."""
        texts = []
        if not os.path.exists(self.config.SCREENSHOTS_DIR):
            logger.warning(f"Screenshots directory {self.config.SCREENSHOTS_DIR} does not exist")
            return texts

        for file in os.listdir(self.config.SCREENSHOTS_DIR):
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in self.supported_image_extensions:
                file_path = os.path.join(self.config.SCREENSHOTS_DIR, file)
                try:
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img)
                    if text.strip():  # Only add if OCR extracted text
                        texts.append({
                            "text": text,
                            "doc_id": file,
                            "chunk_id": "screenshot_0",
                            "doc_type": "screenshot"
                        })
                except Exception as e:
                    logger.error(f"Error processing screenshot {file}: {e}")
                    
        return texts

    def ingest_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Ingest all document types."""
        try:
            result = {
                "policies": self.read_policies(),
                "product_specs": self.read_product_specs(),
                "screenshots": self.read_screenshots(),
                "code": self.read_code()
            }
            
            total_chunks = sum(len(v) for v in result.values())
            logger.info(f"Ingested {total_chunks} total chunks: {[(k, len(v)) for k, v in result.items()]}")
            
            return result
        except Exception as e:
            raise IngestionError(f"Failed to ingest data: {e}")
