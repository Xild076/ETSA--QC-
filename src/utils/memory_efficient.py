"""
Memory-efficient processing utilities for large dataset handling.
Implements streaming, batch processing, and memory cleanup.
"""
import gc
import psutil
import os
from typing import Iterator, List, Any, Optional, Callable
from contextlib import contextmanager
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    used_mb: float
    available_mb: float
    percent_used: float
    
    @classmethod
    def current(cls) -> 'MemoryStats':
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        return cls(
            used_mb=memory.used / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            percent_used=memory.percent
        )

class MemoryManager:
    """Manages memory usage and cleanup during processing."""
    
    def __init__(self, cleanup_threshold_mb: float = 1000):
        self.cleanup_threshold_mb = cleanup_threshold_mb
        self.initial_memory = MemoryStats.current()
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold."""
        current = MemoryStats.current()
        memory_increase = current.used_mb - self.initial_memory.used_mb
        return memory_increase > self.cleanup_threshold_mb
    
    def cleanup_if_needed(self):
        """Perform garbage collection if memory threshold exceeded."""
        if self.check_memory_usage():
            logger.info("Memory threshold exceeded, performing cleanup...")
            gc.collect()
            current = MemoryStats.current()
            logger.info(f"Memory after cleanup: {current.used_mb:.1f} MB ({current.percent_used:.1f}%)")

@contextmanager
def memory_efficient_processing(cleanup_threshold_mb: float = 1000):
    """Context manager for memory-efficient processing."""
    manager = MemoryManager(cleanup_threshold_mb)
    initial = MemoryStats.current()
    logger.info(f"Starting processing with {initial.used_mb:.1f} MB memory usage")
    
    try:
        yield manager
    finally:
        final = MemoryStats.current()
        logger.info(f"Finished processing with {final.used_mb:.1f} MB memory usage")
        manager.cleanup_if_needed()

class StreamingXMLParser:
    """Memory-efficient XML parser for large datasets."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def parse_sentences(self) -> Iterator[dict]:
        """Stream sentences from XML file without loading entire file."""
        try:
            # Use iterparse for memory-efficient parsing
            context = ET.iterparse(self.file_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            current_sentence = {}
            current_text = ""
            current_aspectTerms = []
            current_aspectCategories = []
            
            for event, elem in context:
                if event == 'start':
                    if elem.tag == 'sentence':
                        current_sentence = {'id': elem.get('id', '')}
                    elif elem.tag == 'text':
                        current_text = elem.text or ""
                    elif elem.tag == 'aspectTerm':
                        current_aspectTerms.append({
                            'term': elem.get('term', ''),
                            'polarity': elem.get('polarity', ''),
                            'from': int(elem.get('from', 0)),
                            'to': int(elem.get('to', 0))
                        })
                    elif elem.tag == 'aspectCategory':
                        current_aspectCategories.append({
                            'category': elem.get('category', ''),
                            'polarity': elem.get('polarity', '')
                        })
                
                elif event == 'end':
                    if elem.tag == 'sentence':
                        # Yield complete sentence data
                        yield {
                            'id': current_sentence.get('id', ''),
                            'text': current_text,
                            'aspectTerms': current_aspectTerms[:],  # Copy to avoid reference issues
                            'aspectCategories': current_aspectCategories[:]
                        }
                        
                        # Reset for next sentence
                        current_sentence = {}
                        current_text = ""
                        current_aspectTerms = []
                        current_aspectCategories = []
                        
                        # Clear processed elements to save memory
                        elem.clear()
                        
            # Clean up
            root.clear()
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in {self.file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing {self.file_path}: {e}")

class BatchProcessor:
    """Process data in batches to manage memory usage."""
    
    def __init__(self, batch_size: int = 50, memory_threshold_mb: float = 1000):
        self.batch_size = batch_size
        self.memory_threshold_mb = memory_threshold_mb
        
    def process_batches(self, data_stream: Iterator[Any], 
                       processor_func: Callable[[List[Any]], List[Any]]) -> Iterator[List[Any]]:
        """Process data stream in batches."""
        batch = []
        
        with memory_efficient_processing(self.memory_threshold_mb) as memory_manager:
            for item in data_stream:
                batch.append(item)
                
                if len(batch) >= self.batch_size:
                    # Process batch
                    results = processor_func(batch)
                    yield results
                    
                    # Clear batch and check memory
                    batch = []
                    memory_manager.cleanup_if_needed()
            
            # Process remaining items
            if batch:
                results = processor_func(batch)
                yield results

class EfficientDataLoader:
    """Memory-efficient data loader for ETSA pipeline."""
    
    def __init__(self, batch_size: int = 50, max_memory_mb: float = 1000):
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.memory_manager = MemoryManager(max_memory_mb)
        
    def load_xml_dataset(self, file_path: str) -> Iterator[dict]:
        """Load XML dataset with memory efficiency."""
        parser = StreamingXMLParser(file_path)
        
        for sentence_data in parser.parse_sentences():
            yield sentence_data
            
            # Periodic memory cleanup
            self.memory_manager.cleanup_if_needed()
    
    def load_xml_batches(self, file_path: str) -> Iterator[List[dict]]:
        """Load XML dataset in batches."""
        batch_processor = BatchProcessor(self.batch_size, self.max_memory_mb)
        data_stream = self.load_xml_dataset(file_path)
        
        def identity_processor(batch: List[dict]) -> List[dict]:
            return batch
        
        for batch in batch_processor.process_batches(data_stream, identity_processor):
            yield batch

def process_large_dataset(file_path: str, 
                         processor_func: Callable[[dict], Any],
                         batch_size: int = 50,
                         max_memory_mb: float = 1000) -> Iterator[Any]:
    """
    Process large XML dataset efficiently.
    
    Args:
        file_path: Path to XML file
        processor_func: Function to process each sentence
        batch_size: Number of sentences to process in each batch
        max_memory_mb: Memory threshold for cleanup
    
    Yields:
        Processed results for each sentence
    """
    loader = EfficientDataLoader(batch_size, max_memory_mb)
    
    for batch in loader.load_xml_batches(file_path):
        with memory_efficient_processing(max_memory_mb):
            for sentence_data in batch:
                try:
                    result = processor_func(sentence_data)
                    yield result
                except Exception as e:
                    logger.error(f"Error processing sentence {sentence_data.get('id', 'unknown')}: {e}")
                    yield None

class CacheManager:
    """Manages caches with size limits and cleanup."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with LRU eviction."""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

# Global cache manager instance
global_cache = CacheManager(max_size=1000)

def get_cached_result(key: str, compute_func: Callable[[], Any]) -> Any:
    """Get result from cache or compute and cache it."""
    result = global_cache.get(key)
    if result is None:
        result = compute_func()
        global_cache.set(key, result)
    return result
