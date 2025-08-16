"""
Intelligent caching system for fantasy football data
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import time
import json
from typing import Any, Optional, Dict, List
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class DataCache:
    """Intelligent caching system with automatic invalidation"""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl  # Time to live in seconds
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache metadata: {e}")
    
    def _get_cache_key(self, key: str, params: Dict = None) -> str:
        """Generate cache key with parameter hashing"""
        if params:
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{key}_{param_hash}"
        return key
    
    def _is_cache_valid(self, cache_key: str, ttl: Optional[int] = None) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.metadata:
            return False
        
        cache_info = self.metadata[cache_key]
        cache_time = cache_info.get('timestamp', 0)
        cache_ttl = ttl or cache_info.get('ttl', self.default_ttl)
        
        return (time.time() - cache_time) < cache_ttl
    
    def get(self, key: str, params: Dict = None, ttl: Optional[int] = None) -> Optional[Any]:
        """Get item from cache if valid"""
        cache_key = self._get_cache_key(key, params)
        
        if not self._is_cache_valid(cache_key, ttl):
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Cache hit for {key}")
            return data
        except Exception as e:
            logger.warning(f"Error loading cache for {key}: {e}")
            return None
    
    def set(self, key: str, data: Any, params: Dict = None, ttl: Optional[int] = None) -> None:
        """Store item in cache"""
        cache_key = self._get_cache_key(key, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata[cache_key] = {
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl,
                'original_key': key,
                'params': params
            }
            self._save_metadata()
            
            logger.info(f"Cached {key} ({self._get_size_mb(cache_file):.1f}MB)")
            
        except Exception as e:
            logger.error(f"Error caching {key}: {e}")
    
    def _get_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / 1024 / 1024
        except:
            return 0
    
    def clear_expired(self) -> None:
        """Clear expired cache entries"""
        expired_keys = []
        current_time = time.time()
        
        for cache_key, info in self.metadata.items():
            cache_time = info.get('timestamp', 0)
            cache_ttl = info.get('ttl', self.default_ttl)
            
            if (current_time - cache_time) > cache_ttl:
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            self.delete(cache_key)
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def delete(self, key: str, params: Dict = None) -> None:
        """Delete cache entry"""
        cache_key = self._get_cache_key(key, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            cache_file.unlink()
        
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
    
    def clear_all(self) -> None:
        """Clear all cache entries"""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata.clear()
        self._save_metadata()
        logger.info("Cleared all cache entries")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_files = len(list(self.cache_dir.glob("*.pkl")))
        total_size = sum(self._get_size_mb(f) for f in self.cache_dir.glob("*.pkl"))
        
        valid_entries = sum(1 for key in self.metadata.keys() if self._is_cache_valid(key))
        
        return {
            'total_entries': total_files,
            'valid_entries': valid_entries,
            'total_size_mb': total_size,
            'cache_dir': str(self.cache_dir)
        }

# Global cache instance
cache = DataCache()

def cached(key: str, ttl: int = 3600, use_params: bool = True):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create parameters dict for cache key
            params = None
            if use_params:
                params = {
                    'args': str(args[1:]) if args else '',  # Skip self for methods
                    'kwargs': kwargs
                }
            
            # Try to get from cache
            cached_result = cache.get(key, params, ttl)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(key, result, params, ttl)
            
            return result
        return wrapper
    return decorator

class ModelCache:
    """Specialized cache for ML models and preprocessors"""
    
    def __init__(self, cache_dir: str = "cache/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_pipeline(self, model, scaler, features: List[str], 
                          model_params: Dict, performance_metrics: Dict) -> str:
        """Save complete model pipeline with metadata"""
        timestamp = int(time.time())
        model_id = f"model_{timestamp}"
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'model_params': model_params,
            'performance_metrics': performance_metrics,
            'timestamp': timestamp,
            'model_id': model_id
        }
        
        model_file = self.cache_dir / f"{model_id}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved model pipeline: {model_id}")
        return model_id
    
    def load_latest_model(self) -> Optional[Dict]:
        """Load the most recent model pipeline"""
        model_files = list(self.cache_dir.glob("model_*.pkl"))
        if not model_files:
            return None
        
        # Get most recent model
        latest_file = max(model_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Loaded model: {model_data['model_id']}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def list_models(self) -> List[Dict]:
        """List all cached models with metadata"""
        models = []
        for model_file in self.cache_dir.glob("model_*.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                models.append({
                    'model_id': model_data['model_id'],
                    'timestamp': model_data['timestamp'],
                    'performance': model_data.get('performance_metrics', {}),
                    'file_size_mb': model_file.stat().st_size / 1024 / 1024
                })
            except Exception as e:
                logger.warning(f"Error reading model metadata from {model_file}: {e}")
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)

# Global model cache
model_cache = ModelCache()