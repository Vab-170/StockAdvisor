"""
Caching service to improve performance by avoiding repeated expensive operations
"""
import time
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CacheService:
    """Simple in-memory cache with TTL (Time To Live) support"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = {
            'market_data': 1800,      # 30 minutes for market data (VIX, treasury, etc.)
            'predictions': 900,       # 15 minutes for ML predictions  
            'fundamentals': 3600,     # 1 hour for fundamental data
            'technical': 300,         # 5 minutes for technical indicators
            'stock_data': 300,        # 5 minutes for raw stock price data
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if key not in self._cache:
            return None
            
        cache_item = self._cache[key]
        current_time = time.time()
        
        # Check if expired
        if current_time > cache_item['expires_at']:
            del self._cache[key]
            logger.debug(f"Cache item expired and removed: {key}")
            return None
            
        logger.debug(f"Cache hit: {key}")
        return cache_item['data']
    
    def set(self, key: str, data: Any, cache_type: str = 'predictions') -> None:
        """Set item in cache with TTL"""
        ttl = self._default_ttl.get(cache_type, self._default_ttl['predictions'])
        expires_at = time.time() + ttl
        
        self._cache[key] = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time(),
            'cache_type': cache_type
        }
        logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    def clear(self, pattern: Optional[str] = None) -> None:
        """Clear cache items matching pattern or all if pattern is None"""
        if pattern is None:
            self._cache.clear()
            logger.info("Cache cleared completely")
        else:
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cache cleared for pattern: {pattern}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        total_items = len(self._cache)
        expired_items = sum(1 for item in self._cache.values() 
                          if current_time > item['expires_at'])
        
        by_type = {}
        for item in self._cache.values():
            cache_type = item.get('cache_type', 'unknown')
            by_type[cache_type] = by_type.get(cache_type, 0) + 1
        
        return {
            'total_items': total_items,
            'expired_items': expired_items,
            'active_items': total_items - expired_items,
            'by_type': by_type,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Rough estimate of memory usage in MB"""
        try:
            # Convert cache to JSON string to estimate size
            cache_str = json.dumps(self._cache, default=str)
            size_bytes = len(cache_str.encode('utf-8'))
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count removed"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self._cache.items() 
            if current_time > item['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache items")
        
        return len(expired_keys)

def create_cache_key(*args) -> str:
    """Create a consistent cache key from arguments"""
    # Convert all args to strings and create hash
    key_parts = [str(arg) for arg in args]
    key_string = '|'.join(key_parts)
    
    # Create short hash for cleaner keys
    hash_obj = hashlib.md5(key_string.encode())
    return hash_obj.hexdigest()[:12]

# Global cache instance
cache = CacheService()
