import asyncio
import time
from typing import List, Optional, Dict, Set
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages multiple OpenAI API keys with round-robin distribution and error handling."""
    
    def __init__(self, api_keys: List[str], cooldown_period: int = 300):
        """
        Initialize the API key manager.
        
        Args:
            api_keys: List of OpenAI API keys
            cooldown_period: Time in seconds to wait before retrying a failed key
        """
        self.api_keys = api_keys
        self.cooldown_period = cooldown_period
        self.current_index = 0
        self.failed_keys: Dict[str, float] = {}  # key -> timestamp of failure
        self.lock = Lock()
        
        logger.info(f"Initialized API key manager with {len(api_keys)} keys")
    
    def get_next_key(self) -> Optional[str]:
        """
        Get the next available API key using round-robin strategy.
        
        Returns:
            Next available API key or None if all keys are in cooldown
        """
        with self.lock:
            current_time = time.time()
            
            # Clean up expired cooldowns
            expired_keys = [
                key for key, fail_time in self.failed_keys.items()
                if current_time - fail_time > self.cooldown_period
            ]
            for key in expired_keys:
                del self.failed_keys[key]
                logger.info(f"API key cooldown expired, key is available again")
            
            # Find next available key
            attempts = 0
            while attempts < len(self.api_keys):
                key = self.api_keys[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                
                if key not in self.failed_keys:
                    logger.debug(f"Selected API key index {self.current_index - 1}")
                    return key
                
                attempts += 1
            
            # All keys are in cooldown
            logger.warning("All API keys are in cooldown period")
            return None
    
    def mark_key_failed(self, api_key: str, error_message: str = ""):
        """
        Mark an API key as failed and put it in cooldown.
        
        Args:
            api_key: The failed API key
            error_message: Optional error message for logging
        """
        with self.lock:
            self.failed_keys[api_key] = time.time()
            key_index = self.api_keys.index(api_key) if api_key in self.api_keys else -1
            logger.warning(f"API key (index {key_index}) marked as failed: {error_message}")
    
    def get_available_key_count(self) -> int:
        """Get the number of currently available (not in cooldown) API keys."""
        with self.lock:
            current_time = time.time()
            available_count = 0
            
            for key in self.api_keys:
                if key not in self.failed_keys:
                    available_count += 1
                elif current_time - self.failed_keys[key] > self.cooldown_period:
                    available_count += 1
            
            return available_count
    
    def get_status(self) -> Dict:
        """Get the current status of all API keys."""
        with self.lock:
            current_time = time.time()
            status = {
                "total_keys": len(self.api_keys),
                "available_keys": 0,
                "failed_keys": 0,
                "keys_status": []
            }
            
            for i, key in enumerate(self.api_keys):
                key_status = {
                    "index": i,
                    "key_prefix": key[:10] + "..." if len(key) > 10 else key,
                    "status": "available"
                }
                
                if key in self.failed_keys:
                    fail_time = self.failed_keys[key]
                    time_since_failure = current_time - fail_time
                    
                    if time_since_failure > self.cooldown_period:
                        key_status["status"] = "available"
                        status["available_keys"] += 1
                    else:
                        key_status["status"] = "cooldown"
                        key_status["cooldown_remaining"] = int(self.cooldown_period - time_since_failure)
                        status["failed_keys"] += 1
                else:
                    status["available_keys"] += 1
                
                status["keys_status"].append(key_status)
            
            return status
    
    def reset_all_failures(self):
        """Reset all failed keys (remove from cooldown)."""
        with self.lock:
            self.failed_keys.clear()
            logger.info("All API key failures have been reset")