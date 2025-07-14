import asyncio
import json
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any, List
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError
from .api_key_manager import APIKeyManager

class OpenAIClient:
    """Async OpenAI client with cancellation support and multiple API key management."""
    
    def __init__(self, api_keys: List[str], base_url: str, timeout: int = 90, api_version: Optional[str] = None):
        # Support both single key (backward compatibility) and multiple keys
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        
        self.api_key_manager = APIKeyManager(api_keys)
        self.base_url = base_url
        self.timeout = timeout
        self.api_version = api_version
        self.active_requests: Dict[str, asyncio.Event] = {}
        
        # Keep backward compatibility
        self.api_key = api_keys[0]
    
    def _create_client(self, api_key: str):
        """Create an OpenAI client instance with the given API key."""
        if self.api_version:
            return AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                timeout=self.timeout
            )
        else:
            return AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
    
    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support and automatic key rotation."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            last_exception = None
            attempts = 0
            max_attempts = self.api_key_manager.get_available_key_count()
            
            while attempts < max_attempts:
                # Get next available API key
                api_key = self.api_key_manager.get_next_key()
                if not api_key:
                    # All keys are in cooldown
                    if last_exception:
                        raise last_exception
                    raise HTTPException(status_code=503, detail="All API keys are temporarily unavailable")
                
                try:
                    # Create client with current API key
                    client = self._create_client(api_key)
                    
                    # Create task that can be cancelled
                    completion_task = asyncio.create_task(
                        client.chat.completions.create(**request)
                    )
                    
                    if request_id:
                        # Wait for either completion or cancellation
                        cancel_task = asyncio.create_task(cancel_event.wait())
                        done, pending = await asyncio.wait(
                            [completion_task, cancel_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        
                        # Check if request was cancelled
                        if cancel_task in done:
                            completion_task.cancel()
                            raise HTTPException(status_code=499, detail="Request cancelled by client")
                        
                        completion = await completion_task
                    else:
                        completion = await completion_task
                    
                    # Success! Convert to dict format that matches the original interface
                    return completion.model_dump()
                
                except (AuthenticationError, RateLimitError) as e:
                    # These errors indicate the API key should be marked as failed
                    self.api_key_manager.mark_key_failed(api_key, str(e))
                    last_exception = HTTPException(
                        status_code=401 if isinstance(e, AuthenticationError) else 429,
                        detail=self.classify_openai_error(str(e))
                    )
                    attempts += 1
                    continue
                    
                except BadRequestError as e:
                    # Bad request errors are not key-specific, don't retry
                    raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
                except APIError as e:
                    status_code = getattr(e, 'status_code', 500)
                    # For 5xx errors, we might want to retry with a different key
                    if status_code >= 500:
                        self.api_key_manager.mark_key_failed(api_key, str(e))
                        last_exception = HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
                        attempts += 1
                        continue
                    else:
                        raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
                except Exception as e:
                    # For unexpected errors, try next key
                    self.api_key_manager.mark_key_failed(api_key, str(e))
                    last_exception = HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
                    attempts += 1
                    continue
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
            raise HTTPException(status_code=503, detail="All API keys failed")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support and automatic key rotation."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        try:
            last_exception = None
            attempts = 0
            max_attempts = self.api_key_manager.get_available_key_count()
            
            while attempts < max_attempts:
                # Get next available API key
                api_key = self.api_key_manager.get_next_key()
                if not api_key:
                    # All keys are in cooldown
                    if last_exception:
                        raise last_exception
                    raise HTTPException(status_code=503, detail="All API keys are temporarily unavailable")
                
                try:
                    # Ensure stream is enabled
                    request["stream"] = True
                    if "stream_options" not in request:
                        request["stream_options"] = {}
                    request["stream_options"]["include_usage"] = True
                    
                    # Create client with current API key
                    client = self._create_client(api_key)
                    
                    # Create the streaming completion
                    streaming_completion = await client.chat.completions.create(**request)
                    
                    async for chunk in streaming_completion:
                        # Check for cancellation before yielding each chunk
                        if request_id and request_id in self.active_requests:
                            if self.active_requests[request_id].is_set():
                                raise HTTPException(status_code=499, detail="Request cancelled by client")
                        
                        # Convert chunk to SSE format matching original HTTP client format
                        chunk_dict = chunk.model_dump()
                        chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                        yield f"data: {chunk_json}"
                    
                    # Signal end of stream
                    yield "data: [DONE]"
                    return  # Success, exit the retry loop
                        
                except (AuthenticationError, RateLimitError) as e:
                    # These errors indicate the API key should be marked as failed
                    self.api_key_manager.mark_key_failed(api_key, str(e))
                    last_exception = HTTPException(
                        status_code=401 if isinstance(e, AuthenticationError) else 429,
                        detail=self.classify_openai_error(str(e))
                    )
                    attempts += 1
                    continue
                    
                except BadRequestError as e:
                    # Bad request errors are not key-specific, don't retry
                    raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
                except APIError as e:
                    status_code = getattr(e, 'status_code', 500)
                    # For 5xx errors, we might want to retry with a different key
                    if status_code >= 500:
                        self.api_key_manager.mark_key_failed(api_key, str(e))
                        last_exception = HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
                        attempts += 1
                        continue
                    else:
                        raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
                except Exception as e:
                    # For unexpected errors, try next key
                    self.api_key_manager.mark_key_failed(api_key, str(e))
                    last_exception = HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
                    attempts += 1
                    continue
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
            raise HTTPException(status_code=503, detail="All API keys failed")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
    
    def get_api_key_status(self) -> Dict:
        """Get the current status of all API keys."""
        return self.api_key_manager.get_status()
    
    def reset_api_key_failures(self):
        """Reset all API key failures."""
        self.api_key_manager.reset_all_failures()