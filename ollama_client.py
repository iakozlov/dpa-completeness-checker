import json
import os
import requests
import logging
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = None):
        """Initialize the Ollama client.
        
        Args:
            base_url (str): Base URL for the Ollama API. If None, uses OLLAMA_HOST env var or default.
        """
        if base_url is None:
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = base_url
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config file."""
        try:
            with open("config/ollama_config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Configuration file not found: config/ollama_config.json")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in configuration file")
            raise
            
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model configuration
        """
        if model_name not in self.config["models"]:
            raise ValueError(f"Model {model_name} not found in configuration")
        return self.config["models"][model_name]
        
    def get_representation_config(self, representation_name: str) -> Dict[str, Any]:
        """Get configuration for a specific requirements representation.
        
        Args:
            representation_name (str): Name of the representation
            
        Returns:
            Dict[str, Any]: Representation configuration
        """
        if representation_name not in self.config["requirements_representations"]:
            raise ValueError(f"Representation {representation_name} not found in configuration")
        return self.config["requirements_representations"][representation_name]
        
    def check_health(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def generate(self, prompt: str, model_name: str = None, system_prompt: str = None) -> str:
        """Generate text using Ollama chat endpoint with system and user prompts.
        
        Args:
            prompt (str): The user prompt/query
            model_name (str, optional): Model to use. If None, uses default model
            system_prompt (str, optional): System prompt to guide model behavior
            
        Returns:
            str: Generated response text
        """
        if model_name is None:
            model_name = self.config.get("default_model", "llama3.3:70b")
        
        # Get model configuration
        model_config = self.config["models"].get(model_name, {})
        parameters = model_config.get("parameters", {})
        
        # Prepare messages array
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare request payload for chat endpoint
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": parameters.get("temperature", 0.1),
                "top_p": parameters.get("top_p", 0.9),
                "num_ctx": parameters.get("max_tokens", 4096)
            }
        }
        
        try:
            logger.info(f"Generating text with model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=200
            )
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating text: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            logger.error(f"Response: {response.text}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], model_name: str = None) -> str:
        """Chat with the model using a conversation history.
        
        Args:
            messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys
            model_name (str, optional): Model to use. If None, uses default model
            
        Returns:
            str: Generated response text
        """
        if model_name is None:
            model_name = self.config.get("default_model", "llama3.3:70b")
        
        # Get model configuration
        model_config = self.config["models"].get(model_name, {})
        parameters = model_config.get("parameters", {})
        
        # Prepare request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": parameters.get("temperature", 0.1),
                "top_p": parameters.get("top_p", 0.9),
                "num_ctx": parameters.get("max_tokens", 4096)
            }
        }
        
        try:
            logger.info(f"Chatting with model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in chat: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format: {e}")
            logger.error(f"Response: {response.text}")
            raise
    
    def pull_model(self, model_name: str) -> bool:
        """Pull/download a model from Ollama registry.
        
        Args:
            model_name (str): Name of the model to pull
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=600  # 10 minutes timeout for model downloads
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from configuration."""
        return list(self.config["models"].keys())
    
    def get_requirements_files(self) -> Dict[str, Dict[str, str]]:
        """Get available requirements files configuration."""
        return self.config.get("requirements_files", {})
        
    def translate_requirement(self,
                            requirement: str,
                            representation: str = "deontic",
                            model_name: str = "llama2") -> str:
        """Translate a requirement into the specified representation.
        
        Args:
            requirement (str): The requirement to translate
            representation (str): Target representation type
            model_name (str): Model to use for translation
            
        Returns:
            str: Translated requirement
        """
        rep_config = self.get_representation_config(representation)
        prompt = rep_config["template"].format(requirement=requirement)
        
        return self.generate(prompt, model_name=model_name) 
