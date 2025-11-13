"""
Configuration loader for Quiz Chatbot
This module loads settings from config.ini and provides them to the application
"""

import configparser
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            print(f"Warning: {self.config_file} not found. Using default settings.")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        # Server defaults
        self.config['server'] = {
            'host': '0.0.0.0',
            'port': '8000',
            'reload': 'true'
        }
        
        # Quiz defaults
        self.config['quiz'] = {
            'default_num_questions': '10',
            'min_questions': '5',
            'max_questions': '20'
        }
        
        # PDF processing defaults
        self.config['pdf_processing'] = {
            'chunk_size': '500',
            'chunk_overlap': '50',
            'max_pdf_size_mb': '50'
        }
        
        # Save default config
        with open(self.config_file, 'w') as f:
            self.config.write(f)
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value"""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value"""
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get float configuration value"""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def get_dict(self, section: str) -> Dict[str, str]:
        """Get entire section as dictionary"""
        try:
            return dict(self.config.items(section))
        except configparser.NoSectionError:
            return {}

# Global config instance
config = Config()

# Example usage in quiz_chatbot_backend.py:
"""
from config_loader import config

# Use configuration values
NUM_QUESTIONS = config.getint('quiz', 'default_num_questions', 10)
CHUNK_SIZE = config.getint('pdf_processing', 'chunk_size', 500)
HOST = config.get('server', 'host', '0.0.0.0')
PORT = config.getint('server', 'port', 8000)

# In your QuizGenerator class:
questions = quiz_gen.generate_questions(text, num_questions=NUM_QUESTIONS)

# In your PDFProcessor class:
chunks = processor.chunk_text(text, chunk_size=CHUNK_SIZE)

# When starting the server:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, reload=config.getboolean('server', 'reload', True))
"""
