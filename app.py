from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
import json
import uuid
import time
import threading
import re
import ast
import subprocess
import tempfile
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Any, Optional
import redis
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Enable CORS for cross-origin requests
CORS(app, origins=['http://localhost:3000', 'http://localhost:8000'])

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configuration
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'llama3.1:70b')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')

# Performance optimizations
OLLAMA_TIMEOUT = 180
MAX_CONTEXT_LENGTH = 8192
CONCURRENT_REQUESTS = 5
MAX_SESSIONS = 1000
MAX_MESSAGES_PER_SESSION = 100

# Thread pool for handling requests
executor = ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS)

# Redis for session storage (fallback to in-memory)
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    USE_REDIS = True
    logger.info("Redis connected successfully")
except:
    USE_REDIS = False
    chat_sessions = {}
    logger.warning("Redis unavailable, using in-memory storage")

# Active streaming requests
active_streams = {}

# Code validation patterns
CODE_PATTERNS = {
    'python': r'```python\n(.*?)\n```',
    'javascript': r'```javascript\n(.*?)\n```',
    'java': r'```java\n(.*?)\n```',
    'cpp': r'```cpp\n(.*?)\n```',
    'c': r'```c\n(.*?)\n```',
    'html': r'```html\n(.*?)\n```',
    'css': r'```css\n(.*?)\n```',
    'sql': r'```sql\n(.*?)\n```',
    'bash': r'```bash\n(.*?)\n```',
    'rust': r'```rust\n(.*?)\n```',
    'go': r'```go\n(.*?)\n```',
    'php': r'```php\n(.*?)\n```',
    'typescript': r'```typescript\n(.*?)\n```',
    'kotlin': r'```kotlin\n(.*?)\n```',
    'swift': r'```swift\n(.*?)\n```',
    'ruby': r'```ruby\n(.*?)\n```',
    'dart': r'```dart\n(.*?)\n```',
    'r': r'```r\n(.*?)\n```',
    'matlab': r'```matlab\n(.*?)\n```'
}

@dataclass
class CodeValidationResult:
    is_valid: bool
    language: str
    errors: List[str]
    suggestions: List[str]

class SessionManager:
    """Enhanced session management with Redis support"""
    
    def __init__(self):
        self.redis_client = redis_client if USE_REDIS else None
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session"""
        if USE_REDIS:
            try:
                history = self.redis_client.get(f"chat:{session_id}")
                return json.loads(history) if history else []
            except:
                return []
        else:
            return chat_sessions.get(session_id, [])
    
    def add_message(self, session_id: str, message: Dict):
        """Add a message to session history"""
        history = self.get_session_history(session_id)
        history.append(message)
        
        # Trim history if too long
        if len(history) > MAX_MESSAGES_PER_SESSION:
            history = history[-MAX_MESSAGES_PER_SESSION:]
        
        if USE_REDIS:
            try:
                self.redis_client.setex(
                    f"chat:{session_id}", 
                    timedelta(hours=24),
                    json.dumps(history)
                )
            except Exception as e:
                logger.error(f"Redis error: {e}")
        else:
            chat_sessions[session_id] = history
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        if USE_REDIS:
            try:
                self.redis_client.delete(f"chat:{session_id}")
            except:
                pass
        else:
            chat_sessions.pop(session_id, None)

session_manager = SessionManager()

class CodeValidator:
    """Advanced code validation for multiple languages"""
    
    @staticmethod
    def validate_python(code: str) -> CodeValidationResult:
        """Validate Python code"""
        errors = []
        suggestions = []
        
        try:
            # Parse AST for syntax validation
            ast.parse(code)
            
            # Check for common issues
            if 'print(' not in code and 'return' not in code:
                suggestions.append("Consider adding output or return statements")
            
            if 'import' not in code and any(lib in code for lib in ['np.', 'pd.', 'plt.', 'tf.']):
                errors.append("Missing import statements for used libraries")
            
            return CodeValidationResult(
                is_valid=len(errors) == 0,
                language='python',
                errors=errors,
                suggestions=suggestions
            )
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return CodeValidationResult(
                is_valid=False,
                language='python',
                errors=errors,
                suggestions=suggestions
            )
    
    @staticmethod
    def validate_javascript(code: str) -> CodeValidationResult:
        """Validate JavaScript code"""
        errors = []
        suggestions = []
        
        # Basic syntax checks
        if code.count('{') != code.count('}'):
            errors.append("Mismatched curly braces")
        
        if code.count('(') != code.count(')'):
            errors.append("Mismatched parentheses")
        
        # Check for common issues
        if 'console.log' not in code and 'return' not in code:
            suggestions.append("Consider adding console.log or return statements")
        
        return CodeValidationResult(
            is_valid=len(errors) == 0,
            language='javascript',
            errors=errors,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_java(code: str) -> CodeValidationResult:
        """Validate Java code"""
        errors = []
        suggestions = []
        
        # Check for class definition
        if 'class ' not in code:
            errors.append("Java code should contain a class definition")
        
        # Check for main method in standalone programs
        if 'public static void main' not in code and 'class' in code:
            suggestions.append("Consider adding a main method for executable programs")
        
        # Check braces
        if code.count('{') != code.count('}'):
            errors.append("Mismatched curly braces")
        
        return CodeValidationResult(
            is_valid=len(errors) == 0,
            language='java',
            errors=errors,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_code(code: str, language: str) -> CodeValidationResult:
        """Validate code based on language"""
        validators = {
            'python': CodeValidator.validate_python,
            'javascript': CodeValidator.validate_javascript,
            'java': CodeValidator.validate_java,
        }
        
        if language in validators:
            return validators[language](code)
        else:
            # Basic validation for other languages
            return CodeValidationResult(
                is_valid=True,
                language=language,
                errors=[],
                suggestions=[]
            )

class ClaudeStylePromptProcessor:
    """Process prompts to generate Claude-style responses"""
    
    @staticmethod
    def create_system_prompt() -> str:
        """Create comprehensive system prompt for Claude-style responses"""
        return """You are an advanced AI assistant that provides helpful, accurate, and thoughtful responses. Follow these guidelines:

## Response Style:
- Be conversational yet professional
- Use phrases like "Let me think through this...", "Here's how I'd approach this...", "To break this down..."
- Always provide complete, working code examples
- Explain concepts clearly and thoroughly

## Code Generation Rules:
- ALWAYS show code first in properly formatted code blocks
- Use triple backticks with language specification (```python, ```javascript, etc.)
- Generate complete, runnable code without placeholders
- Include proper error handling and best practices
- Add comments explaining complex logic
- Follow language-specific conventions and style guides

## Multi-Language Support:
- Automatically detect requested programming language
- Support: Python, JavaScript, TypeScript, Java, C++, C, HTML/CSS, PHP, SQL, Bash, Rust, Go, Kotlin, Swift, Ruby, Dart, R, MATLAB, and more
- Provide equivalent implementations when requested
- Validate syntax and logic before responding

## Structure:
1. Brief acknowledgment of the request
2. Complete code implementation
3. Clear explanation of how it works
4. Additional tips or considerations
5. Offer to modify or extend the solution

## Quality Assurance:
- Ensure all code is syntactically correct
- Include necessary imports and dependencies
- Provide realistic, practical examples
- Test logic mentally before responding
- Offer debugging help if needed

Remember: Code first, explanation second. Always prioritize working, complete solutions."""

    @staticmethod
    def enhance_user_prompt(user_message: str) -> str:
        """Enhance user prompt for better code generation"""
        # Detect if user is asking for code in a specific language
        language_requests = {
            'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
            'java': ['java', 'spring', 'maven', 'gradle'],
            'cpp': ['c++', 'cpp', 'c plus plus'],
            'c': ['c language', 'c programming'],
            'html': ['html', 'web page', 'webpage'],
            'css': ['css', 'styling', 'styles'],
            'sql': ['sql', 'database', 'mysql', 'postgresql'],
            'typescript': ['typescript', 'ts'],
            'rust': ['rust', 'cargo'],
            'go': ['golang', 'go language'],
            'php': ['php', 'laravel', 'symfony'],
            'kotlin': ['kotlin', 'android'],
            'swift': ['swift', 'ios'],
            'ruby': ['ruby', 'rails'],
            'dart': ['dart', 'flutter'],
            'bash': ['bash', 'shell script', 'terminal'],
            'r': ['r language', 'r programming', 'rstudio'],
            'matlab': ['matlab', 'octave']
        }
        
        detected_language = None
        user_lower = user_message.lower()
        
        for lang, keywords in language_requests.items():
            if any(keyword in user_lower for keyword in keywords):
                detected_language = lang
                break
        
        # Enhance prompt based on detected language
        if detected_language:
            enhancement = f"\n\nIMPORTANT: The user is asking for {detected_language.upper()} code. Please provide a complete, working {detected_language} implementation with proper syntax, imports, and best practices."
            return user_message + enhancement
        
        return user_message

code_validator = CodeValidator()
prompt_processor = ClaudeStylePromptProcessor()

def cleanup_old_sessions():
    """Clean up old chat sessions"""
    if not USE_REDIS:
        if len(chat_sessions) > MAX_SESSIONS:
            # Remove oldest sessions
            sorted_sessions = sorted(chat_sessions.items(), 
                                   key=lambda x: x[1][-1]['timestamp'] if x[1] else '')
            for session_id, _ in sorted_sessions[:len(chat_sessions) - MAX_SESSIONS]:
                del chat_sessions[session_id]

def prepare_context_with_validation(messages: List[Dict]) -> List[Dict]:
    """Prepare context with code validation"""
    context_messages = []
    current_length = 0
    
    # Add system prompt
    system_prompt = prompt_processor.create_system_prompt()
    context_messages.append({
        'role': 'system',
        'content': system_prompt
    })
    
    # Add conversation history
    for msg in reversed(messages):
        msg_length = len(msg['content'])
        if current_length + msg_length > MAX_CONTEXT_LENGTH:
            break
        
        # Enhance user messages
        if msg['role'] == 'user':
            enhanced_content = prompt_processor.enhance_user_prompt(msg['content'])
            context_messages.insert(-1, {
                'role': msg['role'],
                'content': enhanced_content
            })
        else:
            context_messages.insert(-1, {
                'role': msg['role'],
                'content': msg['content']
            })
        
        current_length += msg_length
    
    return context_messages

def validate_response_code(response: str) -> Dict[str, Any]:
    """Validate code in AI response"""
    validation_results = {}
    
    for language, pattern in CODE_PATTERNS.items():
        matches = re.findall(pattern, response, re.DOTALL)
        for i, code in enumerate(matches):
            result = code_validator.validate_code(code, language)
            validation_results[f"{language}_{i}"] = {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'suggestions': result.suggestions
            }
    
    return validation_results

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/chat/stream', methods=['POST'])
@limiter.limit("30 per minute")
def chat_stream():
    """Enhanced streaming chat endpoint"""
    try:
        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Get user message
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Add user message to history
        session_manager.add_message(session_id, {
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get conversation history
        history = session_manager.get_session_history(session_id)
        
        # Prepare context with enhancements
        context_messages = prepare_context_with_validation(history)
        
        def generate_response():
            try:
                # Mark stream as active
                active_streams[session_id] = True
                
                # Enhanced Ollama request
                response = requests.post(
                    f'{OLLAMA_BASE_URL}/api/chat',
                    json={
                        'model': DEFAULT_MODEL,
                        'messages': context_messages,
                        'stream': True,
                        'options': {
                            'temperature': 0.7,
                            'num_predict': 4096,
                            'num_ctx': MAX_CONTEXT_LENGTH,
                            'num_batch': 512,
                            'num_gpu': 1,
                            'top_k': 40,
                            'top_p': 0.9,
                            'repeat_penalty': 1.1,
                            'seed': -1,
                            'tfs_z': 1.0,
                            'typical_p': 1.0,
                            'presence_penalty': 0.0,
                            'frequency_penalty': 0.0,
                            'mirostat': 0,
                            'mirostat_eta': 0.1,
                            'mirostat_tau': 5.0,
                            'penalize_newline': False,
                            'stop': ['<|im_end|>', '<|endoftext|>']
                        }
                    },
                    stream=True,
                    timeout=OLLAMA_TIMEOUT
                )
                
                if response.status_code != 200:
                    yield f'{{"error": "Ollama API error: {response.status_code}"}}\n'
                    return
                
                full_response = ""
                
                for line in response.iter_lines():
                    if session_id not in active_streams:
                        break
                    
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            
                            if 'message' in data and 'content' in data['message']:
                                content = data['message']['content']
                                full_response += content
                                
                                yield f'{{"content": {json.dumps(content)}}}\n'
                            
                            if data.get('done', False):
                                # Validate code in response
                                validation_results = validate_response_code(full_response)
                                
                                # Add AI response to history
                                session_manager.add_message(session_id, {
                                    'role': 'assistant',
                                    'content': full_response,
                                    'timestamp': datetime.now().isoformat(),
                                    'validation': validation_results
                                })
                                
                                yield f'{{"done": true, "validation": {json.dumps(validation_results)}}}\n'
                                break
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Stream processing error: {e}")
                            yield f'{{"error": "Stream processing error"}}\n'
                            break
                
            except requests.exceptions.ConnectionError:
                yield f'{{"error": "Cannot connect to Ollama. Ensure Ollama is running."}}\n'
            except requests.exceptions.Timeout:
                yield f'{{"error": "Request timeout. Try a shorter message."}}\n'
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                yield f'{{"error": "Unexpected error occurred"}}\n'
            finally:
                active_streams.pop(session_id, None)
        
        return Response(generate_response(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stop', methods=['POST'])
def stop_chat():
    """Stop active chat stream"""
    try:
        if 'session_id' in session:
            session_id = session['session_id']
            active_streams.pop(session_id, None)
            return jsonify({'message': 'Chat stopped successfully'})
        return jsonify({'message': 'No active chat to stop'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate/code', methods=['POST'])
@limiter.limit("20 per minute")
def validate_code_endpoint():
    """Validate code snippet"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        if not code:
            return jsonify({'error': 'Code cannot be empty'}), 400
        
        result = code_validator.validate_code(code, language)
        
        return jsonify({
            'is_valid': result.is_valid,
            'language': result.language,
            'errors': result.errors,
            'suggestions': result.suggestions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/convert/language', methods=['POST'])
@limiter.limit("10 per minute")
def convert_language():
    """Convert code from one language to another"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        from_lang = data.get('from_language', 'python')
        to_lang = data.get('to_language', 'javascript')
        
        if not code:
            return jsonify({'error': 'Code cannot be empty'}), 400
        
        # Create conversion prompt
        conversion_prompt = f"""Convert this {from_lang} code to {to_lang}:

```{from_lang}
{code}
```

Requirements:
- Provide complete, working {to_lang} code
- Maintain the same functionality
- Use {to_lang} best practices and conventions
- Include necessary imports/dependencies
- Add comments explaining key differences"""
        
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Prepare context
        context_messages = [{
            'role': 'system',
            'content': prompt_processor.create_system_prompt()
        }, {
            'role': 'user',
            'content': conversion_prompt
        }]
        
        # Call Ollama
        response = requests.post(
            f'{OLLAMA_BASE_URL}/api/chat',
            json={
                'model': DEFAULT_MODEL,
                'messages': context_messages,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 2048,
                    'num_ctx': 4096
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            response_data = response.json()
            converted_code = response_data['message']['content']
            
            # Validate converted code
            validation_results = validate_response_code(converted_code)
            
            return jsonify({
                'converted_code': converted_code,
                'validation': validation_results
            })
        else:
            return jsonify({'error': 'Code conversion failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    try:
        if 'session_id' in session:
            session_id = session['session_id']
            session_manager.clear_session(session_id)
            active_streams.pop(session_id, None)
        return jsonify({'message': 'Chat cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    try:
        if 'session_id' not in session:
            return jsonify([])
        
        session_id = session['session_id']
        history = session_manager.get_session_history(session_id)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=10)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Could not fetch models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        # Check Ollama connection
        ollama_status = "disconnected"
        model_info = None
        
        try:
            response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
            if response.status_code == 200:
                ollama_status = "connected"
                data = response.json()
                if data.get('models'):
                    model_info = data['models'][0]
        except:
            pass
        
        # Check Redis connection
        redis_status = "connected" if USE_REDIS else "not_configured"
        if USE_REDIS:
            try:
                redis_client.ping()
            except:
                redis_status = "disconnected"
        
        # Get session count
        session_count = len(chat_sessions) if not USE_REDIS else "unknown"
        
        stats = {
            'ollama_status': ollama_status,
            'redis_status': redis_status,
            'active_sessions': session_count,
            'active_streams': len(active_streams),
            'current_model': DEFAULT_MODEL,
            'model_info': model_info,
            'max_context_length': MAX_CONTEXT_LENGTH,
            'concurrent_requests': CONCURRENT_REQUESTS,
            'supported_languages': list(CODE_PATTERNS.keys())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        checks = {
            'flask': True,
            'ollama': False,
            'redis': USE_REDIS,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check Ollama
        try:
            response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
            checks['ollama'] = response.status_code == 200
        except:
            pass
        
        # Check Redis
        if USE_REDIS:
            try:
                redis_client.ping()
                checks['redis'] = True
            except:
                checks['redis'] = False
        
        overall_health = all([checks['flask'], checks['ollama']])
        
        return jsonify({
            'status': 'healthy' if overall_health else 'unhealthy',
            'checks': checks
        }), 200 if overall_health else 503
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Background cleanup task
def cleanup_background():
    """Background cleanup task"""
    while True:
        try:
            time.sleep(600)  # Run every 10 minutes
            cleanup_old_sessions()
            logger.info(f"Cleanup completed. Sessions: {len(chat_sessions) if not USE_REDIS else 'Redis'}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Start background cleanup
cleanup_thread = threading.Thread(target=cleanup_background, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Startup checks
    try:
        # Test Ollama connection
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama connected successfully")
        else:
            logger.warning("⚠️  Ollama connection failed")
    except:
        logger.warning("⚠️  Could not connect to Ollama")
    
    # Test Redis connection
    if USE_REDIS:
        try:
            redis_client.ping()
            logger.info("✅ Redis connected successfully")
        except:
            logger.warning("⚠️  Redis connection failed")
    
    logger.info("🚀 Starting Enhanced Claude-Style AI Backend")
    logger.info(f"📊 Max context length: {MAX_CONTEXT_LENGTH}")
    logger.info(f"🧠 Default model: {DEFAULT_MODEL}")
    logger.info(f"💾 Storage: {'Redis' if USE_REDIS else 'In-Memory'}")
    logger.info(f"🔧 Supported languages: {len(CODE_PATTERNS)}")
    
    app.run(
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=True
    )