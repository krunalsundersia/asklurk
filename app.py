import os, json, logging, sys, re, uuid
from flask import Flask, request, Response, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 300_000
tokens_used = 0

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

app = Flask(__name__)
CORS(app)

KEY = os.getenv("OPENROUTER_API_KEY")
if not KEY:
    logging.error("OPENROUTER_API_KEY missing – export it or add to .env")
    sys.exit(1)

# Define all 6 AI models with their personalities
MODELS = {
    "logic": {"name": "LogicBot", "description": "analytical, structured, step-by-step"},
    "creative": {"name": "CreativeBot", "description": "poetic, metaphorical, emotional"},
    "technical": {"name": "TechBot", "description": "precise, detailed, code-focused"},
    "concise": {"name": "BriefBot", "description": "succinct, to-the-point, efficient"},
    "friendly": {"name": "FriendBot", "description": "warm, supportive, conversational"},
    "expert": {"name": "ExpertBot", "description": "comprehensive, authoritative, in-depth"}
}

# System prompts for each model
SYSTEM_PROMPTS = {
    "logic": "You are LogicBot — analytical, structured, step-by-step. Provide clear, logical reasoning and systematic approaches.",
    "creative": "You are CreativeBot — poetic, metaphorical, emotional. Use imaginative language and creative perspectives.",
    "technical": "You are TechBot — highly technical, precise, focused on implementation details and code. Provide specific technical insights.",
    "concise": "You are BriefBot — extremely concise, direct, and to-the-point. Avoid unnecessary words while maintaining clarity.",
    "friendly": "You are FriendBot — warm, supportive, and conversational. Make the user feel comfortable with a friendly tone.",
    "expert": "You are ExpertBot — comprehensive, authoritative, and in-depth. Provide thorough explanations with expert insights."
}

# OpenRouter models to use
OPENROUTER_MODELS = {
    "logic": "openai/gpt-3.5-turbo",
    "creative": "anthropic/claude-3-sonnet", 
    "technical": "meta-llama/llama-3-70b-instruct",
    "concise": "google/gemini-pro",
    "friendly": "mistralai/mistral-7b-instruct",
    "expert": "microsoft/wizardlm-2-8x22b",
    "asklurk": "openai/gpt-4"  # Use a powerful model for synthesis
}

def generate(bot_name: str, system: str, user: str):
    global tokens_used
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=KEY
        )
        
        # Calculate tokens for the request
        system_tokens = len(enc.encode(system))
        user_tokens = len(enc.encode(user))
        tokens_used += system_tokens + user_tokens
        
        model = OPENROUTER_MODELS.get(bot_name, "deepseek/deepseek-r1-0528:free")
        
        stream = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:5000", 
                "X-Title": "6-Model-Chat"
            },
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True,
        )
        
        bot_tokens = 0
        full_response = ""
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_response += delta
                bot_tokens += len(enc.encode(delta))
                yield f"data: {json.dumps({'bot': bot_name, 'text': delta})}\n\n"
            
            if chunk.choices and chunk.choices[0].finish_reason:
                break
        
        # Update global token count
        tokens_used += bot_tokens
        yield f"data: {json.dumps({'bot': bot_name, 'done': True, 'tokens': tokens_used})}\n\n"
        
    except Exception as exc:
        logging.error(f"Error generating response for {bot_name}: {str(exc)}")
        yield f"data: {json.dumps({'bot': bot_name, 'error': str(exc)})}\n\n"

@app.route("/asklurk", methods=["POST"])
def asklurk():
    data = request.json or {}
    answers = data.get("answers", {})
    prompt = data.get("prompt", "")
    
    # Check if we have responses from all 6 models
    missing_models = [key for key in MODELS.keys() if key not in answers]
    if missing_models:
        return jsonify(best="", error=f"Missing responses from: {', '.join(missing_models)}"), 400
    
    # Merge all responses using AI
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", 
            api_key=KEY
        )
        
        # Prepare the merged content
        merged_content = f"Original question: {prompt}\n\n"
        for key in MODELS.keys():
            merged_content += f"## {MODELS[key]['name']}:\n{answers[key]}\n\n"
        
        # Ask AI to synthesize the best answer
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:5000", 
                "X-Title": "6-Model-Chat"
            },
            model=OPENROUTER_MODELS["asklurk"],
            messages=[
                {
                    "role": "system", 
                    "content": """You are AskLurk - an expert AI synthesizer. Your task is to analyze responses from 6 different AI models and create the single best, most comprehensive answer.

Guidelines:
1. Combine the strengths of each model's approach
2. Maintain accuracy and clarity
3. Provide a well-structured, comprehensive response
4. Don't just repeat what others said - synthesize new insights
5. Keep it engaging and informative"""
                },
                {
                    "role": "user", 
                    "content": f"""Please analyze these 6 AI responses to the question: "{prompt}"

Here are the responses:
{merged_content}

Please provide the best synthesized answer that combines the strengths of all approaches:"""
                }
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        best_answer = response.choices[0].message.content
        
        # Update token usage for AskLurk
        global tokens_used
        asklurk_tokens = len(enc.encode(best_answer))
        tokens_used += asklurk_tokens
        
        return jsonify(best=best_answer, tokens_used=tokens_used)
        
    except Exception as e:
        logging.error(f"AskLurk error: {str(e)}")
        # Fallback: return a combined response
        combined_response = f"## AskLurk Synthesis\n\nBased on analysis of 6 AI models:\n\n"
        for key, response_text in answers.items():
            combined_response += f"**{MODELS[key]['name']}**: {response_text[:200]}...\n\n"
        return jsonify(best=combined_response, error="AI synthesis failed, using combined view")

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify(urls=[], error="No files provided"), 400
    
    files = request.files.getlist('files')
    urls = []
    
    for file in files:
        if file.filename == '':
            continue
        
        # Validate file type
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.pdf', '.txt', '.doc', '.docx'}
        ext = os.path.splitext(file.filename)[1].lower()
        
        if ext not in allowed_extensions:
            continue
            
        # Generate unique filename
        name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(UPLOAD_FOLDER, name)
        
        try:
            file.save(path)
            urls.append(f"/static/uploads/{name}")
        except Exception as e:
            logging.error(f"Error saving file {file.filename}: {str(e)}")
    
    return jsonify(urls=urls)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tokens", methods=["GET"])
def get_tokens():
    """Endpoint to get current token usage"""
    return jsonify({
        "tokens_used": tokens_used,
        "token_limit": TOKEN_LIMIT,
        "remaining_tokens": TOKEN_LIMIT - tokens_used,
        "usage_percentage": (tokens_used / TOKEN_LIMIT) * 100
    })

@app.route("/reset-tokens", methods=["POST"])
def reset_tokens():
    """Endpoint to reset token counter"""
    global tokens_used
    tokens_used = 0
    return jsonify({"message": "Token counter reset", "tokens_used": tokens_used})

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    fileUrls = data.get("fileUrls", [])
    
    if not prompt and not fileUrls:
        return jsonify(error="Empty prompt and no files provided"), 400
    
    # Check token limit
    if tokens_used >= TOKEN_LIMIT:
        return jsonify(error=f"Token limit reached ({tokens_used}/{TOKEN_LIMIT})"), 429
    
    # Prepare prompt with file information
    full_prompt = prompt
    if fileUrls:
        full_prompt += "\n\n[User uploaded files: " + ", ".join(fileUrls) + "]"

    def event_stream():
        # Create generators for all 6 models
        generators = {
            key: generate(key, SYSTEM_PROMPTS[key], full_prompt) 
            for key in MODELS.keys()
        }
        
        # Stream responses from all models concurrently
        active_generators = list(generators.items())
        completed_bots = set()
        
        while active_generators:
            for bot_name, generator in active_generators[:]:
                try:
                    chunk = next(generator)
                    yield chunk
                    
                except StopIteration:
                    active_generators.remove((bot_name, generator))
                    completed_bots.add(bot_name)
                
                except Exception as e:
                    logging.error(f"Error in generator for {bot_name}: {str(e)}")
                    active_generators.remove((bot_name, generator))
                    completed_bots.add(bot_name)
                    yield f"data: {json.dumps({'bot': bot_name, 'error': str(e)})}\n\n"
        
        # Final completion message
        yield f"data: {json.dumps({'all_done': True, 'tokens': tokens_used})}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive", 
            "X-Accel-Buffering": "no"
        },
    )

# Create uploads directory
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)