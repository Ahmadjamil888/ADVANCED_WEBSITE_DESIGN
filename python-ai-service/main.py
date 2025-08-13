from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
tokenizer = None
model = None

def load_model():
    """Load the DeepSeek-R1 model and tokenizer"""
    global tokenizer, model
    try:
        logger.info("Loading DeepSeek-R1 model...")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def generate_response(user_message: str, system_prompt: str = None) -> str:
    """Generate response using DeepSeek-R1 model"""
    try:
        # Prepare messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_message})
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for generating AI responses"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        user_message = data['message']
        system_prompt = data.get('system_prompt', 
            "You are Zehan AI, an advanced AI assistant created by Zehan X Technologies. "
            "You are knowledgeable about AI, machine learning, web development, and business solutions. "
            "Be helpful, professional, and showcase the capabilities of Zehan X Technologies. "
            "Keep responses concise but informative."
        )
        
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please wait for initialization to complete."
            }), 503
        
        # Generate response
        response = generate_response(user_message, system_prompt)
        
        return jsonify({
            "response": response,
            "model": "deepseek-ai/DeepSeek-R1",
            "timestamp": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        "model_name": "deepseek-ai/DeepSeek-R1",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(model.device) if model else None,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    })

if __name__ == '__main__':
    logger.info("Starting Zehan AI Service...")
    
    # Load model on startup
    if load_model():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Exiting...")
        exit(1)