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
    """Load the DistilGPT-2 model and tokenizer (lightweight alternative)"""
    global tokenizer, model
    try:
        logger.info("Loading DistilGPT-2 model...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def generate_response(user_message: str, system_prompt: str = None) -> str:
    """Generate response using DistilGPT-2 model"""
    try:
        # Create a simple prompt for DistilGPT-2
        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        else:
            prompt = f"User: {user_message}\nAssistant:"
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        # Clean up the response
        response = response.split("User:")[0].strip()  # Remove any follow-up user prompts
        
        return response if response else "I'm here to help! Could you please rephrase your question?"
        
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
            "model": "distilbert/distilgpt2",
            "timestamp": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        "model_name": "distilbert/distilgpt2",
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