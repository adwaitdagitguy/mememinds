import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from mememind import MemeMind  # Import the updated MemeMind class

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)  # Enable cross-origin requests

# Initialize MemeMind
meme_generator = MemeMind()

# Create output directory if it doesn't exist
os.makedirs("static/generated", exist_ok=True)

# Serve the HTML file from the static folder
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/generate-meme', methods=['POST'])
def generate_meme():
    data = request.json
    topic = data.get('topic', '')
    
    if not topic:
        return jsonify({'error': 'No topic provided'}), 400
    
    try:
        # Generate the meme image
        meme_image = meme_generator.generate_meme(topic)
        
        # Save the meme to a file
        filename = f"meme_{hash(topic) % 10000}.png"
        filepath = os.path.join("static/generated", filename)
        meme_image.save(filepath)
        
        # Create URL for the image
        image_url = f"/static/generated/{filename}"
        
        # Convert image to base64 for immediate display
        buffered = io.BytesIO()
        meme_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'imageUrl': image_url,
            'imageData': f"data:image/png;base64,{img_str}",
            'topic': topic
        })
    
    except Exception as e:
        print(f"Error generating meme: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/generated/<filename>')
def serve_image(filename):
    return send_from_directory('static/generated', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)