from flask import Flask, request, render_template_string
import os

# We import these INSIDE the function later to make the website open instantly
# import tensorflow as tf 
# import numpy as np

app = Flask(__name__)

# Basic HTML Template for a "User-Centric Interface" 
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Plant Disease Detection</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f7f6; }
        .container { background: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 10px #ccc; }
        h1 { color: #2e7d32; }
        input { margin: 20px 0; }
        .btn { background: #2e7d32; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Plant Disease Detection System</h1>
        <p>Upload a leaf image for rapid, accurate diagnostics[cite: 6].</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required><br>
            <input type="submit" class="btn" value="Analyze Leaf Image">
        </form>
        {% if result %}
            <div style="margin-top:20px; padding:10px; border:2px solid #2e7d32;">
                <h2>Diagnosis: {{ result }}</h2>
                <p><i>Recommendation: Follow targeted treatment strategies[cite: 11].</i></p>
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # LAZY LOADING: Libraries are loaded only when the button is clicked
        import tensorflow as tf
        import numpy as np
        from PIL import Image

        file = request.files['file']
        if file:
            # 1. Load the Model [cite: 21]
            model = tf.keras.models.load_model("plant_model.h5")
            
            # 2. Image Preprocessing
            img = Image.open(file).convert('RGB').resize((128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, 0) 

            # 3. Prediction (Targeting high diagnostic accuracy) [cite: 15]
            class_names = ['Early', 'Healthy', 'Late']
            predictions = model.predict(img_array)
            result = class_names[np.argmax(predictions)]
            
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    # Disable 'debug' and 'reloader' to prevent IDLE from hanging
    app.run(port=8080, debug=False, use_reloader=False)
