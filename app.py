from flask import Flask, render_template, request, redirect, url_for
import os
from anomaly_detector import analyze_video  # Must return a dict with 'accident' and 'congestion'

app = Flask(__name__)
app.config['input_video'] = 'video'

# Ensure the upload folder exists
os.makedirs(app.config['input_video'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'video' not in request.files:
            return '⚠ No video file found in the request'

        file = request.files['video']

        # If no file is selected
        if file.filename == '':
            return '⚠ No selected file'

        # Save the video to input_video/ directory
        filename = file.filename
        video_path = os.path.join(app.config['input_video'], filename)
        try:
            file.save(video_path)
        except Exception as e:
            return f"❌ Failed to save video: {str(e)}"

        # Run the analysis model
        try:
            result = analyze_video(video_path)
            accident = "Yes" if result.get("accident") else "No"
            congestion = result.get("congestion", "Unknown")
        except Exception as e:
            return f"❌ Failed to analyze video: {str(e)}"

        # Pass the results to the result page
        return render_template('result.html', accident=accident, congestion=congestion)

    # GET method just loads the upload form
    return render_template('predict.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)