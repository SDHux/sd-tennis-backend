"""
SD Tennis Lessons — Web Server
Connects the student upload portal to the Claude analysis pipeline.

Setup:
    pip install flask anthropic opencv-python-headless Pillow
    export ANTHROPIC_API_KEY=your_key_here
    python app.py

Deploy options:
    - Railway.app (easiest, ~$5/month, handles video storage)
    - Render.com (free tier available)
    - Heroku
    - Your own VPS (DigitalOcean, Linode)

For production, swap local file storage for S3/Cloudflare R2.
"""

import os
import uuid
import tempfile
from flask_cors import CORS
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from analysis_pipeline import analyze_tennis_video

app = Flask(__name__, static_folder="../frontend")
CORS(app)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# Max upload size: 500MB (adjust based on your server)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'm4v', 'webm'}
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "tennis_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    """Serve the student portal"""
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    Accepts multipart form with:
        - video (file)
        - stroke_type (string)
        - name (string, optional)
        - age (string, optional)
        - level (string)
        - concerns (string, optional)
    
    Returns JSON with analysis text.
    """

    # Validate file upload
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file uploaded'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(video_file.filename):
        return jsonify({
            'success': False,
            'error': 'Unsupported file format. Please upload MP4, MOV, or AVI.'
        }), 400

    # Build student info from form fields
    student_info = {}
    if request.form.get('name'):
        student_info['name'] = request.form['name'].strip()[:50]  # cap length
    if request.form.get('age'):
        try:
            student_info['age'] = int(request.form['age'])
        except ValueError:
            pass
    if request.form.get('level'):
        student_info['level'] = request.form['level']
    if request.form.get('concerns'):
        student_info['concerns'] = request.form['concerns'].strip()[:500]

    stroke_type = request.form.get('stroke_type', 'general')

    # Save video temporarily
    filename = f"{uuid.uuid4().hex}_{secure_filename(video_file.filename)}"
    video_path = UPLOAD_FOLDER / filename

    try:
        video_file.save(str(video_path))

        # Run the analysis pipeline
        result = analyze_tennis_video(
            video_path=str(video_path),
            stroke_type=stroke_type,
            student_info=student_info if student_info else None,
            api_key=ANTHROPIC_API_KEY
        )

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        import traceback
        print(f"EXCEPTION in /api/analyze: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

    finally:
        # Always clean up the uploaded file
        if video_path.exists():
            video_path.unlink()


@app.route('/api/health')
def health():
    """Simple health check for monitoring"""
    return jsonify({
        'status': 'ok',
        'api_key_set': bool(ANTHROPIC_API_KEY),
        'upload_dir': str(UPLOAD_FOLDER)
    })


# ─────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'success': False,
        'error': 'Video file is too large. Please keep it under 500MB.'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'An unexpected server error occurred. Please try again.'
    }), 500


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == '__main__':
    if not ANTHROPIC_API_KEY:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not set. Set it before running.")
        print("   export ANTHROPIC_API_KEY=your_key_here")
    else:
        print("✅ Anthropic API key found")

    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print("🎾 SD Tennis Lessons Analysis Server starting...")
    print("   Visit http://localhost:5000 to open the portal\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
