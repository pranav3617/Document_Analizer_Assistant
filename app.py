import shutil
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, make_response

from rag import session_manager


# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Allowed File Types / Limits
# --------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_MB = 200
SESSION_COOKIE_NAME = "rag_session_id"


# --------------------------------------------------
# Helper: get session id from cookie, or generate one
# --------------------------------------------------

def get_session_id():
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    is_new = session_id is None

    if is_new:
        session_id = uuid.uuid4().hex

    return session_id, is_new


def set_session_cookie(response, session_id):
    # Teams loads this page inside a cross-site iframe. Browsers only
    # send cookies into a cross-site iframe if SameSite=None + Secure,
    # but Secure cookies require HTTPS — so we only use them when the
    # request actually arrived over HTTPS (i.e. your deployed Render
    # URL), and fall back to Lax for plain local http development.
    is_https = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"

    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        httponly=True,
        samesite="None" if is_https else "Lax",
        secure=is_https,
        max_age=60 * 60 * 24  # 1 day
    )
    return response


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    session_id, is_new = get_session_id()

    index_file = TEMPLATE_DIR / "index.html"

    if not index_file.exists():
        return jsonify({"detail": "index.html not found"}), 404

    response = make_response(index_file.read_text(encoding="utf-8"))

    if is_new:
        set_session_cookie(response, session_id)

    return response


# --------------------------------------------------
# Teams Tab Configuration Page
# Teams calls this URL when a user adds this app as a meeting tab.
# --------------------------------------------------

@app.route("/teams-config", methods=["GET"])
def teams_config():
    config_file = TEMPLATE_DIR / "teams_config.html"

    if not config_file.exists():
        return jsonify({"detail": "teams_config.html not found"}), 404

    return make_response(config_file.read_text(encoding="utf-8"))


# --------------------------------------------------
# Upload Documents (accepts one or many files in a single request)
# Flask handles each request in a worker thread by default
# (threaded=True below), so the blocking embedding / vector
# store work here does not freeze other requests.
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_document():

    session_id, is_new = get_session_id()
    rag = session_manager.get_session(session_id)

    # getlist handles both a single file and multiple files sent
    # under the same "file" field name
    files = request.files.getlist("file")
    files = [f for f in files if f and f.filename]

    if not files:
        return jsonify({"detail": "No file selected"}), 400

    session_upload_dir = UPLOAD_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    rejected = []

    for file in files:
        extension = Path(file.filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            rejected.append({"file": file.filename, "reason": "unsupported file type"})
            continue

        file.stream.seek(0, 2)
        size_mb = file.stream.tell() / (1024 * 1024)
        file.stream.seek(0)

        if size_mb > MAX_FILE_SIZE_MB:
            rejected.append({"file": file.filename, "reason": f"exceeds {MAX_FILE_SIZE_MB} MB limit"})
            continue

        file_path = session_upload_dir / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.stream, buffer)

        saved_paths.append(str(file_path))

    if not saved_paths:
        return jsonify({
            "detail": "No valid files to process",
            "rejected": rejected
        }), 400

    try:
        result = rag.process_documents(saved_paths)
        result["files_rejected"] = rejected

        response = jsonify({
            "message": result["message"],
            "details": result
        })

        if is_new:
            set_session_cookie(response, session_id)

        return response

    except Exception as e:
        for path_str in saved_paths:
            p = Path(path_str)
            if p.exists():
                p.unlink()

        return jsonify({"detail": f"Document processing failed: {str(e)}"}), 500


# --------------------------------------------------
# Clear all uploaded documents for this session
# --------------------------------------------------

@app.route("/clear", methods=["POST"])
def clear_documents():

    session_id, is_new = get_session_id()
    rag = session_manager.get_session(session_id)

    rag.clear()

    session_upload_dir = UPLOAD_DIR / session_id
    if session_upload_dir.exists():
        shutil.rmtree(session_upload_dir, ignore_errors=True)

    response = jsonify({"message": "All documents cleared for this session"})

    if is_new:
        set_session_cookie(response, session_id)

    return response


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask_question():

    session_id, is_new = get_session_id()
    rag = session_manager.get_session(session_id)

    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()

    if not question:
        return jsonify({"detail": "Question cannot be empty"}), 400

    # No document check here anymore — the agent can answer using
    # web search or the calculator even with nothing uploaded.

    try:
        answer = rag.ask(question)

        response = jsonify({"answer": answer})

        if is_new:
            set_session_cookie(response, session_id)

        return response

    except Exception as e:
        return jsonify({"detail": f"Failed to generate answer: {str(e)}"}), 500


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    loaded = False

    if session_id and session_id in session_manager._sessions:
        loaded = session_manager._sessions[session_id].document_loaded

    return jsonify({"status": "online", "document_loaded": loaded})


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    import os

    # Render (and most PaaS hosts) inject PORT; default to 8000 locally.
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    # threaded=True lets Flask handle multiple requests concurrently
    # instead of blocking on embedding/LLM calls one at a time.
    # In production this file isn't even used to start the server —
    # gunicorn imports `app` directly (see Procfile).
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)