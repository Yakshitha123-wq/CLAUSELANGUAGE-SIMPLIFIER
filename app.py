from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import spacy
from spacy.matcher import PhraseMatcher
import re
import PyPDF2
from io import BytesIO
from docx import Document
from datetime import datetime
from flask import send_file
from io import BytesIO
from functools import wraps
from sqlalchemy import func
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
import torch


app = Flask(__name__)
# Use environment variables so Docker / compose can configure secrets and DB path
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")
# Accept DATABASE_URL env var (e.g. sqlite:///users.db). Defaults to local sqlite file.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///users.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
nlp = spacy.load("en_core_web_sm")

# --- Legal Clause Detection Model (Legal-BERT) ---
try:
    clause_model_name = "nlpaueb/legal-bert-base-uncased"
    clause_tokenizer = AutoTokenizer.from_pretrained(clause_model_name)
    clause_model = AutoModelForSequenceClassification.from_pretrained(
        clause_model_name,
        num_labels=5
    )

    clause_labels = {
        0: "Confidentiality",
        1: "Termination",
        2: "Indemnity",
        3: "Dispute Resolution",
        4: "Governing Law"
    }

    print("Legal-BERT model loaded successfully.")
except Exception as e:
    print(f"[WARN] Legal-BERT could not load: {e}")
    clause_model, clause_tokenizer, clause_labels = None, None, {}

# --- AI-Based Simplification Model ---
try:
    simp_model_name = "tuner007/pegasus_paraphrase"
    simp_tokenizer = AutoTokenizer.from_pretrained(simp_model_name)
    simp_model = AutoModelForSeq2SeqLM.from_pretrained(simp_model_name)

    simplifier = pipeline(
        "text2text-generation", 
        model=simp_model, 
        tokenizer=simp_tokenizer
    )
    print("Simplification model loaded successfully.")
except Exception as e:
    print(f"[WARN] Simplification model could not load: {e}")
    simplifier = None



def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first!", "warning")
            return redirect(url_for("login"))
        # session['is_admin'] is set at login
        if not session.get("is_admin"):
            flash("Access denied. Admins only!", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function

# --- Models ---
# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # ✅ NEW
    uploads = db.relationship("Upload", backref="user", lazy=True)


class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300))
    text = db.Column(db.Text)
    simplified_text = db.Column(db.Text)
    level = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class GlossaryTerm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    term = db.Column(db.String(100), unique=True, nullable=False)
    meaning = db.Column(db.Text, nullable=False)

# --- Create Tables First ---
with app.app_context():
    db.create_all()

# --- Auto-Create Default Admin (Runs Only If No Admin Exists) ---
with app.app_context():
    if not User.query.filter_by(is_admin=True).first():
        default_admin = User(
            username="admin",
            email="admin@example.com",
            age=25,
            gender="Other",
            country="India",
            state="AdminState",
            password=generate_password_hash("Admin@123"),
            is_admin=True
        )
        db.session.add(default_admin)
        db.session.commit()
        print("✔ Default admin created: username=admin password=Admin@123")
    else:
        print("✔ Admin already exists.")

# --- Legal Terms ---
LEGAL_TERMS = {
    "indemnity": "Security or protection against loss or financial burden.",
    "arbitration": "A method to resolve disputes outside of court.",
    "force majeure": "Unforeseeable events that prevent contract performance.",
    "breach": "Failure to perform obligations under a contract.",
    "jurisdiction": "Authority of a court to interpret and enforce laws.",
    "confidentiality": "Clause ensuring sensitive information remains private.",
    "termination": "The process of ending the contract.",
    "liability": "Legal responsibility for actions or omissions.",
    "warranty": "Guarantee regarding quality or performance of goods/services.",
    "governing law": "Legal system/jurisdiction that applies to the contract.",
    "intellectual property": "Rights related to inventions, designs, and creative works.",
    "non-disclosure": "Agreement to not share confidential information.",
    "consideration": "Something of value exchanged in forming a contract.",
    "assignment": "Transfer of rights or obligations to another party.",
    "dispute resolution": "Methods for settling disagreements."
}

# --- Enhanced Legal Term Recognition ---
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

custom_legal_terms = [
    "non-disclosure agreement",
    "force majeure",
    "governing law",
    "indemnity clause",
    "termination clause",
    "confidential information",
    "intellectual property",
    "arbitration",
    "jurisdiction",
    "liability",
    "payment terms",
    "data protection"
]

patterns = [nlp.make_doc(term) for term in custom_legal_terms]
phrase_matcher.add("LEGAL_TERMS", patterns)

# --- Helper Functions ---
import fitz  # PyMuPDF for better PDF extraction

def extract_text_from_pdf(file_stream):
    """Extract and return text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
        return text.strip()
    except Exception as e:
        return f"[ERROR] Could not extract PDF: {str(e)}"


def extract_text_from_docx(file_stream):
    """Extract and return text from a DOCX file."""
    try:
        doc = Document(file_stream)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"[ERROR] Could not extract DOCX: {str(e)}"


def extract_text(file):
    """Detect file type (PDF, DOCX, TXT) and extract accordingly."""
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "[ERROR] Unsupported file type. Only PDF, DOCX, TXT are supported."



def preprocess_text(text):
    """Clean and normalize the extracted text before NLP processing."""
    if not text:
        return ""

    # Normalize quotes, dashes, and spaces
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Remove non-printable characters and multiple newlines
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove extra punctuation spacing
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def highlight_terms(text, found_terms):
    highlighted = text
    for term in found_terms:
        escaped = re.escape(term)
        highlighted = re.sub(
            rf"({escaped})",
            r'<span class="highlight">\1</span>',
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted


def detect_legal_terms(text):
    found = {}
    for term, meaning in LEGAL_TERMS.items():
        if re.search(rf"\b{term}\b", text, re.IGNORECASE):
            found[term] = meaning
    return found

def enhanced_legal_term_recognition(text):
    doc = nlp(text)
    recognized = {}

    # --- 1. Dictionary-based detection (existing) ---
    for term, meaning in LEGAL_TERMS.items():
        if re.search(rf"\b{term}\b", text, re.IGNORECASE):
            recognized[term] = meaning

    # --- 2. spaCy Entity detection ---
    for ent in doc.ents:
        recognized[ent.text] = f"Detected as entity: {ent.label_}"

    # --- 3. PhraseMatcher detection ---
    matches = phrase_matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end].text
        recognized[span] = "Legal phrase detected"

    return recognized

def simplify_text(text, level="Basic"):
    simplifications_basic = {
        r"\bhereinafter\b": "from now on",
        r"\bthereof\b": "of that",
        r"\bwhereas\b": "considering that",
        r"\baforementioned\b": "mentioned earlier",
    }
    simplifications_intermediate = {
        r"\bhereto\b": "to this",
        r"\bhereunder\b": "under this",
        r"\bprior to\b": "before",
        r"\bsubsequent to\b": "after",
        r"\btherein\b": "in that",
    }
    simplifications_advanced = {
        r"\bnotwithstanding\b": "despite",
        r"\bpursuant to\b": "in accordance with",
    }

    simplified = text
    if level == "Basic":
        rules = simplifications_basic
    elif level == "Intermediate":
        rules = {**simplifications_basic, **simplifications_intermediate}
    else:
        rules = {**simplifications_basic, **simplifications_intermediate, **simplifications_advanced}

    for pattern, replacement in rules.items():
        simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
    return simplified

def ai_simplify_text(text, level="Basic"):
    if not simplifier:
        return "Simplification model not available."

    # Split into manageable sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    simplified_output = []

    for sent in sentences:
        if len(sent.strip()) < 5:
            continue

        max_len = 60 if level == "Basic" else 80 if level == "Intermediate" else 120

        try:
            simplified = simplifier(
                sent,
                max_length=max_len,
                num_beams=5,
                early_stopping=True
            )
            simplified_output.append(simplified[0]["generated_text"])
        except:
            simplified_output.append(sent)

    return " ".join(simplified_output)

def analyze_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def detect_clause_type(text):
    """Detect the clause category using Legal-BERT."""
    if not clause_model or not clause_tokenizer:
        return "Unknown"

    inputs = clause_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = clause_model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return clause_labels.get(predicted_class, "Unknown")


    # Keep it short (token limit)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return clause_labels.get(predicted_class, "Unknown")


def segment_clauses(text):
    """Split text into clauses using simple heuristics."""
    clauses = re.split(r'(?<=[.;:])\s+(?=[A-Z])', text)
    return [c.strip() for c in clauses if len(c.strip()) > 30]


# --- Authentication Routes ---
@app.route("/")
def home():
    # If user is logged in, show a welcome message
    if "user_id" in session:
        user = User.query.get(session["user_id"])
        return render_template("home.html", user=user)
    # Otherwise show the public homepage
    return render_template("home.html")

from sqlalchemy import func

@app.route("/admin")
@admin_required
def admin_dashboard():
    # --- Stats ---
    total_users = User.query.count()
    total_uploads = Upload.query.count()

    # --- Recent Uploads ---
    uploads = Upload.query.order_by(Upload.timestamp.desc()).limit(10).all()

    # --- User List ---
    users = User.query.order_by(User.id.asc()).all()

    # --- Analytics Data ---
    # Uploads per day
    daily_stats = (
        db.session.query(func.date(Upload.timestamp), func.count(Upload.id))
        .group_by(func.date(Upload.timestamp))
        .order_by(func.date(Upload.timestamp))
        .all()
    )
    dates = [str(d[0]) for d in daily_stats]
    upload_counts = [d[1] for d in daily_stats]

    # Simplification levels
    level_stats = (
        db.session.query(Upload.level, func.count(Upload.id))
        .group_by(Upload.level)
        .all()
    )
    levels = [l[0] for l in level_stats]
    level_counts = [l[1] for l in level_stats]

    return render_template(
        "admin.html",
        total_users=total_users,
        total_uploads=total_uploads,
        uploads=uploads,
        users=users,
        dates=dates,
        upload_counts=upload_counts,
        levels=levels,
        level_counts=level_counts
    )


@app.route("/admin/toggle_admin/<int:user_id>", methods=["POST"])
@admin_required
def admin_toggle(user_id):
    # Prevent admin from demoting themselves
    if session.get("user_id") == user_id:
        flash("You cannot change your own admin status.", "warning")
        return redirect(url_for("admin_dashboard"))

    user = User.query.get_or_404(user_id)
    user.is_admin = not bool(user.is_admin)
    db.session.commit()
    status = "promoted to admin" if user.is_admin else "demoted from admin"
    flash(f"User {user.username} {status}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    # Prevent admin from deleting themselves
    if session.get("user_id") == user_id:
        flash("You cannot delete your own account.", "warning")
        return redirect(url_for("admin_dashboard"))

    user = User.query.get_or_404(user_id)

    # Optional: delete uploads explicitly (avoid orphan records)
    Upload.query.filter_by(user_id=user.id).delete()

    db.session.delete(user)
    db.session.commit()
    flash(f"User '{user.username}' deleted.", "success")
    return redirect(url_for("admin_dashboard"))



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        age = request.form["age"]
        gender = request.form["gender"]
        country = request.form["country"]
        state = request.form["state"]
        password = request.form["password"]

        if len(password) < 8:
            flash("Password must be at least 8 characters long!", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            age=age,
            gender=gender,
            country=country,
            state=state,
            password=hashed_pw
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            session["is_admin"] = user.is_admin
            return redirect(url_for("simplify"))
        else:
            flash("Invalid username or password!", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# --- Main Tabs ---
@app.route("/simplify", methods=["GET", "POST"])
def simplify():
    if "user_id" not in session:
        return redirect(url_for("login"))
    results = {}
    if request.method == "POST":
        text_input = request.form.get("contract_text")
        uploaded_file = request.files.get("contract_file")
        simpl_level = request.form.get("simpl_level", "Basic")

        if uploaded_file:
            text_input = extract_text(uploaded_file)

        if text_input:
            cleaned_text = preprocess_text(text_input)
            simplified = ai_simplify_text(cleaned_text, level=simpl_level)
            results["found_terms"] = enhanced_legal_term_recognition(cleaned_text)
            results["simplified_text"] = simplified
            results["original_text"] = text_input
            results["highlighted_text"] = highlight_terms(text_input, results["found_terms"])
            results["simpl_level"] = simpl_level

                    # --- Clause Detection ---
            clauses = segment_clauses(cleaned_text)
            clause_predictions = []
            for clause in clauses:
                detected = detect_clause_type(clause)
                clause_predictions.append({"text": clause, "type": detected})

            results["clauses"] = clause_predictions

            # Save upload in DB
            new_upload = Upload(
                filename=uploaded_file.filename if uploaded_file else "Manual Input",
                text=cleaned_text,
                simplified_text=simplified,
                level=simpl_level,
                user_id=session["user_id"]
            )
            db.session.add(new_upload)
            db.session.commit()

    return render_template("simplify.html", results=results, username=session.get("username"))

@app.route("/uploads")
def uploads():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user_uploads = Upload.query.filter_by(user_id=session["user_id"]).order_by(Upload.timestamp.desc()).all()
    return render_template("uploads.html", uploads=user_uploads)

@app.route("/glossary")
def glossary():
    if "user_id" not in session:
        return redirect(url_for("login"))
    terms = GlossaryTerm.query.order_by(GlossaryTerm.term.asc()).all()
    return render_template("glossary.html", terms=terms)

# --- Admin Glossary Management ---

@app.route("/admin/glossary")
@admin_required
def admin_glossary():
    terms = GlossaryTerm.query.order_by(GlossaryTerm.term.asc()).all()
    return render_template("admin_glossary.html", terms=terms)


@app.route("/admin/glossary/add", methods=["GET", "POST"])
@admin_required
def add_glossary_term():
    if request.method == "POST":
        term = request.form["term"].strip()
        meaning = request.form["meaning"].strip()
        if not term or not meaning:
            flash("Both term and meaning are required!", "danger")
            return redirect(url_for("add_glossary_term"))

        # Prevent duplicates
        if GlossaryTerm.query.filter_by(term=term).first():
            flash("Term already exists!", "warning")
            return redirect(url_for("admin_glossary"))

        new_term = GlossaryTerm(term=term, meaning=meaning)
        db.session.add(new_term)
        db.session.commit()
        flash(f"Added term '{term}'.", "success")
        return redirect(url_for("admin_glossary"))

    return render_template("admin_glossary_add.html")


@app.route("/admin/glossary/edit/<int:term_id>", methods=["GET", "POST"])
@admin_required
def edit_glossary_term(term_id):
    term = GlossaryTerm.query.get_or_404(term_id)
    if request.method == "POST":
        new_term = request.form["term"].strip()
        new_meaning = request.form["meaning"].strip()

        if not new_term or not new_meaning:
            flash("Both term and meaning are required!", "danger")
            return redirect(url_for("edit_glossary_term", term_id=term_id))

        term.term = new_term
        term.meaning = new_meaning
        db.session.commit()
        flash("Glossary term updated!", "success")
        return redirect(url_for("admin_glossary"))

    return render_template("admin_glossary_edit.html", term=term)


@app.route("/admin/glossary/delete/<int:term_id>", methods=["POST"])
@admin_required
def delete_glossary_term(term_id):
    term = GlossaryTerm.query.get_or_404(term_id)
    db.session.delete(term)
    db.session.commit()
    flash(f"Deleted term '{term.term}'.", "success")
    return redirect(url_for("admin_glossary"))


@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    return render_template("profile.html", user=user)


@app.route("/change_password", methods=["POST"])
def change_password():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])
    current = request.form["current_password"]
    new_pw = request.form["new_password"]
    confirm_pw = request.form["confirm_password"]

    if not check_password_hash(user.password, current):
        flash("Current password is incorrect!", "danger")
    elif new_pw != confirm_pw:
        flash("New passwords do not match!", "danger")
    else:
        user.password = generate_password_hash(new_pw)
        db.session.commit()
        flash("Password updated successfully!", "success")

    return redirect(url_for("profile"))

@app.route("/download/<int:upload_id>")
def download_upload(upload_id):
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))

    upload = Upload.query.get_or_404(upload_id)
    if upload.user_id != session["user_id"]:
        flash("You are not authorized to download this file.", "danger")
        return redirect(url_for("history"))

    # Create a BytesIO object for download
    buffer = BytesIO()
    buffer.write(upload.simplified_text.encode('utf-8'))
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Simplified_{upload.filename}.txt",
        mimetype='text/plain'
    )

if __name__ == "__main__":
    # only for local dev if you still run python app.py (not used with gunicorn)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))