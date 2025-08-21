import os
import sqlite3
import logging
import uuid
import re
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_migrate import Migrate
from dotenv import load_dotenv
from tools import (
    fix_windows_encoding, create_llm, test_llm_connection,
    fetch_unseen_emails, agent_classify_and_assign,
    send_email_via_smtp, keyword_based_classify, extract_thread_token,
    extract_text_from_file
)
from models import db, Support, TeamCategory, KnowledgeDocument
import threading
import time
from config import DATABASE_PATH, KNOWLEDGE_BASE_PATH
from agents import create_email_fetcher_agent, create_classifier_agent, create_responder_agent
import certifi
import httpx
from sqlalchemy import inspect
import ssl
from crewai import Crew, Task
import litellm
from litellm import completion
from werkzeug.utils import secure_filename
from datetime import datetime

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY"] = "False"

# Configure SSL to use certifi's CA bundle
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# Patch HTTPX client to explicitly trust certifi's CA
def create_verified_client():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    return httpx.Client(verify=ssl_context)

# Configure LiteLLM to bypass SSL verification (only if absolutely necessary)
litellm.ssl_verify = False
litellm.drop_params = True

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
fix_windows_encoding()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "devkey")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Ensure instance folder and DB file exist
instance_dir = os.path.dirname(DATABASE_PATH)
try:
    os.makedirs(instance_dir, exist_ok=True)
    if not os.access(instance_dir, os.W_OK):
        raise PermissionError(f"No write permission for directory: {instance_dir}")
    if not os.path.exists(DATABASE_PATH):
        sqlite3.connect(DATABASE_PATH).close()
    if not os.access(DATABASE_PATH, os.R_OK | os.W_OK):
        raise PermissionError(f"No read/write permission for database file: {DATABASE_PATH}")
except Exception as e:
    logger.error(f"Database environment error: {e}")
    raise

# Ensure knowledge base directory exists
try:
    os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
    if not os.access(KNOWLEDGE_BASE_PATH, os.W_OK):
        raise PermissionError(f"No write permission for knowledge base directory: {KNOWLEDGE_BASE_PATH}")
except Exception as e:
    logger.error(f"Knowledge base directory error: {e}")
    raise

db.init_app(app)
migrate = Migrate(app, db)

# Initialize DB if not exists
with app.app_context():
    try:
        db.create_all()
        inspector = inspect(db.engine)
        if not inspector.has_table("support"):
            raise Exception("Support table not created")
        if not inspector.has_table("team_categories"):
            raise Exception("TeamCategories table not created")
        if not inspector.has_table("knowledge_documents"):
            raise Exception("KnowledgeDocuments table not created")
        # Add default Internal team and Other category if none exist
        if not TeamCategory.query.filter_by(team_name="Internal", category_name="Other").first():
            internal = TeamCategory(
                team_name="Internal",
                team_email="internalbtrnsfrmd@gmail.com",
                team_description="Default team for unclassified emails",
                category_name="Other",
                category_description="Default other category"
            )
            db.session.add(internal)
            db.session.commit()
            # Create folder for Internal/Other
            os.makedirs(os.path.join(KNOWLEDGE_BASE_PATH, "Internal", "Other"), exist_ok=True)
    except Exception as e:
        logger.error(f"Database init error: {e}")
        raise

# Create LLM and agents
llm = create_llm()
if not test_llm_connection(llm):
    logger.warning("LLM connection failed at startup.")

email_fetcher_agent = create_email_fetcher_agent(llm)
classifier_agent = create_classifier_agent(llm)
responder_agent = create_responder_agent(llm)

# Email polling
POLL_INTERVAL = 60

def generate_thread_token():
    """Generate a unique thread token"""
    return f"SUPPORT-{uuid.uuid4().hex[:8].upper()}"

def poll_and_store():
    with app.app_context():
        while True:
            try:
                new_messages = fetch_unseen_emails()
                if new_messages:
                    for sender, subject, original_subject, body, email_date, is_reply in new_messages:
                        # Check if this is a reply by looking for thread token
                        thread_token = extract_thread_token(body, subject)
                        
                        if thread_token:
                            # Try to find the original thread by token
                            original_thread = Support.query.filter_by(thread_token=thread_token).first()
                            
                            if original_thread:
                                # This is a reply - inherit classification from original thread
                                rec = Support(
                                    sender=sender,
                                    subject=subject,
                                    request=body[:4000],
                                    category=original_thread.category,
                                    team_assigned=original_thread.team_assigned,
                                    status=original_thread.status,
                                    action="Customer reply",
                                    timestamps=email_date,
                                    thread_id=original_thread.thread_id,
                                    thread_token=thread_token,
                                    is_customer_reply=True
                                )
                                db.session.add(rec)
                                db.session.commit()
                                continue
                        
                        # Fallback to subject-based matching if no token found
                        if is_reply:
                            original_thread = Support.query.filter(
                                Support.sender == sender,
                                db.or_(
                                    Support.subject.contains(original_subject),
                                    db.func.substr(Support.subject, 1, 100).contains(original_subject)
                                )
                            ).order_by(Support.timestamps.asc()).first()
                            
                            if original_thread:
                                rec = Support(
                                    sender=sender,
                                    subject=subject,
                                    request=body[:4000],
                                    category=original_thread.category,
                                    team_assigned=original_thread.team_assigned,
                                    status=original_thread.status,
                                    action="Customer reply",
                                    timestamps=email_date,
                                    thread_id=original_thread.thread_id,
                                    thread_token=original_thread.thread_token,
                                    is_customer_reply=True
                                )
                                db.session.add(rec)
                                db.session.commit()
                                continue
                        
                        # Not a reply or original thread not found - create new thread
                        existing = Support.query.filter_by(
                            sender=sender,
                            subject=subject,
                            request=body[:4000]
                        ).first()
                        
                        if existing:
                            continue  # Skip duplicates
                            
                        category, team, confidence, reasoning = agent_classify_and_assign(classifier_agent, subject, body)
                        if not category:
                            category, team = keyword_based_classify(subject + " " + body)
                        
                        # Generate thread token and ID for new conversations
                        thread_token = generate_thread_token()
                        thread_id = f"{sender}_{thread_token}"
                        
                        rec = Support(
                            sender=sender,
                            subject=subject,
                            request=body[:4000],
                            category=category,
                            team_assigned=team,
                            action="Auto-classified",
                            timestamps=email_date,
                            thread_id=thread_id,
                            thread_token=thread_token,
                            is_customer_reply=False
                        )
                        db.session.add(rec)
                    db.session.commit()
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)
                time.sleep(POLL_INTERVAL)

@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    # Get the latest message from each thread using a window function approach
    from sqlalchemy import func, over
    
    # Subquery to rank messages within each thread by timestamp
    ranked_messages = db.session.query(
        Support,
        func.row_number().over(
            partition_by=Support.thread_id,
            order_by=Support.timestamps.desc()
        ).label('rn')
    ).subquery()
    
    # Get only the latest message from each thread and non-threaded messages
    items = db.session.query(ranked_messages).filter(
        (ranked_messages.c.rn == 1) | (ranked_messages.c.thread_id.is_(None))
    ).order_by(ranked_messages.c.timestamps.desc()).all()

    # Create a list of items with their thread counts
    items_with_counts = []
    for item in items:
        # The item is actually a tuple with the Support object and rank
        support_item = item[0] if isinstance(item, tuple) else item
        
        # Calculate the thread message count
        thread_count = (
            Support.query.filter_by(thread_id=support_item.thread_id).count()
            if support_item.thread_id
            else 1
        )
        items_with_counts.append({
            "item": support_item,
            "thread_count": thread_count
        })

    teams_query = db.session.query(TeamCategory.team_name).distinct().all()
    teams = [t[0] for t in teams_query]
    category_lookup = {}
    for tc in TeamCategory.query.all():
        if tc.team_name not in category_lookup:
            category_lookup[tc.team_name] = []
        category_lookup[tc.team_name].append(tc.category_name)
    
    return render_template(
        "dashboard.html",
        items=items_with_counts,
        teams=teams,
        category_lookup=category_lookup
    )

@app.route("/knowledge")
def knowledge_base():
    teams_query = db.session.query(TeamCategory.team_name).distinct().order_by(TeamCategory.team_name).all()
    teams = [t[0] for t in teams_query]
    team_emails = {}
    team_descriptions = {}
    category_lookup = {}
    category_descriptions = {}
    for team in teams:
        category_lookup[team] = []
        category_descriptions[team] = {}
        tc = TeamCategory.query.filter_by(team_name=team).all()
        team_emails[team] = tc[0].team_email
        team_descriptions[team] = tc[0].team_description or ""
        for cat in tc:
            category_lookup[team].append(cat.category_name)
            category_descriptions[team][cat.category_name] = cat.category_description or ""
    
    def get_documents(team, category):
        return KnowledgeDocument.query.filter_by(team_name=team, category_name=category).order_by(KnowledgeDocument.uploaded_at.desc()).all()
    
    return render_template(
        "knowledge.html",
        teams=teams,
        team_emails=team_emails,
        team_descriptions=team_descriptions,
        category_lookup=category_lookup,
        category_descriptions=category_descriptions,
        get_documents=get_documents
    )

@app.route("/knowledge/upload/<team_name>/<category_name>", methods=["POST"])
def upload_document(team_name, category_name):
    team = TeamCategory.query.filter_by(team_name=team_name, category_name=category_name).first()
    if not team:
        flash("Invalid team or category", "danger")
        return redirect(url_for("knowledge_base"))
    
    if 'file' not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for("knowledge_base"))
    
    file = request.files['file']
    if file.filename == '':
        flash("No file selected", "danger")
        return redirect(url_for("knowledge_base"))
    
    allowed_extensions = {'.txt', '.pdf', '.doc', '.docx'}
    if not os.path.splitext(file.filename)[1].lower() in allowed_extensions:
        flash("Invalid file type. Allowed types: txt, pdf, doc, docx", "danger")
        return redirect(url_for("knowledge_base"))
    
    filename = secure_filename(file.filename)
    team_folder = os.path.join(KNOWLEDGE_BASE_PATH, team_name)
    category_folder = os.path.join(team_folder, category_name)
    os.makedirs(category_folder, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(category_folder, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    file_path = os.path.join(category_folder, unique_filename)
    file.save(file_path)
    
    # Extract content from file
    content = extract_text_from_file(file_path)
    
    # Store document metadata and content in the database
    doc = KnowledgeDocument(
        team_name=team_name,
        category_name=category_name,
        filename=unique_filename,
        file_path=file_path,
        uploaded_at=datetime.utcnow(),
        content=content
    )
    db.session.add(doc)
    db.session.commit()
    
    flash(f"Document '{unique_filename}' uploaded successfully", "success")
    return redirect(url_for("knowledge_base"))

@app.route("/knowledge/download/<int:doc_id>")
def download_document(doc_id):
    doc = KnowledgeDocument.query.get(doc_id)
    if not doc:
        flash("Document not found", "danger")
        return redirect(url_for("knowledge_base"))
    
    return send_from_directory(
        directory=os.path.dirname(doc.file_path),
        path=os.path.basename(doc.file_path),
        as_attachment=True
    )

@app.route("/knowledge/update/<int:doc_id>", methods=["POST"])
def update_document(doc_id):
    doc = KnowledgeDocument.query.get(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    new_content = request.form.get("content", doc.content)
    
    doc.content = new_content
    
    try:
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/knowledge/delete/<int:doc_id>", methods=["POST"])
def delete_document(doc_id):
    doc = KnowledgeDocument.query.get(doc_id)
    if not doc:
        flash("Document not found", "danger")
        return redirect(url_for("knowledge_base"))
    
    try:
        os.remove(doc.file_path)
        db.session.delete(doc)
        db.session.commit()
        flash(f"Document '{doc.filename}' deleted successfully", "success")
    except Exception as e:
        flash(f"Error deleting document: {str(e)}", "danger")
    
    return redirect(url_for("knowledge_base"))

@app.route("/update", methods=["POST"])
def update_row():
    row_id = request.form.get("id")
    category = request.form.get("category")
    team = request.form.get("team")
    status = request.form.get("status")
    action = request.form.get("action") or "Updated by user"
    
    rec = Support.query.get(row_id)
    if not rec:
        flash("Record not found", "danger")
        return redirect(url_for("dashboard"))
    
    # Update all messages in the same thread
    if rec.thread_id:
        thread_messages = Support.query.filter(
            Support.thread_id == rec.thread_id
        ).all()
        
        for msg in thread_messages:
            msg.category = category
            msg.team_assigned = team
            msg.status = status
            msg.action = f"Thread {action}"
    else:
        # Single message (no thread)
        rec.category = category
        rec.team_assigned = team
        rec.status = status
        rec.action = action
    
    db.session.commit()
    flash("Updated successfully", "success")
    return redirect(url_for("dashboard"))

@app.route("/assign/<int:row_id>", methods=["POST"])
def assign_to_team(row_id):
    rec = Support.query.get(row_id)
    if not rec:
        flash("Record not found", "danger")
        return redirect(url_for("dashboard"))
    
    # Get all messages in the thread
    messages = [rec]
    if rec.thread_id:
        messages = Support.query.filter(
            Support.thread_id == rec.thread_id
        ).all()
    
    team_name = rec.team_assigned or "Internal"
    team = TeamCategory.query.filter_by(team_name=team_name).first()
    if not team:
        flash("Team not found", "danger")
        return redirect(url_for("dashboard"))
    
    forward_to = team.team_email
    if not forward_to:
        flash("No email configured for team", "danger")
        return redirect(url_for("dashboard"))
    
    # Forward all messages in the thread
    for msg in messages:
        subject = f"FWD: {msg.subject} -- Assigned from Support Portal"
        body = f"Forwarding support request:\n\nFrom: {msg.sender}\nSubject: {msg.subject}\n\nMessage:\n{msg.request}\n\nSupport ID: {msg.customer_mail_id}\nThread Token: {msg.thread_token or 'N/A'}"
        ok, error = send_email_via_smtp(forward_to, subject, body)
        if not ok:
            flash(f"Failed to forward message {msg.customer_mail_id}: {error}", "danger")
            return redirect(url_for("dashboard"))
    
    # Update all messages in the thread
    for msg in messages:
        msg.action = f"Assigned to {team_name} and forwarded to {forward_to}"
    
    db.session.commit()
    flash(f"Assigned {len(messages)} messages and forwarded successfully", "success")
    return redirect(url_for("dashboard"))

@app.route("/request/<int:row_id>")
def view_request(row_id):
    main_rec = Support.query.get(row_id)
    if not main_rec:
        flash("Record not found", "danger")
        return redirect(url_for("dashboard"))
    
    # Get all messages in this thread
    if main_rec.thread_id:
        thread_messages = Support.query.filter(
            Support.thread_id == main_rec.thread_id
        ).order_by(Support.timestamps.desc()).all()
    else:
        thread_messages = [main_rec]
    
    # Ensure all messages in thread have same classification
    if len(thread_messages) > 1:
        # Get the classification from the original message
        original_message = Support.query.filter(
            Support.thread_id == main_rec.thread_id
        ).order_by(Support.timestamps.asc()).first()
        
        # Update all messages in thread to match original classification
        for msg in thread_messages:
            if (msg.category != original_message.category or 
                msg.team_assigned != original_message.team_assigned):
                msg.category = original_message.category
                msg.team_assigned = original_message.team_assigned
                msg.action = "Classification synchronized with thread"
        db.session.commit()
    
    # Load relevant knowledge base documents based on assigned team and category
    knowledge_section = ""
    referenced_docs = []
    if main_rec.team_assigned and main_rec.category:
        docs = KnowledgeDocument.query.filter_by(
            team_name=main_rec.team_assigned,
            category_name=main_rec.category
        ).all()
        if docs:
            knowledge_section = "## Relevant Knowledge Base Documents ##\nUse these to inform your response, but do not mention specific document names in the response.\n\n"
            for doc in docs:
                knowledge_section += f"### {doc.filename} ###\n{doc.content[:2000]}\n\n"
                referenced_docs.append(doc.filename)  # Track document names for UI display

    # Generate response with thread token included
    thread_context = "\n\n--- Previous Messages ---\n"
    for msg in reversed(thread_messages[1:]):  # Skip the first (latest) message
        thread_context += f"\nFrom: {msg.sender}\nDate: {msg.timestamps}\n\n{msg.request}\n"
        if msg.response_text:
            thread_context += f"\nResponse:\n{msg.response_text}\n"
    
    suggested_response = ""
    try:
        prompt = f"""
You are a helpful support agent. Given the following support request thread, generate a concise, polite response to the customer addressing their concern and providing next steps.

Latest Message:
From: {main_rec.sender}
Subject: {main_rec.subject}
Message:
{main_rec.request}

{thread_context}

{knowledge_section}

The response should be professional and include:
1. Acknowledgment of the issue and any previous communications
2. Clear explanation of next steps or solution
3. Offer for further assistance if needed
4. Include the thread reference token: {main_rec.thread_token or 'N/A'}

Format requirements:
- Use formal but friendly tone
- Keep it to 3-6 short paragraphs
- Include the reference token in the response
- End with: "Best regards,\nSupport Team"
- Do not mention specific knowledge base document names in the response, even if you use their information
"""
        task = Task(
            description=prompt,
            expected_output="The response text",
            agent=responder_agent
        )
        crew = Crew(agents=[responder_agent], tasks=[task], verbose=0)
        suggested_response = crew.kickoff()
    except Exception as e:
        suggested_response = f"(LLM error) Could not generate response: {e}\n\nReference: {main_rec.thread_token or 'N/A'}\n\nBest regards,\nSupport Team"
    
    teams_query = db.session.query(TeamCategory.team_name).distinct().all()
    teams = [t[0] for t in teams_query]
    category_lookup = {}
    for tc in TeamCategory.query.all():
        if tc.team_name not in category_lookup:
            category_lookup[tc.team_name] = []
        category_lookup[tc.team_name].append(tc.category_name)
    
    return render_template(
        "response.html", 
        main_rec=main_rec,
        thread_messages=thread_messages,
        suggested_response=suggested_response, 
        category_lookup=category_lookup,
        teams=teams,
        referenced_docs=referenced_docs  # Pass referenced document names to template
    )

@app.route("/send_response/<int:row_id>", methods=["POST"])
def send_response(row_id):
    rec = Support.query.get(row_id)
    if not rec:
        flash("Record not found", "danger")
        return redirect(url_for("dashboard"))
    
    response_text = request.form.get("response_text")
    if not response_text:
        flash("Response body is empty", "danger")
        return redirect(url_for("view_request", row_id=row_id))
    
    subject = f"Re: {rec.subject}"
    ok, error = send_email_via_smtp(rec.sender, subject, response_text)
    if ok:
        # Update all messages in the thread
        if rec.thread_id:
            messages = Support.query.filter(
                Support.thread_id == rec.thread_id
            ).all()
            for msg in messages:
                msg.replied = True
                msg.response_text = response_text
                msg.action = "Response sent by agent via UI"
        else:
            rec.replied = True
            rec.response_text = response_text
            rec.action = "Response sent by agent via UI"
        
        db.session.commit()
        flash("Response sent to customer", "success")
    else:
        flash(f"Failed to send response: {error}", "danger")
    
    return redirect(url_for("dashboard"))

@app.route("/categories_auth", methods=["GET", "POST"])
def categories_auth():
    if request.method == "POST":
        secret_key = request.form.get("secret_key")
        if secret_key == os.getenv("CATEGORIES_SECRET_KEY", "customersupport"):
            return redirect(url_for("categories"))
        else:
            flash("Invalid secret key", "danger")
    return render_template("categories_auth.html")

@app.route("/categories", methods=["GET", "POST"])
def categories():
    if request.method == "POST":
        team_type = request.form.get("team_type")
        category_names = request.form.getlist("category_name[]")
        category_descriptions = request.form.getlist("category_description[]")
        
        logger.debug(f"Received team_type: {team_type}")
        logger.debug(f"Received category_names: {category_names}")
        logger.debug(f"Received category_descriptions: {category_descriptions}")
        
        if not category_names:
            flash("No category names provided", "danger")
            return redirect(url_for("categories"))
        
        if team_type == "new":
            new_team_name = request.form.get("new_team_name", "").strip()
            new_team_email = request.form.get("new_team_email", "").strip()
            new_team_desc = request.form.get("team_description", "").strip()
            
            if not new_team_name or not new_team_email:
                flash("Missing team name or email", "danger")
                return redirect(url_for("categories"))
                
            if TeamCategory.query.filter_by(team_name=new_team_name).first():
                flash("Team already exists", "danger")
                return redirect(url_for("categories"))
                
            # Create team folder
            team_folder = os.path.join(KNOWLEDGE_BASE_PATH, new_team_name)
            os.makedirs(team_folder, exist_ok=True)
            
            # Add default Other category
            other_cat = TeamCategory(
                team_name=new_team_name,
                team_email=new_team_email,
                team_description=new_team_desc,
                category_name="Other",
                category_description="Default other category"
            )
            db.session.add(other_cat)
            os.makedirs(os.path.join(team_folder, "Other"), exist_ok=True)
            
            # Add the new categories if provided and not Other
            added = 0
            for i in range(len(category_names)):
                category_name = category_names[i].strip() if i < len(category_names) else ""
                description = category_descriptions[i].strip() if i < len(category_descriptions) else ""
                
                logger.debug(f"Processing category {i+1}: name='{category_name}', description='{description}'")
                
                if not category_name:
                    logger.warning(f"Skipping empty category name at index {i}")
                    continue
                    
                if category_name.lower() == "other":
                    logger.warning(f"Skipping 'Other' category at index {i}")
                    flash("Cannot create 'Other' category manually", "danger")
                    continue
                    
                if TeamCategory.query.filter_by(team_name=new_team_name, category_name=category_name).first():
                    logger.warning(f"Category '{category_name}' already exists for team '{new_team_name}'")
                    flash(f"Category '{category_name}' already exists for this team", "danger")
                    continue
                    
                cat = TeamCategory(
                    team_name=new_team_name,
                    team_email=new_team_email,
                    team_description=new_team_desc,
                    category_name=category_name,
                    category_description=description
                )
                db.session.add(cat)
                os.makedirs(os.path.join(team_folder, category_name), exist_ok=True)
                added += 1
                
            db.session.commit()
            flash(f"New team and {added} categories added", "success") if added > 0 else flash("No valid categories added for new team", "info")
            
        else:  # existing team
            team_name = request.form.get("team_name", "").strip()
            
            if not team_name:
                flash("Missing team name", "danger")
                return redirect(url_for("categories"))
                
            # Check if team exists
            team_entries = TeamCategory.query.filter_by(team_name=team_name).all()
            if not team_entries:
                flash("Team not found", "danger")
                return redirect(url_for("categories"))
                
            # Create team folder if it doesn't exist
            team_folder = os.path.join(KNOWLEDGE_BASE_PATH, team_name)
            os.makedirs(team_folder, exist_ok=True)
            
            # Add the new categories
            added = 0
            for i in range(len(category_names)):
                category_name = category_names[i].strip() if i < len(category_names) else ""
                description = category_descriptions[i].strip() if i < len(category_descriptions) else ""
                
                logger.debug(f"Processing category {i+1}: name='{category_name}', description='{description}'")
                
                if not category_name:
                    logger.warning(f"Skipping empty category name at index {i}")
                    continue
                    
                if category_name.lower() == "other":
                    logger.warning(f"Skipping 'Other' category at index {i}")
                    flash("Cannot create 'Other' category manually", "danger")
                    continue
                    
                # Check if category already exists for this team
                if TeamCategory.query.filter_by(team_name=team_name, category_name=category_name).first():
                    logger.warning(f"Category '{category_name}' already exists for team '{team_name}'")
                    flash(f"Category '{category_name}' already exists for this team", "danger")
                    continue
                    
                # Add the new category
                cat = TeamCategory(
                    team_name=team_name,
                    team_email=team_entries[0].team_email,
                    team_description=team_entries[0].team_description,
                    category_name=category_name,
                    category_description=description
                )
                db.session.add(cat)
                os.makedirs(os.path.join(team_folder, category_name), exist_ok=True)
                added += 1
            
            db.session.commit()
            flash(f"{added} new categories added", "success") if added > 0 else flash("No valid categories added", "info")
            
        return redirect(url_for("categories"))
    
    # GET request handling
    teams_query = db.session.query(TeamCategory.team_name).distinct().order_by(TeamCategory.team_name).all()
    teams = [t[0] for t in teams_query]
    team_emails = {}
    team_descriptions = {}
    for team in teams:
        tc = TeamCategory.query.filter_by(team_name=team).first()
        team_emails[team] = tc.team_email
        team_descriptions[team] = tc.team_description or ""
    categories_list = TeamCategory.query.order_by(TeamCategory.team_name).all()
    return render_template("categories.html", 
                         teams=teams, 
                         team_emails=team_emails, 
                         team_descriptions=team_descriptions, 
                         categories_list=categories_list)

@app.route("/categories/delete/<int:cat_id>", methods=["POST"])
def delete_category(cat_id):
    cat = TeamCategory.query.get(cat_id)
    if cat and cat.category_name != "Other":
        # Check if there are any documents in the category
        docs = KnowledgeDocument.query.filter_by(team_name=cat.team_name, category_name=cat.category_name).all()
        if docs:
            flash("Cannot delete category with existing documents", "danger")
            return redirect(url_for("categories"))
        # Delete the category folder if empty
        category_folder = os.path.join(KNOWLEDGE_BASE_PATH, cat.team_name, cat.category_name)
        try:
            if os.path.exists(category_folder):
                os.rmdir(category_folder)
        except OSError:
            flash("Cannot delete category folder; ensure it is empty", "danger")
            return redirect(url_for("categories"))
        db.session.delete(cat)
        db.session.commit()
        flash("Category deleted", "success")
    else:
        flash("Cannot delete this category", "danger")
    return redirect(url_for("categories"))

@app.route("/teams/update/<team_name>", methods=["POST"])
def update_team(team_name):
    team_name = team_name  # This is the old team_name
    team_entries = TeamCategory.query.filter_by(team_name=team_name).all()
    if not team_entries:
        return jsonify({"error": "Team not found"}), 404
    
    new_name = request.form.get("team_name", team_name).strip()
    new_email = request.form.get("team_email", team_entries[0].team_email)
    new_description = request.form.get("team_description", team_entries[0].team_description)
    
    if new_name != team_name and TeamCategory.query.filter_by(team_name=new_name).first():
        return jsonify({"error": "Team name already exists"}), 400
    
    # Rename team folder if name changed
    if new_name != team_name:
        old_folder = os.path.join(KNOWLEDGE_BASE_PATH, team_name)
        new_folder = os.path.join(KNOWLEDGE_BASE_PATH, new_name)
        if os.path.exists(old_folder):
            os.rename(old_folder, new_folder)
    
    for entry in team_entries:
        entry.team_name = new_name
        entry.team_email = new_email
        entry.team_description = new_description
    
    try:
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/categories/update/<int:cat_id>", methods=["POST"])
def update_category(cat_id):
    category = TeamCategory.query.get(cat_id)
    if not category:
        return jsonify({"error": "Category not found"}), 404
    
    if category.category_name == "Other":
        return jsonify({"error": "Cannot modify 'Other' category"}), 400
    
    new_category_name = request.form.get("category_name", category.category_name).strip()
    new_description = request.form.get("category_description", category.category_description)
    
    if new_category_name != category.category_name and TeamCategory.query.filter_by(team_name=category.team_name, category_name=new_category_name).first():
        return jsonify({"error": "Category name already exists for this team"}), 400
    
    # Rename category folder if name changed
    if new_category_name != category.category_name:
        old_folder = os.path.join(KNOWLEDGE_BASE_PATH, category.team_name, category.category_name)
        new_folder = os.path.join(KNOWLEDGE_BASE_PATH, category.team_name, new_category_name)
        if os.path.exists(old_folder):
            os.rename(old_folder, new_folder)
    
    category.category_name = new_category_name
    category.category_description = new_description
    
    try:
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    t = threading.Thread(target=poll_and_store, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=True)