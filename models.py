# models.py file
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Support(db.Model):
    __tablename__ = "support"
    customer_mail_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    sender = db.Column(db.String(256), nullable=False)
    subject = db.Column(db.String(512), nullable=False)
    request = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(128), nullable=True)
    team_assigned = db.Column(db.String(64), nullable=True)
    status = db.Column(db.String(20), default="pending")
    timestamps = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    action = db.Column(db.String(256), nullable=True)
    replied = db.Column(db.Boolean, default=False)
    response_text = db.Column(db.Text, nullable=True)
    thread_id = db.Column(db.String(128), nullable=True)
    thread_token = db.Column(db.String(16), nullable=True)  # New field for thread token
    is_customer_reply = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "customer_mail_id": self.customer_mail_id,
            "sender": self.sender,
            "subject": self.subject,
            "request": self.request,
            "category": self.category,
            "team_assigned": self.team_assigned,
            "status": self.status,
            "timestamps": self.timestamps,
            "action": self.action,
            "replied": self.replied,
            "thread_id": self.thread_id,
            "thread_token": self.thread_token,
            "is_customer_reply": self.is_customer_reply
        }

class TeamCategory(db.Model):
    __tablename__ = "team_categories"
    id = db.Column(db.Integer, primary_key=True)
    team_name = db.Column(db.String(64), nullable=False)
    team_email = db.Column(db.String(256), nullable=False)
    team_description = db.Column(db.Text, default="")
    category_name = db.Column(db.String(128), nullable=False)
    category_description = db.Column(db.Text, nullable=True)
    __table_args__ = (db.UniqueConstraint('team_name', 'category_name', name='uix_team_category'),)