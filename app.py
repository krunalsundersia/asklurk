from flask import Flask, render_template, request, jsonify, session
import os
import logging
from datetime import datetime, timedelta
import random
import firebase_admin
from firebase_admin import credentials, auth, firestore
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-here")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
try:
    firebase_credentials = {
        "type": "service_account",
        "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
        "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase Admin SDK initialized successfully")
    
except Exception as e:
    logger.warning(f"Firebase Admin SDK initialization failed: {e}")
    db = None

# In-memory storage for OTP (use Redis in production)
otp_storage = {}

@app.route('/')
def index():
    return render_template('index.html')

def verify_firebase_token(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None

# OTP Verification Routes
@app.route('/api/auth/send-otp', methods=['POST'])
def send_otp():
    """Send OTP to email for verification"""
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Generate 6-digit OTP
        otp = str(random.randint(100000, 999999))
        
        # Store OTP with timestamp (valid for 10 minutes)
        otp_storage[email] = {
            'otp': otp,
            'created_at': datetime.now(),
            'attempts': 0
        }
        
        # TODO: Integrate with your email service (SendGrid, SMTP, etc.)
        # For now, we'll log the OTP (remove this in production)
        logger.info(f"OTP for {email}: {otp}")
        
        # Simulate email sending
        print(f"📧 OTP sent to {email}: {otp}")
        
        return jsonify({
            'success': True,
            'message': 'OTP sent successfully',
            'debug_otp': otp  # Remove this in production
        })
        
    except Exception as e:
        logger.error(f"Send OTP error: {e}")
        return jsonify({'error': 'Failed to send OTP'}), 500

@app.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP and create user account"""
    try:
        data = request.json
        email = data.get('email')
        otp = data.get('otp')
        name = data.get('name', '')
        password = data.get('password')
        
        if not email or not otp:
            return jsonify({'error': 'Email and OTP are required'}), 400
        
        # Check if OTP exists and is valid
        if email not in otp_storage:
            return jsonify({'error': 'OTP not found or expired'}), 400
        
        otp_data = otp_storage[email]
        
        # Check if OTP is expired (10 minutes)
        if datetime.now() - otp_data['created_at'] > timedelta(minutes=10):
            del otp_storage[email]
            return jsonify({'error': 'OTP has expired'}), 400
        
        # Check attempts
        if otp_data['attempts'] >= 3:
            del otp_storage[email]
            return jsonify({'error': 'Too many failed attempts'}), 400
        
        # Verify OTP
        if otp_data['otp'] != otp:
            otp_data['attempts'] += 1
            return jsonify({'error': 'Invalid OTP'}), 400
        
        # OTP verified successfully - create user account
        try:
            # Create user in Firebase Auth
            user = auth.create_user(
                email=email,
                password=password,
                display_name=name
            )
            
            # Send email verification
            auth.generate_email_verification_link(email)
            
            # Create user document in Firestore
            if db:
                user_ref = db.collection('users').document(user.uid)
                user_ref.set({
                    'email': email,
                    'name': name,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'email_verified': False,
                    'last_login': firestore.SERVER_TIMESTAMP
                })
            
            # Clean up OTP
            del otp_storage[email]
            
            return jsonify({
                'success': True,
                'message': 'Account created successfully',
                'user': {
                    'uid': user.uid,
                    'email': user.email,
                    'name': name
                }
            })
            
        except Exception as firebase_error:
            logger.error(f"Firebase user creation error: {firebase_error}")
            return jsonify({'error': 'Failed to create user account'}), 500
        
    except Exception as e:
        logger.error(f"Verify OTP error: {e}")
        return jsonify({'error': 'Failed to verify OTP'}), 500

@app.route('/api/auth/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP to email"""
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        # Generate new OTP
        otp = str(random.randint(100000, 999999))
        
        # Update OTP storage
        otp_storage[email] = {
            'otp': otp,
            'created_at': datetime.now(),
            'attempts': 0
        }
        
        # Log OTP (remove in production)
        logger.info(f"New OTP for {email}: {otp}")
        print(f"📧 New OTP sent to {email}: {otp}")
        
        return jsonify({
            'success': True,
            'message': 'OTP resent successfully',
            'debug_otp': otp  # Remove this in production
        })
        
    except Exception as e:
        logger.error(f"Resend OTP error: {e}")
        return jsonify({'error': 'Failed to resend OTP'}), 500

# Protected API routes
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not id_token:
            return jsonify({'error': 'Unauthorized'}), 401
        
        decoded_token = verify_firebase_token(id_token)
        if not decoded_token:
            return jsonify({'error': 'Unauthorized'}), 401
        
        user_id = decoded_token['uid']
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Save to Firestore
        chat_data = {
            'user_id': user_id,
            'message': message,
            'response': f"Echo: {message}",  # Replace with actual AI response
            'timestamp': datetime.now().isoformat()
        }
        
        if db:
            db.collection('chats').add(chat_data)
        
        return jsonify({
            'response': f"Echo: {message}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/user/profile', methods=['GET'])
def get_user_profile():
    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not id_token:
            return jsonify({'error': 'Unauthorized'}), 401
        
        decoded_token = verify_firebase_token(id_token)
        if not decoded_token:
            return jsonify({'error': 'Unauthorized'}), 401
        
        return jsonify({
            'user': {
                'uid': decoded_token['uid'],
                'email': decoded_token.get('email'),
                'name': decoded_token.get('name', 'User'),
                'email_verified': decoded_token.get('email_verified', False)
            }
        })
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
