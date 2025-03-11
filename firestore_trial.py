import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the path to the Firebase service account key from environment variable
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not cred_path:
    raise ValueError("No Firebase credentials path found in environment variables.")

# Initialize the Firebase Admin SDK with the service account
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Example usage of Firestore
# getting details from firestore
doc_ref = db.collection('adminData').document('6Ho7STK5cnbeOWE6zGzqX29CyJn1')
print(doc_ref.get().to_dict())