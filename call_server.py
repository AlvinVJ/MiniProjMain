from flask import Flask, Response, request, jsonify
from twilio.twiml.voice_response import VoiceResponse
from google.cloud import firestore
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
db = firestore.Client()

@app.route("/get_links", methods=["POST"])
def get_links():   
    try:
        data = request.get_json()
        uid = data.get("uid", "").strip()  # Ensure we get a valid UID

        if not uid:
            return jsonify({"error": "UID is required"}), 400

        if doc.exists:
            l = get_server_ip()
            user_data = doc.to_dict()
            links = []
            for i in range(len(user_data["camera_index"])):
                links.append(f"rtsp://{l[0]}:{l[1]}/stream_{uid}_cam{user_data['camera_index'][i]}")
            return jsonify({"links" : links}), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route("/get_server_ip", methods=["POST"])
def get_server_ip():
    users_ref = db.collection("adminData").document(uid)
    doc = users_ref.get()
    server_ip = db.collection("config").document("mediamtx_data").get().to_dict()["ip"]
    mediamtx_port = db.collection("config").document("mediamtx_data").get().to_dict()["port"]
    flask_port = db.collection("config").document("flask_data").get().to_dict()["port"]
    return server_ip, mediamtx_port, flask_port

@app.route("/get_uid", methods=["POST"])
def get_uid():
    try:
        data = request.get_json()
        username = data.get("username", "").strip() # Ensure we get a valid phone number
        password = data.get("password", "").strip() # Ensure we get a valid phone number

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        doc = db.collection("adminData").where("phoneNumber", "==", phone).stream()
        for d in doc:
            return jsonify({"uid": d.id}), 200

        return jsonify({"error": "User not found"}), 404

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

@app.route("/voice", methods=["GET", "POST"])  # ✅ Force POST method
def voice():
    print("Received Twilio request")  # ✅ Debugging log
    response = VoiceResponse()
    response.say("Automated call from safe solo life. a fall has been detected at registered location. verify person's safety")
    return Response(str(response), mimetype="application/xml")

@app.route("/ping", methods = ["POST"])
def ping():
        print("Server has been pinged")
        return jsonify({"message": "Pong"})

if __name__ == "__main__":
    app.run(port=5000, debug = 'True', host = "0.0.0.0") # ✅ Use 0.0.0.0