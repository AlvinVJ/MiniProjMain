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
        print(uid)
        if not uid:
            return jsonify({"error": "UID is required"}), 400

        users_ref = db.collection("adminData").document(uid)
        doc = users_ref.get()
        server_ip = db.collection("config").document("mediamtx_data").get().to_dict()["ip"]
        port = db.collection("config").document("mediamtx_data").get().to_dict()["port"]

        if doc.exists:
            user_data = doc.to_dict()
            links = []
            for i in range(len(user_data["camera_index"])):
                links.append(f"rtsp://{server_ip}:{port}/stream_{uid}_cam{i+1}")
            return jsonify({"links" : links}), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "An error occurred", "details": str(e)}), 500


@app.route("/setUserDetails", methods = ["POST"])
def setUserDetails():
    try:
        data = request.get_json()
        print(data)
        users_ref = db.collection("adminData")
        user_ref_doc = users_ref.document(data["localId"])
        user_ref_doc.set({
                "userName": data["username"],
                "phoneNumber": data["phoneNumber"],
                "email": data["email"]

            })
        cameras_ref = user_ref_doc.collection("cameras").document("camera_details")
        d = {}
        for i in range(int(data["numCameras"])):
            d[f"cam{i+1}Idx"] = data["cameras"][i]["index"]
            d[f"cam{i+1}Location"] = data["cameras"][i]["location"]
            d[f"cam{i+1}Name"] = data["cameras"][i]["name"]
        cameras_ref.set(d)

        return jsonify({"message": "success"})

    except Exception as e:
        print(Exception)
        return jsonify({"message": "An error occurred"})

@app.route("/voice/<location>", methods=["GET", "POST"])  # ✅ Force POST method
def voice(location):
    print("Received Twilio request")  # ✅ Debugging log
    response = VoiceResponse()
    response.say(f"Automated call from safe solo life. a fall has been detected at {location}. verify person's safety")
    return Response(str(response), mimetype="application/xml")

@app.route("/ping", methods = ["POST"])
def ping():
        print("Server has been pinged")
        return jsonify({"message": "Pong"})

if __name__ == "__main__":
    app.run(port=5000, debug = 'True', host = "0.0.0.0") # ✅ Use 0.0.0.0