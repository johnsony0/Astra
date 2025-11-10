import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import json

# --- 1. App and DB Configuration ---
# Set up a database file named 'flow_data.db' in the current directory
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'flow_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 2. Database Model (Schema) ---
# NOTE: The CICFlowMeter output has over 80 features. 
# For simplicity, we are only defining a few key columns here.
# You MUST expand this model to include ALL the features you plan to use (e.g., 'Flow Duration', 'Min Packet Length', etc.)

class FlowRecord(db.Model):
    # Unique ID for the database record
    id = db.Column(db.Integer, primary_key=True)
    
    # Key features from the flow
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    src_ip = db.Column(db.String(50))
    dst_ip = db.Column(db.String(50))
    flow_duration = db.Column(db.Float)
    total_fwd_packets = db.Column(db.Integer)
    
    # To store the ENTIRE JSON payload for unmodeled features
    full_payload = db.Column(db.Text) 

# --- 3. Database Initialization ---
# Create the database file and table if they don't exist
with app.app_context():
    db.create_all()
    ...


# --- 4. POST Endpoint for CICFlowMeter ---
@app.route('/write_flow', methods=['POST'])
def receive_flow_data():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    flow_json = request.get_json()
    
    # Optional: Run ML Model here
    # prediction = your_model.predict(flow_json)
    
    try:
        # Create a new record using the received JSON data
        new_record = FlowRecord(
            src_ip=flow_json.get('Source IP'),
            dst_ip=flow_json.get('Destination IP'),
            flow_duration=flow_json.get('Flow Duration'),
            total_fwd_packets=flow_json.get('Total Fwd Packets'),
            full_payload=json.dumps(flow_json) 
        )
        
        db.session.add(new_record)
        db.session.commit()
        
        # Respond back to the cicflowmeter tool
        return jsonify({"status": "success", "message": "Flow data recorded."}), 201

    except Exception as e:
        db.session.rollback()
        print(f"Error processing flow data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run the server on the exact port specified in the cicflowmeter -u command
    app.run(host='0.0.0.0', port=8080, debug=True)