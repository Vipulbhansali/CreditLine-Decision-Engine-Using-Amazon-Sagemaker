from flask import Flask, Response, jsonify
import subprocess

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy."""
    print("Ping endpoint was hit")  # Debug statement
    try:
        status = 200  # HTTP 200 status code indicates the container is healthy
        response = jsonify(status='healthy')
    except Exception as e:
        print(f"Error in /ping endpoint: {e}")
        status = 500
        response = jsonify(status='unhealthy', error=str(e))
    return response, status

@app.route('/invocations', methods=['POST'])
def invocations():
    """Run the script for batch inference."""
    print("Invocations endpoint was hit")  # Debug statement
    try:
        subprocess.call(['python3', 'script.py'])
        return Response(status=200)
    except Exception as e:
        print(f"Error in /invocations endpoint: {e}")
        return Response(status=500)

if __name__ == '__main__':
    print("Starting Flask app")  # Debug statement
    app.run(host='0.0.0.0', port=8080)
