from flask import Flask, request, jsonify
app = Flask(__name__)

# Sample dictionary of valid license keys (you can replace this with your own data)
valid_license_keys = {
    'admin': {'key': '135A345QW', 'MBnum' : ''},
    'aplusrf': {'key': 'TEST', 'MBnum' : ''},
}

@app.route('/verify_license', methods=['POST'])
def verify_license():
    data = request.json
    user = data.get('user')
    user_input_key = data.get('key')

    # Check if the user exists and has a valid license key
    if user in valid_license_keys and user_input_key == valid_license_keys[user]['key']:
        response_data = {"valid": True, "message": "라이센스 키가 유효합니다."}
    else:
        response_data = {"valid": False, "message": "라이센스 키가 유효하지 않습니다."}

    return jsonify(response_data)

if __name__ == '__main__':
    app.run()
