# visualizer_server.py
import json
import time
import threading
import math
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# 全局变量存储最新的环境状态（用于前端轮询）
latest_env_state = {
    "uavs": [],
    "weapon": {"pos": [0, 0], "top_project_position": [1, 0], "state": 0},
    "target": [0, 0],
    "bullets": [],
    "rewards": [0, 0],
    "step": 0,
    "info": {}
}

@app.route('/')
def index():
    # 渲染 HTML 页面（我们稍后会在第二步写这个 HTML）
    return render_template('debug_panel.html')

@app.route('/api/state')
def get_state():
    return jsonify(latest_env_state)

@app.route('/api/update', methods=['POST'])
def update_state():
    global latest_env_state
    data = request.json
    latest_env_state = data
    return "OK"

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# 如果你直接运行这个文件，启动服务器
if __name__ == '__main__':
    run_server()