from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 建立必要資料夾
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # 禁用圖片快取

@app.route('/', methods=['GET', 'POST'])
def index():
    n = 5  # 預設網格大小
    if request.method == 'POST':
        try:
            n = int(request.form.get('n', 5))
            n = max(3, min(n, 10))  # 限制 n 在 3~10
        except:
            pass
    return render_template('index.html', n=n)

@app.route('/api/generate', methods=['POST'])
def generate_policy_image():
    print("🔁 收到 POST 請求 /api/generate")

    data = request.get_json()
    n = int(data['n'])
    start = tuple(data['start'])
    end = tuple(data['end'])
    walls = [tuple(w) for w in data['walls']]

    print("n =", n)
    print("start =", start)
    print("end =", end)
    print("walls =", walls)

    # 初始化
    actions = [(-1,0), (0,1), (1,0), (0,-1)]  # 上、右、下、左
    gamma = 0.9
    value = np.zeros((n, n))
    policy = np.random.choice(4, size=(n, n))

    # Value Iteration
    for _ in range(100):
        new_value = value.copy()
        for i in range(n):
            for j in range(n):
                if (i, j) == end or (i, j) in walls:
                    continue
                a = policy[i, j]
                ni, nj = i + actions[a][0], j + actions[a][1]
                if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in walls:
                    r = -1
                else:
                    ni, nj = i, j
                    r = -5
                new_value[i, j] = r + gamma * value[ni, nj]
        if np.max(np.abs(new_value - value)) < 1e-3:
            break
        value = new_value

    # 繪製圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    for ax in [ax1, ax2]:
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(np.arange(n+1))
        ax.set_yticks(np.arange(n+1))
        ax.grid(True)

    ax1.set_title("Value Matrix")
    for i in range(n):
        for j in range(n):
            if (i, j) in walls:
                ax1.add_patch(plt.Rectangle((j, n-1-i), 1, 1, color='gray'))
            else:
                ax1.text(j + 0.5, n - 1 - i + 0.5, f"{value[i, j]:.2f}", ha='center', va='center')

    ax2.set_title("Policy Matrix")
    for i in range(n):
        for j in range(n):
            if (i, j) in walls:
                ax2.add_patch(plt.Rectangle((j, n-1-i), 1, 1, color='gray'))
            elif (i, j) == end:
                ax2.text(j + 0.5, n - 1 - i + 0.5, '★', ha='center', va='center', fontsize=16, color='red')
            else:
                a = policy[i, j]
                dx, dy = actions[a][1], -actions[a][0]
                ax2.arrow(j + 0.5, n - 1 - i + 0.5, 0.3 * dx, 0.3 * dy,
                          head_width=0.15, head_length=0.15, fc='black', ec='black')

    plt.tight_layout()

    # 儲存圖片
    # ✅ 確保 static 路徑正確
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')

    os.makedirs(STATIC_FOLDER, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
    image_filename = f'result_{ts}.png'
    image_path = os.path.join(STATIC_FOLDER, image_filename)

    print(f"📂 絕對儲存路徑：{image_path}")

    try:
        plt.savefig(image_path)
        print(f"✅ 圖片儲存成功: {image_path}")
    except Exception as e:
        print(f"❌ 儲存圖片失敗: {e}")

    if not os.path.exists(image_path):
        print(f"⚠️ 圖片實際未產生：{image_path}")
    else:
        print(f"✅ 圖片真的存在：{image_path}")

    plt.close()

    # ✅ 傳給前端的 URL 要用 Web 格式
    image_url = f"/static/{image_filename}"
    return jsonify({"image_url": image_url})



if __name__ == '__main__':
    app.run(debug=True)
