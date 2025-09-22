from flask import Flask, render_template, jsonify, request, session
import gspread
from google.oauth2.service_account import Credentials
import os
import random
import datetime

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")

# Google Sheets setup
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
SPREADSHEET_ID = os.getenv("GSHEET_SPREADSHEET_ID")
CREDENTIALS_FILE = "other.json"

def get_sheet():
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        return client.open_by_key(SPREADSHEET_ID).sheet1
    except Exception as e:
        print("⚠️ Google Sheets unavailable:", e)
        return None

QUESTIONS = [
    {"id": "Q1", "text": "請問這裡是住家電話還是公司電話?",
     "options": ["住家電話", "住商合一", "公司電話（終止訪問）"]},
    {"id": "Q2", "text": "請問您是否已年滿20歲，並設籍在高雄市具有投票權?",
     "options": ["是", "沒有/拒答"]},
    {"id": "Q3", "text": "請問您目前設籍在高雄哪一個行政區?",
     "options": [f"行政區{i}" for i in range(1, 39)] + ["拒答（終止訪問）"]},
    {"id": "Q4", "text": "請問您知不知道目前有誰想要參選下一屆市長?（可複選）",
     "options": ["許智傑", "邱議瑩", "林岱樺", "賴瑞隆", "柯志恩", "其他人（請訪員紀錄）"],
     "multi": True},
    {"id": "Q5", "text": "目前在民進黨內，有意參選的4個人中，您最希望哪一個人?",
     "options": ["許智傑", "邱議瑩", "林岱樺", "賴瑞隆", "無法選出任何人"], "shuffle": True},
    {"id": "Q6", "text": "根據報導，目前可能參選下屆高雄市長的有民進黨許智傑、國民黨柯志恩，您最希望哪一位?",
     "options": ["許智傑", "柯志恩", "無法選出任何一人"], "shuffle": True},
    {"id": "Q7", "text": "根據報導，目前可能參選下屆高雄市長的有民進黨賴瑞隆、國民黨柯志恩，您最希望哪一位?",
     "options": ["賴瑞隆", "柯志恩", "無法選出任何一人"], "shuffle": True},
    {"id": "Q8", "text": "根據報導，目前可能參選下屆高雄市長的有民進黨邱議瑩、國民黨柯志恩，您最希望哪一位?",
     "options": ["邱議瑩", "柯志恩", "無法選出任何一人"], "shuffle": True},
    {"id": "Q9", "text": "根據報導，目前可能參選下屆高雄市長的有民進黨林岱樺、國民黨柯志恩，您最希望哪一位?",
     "options": ["林岱樺", "柯志恩", "無法選出任何一人"], "shuffle": True},
    {"id": "Q10", "text": "哪一個政黨的理念與主張與您較為接近?",
     "options": ["民進黨", "國民黨", "台灣民眾黨", "時代力量", "臺灣基進", "綠黨", "其他（請訪員紀錄）"]},
    {"id": "Q11", "text": "請問您的教育程度?",
     "options": ["小學及以下", "國中", "高中、高職", "專科", "大學", "研究所以上", "拒答"]},
    {"id": "Q12", "text": "請問您大約幾歲?",
     "options": ["20-24歲", "25-29歲", "30-34歲", "35-39歲", "40-44歲", "45-49歲",
                 "50-54歲", "55-59歲", "60-64歲", "70歲以上", "拒答"]},
    {"id": "Q13", "text": "請問您的生理性別?",
     "options": ["男性", "女性"]},
]

@app.route("/")
def index():
    session.clear()
    session["step"] = 0
    session["answers"] = {}
    return render_template("survey.html", total=len(QUESTIONS))

@app.route("/question/<int:step>")
def get_question(step):
    if step > len(QUESTIONS):
        return jsonify({"options": []})
    q = QUESTIONS[step - 1].copy()
    if q.get("shuffle"):
        fixed = [opt for opt in q["options"] if "無法" in opt]
        others = [opt for opt in q["options"] if opt not in fixed]
        random.shuffle(others)
        q["options"] = others + fixed
    return jsonify(q)

@app.route("/submit/<int:step>", methods=["POST"])
def submit_question(step):
    q = QUESTIONS[step - 1]
    ans = request.form.getlist("answer") if q.get("multi") else request.form.get("answer")
    session["answers"][q["id"]] = ans

    # Early termination logic
    terminate = False
    if q["id"] == "Q1" and ans == "公司電話（終止訪問）":
        terminate = True
    elif q["id"] == "Q2" and ans == "沒有/拒答":
        terminate = True
    elif q["id"] == "Q3" and ("拒答" in ans if isinstance(ans, str) else any("拒答" in a for a in ans)):
        terminate = True

    if terminate:
        save_to_gsheet(session["answers"])
        session.clear()
        return jsonify({"terminated": True})

    session["step"] = step
    return jsonify({"terminated": False})

@app.route("/finish", methods=["POST"])
def finish():
    # Save all answers when finishing
    save_to_gsheet(session.get("answers", {}))
    session.clear()
    return jsonify({"success": True})

def save_to_gsheet(answers):
    try:
        sheet = get_sheet()
        if sheet:
            row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            for q in QUESTIONS:
                ans = answers.get(q["id"], "")
                if isinstance(ans, list):
                    ans = ", ".join(ans)
                row.append(ans)
            sheet.append_row(row)
            print("✅ Saved to Google Sheets")
    except Exception as e:
        print("⚠️ Failed to save to Google Sheets:", e)

if __name__ == "__main__":
    app.run(debug=True)
