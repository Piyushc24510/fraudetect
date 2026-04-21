from dotenv import load_dotenv
load_dotenv()

import csv, hashlib, io, json, os, re, smtplib, sqlite3, urllib.error, urllib.request
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from urllib.parse import urlparse

import nltk, pickle
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer

try:
    import bcrypt as _bcrypt; _BCRYPT = True
except ImportError:
    _BCRYPT = False

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    _LC = True
except ImportError:
    _LC = False

# ── APP ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-in-prod")
app.config.update(PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

DB = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_detection.db"))

@contextmanager
def get_db():
    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
    try: yield conn
    finally: conn.close()

# ── PASSWORD ──────────────────────────────────────────────────────────────────
def _hash(p):
    return _bcrypt.hashpw(p.encode(), _bcrypt.gensalt()).decode() if _BCRYPT else hashlib.sha256(p.encode()).hexdigest()

def _check(plain, stored):
    if _BCRYPT:
        try: return _bcrypt.checkpw(plain.encode(), stored.encode())
        except: pass
    return hashlib.sha256(plain.encode()).hexdigest() == stored

# ── DB INIT ───────────────────────────────────────────────────────────────────
def init_db():
    with get_db() as c:
        c.cursor().executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL, name TEXT NOT NULL, email TEXT,
                role TEXT DEFAULT 'user', joined TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, review TEXT NOT NULL, review_full TEXT,
                is_fake INTEGER NOT NULL, confidence REAL NOT NULL, checked_by TEXT NOT NULL,
                timestamp TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS idx_hcb ON history(checked_by);
            CREATE INDEX IF NOT EXISTS idx_hts ON history(timestamp);
        """)
        if not c.execute("SELECT id FROM users WHERE username='admin'").fetchone():
            c.execute("INSERT INTO users (username,password,name,role,joined) VALUES (?,?,?,?,?)",
                      ("admin", _hash("admin123"), "Administrator", "admin", datetime.now().strftime("%Y-%m-%d")))
        c.commit()

# ── EMAIL ─────────────────────────────────────────────────────────────────────
ES, EP, ER = os.environ.get("NOTIFY_EMAIL_SENDER",""), os.environ.get("NOTIFY_EMAIL_PASSWORD",""), os.environ.get("NOTIFY_EMAIL_RECEIVER","")
EMAIL_ON = os.environ.get("NOTIFY_EMAIL_ENABLED","false").lower() == "true"

def send_email(username, name, role):
    if not EMAIL_ON or not all([ES, EP, ER]): return
    try:
        msg = MIMEMultipart("alternative"); msg["Subject"] = f"New Registration — {name}"; msg["From"] = ES; msg["To"] = ER
        msg.attach(MIMEText(f"<b>{name}</b> ({username}) registered as {role} at {datetime.now().strftime('%d %b %Y %I:%M %p')}", "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(ES, EP); s.sendmail(ES, ER, msg.as_string())
    except Exception as e: print(f"Email error: {e}")

# ── ML MODEL ──────────────────────────────────────────────────────────────────
def load_model():
    for mp, vp in [("model.pkl","vectorizer.pkl"),("/tmp/model.pkl","/tmp/vectorizer.pkl")]:
        try:
            with open(mp,"rb") as f: m = pickle.load(f)
            with open(vp,"rb") as f: v = pickle.load(f)
            return m, v
        except: continue
    return None, None

model, vectorizer = load_model()

nltk.data.path.append("/tmp/nltk_data")
for p in ["punkt","stopwords","vader_lexicon","averaged_perceptron_tagger"]:
    try: nltk.download(p, download_dir="/tmp/nltk_data", quiet=True)
    except: pass

try:
    stemmer = PorterStemmer(); stop_words = set(stopwords.words("english")); sia = SentimentIntensityAnalyzer(); _NLP = True
except: stemmer = stop_words = sia = None; _NLP = False

# ── AUTH ──────────────────────────────────────────────────────────────────────
def _safe(t): r = urlparse(t); return not r.netloc and not r.scheme

def login_required(f):
    @wraps(f)
    def d(*a, **k):
        if "user" not in session: return redirect(url_for("login", next=request.url))
        session.modified = True; return f(*a, **k)
    return d

def admin_required(f):
    @wraps(f)
    def d(*a, **k):
        if "user" not in session: return redirect(url_for("login"))
        if session.get("user_role") != "admin": flash("Admin only.","error"); return redirect(url_for("dashboard"))
        session.modified = True; return f(*a, **k)
    return d

# ── DB HELPERS ────────────────────────────────────────────────────────────────
def get_history(user=None):
    with get_db() as c:
        rows = c.execute("SELECT * FROM history WHERE checked_by=? ORDER BY id DESC",(user,)).fetchall() if user \
               else c.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

def add_history(review, full, fake, conf, by):
    with get_db() as c:
        c.execute("INSERT INTO history (review,review_full,is_fake,confidence,checked_by,timestamp) VALUES (?,?,?,?,?,?)",
                  (review, full, fake, conf, by, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))); c.commit()

# ── CLAUDE API ────────────────────────────────────────────────────────────────
def _prompt(text):
    return f"""You are a strict fraud detection expert. Analyze this review for authenticity.

Review: \"\"\"{text}\"\"\"

FAKE signals: vague praise, generic language, no negatives, too short, template feel, excessive punctuation, no personal context.
GENUINE signals: specific details, balanced pros/cons, personal context, natural language, temporal markers.

If 3+ fake signals → FAKE. When in doubt → FAKE.

Respond ONLY with valid JSON:
{{
  "verdict": "FAKE" or "GENUINE",
  "confidence": <0-100>,
  "reason": "<one sentence>",
  "red_flags": ["..."],
  "green_flags": ["..."]
}}"""

def call_claude(text, key):
    try:
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":400,
                             "messages":[{"role":"user","content":_prompt(text)}]}).encode(),
            headers={"Content-Type":"application/json","x-api-key":key,"anthropic-version":"2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r: raw = json.loads(r.read())
        txt = "".join(b.get("text","") for b in raw.get("content",[]) if b.get("type")=="text").strip()
        txt = re.sub(r"^```(?:json)?\s*","",txt,flags=re.I); txt = re.sub(r"```\s*$","",txt).strip()
        res = json.loads(txt); v = str(res.get("verdict","GENUINE")).upper()
        return {"verdict": v if v in ("FAKE","GENUINE") else "GENUINE",
                "confidence": max(0,min(100,int(res.get("confidence",75)))),
                "reason": str(res.get("reason","Analysis complete.")),
                "red_flags": res.get("red_flags",[]) if isinstance(res.get("red_flags"),list) else [],
                "green_flags": res.get("green_flags",[]) if isinstance(res.get("green_flags"),list) else []}
    except urllib.error.HTTPError as e: print(f"API {e.code}: {e.read().decode()}"); return None
    except Exception as e: print(f"Claude error: {e}"); return None

# ── TEXT METRICS ──────────────────────────────────────────────────────────────
_EXTREME = {"worst","best","terrible","amazing","horrible","perfect","awful","fantastic","garbage",
            "excellent","pathetic","outstanding","incredible","unbelievable","phenomenal","extraordinary",
            "superb","flawless","brilliant","magnificent","spectacular","breathtaking"}
_GENERIC = ["highly recommend","five stars","love it","best ever","amazing product","great value",
            "definitely recommend","must buy","absolutely love","perfect product","exceeded expectations",
            "will buy again","worth every penny","life changing","game changer","must have",
            "love this product","great product","works great","works perfectly","no complaints",
            "zero complaints","10/10","5 stars","couldn't be happier"]
_GENUINE = ["however","although","but","except","issue","problem","concern","could be better","wish",
            "downside","drawback","disappointed","took a while","at first","after a few","been using",
            "months ago","week ago","compared to","switched from","replaced my","update:","edit:","returned"]
_NEG = ["but","however","although","except","issue","problem","concern","not","don't","doesn't",
        "didn't","wasn't","aren't","bad","poor","slow","hard","difficult","wish","disappointed","unfortunately"]

def metrics(text):
    lo = text.lower(); words = text.split(); wc = len(words)
    return dict(wc=wc, uniq=round(len(set(lo.split()))/max(wc,1)*100,1),
                exc=text.count("!"), caps=round(sum(c.isupper() for c in text)/max(len(text),1)*100,1),
                ext=sum(1 for w in lo.split() if w.strip(".,!?") in _EXTREME),
                gen=sum(1 for p in _GENERIC if p in lo), real=sum(1 for p in _GENUINE if p in lo),
                rep=[w for w,c in Counter([w.strip(".,!?") for w in lo.split() if len(w)>3]).items() if c>=3],
                neg=any(w in lo for w in _NEG))

def rule_based(m):
    red, green = [], []
    if m["exc"]>2:  red.append(f"Excessive exclamation marks ({m['exc']})")
    if m["caps"]>15: red.append(f"High CAPS ratio ({m['caps']}%)")
    if m["ext"]>=2: red.append(f"Heavy superlatives ({m['ext']})")
    if m["wc"]<=8:  red.append(f"Too short ({m['wc']} words)")
    elif m["wc"]<=20: red.append("Very brief — lacks detail")
    if m["uniq"]<40: red.append(f"Low word diversity ({m['uniq']}%)")
    if m["gen"]>=2: red.append(f"Multiple generic phrases ({m['gen']})")
    elif m["gen"]==1: red.append("Generic promotional phrase")
    if len(m["rep"])>=2: red.append(f"Repetitive words: {', '.join(m['rep'][:3])}")
    if not m["neg"] and m["wc"]>10: red.append("Suspiciously all-positive")
    if m["real"]==0 and m["wc"]>15: red.append("No personal context or temporal markers")
    if m["wc"]>50:  green.append(f"Detailed ({m['wc']} words)")
    if m["uniq"]>65: green.append(f"Good vocabulary diversity ({m['uniq']}%)")
    if m["exc"]<=1: green.append("Measured punctuation")
    if m["neg"]:    green.append("Mentions negatives/caveats")
    if m["real"]>=2: green.append(f"Personal context present ({m['real']} indicators)")
    if m["gen"]==0: green.append("No generic phrases")
    if m["ext"]==0: green.append("No superlatives")
    fs = ((m["exc"]>2)*12+(m["caps"]>15)*12+(m["ext"]>=2)*10+(m["wc"]<=8)*30+(m["wc"]<=20)*10+
          (m["uniq"]<40)*12+(m["gen"]>=2)*15+(m["gen"]==1)*8+(len(m["rep"])>=2)*8+
          (not m["neg"] and m["wc"]>10)*12+(m["real"]==0 and m["wc"]>15)*10)
    fs = max(0, fs - (m["neg"]*8+( m["real"]>=2)*10+(m["wc"]>50)*5+(m["gen"]==0)*5))
    v = "FAKE" if fs>=25 else "GENUINE"
    return dict(verdict=v, confidence=min(93,48+fs) if v=="FAKE" else min(82,65-fs//2),
                reason=f"Rule-based: {len(red)} fake, {len(green)} genuine signals. Add tool for analysis.",
                red_flags=red, green_flags=green)

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def home(): return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if "user" in session: return redirect(url_for("dashboard"))
    err = None
    if request.method == "POST":
        u, p = request.form.get("username","").strip(), request.form.get("password","")
        with get_db() as c: user = c.execute("SELECT * FROM users WHERE username=?",(u,)).fetchone()
        if user and _check(p, user["password"]):
            session.permanent = True
            session.update(user=u, user_name=user["name"], user_role=user["role"])
            nxt = request.args.get("next","")
            return redirect(nxt if nxt and _safe(nxt) else url_for("dashboard"))
        err = "Invalid username or password"
    return render_template("login.html", error=err, success=request.args.get("success"))

@app.route("/register", methods=["GET","POST"])
def register():
    if "user" in session: return redirect(url_for("dashboard"))
    err = None
    if request.method == "POST":
        u = request.form.get("username","").strip(); p = request.form.get("password","")
        cp = request.form.get("confirm_password",""); n = request.form.get("name","").strip()
        if not all([u,p,cp,n]): err = "All fields required"
        elif len(u)<3 or not re.match(r"^\w+$",u): err = "Username: 3+ chars, letters/numbers/underscore only"
        elif p != cp: err = "Passwords do not match"
        elif len(p)<8: err = "Password must be 8+ characters"
        elif not re.search(r"[A-Z]",p) or not re.search(r"\d",p): err = "Password needs uppercase + number"
        else:
            with get_db() as c:
                if c.execute("SELECT id FROM users WHERE username=?",(u,)).fetchone(): err = "Username taken"
                else:
                    c.execute("INSERT INTO users (username,password,name,role,joined) VALUES (?,?,?,?,?)",
                              (u,_hash(p),n,"user",datetime.now().strftime("%Y-%m-%d"))); c.commit()
            if not err:
                send_email(u, n, "user")
                return redirect(url_for("login", success="Registration successful! Please sign in."))
    return render_template("register.html", error=err)

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    hist = get_history(); total = len(hist); fake = sum(1 for h in hist if h.get("is_fake"))
    us = {}
    for h in hist:
        s = us.setdefault(h.get("checked_by","?"),{"total":0,"fake":0,"genuine":0})
        s["total"]+=1; s["fake" if h.get("is_fake") else "genuine"]+=1
    ph = [h for h in hist if h.get("checked_by")==session.get("user")]
    pt = len(ph); pf = sum(1 for h in ph if h.get("is_fake"))
    return render_template("dashboard_advanced.html", user_name=session.get("user_name"),
        user_role=session.get("user_role"), history=hist[:10], total=total, fake=fake,
        genuine=total-fake, fake_percent=round(fake/total*100,1) if total else 0,
        user_stats=us, personal_stats={"total":pt,"fake":pf,"genuine":pt-pf} if pt else None)

@app.route("/check", methods=["GET","POST"])
@login_required
def check():
    result = None
    if request.method == "POST":
        text = request.form.get("review","").strip()
        if not text:
            flash("Please enter a review.","warning")
            return render_template("check.html", user_name=session.get("user_name"), user_role=session.get("user_role"), result=None)
        m = metrics(text); key = os.environ.get("ANTHROPIC_API_KEY","")
        ai = call_claude(text, key) if key else None
        if ai is None: ai = rule_based(m)
        fake = ai["verdict"]=="FAKE"; conf = float(ai["confidence"])
        fp = gp = None
        if model and vectorizer and _NLP:
            try:
                clean = re.sub(r"[^a-zA-Z]"," ",text.lower())
                vec = vectorizer.transform([" ".join(stemmer.stem(w) for w in clean.split() if w and w not in stop_words)])
                pr = model.predict_proba(vec)[0]; fp = round(pr[1]*100,2); gp = round(pr[0]*100,2)
            except: pass
        result = {"text":text,"review":text[:100],"review_full":text,"is_fake":fake,"confidence":conf,
                  "fake_probability": fp if fp is not None else (conf if fake else 100-conf),
                  "genuine_probability": gp if gp is not None else (100-conf if fake else conf),
                  "explanation":{"summary":ai["reason"],"red_flags":ai["red_flags"],"green_flags":ai["green_flags"],
                      "technical_details":{"word_count":m["wc"],"unique_words":m["uniq"],
                          "sentiment_score":round(conf if fake else -conf,1),
                          "exclamations":m["exc"],"caps_ratio":m["caps"],"extreme_words":m["ext"]}}}
        add_history(result["review"], text, 1 if fake else 0, conf, session.get("user"))
    return render_template("check.html", user_name=session.get("user_name"), user_role=session.get("user_role"), result=result)

@app.route("/history")
@login_required
def history_view():
    role = session.get("user_role")
    return render_template("history.html", user_name=session.get("user_name"), user_role=role,
                           history=get_history() if role=="admin" else get_history(session.get("user")))

@app.route("/analytics")
@login_required
def analytics():
    try:
        s = datetime.strptime(request.args["start_date"],"%Y-%m-%d") if request.args.get("start_date") else datetime.now()-timedelta(days=30)
        e = datetime.strptime(request.args["end_date"],"%Y-%m-%d") if request.args.get("end_date") else datetime.now()
    except: s = datetime.now()-timedelta(days=30); e = datetime.now()
    filt = [h for h in get_history() if s <= datetime.strptime(h["timestamp"],"%Y-%m-%d %H:%M:%S") <= e]
    total = len(filt); fake = sum(1 for h in filt if h.get("is_fake")); us = {}
    for h in filt:
        st = us.setdefault(h.get("checked_by","?"),{"total":0,"fake":0,"genuine":0})
        st["total"]+=1; st["fake" if h.get("is_fake") else "genuine"]+=1
    return render_template("analytics.html", user_name=session.get("user_name"), user_role=session.get("user_role"),
        total=total, fake=fake, genuine=total-fake, history=filt, user_stats=us,
        start_date=s.strftime("%Y-%m-%d"), end_date=e.strftime("%Y-%m-%d"))

@app.route("/users")
@admin_required
def user_management():
    with get_db() as c:
        users = [dict(r) for r in c.execute("SELECT id,username,name,role,joined FROM users ORDER BY id DESC").fetchall()]
        rc = {r["checked_by"]:r["total"] for r in c.execute("SELECT checked_by,COUNT(*) as total FROM history GROUP BY checked_by").fetchall()}
    for u in users: u["review_count"] = rc.get(u["username"],0)
    return render_template("user_management.html", user_name=session.get("user_name"),
        user_role=session.get("user_role"), users=users, current_user=session.get("user"))

@app.route("/users/delete/<int:uid>", methods=["POST"])
@admin_required
def delete_user(uid):
    with get_db() as c:
        row = c.execute("SELECT username FROM users WHERE id=?",(uid,)).fetchone()
        if row and row["username"]==session.get("user"): flash("Cannot delete own account.","error"); return redirect(url_for("user_management"))
        c.execute("DELETE FROM users WHERE id=?",(uid,)); c.commit()
    return redirect(url_for("user_management"))

@app.route("/users/role/<int:uid>", methods=["POST"])
@admin_required
def change_role(uid):
    role = request.form.get("role")
    if role not in ("admin","user"): flash("Invalid role.","error"); return redirect(url_for("user_management"))
    with get_db() as c: c.execute("UPDATE users SET role=? WHERE id=?",(role,uid)); c.commit()
    return redirect(url_for("user_management"))

@app.route("/user-dashboard")
@login_required
def user_dashboard(): return redirect(url_for("dashboard"))

# ── CHATBOT ───────────────────────────────────────────────────────────────────
_SYS = "You are FraudDetect AI, an expert fraud detection assistant. Help users analyze reviews, explain fraud signals, and answer questions about review authenticity. Be professional and accurate."

def get_chat_model():
    key = os.environ.get("ANTHROPIC_API_KEY","")
    return ChatAnthropic(anthropic_api_key=key, model="claude-sonnet-4-20250514", temperature=0.7, max_tokens=1000) if _LC and key else None

@app.route("/chat")
@login_required
def chat():
    if not _LC: flash("Install: pip install langchain langchain-anthropic","error"); return redirect(url_for("dashboard"))
    return render_template("chat.html", user_name=session.get("user_name"), user_role=session.get("user_role"))

@app.route("/api/chat", methods=["POST"])
@login_required
def chat_api():
    if not _LC: return jsonify({"error":"LangChain not available"}), 503
    data = request.get_json(silent=True) or {}; msg = (data.get("message") or "").strip()
    if not msg: return jsonify({"error":"No message"}), 400
    cid = data.get("conversation_id","default"); hk = f"chat_{session.get('user')}_{cid}"
    cm = get_chat_model()
    if not cm: return jsonify({"error":"ANTHROPIC_API_KEY not set"}), 503
    hist = session.get(hk, [])
    messages = [SystemMessage(content=_SYS)]
    for x in hist[-5:]: messages += [HumanMessage(content=x["user"]), AIMessage(content=x["bot"])]
    messages.append(HumanMessage(content=msg))
    try:
        bot = cm.invoke(messages).content.strip()
        hist.append({"user":msg,"bot":bot,"timestamp":datetime.now().isoformat()})
        session[hk] = hist[-20:]
        return jsonify({"response":bot,"conversation_id":cid})
    except Exception as e: print(f"Chat error: {e}"); return jsonify({"error":"Chat failed"}), 500

@app.route("/api/chat/history")
@login_required
def chat_history():
    cid = request.args.get("conversation_id","default")
    return jsonify({"history": session.get(f"chat_{session.get('user')}_{cid}", [])})

@app.route("/api/chat/clear", methods=["POST"])
@login_required
def clear_chat():
    cid = request.args.get("conversation_id","default"); session.pop(f"chat_{session.get('user')}_{cid}", None)
    return jsonify({"success":True})

# ── EXPORTS ───────────────────────────────────────────────────────────────────
@app.route("/export/csv")
@login_required
def export_csv():
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["ID","Review","Result","Confidence (%)","Checked By","Timestamp"])
    for r in get_history():
        w.writerow([r.get("id"),r.get("review_full") or r.get("review"),"Fake" if r.get("is_fake") else "Genuine",r.get("confidence"),r.get("checked_by"),r.get("timestamp")])
    out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":f"attachment; filename=fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"})

@app.route("/export/excel")
@login_required
def export_excel():
    try: import openpyxl; from openpyxl.styles import Alignment, Font, PatternFill
    except: flash("Run: pip install openpyxl","error"); return redirect(url_for("dashboard"))
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Fraud History"
    hdr = ["ID","Review","Result","Confidence (%)","Checked By","Timestamp"]
    for i,h in enumerate(hdr,1):
        c = ws.cell(1,i,h); c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill(start_color="0D0D0D",end_color="0D0D0D",fill_type="solid"); c.alignment=Alignment(horizontal="center")
    for i,r in enumerate(get_history(),2):
        for j,v in enumerate([r.get("id"),r.get("review_full") or r.get("review"),"Fake" if r.get("is_fake") else "Genuine",r.get("confidence"),r.get("checked_by"),r.get("timestamp")],1):
            ws.cell(i,j,v)
    for col in ws.columns: ws.column_dimensions[col[0].column_letter].width = min(max((len(str(c.value)) if c.value else 0) for c in col)+4,60)
    out = io.BytesIO(); wb.save(out); out.seek(0)
    return Response(out.getvalue(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition":f"attachment; filename=fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"})

@app.route("/export/pdf")
@login_required
def export_pdf():
    try:
        from reportlab.lib import colors; from reportlab.lib.pagesizes import landscape, A4
        from reportlab.lib.styles import getSampleStyleSheet; from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except: flash("Run: pip install reportlab","error"); return redirect(url_for("dashboard"))
    out = io.BytesIO(); doc = SimpleDocTemplate(out, pagesize=landscape(A4), topMargin=1.5*cm, bottomMargin=1.5*cm)
    ss = getSampleStyleSheet()
    rows = [["#","Review","Result","Confidence","Checked By","Timestamp"]]
    for r in get_history():
        rv = str(r.get("review_full") or r.get("review",""))
        rows.append([str(r.get("id","")), rv[:60]+("…" if len(rv)>60 else ""),
                     "FAKE" if r.get("is_fake") else "GENUINE", f"{r.get('confidence',0):.1f}%",
                     str(r.get("checked_by","")), str(r.get("timestamp",""))])
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0d0d0d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),10),("FONTSIZE",(0,1),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f9f9f7")]),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#e2e2de")),("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("PADDING",(0,0),(-1,-1),4)]))
    doc.build([Paragraph("FraudDetect — History Report",ss["Title"]),
               Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",ss["Normal"]),
               Spacer(1,0.5*cm), t])
    out.seek(0)
    return Response(out.getvalue(), mimetype="application/pdf",
        headers={"Content-Disposition":f"attachment; filename=fraud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"})

# ── MISC ──────────────────────────────────────────────────────────────────────
@app.route("/clear-history", methods=["POST"])
@admin_required
def clear_history():
    with get_db() as c: c.execute("DELETE FROM history"); c.commit()
    return redirect(url_for("history_view"))

@app.route("/api/live-activity")
@login_required
def live_activity():
    limit = min(int(request.args.get("limit",10)),500); hist = get_history()
    total = len(hist); fake = sum(1 for h in hist if h.get("is_fake"))
    return jsonify({"total":total, "activity":[
        {"review":(h.get("review_full") or h.get("review",""))[:100],"is_fake":bool(h.get("is_fake")),
         "confidence":round(float(h.get("confidence") or 0),1),"checked_by":h.get("checked_by",""),"timestamp":h.get("timestamp","")}
        for h in hist[:limit]], "stats":{"total":total,"fake":fake,"genuine":total-fake,
        "fake_percent":round(fake/total*100,1) if total else 0,"genuine_percent":round((total-fake)/total*100,1) if total else 0}})

@app.route("/api/ai-check", methods=["POST"])
@login_required
def ai_check():
    data = request.get_json(silent=True) or {}; text = (data.get("review") or "").strip()
    if not text: return jsonify({"error":"No review text"}), 400
    key = os.environ.get("ANTHROPIC_API_KEY","")
    if not key: return jsonify({"error":"ANTHROPIC_API_KEY not configured"}), 503
    res = call_claude(text, key)
    return jsonify(res) if res else (jsonify({"error":"AI analysis failed"}), 500)

@app.errorhandler(404)
def not_found(e): return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e): return render_template("404.html"), 500

# ── RUN ───────────────────────────────────────────────────────────────────────
init_db()
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT",5000)), use_reloader=False, threaded=True)
