import streamlit as st
import sqlite3, hashlib, cv2, numpy as np, time, os, requests, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
from datetime import datetime

# ---------------- CONFIG ----------------
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

ALERT_COOLDOWN = 20

st.set_page_config(page_title="Smart Monitoring System", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,email TEXT,phone TEXT,gender TEXT,
username TEXT UNIQUE,password TEXT,
role TEXT DEFAULT 'user',status TEXT DEFAULT 'active')""")

c.execute("""CREATE TABLE IF NOT EXISTS alerts (
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT,alert_type TEXT,time TEXT,image_path TEXT)""")
conn.commit()

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), YOLO("fire_yolov8.pt")

model, fire_model = load_models()

# ---------------- UTILS ----------------
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def check_login(u, p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (u, hash_password(p)))
    return c.fetchone()

# ---------------- ALERT FUNCTIONS ----------------
def send_email(subject, message, image_path=None):
    try:
        c.execute("SELECT email FROM users WHERE username=?",
                  (st.session_state.username,))
        result = c.fetchone()
        if not result:
            return

        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = result[0]
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img = MIMEImage(f.read())
                msg.attach(img)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, result[0], msg.as_string())
        server.quit()

    except Exception as e:
        print("Email error:", e)

def send_sms(message):
    try:
        c.execute("SELECT phone FROM users WHERE username=?",
                  (st.session_state.username,))
        result = c.fetchone()
        if not result:
            return

        url = "https://www.fast2sms.com/dev/bulkV2"
        payload = {
            "authorization": FAST2SMS_API_KEY,
            "message": message,
            "route": "v3",
            "numbers": result[0]
        }
        headers = {"cache-control": "no-cache"}
        requests.post(url, data=payload, headers=headers)

    except Exception as e:
        print("SMS error:", e)

def trigger_alert(text, frame):
    now = time.time()
    if now - st.session_state.last_alert_time < ALERT_COOLDOWN:
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img = f"alert_{int(time.time())}.jpg"
    cv2.imwrite(img, frame)

    c.execute("INSERT INTO alerts VALUES(NULL,?,?,?,?)",
              (st.session_state.username, text, ts, img))
    conn.commit()

    send_email("Smart Alert", text, img)
    send_sms(text)

    st.session_state.last_alert_time = now

# ---------------- DETECTION ----------------
def detect_events(results, fire_results, conf):
    alerts = []

    for box in results[0].boxes:
        if float(box.conf[0]) >= conf:
            cls = int(box.cls[0])
            if cls == 0:
                alerts.append("Person Detected")

    for box in fire_results[0].boxes:
        if float(box.conf[0]) >= conf:
            alerts.append("Fire Detected")

    return list(set(alerts))

# ---------------- CAMERA ----------------
def run_camera(src, conf):
    cap = cv2.VideoCapture(src)
    frame_box = st.empty()
    alert_box = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        fire_results = fire_model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        alerts = detect_events(results, fire_results, conf)

        frame_box.image(annotated, channels="BGR")

        if alerts:
            text = " | ".join(alerts)
            alert_box.error(text)
            trigger_alert(text, frame)
        else:
            alert_box.success("No threats detected")

    cap.release()

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.title("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = check_login(u, p)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = user[5]
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()

# ---------------- REGISTER PAGE ----------------
def register_page():
    st.title("Register")
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Submit"):
        try:
            c.execute("INSERT INTO users (name,email,phone,gender,username,password) VALUES (?,?,?,?,?,?)",
                      (name, email, phone, gender, username, hash_password(password)))
            conn.commit()
            st.success("Registered Successfully")
            st.session_state.page = "login"
            st.rerun()
        except:
            st.error("Username already exists")

# ---------------- DASHBOARD ----------------
def dashboard():
    st.sidebar.write(f"Welcome {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    st.title("AI Monitoring Dashboard")

    source = st.selectbox("Select Source", ["IP Camera (RTSP)", "Upload Video"])
    conf = st.slider("Confidence", 0.1, 0.9, 0.5)

    if source == "IP Camera (RTSP)":
        rtsp = st.text_input("RTSP URL")
        if st.button("Start"):
            run_camera(rtsp, conf)

    elif source == "Upload Video":
        file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if file:
            path = f"temp_{int(time.time())}.mp4"
            with open(path, "wb") as f:
                f.write(file.read())
            if st.button("Play"):
                run_camera(path, conf)

# ---------------- ROUTER ----------------
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.logged_in:
    dashboard()
