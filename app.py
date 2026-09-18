import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from googletrans import Translator
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
# --- Custom CSS ---
st.markdown("""
<style>
    /* Import Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');

    :root {
        --primary: #00f2ff;
        --secondary: #7000ff;
        --accent: #ff0055;
        --bg-dark: #050510;
        --glass: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    /* Base Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: var(--bg-dark);
        color: #e0e0e0;
    }

    /* Animated Background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, rgba(112, 0, 255, 0.1), transparent 25%),
                    radial-gradient(circle at 85% 30%, rgba(0, 242, 255, 0.1), transparent 25%);
        background-color: #0e1117;
    }

    /* Sidebar Glass - Lighter & Colorful */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 30, 50, 0.6) 0%, rgba(30, 30, 50, 0.3) 100%);
        backdrop-filter: blur(25px);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 10px 0 30px rgba(0,0,0,0.1);
    }

    /* Typography - Colorful Title */
    h1 {
        background: linear-gradient(90deg, #FF69B4, #FFD700, #00BFFF, #7B68EE);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s linear infinite;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0 3rem 0;
        text-shadow: 0 0 20px rgba(255, 105, 180, 0.3);
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Glass Cards */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 3rem 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .stCard:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 
                    0 0 20px rgba(0, 242, 255, 0.1);
        border-color: rgba(0, 242, 255, 0.3);
    }

    /* Metrics Styling */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #fff;
        line-height: 1.2;
    }

    .detected-text {
        background: linear-gradient(to bottom, #fff, var(--primary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(0, 242, 255, 0.5));
    }
    
    .translation-text {
        background: linear-gradient(to bottom, #fff, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.5));
    }

    /* Camera Feed Glow */
    div[data-testid="stImage"] {
        border-radius: 24px;
        overflow: hidden;
        border: 2px solid var(--glass-border);
        box-shadow: 0 0 40px rgba(0, 242, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stImage"]:hover {
        border-color: var(--primary);
        box-shadow: 0 0 60px rgba(0, 242, 255, 0.3);
    }

    /* UI Components */
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--glass) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px;
        color: white;
    }
    
    div[class*="stCheckbox"] label, div[class*="stToggle"] label {
        color: #e0e0e0;
        font-weight: 500;
    }

    /* Hide standard chrome */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Initialize Resources ---
if 'last_spoken' not in st.session_state:
    st.session_state.last_spoken = ""

@st.cache_resource
def load_resources():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']
    except FileNotFoundError:
        return None, None, None, None
        
    translator = Translator()
    return mp_hands, mp_drawing, mp_drawing_styles, hands, model, translator

mp_hands, mp_drawing, mp_drawing_styles, hands, model, translator = load_resources()

if not model:
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

# --- TTS Function (Threaded) ---


# --- Valid Languages ---
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-cn",
    "Korean": "ko",
    "Punjabi": "pa"
}

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    target_lang = st.selectbox("Target Language", list(LANGUAGES.keys()))
    auto_speak = st.toggle("🔊 Auto-Speak Translation", value=True)
    st.divider()
    run_camera = st.toggle("📷 Start Camera", value=False)
    
    st.markdown("### Instructions")
    st.info("1. Enable Camera\n2. Show hand sign clearly\n3. Wait for translation")

# --- Main Content ---
st.title("🤟 Real-Time Sign Language Translator")

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown("### Camera Feed")
    cam_placeholder = st.empty()

with col2:
    st.markdown("### Translation Results")
    result_placeholder = st.empty()

# --- Main Loop ---
if run_camera:
    cap = cv2.VideoCapture(0)
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break
            
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        predicted_char = "..."
        translated_text = "..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract features
                data_aux = []
                x_ = []
                y_ = []
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                
                # Predict
                if len(data_aux) == 42:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_char = prediction[0]
                        
                        # Translate
                        lang_code = LANGUAGES[target_lang]
                        if lang_code == "en":
                            translated_text = predicted_char
                        else:
                            trans = translator.translate(predicted_char, dest=lang_code)
                            translated_text = trans.text
                    except Exception as e:
                        pass
        
        # Update Camera Feed
        cam_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # Update UI Results
        result_placeholder.markdown(f"""
        <div class="stCard">
            <div class="metric-container">
                <div class="metric-label">Detected Sign</div>
                <div class="metric-value detected-text">{predicted_char}</div>
            </div>
            <br>
            <div class="metric-container">
                <div class="metric-label">Translation ({target_lang})</div>
                <div class="metric-value translation-text">{translated_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle Instant TTS (Web Speech API)
        if auto_speak and translated_text != "..." and translated_text != st.session_state.last_spoken:
            st.session_state.last_spoken = translated_text
            
            # JavaScript Injection for Instant Browser Speech
            lang_code = LANGUAGES[target_lang]
            safe_text = translated_text.replace("'", "\\'")
            
            js = f"""
            <script>
                function speak() {{
                    const msg = new SpeechSynthesisUtterance('{safe_text}');
                    msg.lang = "{lang_code}";
                    window.speechSynthesis.cancel(); // Stop previous
                    window.speechSynthesis.speak(msg);
                }}
                speak();
            </script>
            """
            import streamlit.components.v1 as components
            components.html(js, height=0)
                
        # Small delay to reduce CPU usage
        pass 

    cap.release()
else:
    cam_placeholder.info("Camera is disabled. Toggle 'Start Camera' in settings.")
    result_placeholder.empty()

