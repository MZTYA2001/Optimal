import streamlit as st
import uuid
from datetime import datetime, timedelta
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from streamlit_mic_recorder import speech_to_text
import fitz  # PyMuPDF
from PIL import Image
import io

# API Keys
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

def initialize_ui_texts():
    """تهيئة نصوص واجهة المستخدم بجميع اللغات المدعومة"""
    return {
        "العربية": {
            "page": "صفحة",
            "today": "اليوم",
            "yesterday": "أمس",
            "new_chat": "محادثة جديدة",
            "previous_chats": "المحادثات السابقة",
            "welcome_title": "ChatBot شركة غاز البصرة",
            "welcome_message": "مرحباً بك في ChatBot شركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.",
            "how_to_use": "كيفية الاستخدام:",
            "instructions": "• اكتب سؤالك في الأسفل أو استخدم الميكروفون للتحدث.\n• ستتلقى إجابات بناءً على المعلومات المتوفرة.",
            "input_placeholder": "اكتب سؤالك هنا...",
            "error_pdf": "خطأ في معالجة ملف PDF: ",
            "error_question": "خطأ في معالجة السؤال: ",
            "page_references": "مراجع الصفحات"
        },
        "English": {
            "page": "Page",
            "today": "Today",
            "yesterday": "Yesterday",
            "new_chat": "New Chat",
            "previous_chats": "Previous Chats",
            "welcome_title": "BGC ChatBot",
            "welcome_message": "Welcome to the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.",
            "how_to_use": "How to use:",
            "instructions": "• Type your question below or use the microphone to speak.\n• You will receive answers based on available information.",
            "input_placeholder": "Type your question here...",
            "error_pdf": "Error processing PDF: ",
            "error_question": "Error processing question: ",
            "page_references": "Page References"
        }
    }

def initialize_session_state():
    """تهيئة متغيرات حالة الجلسة"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interface_language" not in st.session_state:
        st.session_state.interface_language = "English"

def hide_streamlit_elements():
    """إخفاء عناصر واجهة Streamlit غير المرغوب فيها"""
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def setup_page_config():
    """تهيئة إعدادات الصفحة"""
    st.set_page_config(
        page_title="BGC ChatBot",
        page_icon="🤖",
        layout="wide"
    )

def initialize_llm():
    """تهيئة نموذج اللغة"""
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        return ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    return None

def load_embeddings():
    """تحميل التضمينات"""
    if "vectors" not in st.session_state:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.vectors = FAISS.load_local(
                "embeddings",
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            st.session_state.vectors = None

def create_chat_prompt():
    """إنشاء قالب المحادثة"""
    return PromptTemplate(
        template="""You are a specialized assistant in work procedures and risk management...
        [Your existing prompt template here]
        """,
        input_variables=["context", "input"]
    )

def format_chat_date(timestamp):
    """تنسيق تاريخ المحادثة"""
    today = datetime.now().date()
    chat_date = timestamp.date()
    
    if chat_date == today:
        return UI_TEXTS[st.session_state.interface_language]['today']
    elif chat_date == today - timedelta(days=1):
        return UI_TEXTS[st.session_state.interface_language]['yesterday']
    else:
        return timestamp.strftime('%Y-%m-%d')

def create_new_chat():
    """إنشاء محادثة جديدة"""
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.chat_history[chat_id] = {
        'messages': [],
        'first_message': None,
        'timestamp': datetime.now(),
        'memory': ConversationBufferMemory(memory_key="history", return_messages=True),
        'last_context': None
    }

def load_chat(chat_id):
    """تحميل محادثة محددة"""
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']

def create_sidebar():
    """إنشاء الشريط الجانبي"""
    with st.sidebar:
        st.title("Settings")
        
        # اختيار اللغة
        interface_language = st.selectbox(
            "Interface Language",
            ["English", "العربية"],
            key="interface_language"
        )
        
        # إدخال صوتي
        st.header("Voice Input")
        voice_input = speech_to_text(
            "🎤",
            "⏹️ Stop",
            language="ar" if interface_language == "العربية" else "en",
            use_container_width=True,
            just_once=True,
            key="mic_button"
        )
        
        # زر محادثة جديدة
        if st.button("New Chat"):
            create_new_chat()
            st.rerun()
        
        # عرض المحادثات السابقة
        st.header("Previous Chats")
        display_chat_history()

def create_main_layout():
    """إنشاء التخطيط الرئيسي للصفحة"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("logo.png", width=200)
    
    with col2:
        hide_streamlit_elements()
        st.title(UI_TEXTS[st.session_state.interface_language]['welcome_title'])
        st.write(UI_TEXTS[st.session_state.interface_language]['welcome_message'])
        st.write(UI_TEXTS[st.session_state.interface_language]['how_to_use'])
        st.write(UI_TEXTS[st.session_state.interface_language]['instructions'])

def display_chat_messages():
    """عرض رسائل المحادثة"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "references" in message and message["references"]:
                display_references(message["references"])

def handle_user_input():
    """معالجة إدخال المستخدم"""
    if prompt := st.chat_input(UI_TEXTS[st.session_state.interface_language]['input_placeholder']):
        process_input(prompt)

def handle_voice_input(voice_input):
    """معالجة الإدخال الصوتي"""
    if voice_input:
        process_input(voice_input)

def process_input(user_input):
    """معالجة الإدخال وإنشاء الرد"""
    # إضافة رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # معالجة السؤال
    response = process_user_input(user_input)
    
    # إضافة رد المساعد
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "references": response.get("references", [])
    })
    
    # تحديث سجل المحادثة
    st.session_state.chat_history[st.session_state.current_chat_id] = {
        'messages': st.session_state.messages,
        'last_context': response.get("context", None)
    }

def main():
    """الدالة الرئيسية للتطبيق"""
    # تهيئة الحالة الأولية والإعدادات
    initialize_session_state()
    setup_page_config()
    
    # تهيئة النصوص والنماذج
    global UI_TEXTS
    UI_TEXTS = initialize_ui_texts()
    llm = initialize_llm()
    load_embeddings()
    
    # إنشاء محادثة جديدة إذا لم تكن هناك محادثة حالية
    if st.session_state.current_chat_id is None:
        create_new_chat()
    
    # إنشاء واجهة المستخدم
    create_sidebar()
    create_main_layout()
    
    # عرض المحادثة ومعالجة الإدخال
    display_chat_messages()
    handle_user_input()

if __name__ == "__main__":
    main()
