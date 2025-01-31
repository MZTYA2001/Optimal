import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import speech_to_text
import fitz
import pdfplumber
from datetime import datetime, timedelta
import uuid

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# تعريف النصوص حسب اللغة
UI_TEXTS = {
    "العربية": {
        "page": "صفحة",
        "error_pdf": "حدث خطأ أثناء معالجة ملف PDF: ",
        "error_question": "حدث خطأ أثناء معالجة السؤال: ",
        "input_placeholder": "اكتب سؤالك هنا...",
        "source": "المصدر",
        "page_number": "صفحة رقم",
        "welcome_title": "محمد الياسين | بوت الدردشة BGC",
        "page_references": "مراجع الصفحات",
        "new_chat": "محادثة جديدة",
        "today": "اليوم",
        "yesterday": "أمس",
        "previous_chats": "المحادثات السابقة",
        "welcome_message": """
        **مرحبًا!**  
        هذا بوت الدردشة الخاص بشركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.  
        
        **كيفية الاستخدام:**  
        - اكتب سؤالك في الأسفل أو استخدم الميكروفون للتحدث.  
        - سيتم الرد عليك بناءً على المعلومات المتاحة.  
        """
    },
    "English": {
        "page": "Page",
        "error_pdf": "Error processing PDF file: ",
        "error_question": "Error processing question: ",
        "input_placeholder": "Type your question here...",
        "source": "Source",
        "page_number": "Page number",
        "welcome_title": "Mohammed Al-Yaseen | BGC ChatBot",
        "page_references": "Page References",
        "new_chat": "New Chat",
        "today": "Today",
        "yesterday": "Yesterday",
        "previous_chats": "Previous Chats",
        "welcome_message": """
        **Welcome!**  
        This is the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.  
        
        **How to use:**  
        - Type your question below or use the microphone to speak.  
        - You will receive answers based on available information.  
        """
    }
}

# PDF Search and Screenshot Class
class PDFSearchAndDisplay:
    def __init__(self):
        self.fitz = fitz
        self.pdfplumber = pdfplumber

    def capture_screenshots(self, pdf_path, pages):
        screenshots = []
        try:
            doc = self.fitz.open(pdf_path)
            for page_num, _ in pages:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    screenshots.append(pix.tobytes())
            doc.close()
        except Exception as e:
            st.error(f"{UI_TEXTS[interface_language]['error_pdf']}{str(e)}")
        return screenshots

# Initialize session state variables
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Format chat date
def format_chat_date(timestamp):
    today = datetime.now().date()
    chat_date = timestamp.date()
    if chat_date == today:
        return UI_TEXTS[interface_language]['today']
    elif chat_date == today - timedelta(days=1):
        return UI_TEXTS[interface_language]['yesterday']
    else:
        return timestamp.strftime('%Y-%m-%d')

# Format chat title
def format_chat_title(chat):
    display_text = chat['first_message']
    if display_text:
        display_text = display_text[:50] + '...' if len(display_text) > 50 else display_text
    else:
        display_text = UI_TEXTS[interface_language]['new_chat']
    return display_text

# Update chat title
def update_chat_title(chat_id, message):
    if chat_id in st.session_state.chat_history:
        title = message.strip().replace('\n', ' ')
        title = title[:50] + '...' if len(title) > 50 else title
        st.session_state.chat_history[chat_id]['first_message'] = title

# Create new chat
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    new_memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    st.session_state.chat_history[chat_id] = {
        'messages': [],
        'first_message': None,
        'timestamp': datetime.now(),
        'memory': new_memory,
        'last_context': None
    }
    new_memory.clear()

# Load chat
def load_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']

# Initialize the PDFSearchAndDisplay class
def initialize_pdf_searcher():
    pdf_path = "BGC.pdf"
    return PDFSearchAndDisplay()

# Sidebar configuration
def configure_sidebar():
    with st.sidebar:
        global interface_language
        interface_language = st.selectbox("Interface Language", ["English", "العربية"])
        if interface_language == "العربية":
            apply_css_direction("rtl")
            st.title("الإعدادات")
        else:
            apply_css_direction("ltr")
            st.title("Settings")

        if groq_api_key and google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
            if "vectors" not in st.session_state:
                with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    embeddings_path = "embeddings"
                    try:
                        st.session_state.vectors = FAISS.load_local(
                            embeddings_path,
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء تحميل التضميدات: {str(e)}" if interface_language == "العربية" else f"Error loading embeddings: {str(e)}")
                        st.session_state.vectors = None

            input_lang_code = "ar" if interface_language == "العربية" else "en"
            voice_input = speech_to_text(
                start_prompt="🎤",
                stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
                language=input_lang_code,
                use_container_width=True,
                just_once=True,
                key="mic_button",
            )
        else:
            st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

# Main area for chat interface
def display_chat_interface():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("BGC Logo Colored.svg", width=100)
    with col2:
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        st.title(UI_TEXTS[interface_language]['welcome_title'])
        st.write(UI_TEXTS[interface_language]['welcome_message'])

# Sidebar for chat history
def display_chat_history():
    with st.sidebar:
        if st.button(UI_TEXTS[interface_language]['new_chat'], use_container_width=True):
            create_new_chat()
            st.rerun()
        st.markdown("---")
        st.markdown(f"### {UI_TEXTS[interface_language]['previous_chats']}")
        chats_by_date = {}
        for chat_id, chat_data in st.session_state.chat_history.items():
            date = chat_data['timestamp'].date()
            if date not in chats_by_date:
                chats_by_date[date] = []
            chats_by_date[date].append((chat_id, chat_data))
        for date in sorted(chats_by_date.keys(), reverse=True):
            chats = chats_by_date[date]
            st.markdown(f"#### {format_chat_date(chats[0][1]['timestamp'])}")
            for chat_id, chat_data in sorted(chats, key=lambda x: x[1]['timestamp'], reverse=True):
                if st.sidebar.button(
                    format_chat_title(chat_data),
                    key=f"chat_{chat_id}",
                    use_container_width=True
                ):
                    load_chat(chat_id)

# Display chat message
def display_chat_message(message, with_refs=False):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if with_refs and "references" in message and message.get("references"):
            display_references(message.get("references"))

# Display references
def display_references(refs):
    references = []
    if isinstance(refs, dict) and "references" in refs:
        references = refs["references"]
    elif isinstance(refs, list):
        references = refs
    page_info = []
    for ref in references:
        if "page" in ref and ref["page"] is not None:
            page_info.append(ref["page"])
    if page_info:
        unique_pages = sorted(set(page_info))
        with st.expander(UI_TEXTS[interface_language]["page_references"]):
            cols = st.columns(2)
            for idx, page_num in enumerate(unique_pages):
                col_idx = idx % 2
                with cols[col_idx]:
                    screenshots = pdf_searcher.capture_screenshots(pdf_path, [(page_num, "")])
                    if screenshots:
                        st.image(screenshots[0], use_container_width=True)
                        st.markdown(f"**{UI_TEXTS[interface_language]['page']} {page_num}**")

# Process user input
def process_user_input(user_input, is_first_message=False):
    try:
        context = get_relevant_context(query=user_input)
        current_memory = st.session_state.chat_history[st.session_state.current_chat_id]['memory']
        follow_up_phrases = ["tell me more", "what else", "explain more", "give me more details", "و بعد", "شنو بعد", "اكو شي ثاني"]
        is_follow_up = any(phrase in user_input.lower() for phrase in follow_up_phrases)
        if is_follow_up and len(st.session_state.messages) >= 2:
            last_context = st.session_state.chat_history[st.session_state.current_chat_id].get('last_context')
            if last_context:
                context = last_context
        response = create_chat_response(
            user_input,
            context,
            current_memory,
            interface_language
        )
        st.session_state.chat_history[st.session_state.current_chat_id]['last_context'] = context
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "references": response.get("references", [])
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
        if not any(phrase in response["answer"].lower() for phrase in negative_phrases):
            st.chat_message("assistant").markdown(response["answer"])
            if response.get("references"):
                display_references(response)
        else:
            st.chat_message("assistant").markdown(response["answer"])
        if is_first_message:
            update_chat_title(st.session_state.current_chat_id, user_input)
            st.rerun()
    except Exception as e:
        st.error(f"{UI_TEXTS[interface_language]['error_question']}{str(e)}")

# Main execution
initialize_session_state()
pdf_searcher = initialize_pdf_searcher()
configure_sidebar()
display_chat_interface()
display_chat_history()

if st.session_state.current_chat_id is None:
    create_new_chat()

for message in st.session_state.messages:
    if message["role"] == "assistant" and "references" in message:
        display_chat_message(message, with_refs=True)
    else:
        display_chat_message(message)

human_input = st.chat_input(UI_TEXTS[interface_language]['input_placeholder'])

if human_input:
    user_message = {"role": "user", "content": human_input}
    st.session_state.messages.append(user_message)
    is_first_message = len(st.session_state.messages) == 1
    if is_first_message:
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = human_input
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    display_chat_message(user_message)
    process_user_input(human_input, is_first_message)

if voice_input:
    user_message = {"role": "user", "content": voice_input}
    st.session_state.messages.append(user_message)
    is_first_message = len(st.session_state.messages) == 1
    if is_first_message:
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = voice_input
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    display_chat_message(user_message)
    process_user_input(voice_input, is_first_message)
