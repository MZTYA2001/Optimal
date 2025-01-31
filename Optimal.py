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
from streamlit_mic_recorder import speech_to_text  # Import speech-to-text function
import fitz  # PyMuPDF for capturing screenshots
import pdfplumber  # For searching text in PDF
from datetime import datetime, timedelta

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  # Page title
    page_icon="BGC Logo Colored.svg",  # New page icon
    layout="wide"  # Page layout
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
        """تهيئة الكلاس"""
        self.fitz = fitz
        self.pdfplumber = pdfplumber

    def capture_screenshots(self, pdf_path, pages):
        """التقاط صور من صفحات PDF محددة"""
        screenshots = []
        try:
            doc = self.fitz.open(pdf_path)
            for page_num, _ in pages:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    # تحويل الصفحة إلى صورة بدقة عالية
                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    screenshots.append(pix.tobytes())
            doc.close()
        except Exception as e:
            st.error(f"{UI_TEXTS[interface_language]['error_pdf']}{str(e)}")
        return screenshots

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")  # Right-to-left for Arabic
        st.title("الإعدادات")  # Sidebar title in Arabic
    else:
        apply_css_direction("ltr")  # Left-to-right for English
        st.title("Settings")  # Sidebar title in English

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # تعريف القالب الأساسي للدردشة
        def create_chat_prompt():
            return PromptTemplate(
                template="""أنت مساعد مفيد لشركة غاز البصرة (BGC). مهمتك هي الإجابة على الأسئلة بناءً على السياق المقدم حول BGC. اتبع هذه القواعد بدقة:

                1. قدم إجابات دقيقة ومباشرة
                2. استخدم فقط المعلومات من السياق المقدم
                3. إذا لم تكن متأكداً، قل ذلك بصراحة
                4. حافظ على لغة مهنية ومحترفة

                السياق المقدم:
                {context}

                السؤال: {input}

                تذكر أن تقدم إجابة:
                1. دقيقة ومستندة إلى الوثائق
                2. مباشرة وواضحة
                3. مهنية ومنظمة
                """,
                input_variables=["context", "input"]
            )

        def create_custom_chain(llm, prompt):
            """إنشاء سلسلة معالجة المستندات"""
            return create_stuff_documents_chain(
                llm=llm,
                prompt=prompt
            )

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "embeddings"  # Path to your embeddings folder
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
                    )
                except Exception as e:
                    st.error(f"حدث خطأ أثناء تحميل التضميدات: {str(e)}" if interface_language == "العربية" else f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

        # Microphone button in the sidebar
        st.markdown("### الإدخال الصوتي" if interface_language == "العربية" else "### Voice Input")
        input_lang_code = "ar" if interface_language == "العربية" else "en"  # Set language code based on interface language
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language=input_lang_code,  # Language (en for English, ar for Arabic)
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        # Reset button in the sidebar
        if st.button("إعادة تعيين الدردشة" if interface_language == "العربية" else "Reset Chat"):
            st.session_state.messages = []  # Clear chat history
            st.session_state.memory.clear()  # Clear memory
            st.success("تمت إعادة تعيين الدردشة بنجاح." if interface_language == "العربية" else "Chat has been reset successfully.")
            st.rerun()  # Rerun the app to reflect changes immediately
    else:
        st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main area for chat interface
# Use columns to display logo and title side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Display the logo in the first column
with col1:
    st.image("BGC Logo Colored.svg", width=100)  # Adjust the width as needed

# Display the title and description in the second column
with col2:
    st.title(UI_TEXTS[interface_language]['welcome_title'])
    st.write(UI_TEXTS[interface_language]['welcome_message'])

# Initialize session state for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def create_new_chat():
    """إنشاء محادثة جديدة"""
    chat_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    if chat_id not in st.session_state.chat_history:
        st.session_state.chat_history[chat_id] = {
            'messages': [],
            'timestamp': datetime.now(),
            'first_message': UI_TEXTS[interface_language]['new_chat']
        }
    return chat_id

def update_chat_title(chat_id, message):
    """تحديث عنوان المحادثة"""
    if chat_id in st.session_state.chat_history:
        # تنظيف الرسالة وتقصيرها إذا كانت طويلة
        title = message.strip().replace('\n', ' ')
        title = title[:50] + '...' if len(title) > 50 else title
        st.session_state.chat_history[chat_id]['first_message'] = title
        st.rerun()

def load_chat(chat_id):
    """تحميل محادثة محددة"""
    if chat_id in st.session_state.chat_history:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]['messages']
        st.rerun()  # استخدام st.rerun بدلاً من experimental_rerun

def format_chat_title(chat):
    """تنسيق عنوان المحادثة"""
    # استخدام الموضوع إذا كان موجوداً، وإلا استخدام أول رسالة
    display_text = chat['first_message']
    if display_text:
        display_text = display_text[:50] + '...' if len(display_text) > 50 else display_text
    else:
        display_text = UI_TEXTS[interface_language]['new_chat']
    return display_text

def format_chat_date(timestamp):
    """تنسيق تاريخ المحادثة"""
    today = datetime.now().date()
    chat_date = timestamp.date()
    
    if chat_date == today:
        return UI_TEXTS[interface_language]['today']
    elif chat_date == today - timedelta(days=1):
        return UI_TEXTS[interface_language]['yesterday']
    else:
        return timestamp.strftime('%Y-%m-%d')

# Sidebar for chat history
with st.sidebar:
    # New Chat button
    if st.button(UI_TEXTS[interface_language]['new_chat'], use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    st.markdown(f"### {UI_TEXTS[interface_language]['previous_chats']}")
    
    # Group chats by date
    chats_by_date = {}
    for chat_id, chat_data in st.session_state.chat_history.items():
        date = chat_data['timestamp'].date()
        if date not in chats_by_date:
            chats_by_date[date] = []
        chats_by_date[date].append((chat_id, chat_data))
    
    # Display chats grouped by date
    for date in sorted(chats_by_date.keys(), reverse=True):
        chats = chats_by_date[date]
        
        # عرض التاريخ كعنوان
        st.markdown(f"#### {format_chat_date(chats[0][1]['timestamp'])}")
        
        # عرض المحادثات تحت كل تاريخ
        for chat_id, chat_data in sorted(chats, key=lambda x: x[1]['timestamp'], reverse=True):
            if st.sidebar.button(
                format_chat_title(chat_data),
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                load_chat(chat_id)

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# List of negative phrases to check for unclear or insufficient answers
negative_phrases = [
    "I'm sorry",
    "عذرًا",
    "لا أملك معلومات كافية",
    "I don't have enough information",
    "لم أتمكن من فهم سؤالك",
    "I couldn't understand your question",
    "لا يمكنني الإجابة على هذا السؤال",
    "I cannot answer this question",
    "يرجى تقديم المزيد من التفاصيل",
    "Please provide more details",
    "غير واضح",
    "Unclear",
    "غير متأكد",
    "Not sure",
    "لا أعرف",
    "I don't know",
    "غير متاح",
    "Not available",
    "غير موجود",
    "Not found",
    "غير معروف",
    "Unknown",
    "غير محدد",
    "Unspecified",
    "غير مؤكد",
    "Uncertain",
    "غير كافي",
    "Insufficient",
    "غير دقيق",
    "Inaccurate",
    "غير مفهوم",
    "Not clear",
    "غير مكتمل",
    "Incomplete",
    "غير صحيح",
    "Incorrect",
    "غير مناسب",
    "Inappropriate",
    "Please provide me",  # إضافة هذه العبارة
    "يرجى تزويدي",  # إضافة هذه العبارة
    "Can you provide more",  # إضافة هذه العبارة
    "هل يمكنك تقديم المزيد"  # إضافة هذه العبارة
]

def clean_text(text):
    """تنظيف النص من الأخطاء والفراغات الزائدة"""
    # إزالة الفراغات الزائدة
    text = ' '.join(text.split())
    # إزالة علامات التنسيق غير المرغوبة
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text

def extract_complete_sentences(text, max_length=200):
    """استخراج جمل كاملة من النص"""
    # تقسيم النص إلى جمل
    sentences = text.split('.')
    complete_text = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # التأكد من أن الجملة تبدأ بحرف كبير وتنتهي بنقطة
        if sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'):
            sentence += '.'
            
        # إضافة الجملة إذا كانت ضمن الحد الأقصى للطول
        if current_length + len(sentence) <= max_length:
            complete_text.append(sentence)
            current_length += len(sentence)
        else:
            break
            
    return ' '.join(complete_text)

def get_relevant_context(query, retriever=None):
    """الحصول على السياق المناسب من الملفات PDF"""
    try:
        if retriever is None and "vectors" in st.session_state:
            retriever = st.session_state.vectors.as_retriever()
            
        if retriever:
            # البحث عن المستندات ذات الصلة
            docs = retriever.get_relevant_documents(query)
            
            # تنظيم السياق
            organized_context = []
            for doc in docs:
                organized_context.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", None),
                    "source": doc.metadata.get("source", None)
                })
            
            return {"references": organized_context}
        
        return {"references": []}
            
    except Exception as e:
        st.error(f"Error getting context: {str(e)}")
        return {"references": []}

def create_chat_response(query, context, memory, language):
    """إنشاء إجابة للمحادثة باستخدام Groq"""
    try:
        # تحضير السياق من المراجع
        references_text = ""
        if context and "references" in context:
            for ref in context["references"]:
                if ref["content"]:
                    references_text += f"\n{ref['content']}"

        # بناء الرسالة للنموذج
        messages = []
        
        # إضافة السياق إذا وجد
        if references_text:
            messages.append({
                "role": "system",
                "content": f"You are a helpful assistant. Use this context to answer the question:\n{references_text}"
            })
        
        # إضافة الذاكرة السابقة
        if memory:
            chat_history = memory.load_memory_variables({})
            if "history" in chat_history:
                messages.extend(chat_history["history"])
        
        # إضافة السؤال الحالي
        messages.append({
            "role": "user",
            "content": query
        })
        
        # الحصول على الإجابة من Groq
        response = llm.invoke(messages)
        
        # تنظيم الإجابة
        answer = response.content
        
        # إضافة الإجابة إلى الذاكرة
        if memory:
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)
        
        return {
            "answer": answer,
            "references": context.get("references", []) if context else []
        }
        
    except Exception as e:
        st.error(f"Error creating response: {str(e)}")
        return {
            "answer": UI_TEXTS[language]['error_response'],
            "references": []
        }

def process_user_input(user_input, is_first_message=False):
    """معالجة إدخال المستخدم وإنشاء الرد"""
    try:
        # تحضير السياق من الملفات PDF
        context = get_relevant_context(query=user_input)
        
        # إنشاء الإجابة باستخدام OpenAI
        response = create_chat_response(
            user_input,
            context,
            st.session_state.memory,
            interface_language
        )
        
        # إضافة الإجابة إلى سجل المحادثة
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "references": response.get("references", [])
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
        
        # عرض الإجابة مع المراجع
        display_response_with_references(response, response["answer"])
        
        # إذا كانت أول رسالة، قم بإعادة تحميل الواجهة
        if is_first_message:
            st.rerun()
            
    except Exception as e:
        st.error(f"{UI_TEXTS[interface_language]['error_question']}{str(e)}")

def display_references(refs):
    """عرض المراجع والصور من ملفات PDF"""
    if refs and isinstance(refs, dict) and "references" in refs:
        page_info = []
        for ref in refs["references"]:
            if "page" in ref and ref["page"] is not None:
                page_info.append(ref["page"])

        if page_info:
            with st.expander(UI_TEXTS[interface_language]["page_references"]):
                cols = st.columns(2)
                for idx, page_num in enumerate(sorted(set(page_info))):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        screenshots = pdf_searcher.capture_screenshots(pdf_path, [(page_num, "")])
                        if screenshots:
                            st.image(screenshots[0], use_container_width=True)
                            st.markdown(f"**{UI_TEXTS[interface_language]['page']} {page_num}**")

def display_chat_message(message, with_refs=False):
    """عرض رسالة المحادثة"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if with_refs and "references" in message:
            display_references(message)

def display_response_with_references(response, answer):
    """عرض الإجابة مع المراجع"""
    if not any(phrase in answer.lower() for phrase in negative_phrases):
        # إضافة المراجع إلى الرسالة
        message = {
            "role": "assistant",
            "content": answer,
            "references": response
        }
        display_chat_message(message, with_refs=True)
    else:
        # إذا كان الرد يحتوي على عبارات سلبية، نعرض الرد فقط
        display_chat_message({
            "role": "assistant",
            "content": answer
        })

# عرض سجل المحادثة
for message in st.session_state.messages:
    if message["role"] == "assistant" and "references" in message:
        display_chat_message(message, with_refs=True)
    else:
        display_chat_message(message)

# حقل إدخال النص
human_input = st.chat_input(UI_TEXTS[interface_language]['input_placeholder'])

# معالجة الإدخال النصي
if human_input:
    user_message = {"role": "user", "content": human_input}
    st.session_state.messages.append(user_message)
    
    # تحديث عنوان المحادثة وإظهار الإجابة إذا كانت أول رسالة
    is_first_message = len(st.session_state.messages) == 1
    if is_first_message:
        # تحديث عنوان المحادثة
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = human_input
    
    # تحديث سجل المحادثة
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    
    # عرض رسالة المستخدم
    display_chat_message(user_message)
    
    # معالجة السؤال وإظهار الإجابة
    process_user_input(human_input, is_first_message)

# معالجة الإدخال الصوتي
if voice_input:
    user_message = {"role": "user", "content": voice_input}
    st.session_state.messages.append(user_message)
    
    # تحديث عنوان المحادثة وإظهار الإجابة إذا كانت أول رسالة
    is_first_message = len(st.session_state.messages) == 1
    if is_first_message:
        # تحديث عنوان المحادثة
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = voice_input
    
    # تحديث سجل المحادثة
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    
    # عرض رسالة المستخدم
    display_chat_message(user_message)
    
    # معالجة السؤال وإظهار الإجابة
    process_user_input(voice_input, is_first_message)

# Create new chat if no chat is selected
if st.session_state.current_chat_id is None:
    create_new_chat()
