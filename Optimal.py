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
import uuid

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

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

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

def format_chat_title(chat):
    """تنسيق عنوان المحادثة"""
    # استخدام الموضوع إذا كان موجوداً، وإلا استخدام أول رسالة
    display_text = chat['first_message']
    if display_text:
        display_text = display_text[:50] + '...' if len(display_text) > 50 else display_text
    else:
        display_text = UI_TEXTS[interface_language]['new_chat']
    return display_text

def update_chat_title(chat_id, message):
    """تحديث عنوان المحادثة"""
    if chat_id in st.session_state.chat_history:
        # تنظيف الرسالة وتقصيرها إذا كانت طويلة
        title = message.strip().replace('\n', ' ')
        title = title[:50] + '...' if len(title) > 50 else title
        st.session_state.chat_history[chat_id]['first_message'] = title

def create_new_chat():
    """إنشاء محادثة جديدة مستقلة تماماً"""
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    
    # إنشاء ذاكرة جديدة فارغة لكل محادثة
    new_memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    
    st.session_state.chat_history[chat_id] = {
        'messages': [],
        'first_message': None,
        'timestamp': datetime.now(),
        'memory': new_memory
    }
    
    # تنظيف الذاكرة عند بدء محادثة جديدة
    new_memory.clear()

def load_chat(chat_id):
    """تحميل محادثة محددة"""
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = st.session_state.chat_history[chat_id]['messages']

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

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
                template="""You are a specialized assistant in work procedures and risk management, with a focus on the oil and gas sector, particularly Basrah Gas Company (BGC). Your responses must be strictly based on the content within the uploaded documents, adhering to the following rules:

                1. Strict Adherence to Provided Content
                - Extract answers only from the numbered sections in the document (e.g., #1 Blasting & Painting, #25 Permit to Work)
                - Choose the closest relevant answer in the document and avoid using information from distant sections unless necessary
                - When responding, always mention the WMP number and title of the referenced section
                - For page references, only cite the most relevant page where the answer was found. Only mention multiple pages if the question specifically requires information from different sections

                2. Concise and Precise Responses
                - Answer only what is asked, without adding unnecessary details unless explicitly requested
                - Avoid lengthy explanations or including information not found in the document

                3. Logical Industry-Based Reasoning (When Necessary)
                - If no direct answer is available, use internal industry knowledge to provide a logical, best-practice-based response
                - Clearly indicate when an answer is based on reasoning rather than direct document references

                4. Handling Unclear or Context-Dependent Questions
                - If a question is unclear or vague, do not respond
                - If a question requires prior context that is missing, reply: "It seems your question relies on prior context that is not available. Could you please clarify so I can assist you more accurately? 😊"

                5. Ignoring Out-of-Scope Questions
                - If a question falls outside the content of the uploaded document, do not respond at all

                6. Structured and Clear Formatting
                - Use subheadings and numbering to organize responses clearly
                - Highlight key industry terms such as (PTW, JHA, LSR) for better readability

                7. Language Adaptation
                - Respond in English, Modern Standard Arabic, or Iraqi Arabic, depending on the language of the question
                - If the question is in Iraqi Arabic, provide a simple and understandable response while maintaining technical accuracy

                Context:
                {context}

                Question: {input}

                Remember to:
                1. Always cite the WMP number and section title
                2. Keep responses focused and document-based
                3. Use appropriate formatting and highlighting
                4. Match the language of the question
                5. Only reference the most relevant page unless multiple pages are specifically needed
                """,
                input_variables=["context", "input"]
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

    else:
        st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

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
    """إنشاء رد باستخدام Groq مع التحقق من وجود سياق"""
    
    # إذا لم يكن هناك سياق من الملف، نطلب من المستخدم طرح سؤال متعلق بمحتوى الملف
    if not context or not context.get("references", []):
        no_context_message = {
            "العربية": "عذراً، لا يمكنني الإجابة على هذا السؤال. الرجاء طرح سؤال يتعلق بمحتوى الملف.",
            "English": "Sorry, I cannot answer this question. Please ask a question related to the file content."
        }
        return {
            "answer": no_context_message[language],
            "references": []
        }

    # تحضير السياق للنموذج
    context_text = "\n".join([
        f"Content from page {ref.get('page', 'N/A')}: {ref.get('content', '')}"
        for ref in context.get("references", [])
    ])
    
    # بناء المطالبة مع التأكيد على استخدام محتوى الملف فقط
    if language == "العربية":
        system_prompt = """أنت مساعد ذكي يجيب على الأسئلة باللغة العربية فقط من محتوى الملف المقدم.
        - استخدم السياق المعطى فقط للإجابة على الأسئلة.
        - لا تستخدم أي معلومات خارجية أو معرفة سابقة.
        - إذا كان السؤال خارج نطاق السياق المعطى، اطلب من المستخدم طرح سؤال يتعلق بمحتوى الملف.
        - قدم إجابات دقيقة ومختصرة مبنية فقط على المحتوى المتوفر."""
    else:
        system_prompt = """You are an intelligent assistant that only answers questions from the provided file content in English.
        - Use only the given context to answer questions.
        - Do not use any external information or prior knowledge.
        - If the question is outside the scope of the given context, ask the user to pose a question related to the file content.
        - Provide accurate and concise answers based only on the available content."""

    # إرسال السياق والسؤال فقط، بدون المحادثة السابقة
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]

    try:
        # إنشاء الإجابة باستخدام Groq
        response = llm.invoke(messages)
        
        # تحديث الذاكرة
        memory.save_context(
            {"input": query},
            {"output": response.content}
        )

        return {
            "answer": response.content,
            "references": context.get("references", [])
        }

    except Exception as e:
        error_message = {
            "العربية": f"عذراً، حدث خطأ: {str(e)}",
            "English": f"Sorry, an error occurred: {str(e)}"
        }
        return {
            "answer": error_message[language],
            "references": []
        }

def display_references(refs):
    """عرض المراجع والصور من ملفات PDF"""
    # تحويل المراجع إلى تنسيق موحد
    references = []
    if isinstance(refs, dict) and "references" in refs:
        references = refs["references"]
    elif isinstance(refs, list):
        references = refs

    # جمع أرقام الصفحات الفريدة
    page_info = []
    for ref in references:
        if "page" in ref and ref["page"] is not None:
            page_info.append(ref["page"])

    # عرض الصفحات الفريدة فقط
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

def display_chat_message(message, with_refs=False):
    """عرض رسالة المحادثة"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if with_refs and "references" in message and message.get("references"):
            display_references(message.get("references"))

def display_response_with_references(response, answer):
    """عرض الإجابة مع المراجع"""
    if not any(phrase in answer.lower() for phrase in negative_phrases):
        # عرض الإجابة والمراجع
        st.chat_message("assistant").markdown(answer)
        if response.get("references"):
            display_references(response)
    else:
        # إذا كان الرد يحتوي على عبارات سلبية، نعرض الرد فقط
        st.chat_message("assistant").markdown(answer)

def process_user_input(user_input, is_first_message=False):
    """معالجة إدخال المستخدم وإنشاء الرد"""
    try:
        # تحضير السياق من الملفات PDF
        context = get_relevant_context(query=user_input)
        
        # استخدام الذاكرة الخاصة بالمحادثة الحالية
        current_memory = st.session_state.chat_history[st.session_state.current_chat_id]['memory']
        
        # إنشاء الإجابة باستخدام Groq
        response = create_chat_response(
            user_input,
            context,
            current_memory,  # استخدام الذاكرة الخاصة بالمحادثة
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
        
        # عرض الإجابة مع المراجع فوراً
        if not any(phrase in response["answer"].lower() for phrase in negative_phrases):
            st.chat_message("assistant").markdown(response["answer"])
            if response.get("references"):
                display_references(response)
        else:
            st.chat_message("assistant").markdown(response["answer"])
        
        # إذا كانت أول رسالة، قم بتحديث العنوان وإعادة التحميل
        if is_first_message:
            update_chat_title(st.session_state.current_chat_id, user_input)
            st.rerun()
            
    except Exception as e:
        st.error(f"{UI_TEXTS[interface_language]['error_question']}{str(e)}")

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

# Create new chat if no chat is selected
if st.session_state.current_chat_id is None:
    create_new_chat()

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
        st.session_state.chat_history[st.session_state.current_chat_id]['first_message'] = voice_input
    
    # تحديث سجل المحادثة
    st.session_state.chat_history[st.session_state.current_chat_id]['messages'] = st.session_state.messages
    
    # عرض رسالة المستخدم
    display_chat_message(user_message)
    
    # معالجة السؤال وإظهار الإجابة
    process_user_input(voice_input, is_first_message)
