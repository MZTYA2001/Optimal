import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import uuid
import re
import base64
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io

try:
    from streamlit_mic_recorder import speech_to_text
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    st.warning("ميزة الإدخال الصوتي غير متوفرة. يرجى تثبيت مكتبة streamlit-mic-recorder.")

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
        self.current_pdf = None
        self.current_pdf_path = None
        self.page_images = {}
        
    def load_pdf(self, pdf_path):
        """تحميل ملف PDF وتخزين الصور"""
        try:
            self.current_pdf_path = pdf_path
            self.current_pdf = self.fitz.open(pdf_path)
            
            # حفظ صور الصفحات
            for page_num in range(len(self.current_pdf)):
                page = self.current_pdf[page_num]
                pix = page.get_pixmap()
                img_path = os.path.join(PDF_IMAGES_DIR, f'page_{page_num + 1}.png')
                pix.save(img_path)
                
                # تخزين الصورة في الذاكرة
                with open(img_path, 'rb') as img_file:
                    self.page_images[page_num + 1] = base64.b64encode(img_file.read()).decode('utf-8')
                    
            return True
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return False
            
    def get_page_image(self, page_num):
        """الحصول على صورة صفحة معينة"""
        return self.page_images.get(page_num)
        
    def search_text(self, query):
        """البحث عن نص في الملف"""
        results = []
        if not self.current_pdf_path:
            return results
            
        with self.pdfplumber.open(self.current_pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if query.lower() in text.lower():
                    results.append({
                        'page': page_num,
                        'content': text,
                        'image': self.get_page_image(page_num)
                    })
                    
        return results
        
    def close(self):
        """إغلاق الملف"""
        if self.current_pdf:
            self.current_pdf.close()
            self.current_pdf = None
            self.current_pdf_path = None
            self.page_images.clear()

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
        'memory': new_memory,
        'last_context': None
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
        llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768"
        )

        # تعريف المسارات
        PDF_DIR = "pdfs"
        PDF_IMAGES_DIR = os.path.join(PDF_DIR, "images")

        # التأكد من وجود المجلدات
        os.makedirs(PDF_IMAGES_DIR, exist_ok=True)

        # تعريف القالب الأساسي للدردشة
        def create_chat_prompt():
            return PromptTemplate(
                template="""You are a specialized assistant in work procedures and risk management, with a focus on the oil and gas sector, particularly Basrah Gas Company (BGC). Your responses must be strictly based on the content within the uploaded documents, adhering to the following rules:

                1. Strict Adherence to Provided Content
                - Extract answers only from the numbered sections in the document (e.g., #1 Blasting & Painting, #25 Permit to Work)
                - Choose the closest relevant answer in the document and avoid using information from distant sections unless necessary
                - When responding, always mention the WMP number and title of the referenced section
                - For page references, only cite the most relevant page where the answer was found. Only mention multiple pages if the question specifically requires information from different sections
                - DO NOT provide any information that is not explicitly stated in the document
                - If a question cannot be answered using the document content, DO NOT provide an answer or alternatives

                2. Handling Out-of-Scope Questions
                - If a question falls outside the content of the uploaded document, respond with ONLY:
                  العربية: "عذراً، هذا السؤال خارج نطاق محتوى الملف."
                  English: "Sorry, this question is outside the scope of the file content."
                - DO NOT attempt to answer questions that are not covered in the document
                - DO NOT suggest alternative answers or provide general knowledge

                3. Concise and Precise Responses
                - Answer only what is asked, without adding unnecessary details unless explicitly requested
                - Avoid lengthy explanations or including information not found in the document
                - Keep responses focused on the specific document content being referenced

                4. Handling Unclear or Context-Dependent Questions
                - If a question is unclear or vague, respond with example questions from the document content
                - If a question requires prior context that is missing, ask for clarification
                - DO NOT make assumptions or provide general information

                5. Language Adaptation
                - Respond in English, Modern Standard Arabic, or Iraqi Arabic, depending on the language of the question
                - If the question is in Iraqi Arabic, provide a simple and understandable response while maintaining technical accuracy
                - Maintain the same strict adherence to document content regardless of language used

                Context:
                {context}

                Question: {input}

                Remember:
                1. ONLY use information explicitly stated in the document
                2. If information is not in the document, DO NOT provide an answer
                3. DO NOT add any external knowledge or suggestions
                4. Only show page references when providing actual content from the document
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
        if VOICE_INPUT_AVAILABLE:
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
            voice_input = None

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

def get_context(question, num_pages=3):
    """
    الحصول على السياق المناسب للسؤال
    Args:
        question (str): سؤال المستخدم
        num_pages (int): عدد الصفحات المراد إرجاعها
    Returns:
        dict: قاموس يحتوي على المراجع المناسبة
    """
    try:
        # استخدام نموذج التشابه للعثور على أفضل الصفحات
        results = embeddings_search(question)
        
        # تحديد أفضل الصفحات
        top_pages = results[:num_pages] if results else []
        
        if not top_pages:
            return None
            
        references = []
        for page in top_pages:
            page_num = page.get('page')
            try:
                # قراءة صورة الصفحة
                image_path = os.path.join(PDF_IMAGES_DIR, f'page_{page_num}.png')
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                else:
                    image_data = None
                    
                references.append({
                    'page': page_num,
                    'content': page.get('content', ''),
                    'image': image_data
                })
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue
        
        return {"references": references} if references else None
        
    except Exception as e:
        print(f"Error in get_context: {str(e)}")
        return None

def detect_language(text):
    """
    تحديد لغة النص المدخل (عربي أو إنجليزي)
    """
    # تحقق من وجود حروف عربية في النص
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    if arabic_chars:
        return "العربية"
    return "English"

def get_interaction_styles():
    """
    إرجاع أنماط CSS للأزرار التفاعلية
    """
    return """
    <style>
    .interaction-buttons {
        display: flex;
        gap: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    
    .interaction-buttons button {
        padding: 8px 16px;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .btn-like {
        background-color: #e7f5e7;
        color: #2e7d32;
    }
    
    .btn-like:hover, .btn-like.active {
        background-color: #2e7d32;
        color: white;
    }
    
    .btn-dislike {
        background-color: #fde7e7;
        color: #c62828;
    }
    
    .btn-dislike:hover, .btn-dislike.active {
        background-color: #c62828;
        color: white;
    }
    
    .btn-copy {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    
    .btn-copy:hover, .btn-copy.active {
        background-color: #1565c0;
        color: white;
    }
    
    .btn-share {
        background-color: #f3e5f5;
        color: #6a1b9a;
    }
    
    .btn-share:hover, .btn-share.active {
        background-color: #6a1b9a;
        color: white;
    }
    
    .page-references {
        margin-top: 10px;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 8px;
        font-size: 14px;
    }
    
    .page-references h4 {
        margin: 0 0 8px 0;
        color: #666;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .page-references h4::before {
        content: '▶';
        display: inline-block;
        transition: transform 0.3s;
    }
    
    .page-references h4.expanded::before {
        transform: rotate(90deg);
    }
    
    .references-content {
        display: none;
        margin-top: 10px;
    }
    
    .references-content.show {
        display: block;
    }
    
    .reference-item {
        margin-bottom: 15px;
        padding: 10px;
        background: white;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .reference-item p {
        margin: 0 0 8px 0;
        color: #333;
    }
    
    .reference-item img {
        max-width: 100%;
        border-radius: 4px;
        margin-top: 8px;
    }
    
    @media (max-width: 600px) {
        .interaction-buttons {
            flex-direction: column;
        }
        .interaction-buttons button {
            width: 100%;
            justify-content: center;
        }
    }
    </style>
    """

def create_chat_response(question, context=None, memory=None):
    try:
        # تحديد لغة السؤال
        language = detect_language(question)
        
        # بناء المطالبة العامة مع دعم اللغتين
        system_prompt = """You are a specialized assistant that answers using only the file content.
        - Detect the question language and respond in the same language (Arabic/English)
        - Use only the information present in the provided context
        - Do not provide any information not found in the file
        - Do not use any external knowledge or alternative suggestions
        - If the question is unclear or too general, respond with:
          - For Arabic: "نحتاج إلى سؤال أكثر تحديداً. يمكنك أن تسأل عن:
            • إجراءات السلامة المحددة (مثل: العمل في المرتفعات، الأماكن المغلقة)
            • متطلبات تصاريح العمل
            • معدات الحماية الشخصية
            • إجراءات الطوارئ"
          - For English: "We need a more specific question. You can ask about:
            • Specific safety procedures (e.g., working at heights, confined spaces)
            • Work permit requirements
            • Personal protective equipment
            • Emergency procedures"
        - When asked for more information, use only the content available in the context"""
        
        # نصوص الأزرار حسب اللغة
        interaction_buttons = {
            "العربية": {
                "like": "👍 أعجبني",
                "dislike": "👎 لم يعجبني",
                "copy": "📋 نسخ الإجابة",
                "share": "🔗 مشاركة"
            },
            "English": {
                "like": "👍 Like",
                "dislike": "👎 Dislike",
                "copy": "📋 Copy Answer",
                "share": "🔗 Share"
            }
        }
        
        # تحديد نص الأزرار حسب اللغة
        buttons = interaction_buttons[language]
        
        # تحضير السياق والذاكرة
        context_text = ""
        if memory and memory.buffer_as_messages:
            # إضافة السياق السابق من الذاكرة
            previous_messages = memory.buffer_as_messages[-2:]  # آخر سؤال وجواب
            context_text = "Previous conversation:\n"
            for msg in previous_messages:
                context_text += f"{msg.type}: {msg.content}\n"
        
        if context and context.get("references"):
            context_text += "\nCurrent context:\n" + "\n".join([
                f"Content from page {ref.get('page', 'N/A')}: {ref.get('content', '')}"
                for ref in context.get("references", [])
            ])
        
        # إعداد الرسائل للنموذج
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]
        
        # إنشاء الإجابة
        response = llm.invoke(messages)
        
        # تجهيز المراجع
        references_text = ""
        references_html = ""
        if context and context.get("references"):
            # ترتيب المراجع حسب رقم الصفحة
            sorted_references = sorted(context["references"], key=lambda x: int(x.get('page', 0)))
            
            references_text = "\n\nPage References:\n" + "\n".join([
                f"• Page {ref['page']}"
                for ref in sorted_references
            ])
            
            refs_title = "المراجع:" if language == "العربية" else "References:"
            
            # التأكد من وجود الصور وإنشاء HTML للمراجع
            refs_content = ""
            for ref in sorted_references:
                page_num = ref.get('page', 'N/A')
                image_data = ref.get('image', '')
                
                if image_data:
                    refs_content += f'''
                    <div class="reference-item">
                        <p>Page {page_num}</p>
                        <p>{ref.get('content', '')}</p>
                        <img src="data:image/png;base64,{image_data}" 
                             alt="Page {page_num}" 
                             loading="lazy"
                             onclick="window.open(this.src)"/>
                    </div>
                    '''
                else:
                    refs_content += f'''
                    <div class="reference-item">
                        <p>Page {page_num}</p>
                        <p>{ref.get('content', '')}</p>
                    </div>
                    '''
            
            if refs_content:
                references_html = f"""
                <div class="page-references">
                    <h4 onclick="toggleReferences(this)">{refs_title}</h4>
                    <div class="references-content">
                        {refs_content}
                    </div>
                </div>
                """
        
        # إضافة أزرار التفاعل مع JavaScript للتحكم في الحالة
        buttons_html = f"""
        {get_interaction_styles()}
        <div class="interaction-buttons" id="buttons_{id(response)}">
            <button onclick="handleLike(this)" class="btn-like">{buttons['like']}</button>
            <button onclick="handleDislike(this)" class="btn-dislike">{buttons['dislike']}</button>
            <button onclick="handleCopy(this)" class="btn-copy" data-full-answer="{response.content}{references_text}">{buttons['copy']}</button>
            <button onclick="handleShare(this)" class="btn-share">{buttons['share']}</button>
        </div>
        {references_html}
        <script>
        const currentLanguage = "{language}";
        
        function toggleReferences(header) {{
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            content.classList.toggle('show');
        }}
        
        function handleLike(button) {{
            const buttonsContainer = button.parentElement;
            const dislikeButton = buttonsContainer.querySelector('.btn-dislike');
            dislikeButton.classList.remove('active');
            button.classList.toggle('active');
        }}
        
        function handleDislike(button) {{
            const buttonsContainer = button.parentElement;
            const likeButton = buttonsContainer.querySelector('.btn-like');
            likeButton.classList.remove('active');
            button.classList.toggle('active');
        }}
        
        function handleCopy(button) {{
            const textToCopy = button.getAttribute('data-full-answer');
            navigator.clipboard.writeText(textToCopy).then(() => {{
                button.classList.add('active');
                const originalText = button.textContent;
                button.textContent = currentLanguage === 'العربية' ? '✓ تم النسخ' : '✓ Copied';
                setTimeout(() => {{
                    button.classList.remove('active');
                    button.textContent = originalText;
                }}, 2000);
            }});
        }}
        
        function handleShare(button) {{
            const textToShare = button.parentElement.querySelector('.btn-copy').getAttribute('data-full-answer');
            if (navigator.share) {{
                navigator.share({{
                    title: 'Safety Answer',
                    text: textToShare
                }}).then(() => {{
                    button.classList.add('active');
                    setTimeout(() => button.classList.remove('active'), 2000);
                }});
            }}
        }}
        </script>
        """
        
        # تحديث الذاكرة
        if memory:
            memory.save_context(
                {"input": question},
                {"output": response.content}
            )
        
        return {
            "answer": response.content,
            "references": sorted_references if context and context.get("references") else [],
            "buttons_html": buttons_html
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

def embeddings_search(query):
    """
    البحث عن الصفحات المناسبة باستخدام التشابه النصي
    Args:
        query (str): النص المراد البحث عنه
    Returns:
        list: قائمة بالصفحات المناسبة
    """
    try:
        if "vectors" not in st.session_state:
            return []
            
        # استخدام FAISS للبحث
        retriever = st.session_state.vectors.as_retriever()
        docs = retriever.get_relevant_documents(query)
        
        # تنظيم النتائج
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", None),
                "source": doc.metadata.get("source", None)
            })
            
        return results
        
    except Exception as e:
        print(f"Error in embeddings_search: {str(e)}")
        return []

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
        context = get_context(user_input)
        
        # استخدام الذاكرة الخاصة بالمحادثة الحالية
        current_memory = st.session_state.chat_history[st.session_state.current_chat_id]['memory']
        
        # إنشاء الإجابة باستخدام Groq
        response = create_chat_response(
            user_input,
            context,
            current_memory
        )
        
        # حفظ السياق الحالي للاستخدام في الأسئلة المتابعة
        st.session_state.chat_history[st.session_state.current_chat_id]['last_context'] = context
        
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

def handle_user_input():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "buttons_html" in message:
                st.components.v1.html(message["buttons_html"], height=100)

    if user_input := st.chat_input("اكتب سؤالك هنا..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            response = create_chat_response(
                user_input,
                get_context(user_input),
                st.session_state.chat_history[st.session_state.current_chat_id]['memory']
            )
            
            st.markdown(response["answer"])
            if "buttons_html" in response:
                st.components.v1.html(response["buttons_html"], height=100)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "buttons_html": response.get("buttons_html", "")
            })

# Main area for chat interface
# Use columns to display logo and title side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Display the logo in the first column
with col1:
    st.image("BGC Logo Colored.svg", width=100)  # Adjust the width as needed

# Display the title and description in the second column
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
handle_user_input()

# معالجة الإدخال الصوتي
if VOICE_INPUT_AVAILABLE and voice_input:
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
