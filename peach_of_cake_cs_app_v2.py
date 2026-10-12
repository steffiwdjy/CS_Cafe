import os
import streamlit as st
import tempfile
# Menggunakan import terbaru sesuai dengan versi LangChain terkini (langchain-community)
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================
st.set_page_config(page_title="Peach of Cake CS", page_icon="🍑", layout="centered")

# Menggunakan Streamlit Secrets untuk API Key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("⚠️ OPENAI_API_KEY tidak ditemukan di Streamlit Secrets.")
    st.stop()

# ==========================================
# FUNGSI INISIALISASI (DI-CACHE)
# ==========================================
@st.cache_resource(show_spinner="Memuat basis pengetahuan toko...")
def init_vectorstore(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

def init_chatbot(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return qa_chain

# ==========================================
# ANTARMUKA PENGGUNA (UI) STREAMLIT
# ==========================================
st.title("🍑 Peach of Cake CS 🍰")
st.markdown("Selamat datang di Layanan Pelanggan kami! Silakan unggah katalog/menu PDF Anda di *sidebar*, lalu mulai mengobrol dengan bot kami.")

# Sidebar untuk upload PDF
with st.sidebar:
    st.header("⚙️ Pengaturan Toko")
    uploaded_file = st.file_uploader("Unggah Katalog PDF (Menu/FAQ)", type="pdf")
    
    if st.button("Hapus Riwayat Obrolan"):
        st.session_state.messages = []
        if 'chatbot' in st.session_state:
            del st.session_state['chatbot']
        st.rerun()

# Memeriksa apakah file sudah diunggah
if uploaded_file is None:
    st.info("👈 Silakan unggah file PDF katalog pada menu sebelah kiri terlebih dahulu untuk memulai.")
    st.stop()

# Memproses file yang diunggah
if "vectorstore" not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Memproses katalog Anda..."):
        st.session_state.vectorstore = init_vectorstore(tmp_path)
        st.session_state.uploaded_file_name = uploaded_file.name
        os.remove(tmp_path)

# Inisialisasi state untuk chatbot dan riwayat pesan
if "chatbot" not in st.session_state:
    st.session_state.chatbot = init_chatbot(st.session_state.vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya CS dari Peach of Cake. Ada yang bisa saya bantu terkait menu atau pesanan Anda hari ini?"}
    ]

# Menampilkan riwayat obrolan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input obrolan dari pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang kue atau layanan kami..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mengetik..."):
            try:
                response = st.session_state.chatbot({"question": prompt})
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {e}"
                st.error(error_msg)
