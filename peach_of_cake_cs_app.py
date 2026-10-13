import os
import streamlit as st
import tempfile
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================
st.set_page_config(page_title="Peach of Cake CS", page_icon="ðŸ‘", layout="centered")

# Inisialisasi API Key dari Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
    # Karena Langchain Google menggunakan variabel lingkungan spesifik
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Klien OpenAI khusus untuk text generation (memanggil Gemini)
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
else:
    st.error("âš ï¸ OPENAI_API_KEY (Gemini API Key) tidak ditemukan di Streamlit Secrets.")
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
    
    # PERBAIKAN: Gunakan text-embedding-004 tanpa prefix 'models/'
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ==========================================
# ANTARMUKA PENGGUNA (UI) STREAMLIT
# ==========================================
st.title("ðŸ‘ Peach of Cake CS ðŸ°")
st.markdown("Selamat datang di Layanan Pelanggan kami! Silakan unggah katalog/menu PDF Anda di *sidebar*, lalu mulai mengobrol dengan bot kami.")

with st.sidebar:
    st.header("âš™ï¸ Pengaturan Toko")
    uploaded_file = st.file_uploader("Unggah Katalog PDF (Menu/FAQ)", type="pdf")
    
    if st.button("Hapus Riwayat Obrolan"):
        st.session_state.messages = []
        st.rerun()

if uploaded_file is None:
    st.info("ðŸ‘ˆ Silakan unggah file PDF katalog pada menu sebelah kiri terlebih dahulu untuk memulai.")
    st.stop()

# Memproses PDF
if "vectorstore" not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Memproses katalog Anda..."):
        try:
            st.session_state.vectorstore = init_vectorstore(tmp_path)
            st.session_state.uploaded_file_name = uploaded_file.name
        except Exception as e:
            st.error(f"Gagal memproses PDF: {e}")
            st.stop()
        finally:
            os.remove(tmp_path)

# Inisialisasi riwayat pesan
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya CS dari Peach of Cake. Ada yang bisa saya bantu terkait menu atau pesanan Anda hari ini?"}
    ]

# Tampilkan riwayat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Tangkap input pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang kue atau layanan kami..."):
    # Simpan dan tampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            try:
                # 1. Ambil dokumen yang relevan dari Vector Store
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # 2. Susun System Prompt dengan konteks dari PDF
                system_prompt = (
                    "Anda adalah Customer Service yang ramah dari toko kue 'Peach of Cake'. "
                    "Gunakan informasi berikut dari katalog toko untuk menjawab pertanyaan pelanggan. "
                    "Jika informasi yang ditanyakan tidak ada di dalam konteks ini, sampaikan dengan sopan bahwa Anda tidak mengetahuinya.\n\n"
                    f"INFORMASI KATALOG:\n{context_text}"
                )
                
                # 3. Bangun payload pesan untuk OpenAI (Gemini)
                api_messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages[-6:]:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                    
                # 4. Panggil API menggunakan model Gemini yang gratis
                response = client.chat.completions.create(
                    model="gemini-2.5-flash",
                    messages=api_messages,
                    temperature=0.2
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Maaf, terjadi kesalahan sistem: {e}")
