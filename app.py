import streamlit as st
import fitz
import re
import io
import os
import collections
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

try:
    from gtts import gTTS
    VOICE_OUTPUT = True
except:
    VOICE_OUTPUT = False

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Smart Doc Analyser", layout="wide", page_icon="📚")
# --- ONLY STYLE IMPROVED FOR TOPICS & FEATURES - BG SAME ---
st.markdown("""
<style>
.stApp { background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%); }
.main-title { font-size: 48px; font-weight: 900; text-align: center; color: #1A237E; }
.subtitle { text-align: center; color: #311B92; font-size: 18px; font-weight: 600; }
.glass-card { background: rgba(255, 255, 255, 0.88); backdrop-filter: blur(10px); padding: 25px; border-radius: 20px; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15); margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.6); }
.metric-card { background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 20px; color: white; text-align: center; }

/* TOPICS - NEW MATURE STYLE ONLY */
.topic-chip {
    display: inline-block;
    background: white;
    border: 1px solid #d1c4e9;
    color: #4A148C;
    padding: 6px 14px;
    border-radius: 999px;
    margin: 5px;
    font-weight: 600;
    font-size: 13px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    transition: all 0.2s ease;
}
.topic-chip:hover { background: #4A148C; color: white; transform: translateY(-1px); }

/* FEATURES BOX - MATURE */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.75)!important;
    border: 1px solid rgba(255,255,255,0.8)!important;
    border-radius: 12px!important;
    backdrop-filter: blur(8px);
}

.answer-box { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 25px; border-radius: 20px; border-left: 8px solid #6C63FF; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Smart Doc Analyser 📚</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">PDF • Photo to Text • Voice Q&A • Topics • Summary</div>', unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'voice_question' not in st.session_state:
    st.session_state.voice_question = ""

# --- SUMMARY FIX ONLY - LOGIC SAME ---
def analyze(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 25]
    if len(sentences) < 4: return None

    # For summary - filter junk only for summary scoring
    junk_words = ["PBR VITS", "AUTONOMOUS", "TEXT BOOK", "Course Instructor", "B.TECH", "UNIT-", "Dept. of", "Pearson Education"]
    clean_for_summary = [s for s in sentences if not any(j.lower() in s.lower() for j in junk_words) and not s.isupper() and len(s.split())>7]
    if len(clean_for_summary) < 4:
        clean_for_summary = sentences

    chunks = [" ".join(sentences[i:i+4]) for i in range(0, len(sentences), 4)]
    summary_chunks = [" ".join(clean_for_summary[i:i+4]) for i in range(0, len(clean_for_summary), 4)]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(chunks)
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-7:][::-1]] for topic in lda.components_]
    all_words = re.findall(r'\w+', text.lower())
    keywords = collections.Counter(all_words).most_common(8)
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(s)['compound'] for s in sentences[:20]]

    # Summary from clean chunks, skip first 2 if many
    if len(summary_chunks) > 5:
        use_chunks = summary_chunks[2:]
    else:
        use_chunks = summary_chunks
    vec2 = TfidfVectorizer(stop_words='english').fit(use_chunks)
    X2 = vec2.transform(use_chunks)
    scores = X2.sum(axis=1).A1
    top_idx = sorted(scores.argsort()[-3:][::-1])
    summary = " ".join([use_chunks[i] for i in top_idx])[:700]

    return chunks, topics, sentiments, summary, keywords, len(all_words), len(sentences)

with st.sidebar:
    st.markdown("### ✨ Features")
    st.info("✅ Multiple PDF Upload\n✅ Photo to Text (OCR)\n✅ Voice Input & Output\n✅ Smart Q&A\n✅ Topics & Summary\n✅ Report Download")
    st.divider()
    st.subheader("📸 Photo to Text")
    img_file = st.file_uploader("Upload Image (JPG/PNG)", type=['png','jpg','jpeg'])
    ocr_text = ""
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if OCR_AVAILABLE:
            with st.spinner("Extracting text from photo..."):
                try:
                    ocr_text = pytesseract.image_to_string(image)
                    st.success("Text Extracted!")
                    st.text_area("OCR Result", ocr_text[:500], height=100)
                except Exception as e:
                    st.error(f"OCR Error: {e}. Please install tesseract.")
        else:
            st.warning("Install pytesseract for OCR")
    st.divider()
    st.subheader("🎙️ Voice Input")
    audio_file = st.audio_input("Record your question")
    if audio_file is not None:
        try:
            import speech_recognition as sr
            with open("temp.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            r = sr.Recognizer()
            with sr.AudioFile("temp.wav") as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='en-IN')
                st.success(f"You said: {text}")
                st.session_state.voice_question = text
                st.toast(f"Voice recognized: {text}", icon="🎙️")
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
        except Exception as e:
            st.error(f"Voice Error: {e}")
    st.divider()
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.voice_question = ""
        st.toast("Cleared!", icon="🗑️")
        st.rerun()

full_text = ""
c1, c2 = st.columns(2)
with c1:
    uploaded = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)
with c2:
    typed = st.text_area("✍️ Or Paste Text", height=160, placeholder="Paste your document text here...")

if uploaded:
    full_text = "".join(["".join([p.get_text() for p in fitz.open(stream=f.read(), filetype="pdf")]) for f in uploaded])
elif ocr_text:
    full_text = ocr_text
elif typed.strip():
    full_text = typed

if full_text:
    result = analyze(full_text)
    if not result:
        st.error("Please provide more text! At least 4-5 sentences needed.")
    else:
        chunks, topics, sentiments, summary, keywords, wc, sc = result
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
        st.toast(f"Analyzed {wc} words!", icon="🎉")
        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><h2>📝 {wc}</h2><p>Words</p></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><h2>📄 {sc}</h2><p>Sentences</p></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><h2>🧩 {len(chunks)}</h2><p>Chunks</p></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><h2>⏱️ {wc//180+1} min</h2><p>Read Time</p></div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["🎨 Topics", "💬 Q&A + Voice", "☁️ Summary", "📥 Report"])
        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📑 Document Topics")
            for i, t in enumerate(topics):
                st.write(f"**Topic {i+1}**")
                st.markdown("".join([f'<span class="topic-chip">#{w}</span>' for w in t]), unsafe_allow_html=True)
            st.divider()
            for w,c in keywords:
                if len(w)>3: st.markdown(f"**{w}** - {c} times")
            st.markdown('</div>', unsafe_allow_html=True)
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("💬 Ask Question (Text or Voice)")
            q_input = st.text_input("Your Question:", value=st.session_state.voice_question, placeholder="Ex: What is CFG? Or use voice from sidebar")
            enable_voice_out = st.checkbox("🔊 Enable Voice Answer (Text to Speech)")
            if q_input:
                vec = TfidfVectorizer().fit(chunks + [q_input])
                v = vec.transform(chunks + [q_input])
                sim = cosine_similarity(v[-1], v[:-1])
                best = sim.argmax()
                score = sim[0][best]
                ans = chunks[best]
                st.session_state.chat_history.append((q_input, score))
                if score > 0.1:
                    st.markdown(f'<div class="answer-box"><h3>✅ Answer ({score*100:.1f}% match)</h3><p style="font-size:17px;">{ans}</p></div>', unsafe_allow_html=True)
                    st.toast("Answer found!", icon="✨")
                    if enable_voice_out and VOICE_OUTPUT:
                        try:
                            tts = gTTS(text=ans[:400], lang='en', slow=False)
                            buf = io.BytesIO()
                            tts.write_to_fp(buf)
                            buf.seek(0)
                            st.audio(buf, format='audio/mp3')
                            st.caption("🔊 Playing answer...")
                        except Exception as e:
                            st.error(f"Voice output error: {e}")
                else:
                    st.warning("Closest result:")
                    st.info(ans)
            if st.session_state.chat_history:
                st.write("**Recent:**")
                for qq, ss in st.session_state.chat_history[-4:]:
                    st.caption(f"👉 {qq} - {ss*100:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with tab3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📝 Summary")
            st.success(summary)
            wc_img = WordCloud(width=900, height=400, background_color="white", colormap="coolwarm").generate(full_text[:4000])
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc_img, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with tab4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            report = f"Report\nWords:{wc}\nTopics:{topics}\nSummary:{summary}"
            st.download_button("💾 Download TXT", report, "Report.txt")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0,10, report.encode('latin-1','replace').decode('latin-1'))
            pdf_bytes = bytes(pdf.output())
            st.download_button("📄 Download PDF", pdf_bytes, "Report.pdf", mime="application/pdf")
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="glass-card" style="text-align:center;"><h2>👋 Upload PDF / Photo / Text to Start!</h2><p>Supports PDF, Image OCR, Voice Input & Output</p></div>', unsafe_allow_html=True)