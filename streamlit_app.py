"""
Simple VTON Frontend – UI 2.0
Same API contract, smoother experience.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from PIL import Image

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Fashion Studio",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- CSS ----------
st.markdown(
    """
<style>
    /* Root colour palette */
    :root {
        --primary: #6a11cb;
        --secondary: #2575fc;
        --bg: #f9fafb;
        --card: #ffffff;
        --text: #1f2937;
        --muted: #9ca3af;
        --success: #10b981;
        --error: #ef4444;
    }

    /* Global tweaks */
    html, body, .main, .block-container { background-color: var(--bg); }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; color: var(--text); }
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0 0.5rem 0;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Cards */
    .result-card, .uploader-card {
        background: var(--card);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .result-card:hover, .uploader-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Session Helpers ----------
def init_state():
    for key in ("user_image", "processing", "results", "scroll"):
        if key not in st.session_state:
            st.session_state[key] = None if key != "processing" else False

init_state()

# ---------- App ----------
class SimpleVTONApp:
    def __init__(self):
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        if not self.api_base_url.startswith(("http://", "https://")):
            self.api_base_url = f"http://{self.api_base_url}"
        self.api_base_url = self.api_base_url.rstrip("/")

    # ---------- UI ----------
    def run(self):
        st.markdown(
            '<h1 class="main-header">👗 AI Fashion Studio</h1>', unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center; color:var(--muted); font-size:1.1rem; margin-bottom:2rem;'>"
            "Upload your photo → pick preferences → instant recommendations & try-on</p>",
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="uploader-card">', unsafe_allow_html=True)
            st.subheader("📸 Upload Your Photo")
            uploaded_file = st.file_uploader(
                "Drag & drop or click to select",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                key="uploader",
            )

            if uploaded_file:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, caption="Preview", use_container_width=True)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    st.session_state.user_image = base64.b64encode(buf.getvalue()).decode()
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.session_state.user_image = None
            else:
                st.info("👆 Please upload a photo", icon="ℹ️")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="uploader-card">', unsafe_allow_html=True)
            st.subheader("⚙️ Style Preferences")
            gender = st.selectbox(
                "Gender",
                ["male", "female", "unisex"],
                help="Used to filter catalogue items.",
            )
            style = st.selectbox(
                "Style",
                ["casual", "formal", "business", "sporty", "trendy"],
                help="Defines the overall vibe.",
            )
            season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])
            occasion = st.selectbox(
                "Occasion",
                ["everyday", "work", "party", "date", "formal"],
                help="Tailors to the specific event.",
            )

            st.markdown("---")
            go = st.button(
                "🪄 Get Recommendations & Try-On",
                type="primary",
                disabled=not st.session_state.user_image or st.session_state.processing,
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if go and st.session_state.user_image:
            st.session_state.processing = True
            self.process_recommendations_and_tryon(gender, style, season, occasion)

        if st.session_state.results:
            self.show_results()

    # ---------- Processing ----------
    def process_recommendations_and_tryon(self, gender, style, season, occasion):
        placeholder = st.empty()
        with placeholder.container():
            st.info("🔄 Processing your request…")
            bar = st.progress(0)

        payload = {
            "user_image": st.session_state.user_image,
            "preferences": {
                "style": style,
                "gender": gender,
                "season": season,
                "occasion": occasion,
            },
            "max_recommendations": 3,
            "include_vton": True,
        }

        try:
            bar.progress(20)
            resp = self.api_call("/recommend-and-tryon", payload)
            bar.progress(100)
            if resp.get("success"):
                st.session_state.results = resp
                st.session_state.scroll = True  # trigger scroll
            else:
                st.error(f"❌ {resp.get('message','Unknown error')}")
        except Exception as e:
            st.error(f"❌ {e}")
        finally:
            st.session_state.processing = False
            placeholder.empty()

    # ---------- Results ----------
    def show_results(self):
        if st.session_state.scroll:
            st.session_state.scroll = False
            st.html(
                """
                <script>
                setTimeout(() => document.querySelector('[data-testid="stMain"]').scrollIntoView({
                    behavior: 'smooth', block: 'center'
                }), 100);
                </script>
            """
            )

        st.markdown("---")
        st.subheader("🎯 Your Results")

        results = st.session_state.results
        left, right = st.columns([1, 1], gap="large")

        # Style Analysis
        analysis = results.get("user_analysis", {})
        if analysis:
            st.markdown("### 👤 Style Analysis")
            c1, c2, c3 = st.columns(3)
            with c1:
                if "dominant_colors" in analysis:
                    st.write("**Color Palette**")
                    colors = analysis["dominant_colors"][:5]
                    st.html(
                        "<div style='display:flex;gap:6px;margin-top:6px;'>"
                        + "".join(
                            f"<div style='width:24px;height:24px;border-radius:50%;background:{c};border:1px solid #ddd;'></div>"
                            for c in colors
                        )
                        + "</div>"
                    )
            with c2:
                if "body_shape" in analysis:
                    st.write(f"**Body Type:** {analysis['body_shape'].title()}")
            with c3:
                if "season_compatibility" in analysis:
                    st.write(f"**Season:** {analysis['season_compatibility'].title()}")
            st.markdown("---")

        # Recommendations
        with left:
            st.markdown("### 👔 Recommended Clothing")
            for rec in results.get("recommendations", []):
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    conf = int(rec.get("confidence", 0) * 100)
                    st.markdown(f"**{rec['type'].replace('_',' ').title()}** – {conf}% match")
                    st.image(rec["image"], use_container_width=True)
                    meta = rec.get("metadata", {})
                    st.caption(f"**Brand:** {meta.get('brand','Unknown')}")
                    if meta.get("price", 0) > 0:
                        st.caption(f"**Price:** ${meta['price']:.2f}")
                    reasons = meta.get("reasons", [])
                    if reasons:
                        with st.expander("Why recommended?"):
                            for r in reasons[:3]:
                                st.write(f"• {r}")
                    if link := meta.get("product_link"):
                        st.link_button("🛒 Buy Now", link, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        # Try-on
        with right:
            st.markdown("### 🪞 Virtual Try-On")
            vton = results.get("vton_result", {})
            if vton and vton.get("success"):
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                img = vton.get("final_image_url") or vton.get("final_image")
                if img:
                    st.image(img, caption="Your Virtual Try-On", use_container_width=True)
                else:
                    st.info("Try-on ready but image not available")
                st.caption(f"**Processing Time:** {vton.get('processing_time',0):.1f}s")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(vton.get("error", "Virtual try-on failed"))

        st.markdown("---")
        _, c, _ = st.columns([1, 1, 1])
        with c:
            if st.button("🔄 Start Over", type="secondary", use_container_width=True):
                for key in ("user_image", "results"):
                    st.session_state[key] = None
                st.rerun()

    # ---------- API ----------
    def api_call(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base_url}{endpoint}"
        try:
            r = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=400,
            )
            if r.status_code == 200:
                return r.json()
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise Exception(f"{r.status_code}: {detail}")
        except requests.Timeout:
            raise Exception("Request timed out")
        except requests.ConnectionError:
            raise Exception("Backend unreachable – is it running?")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("AI Fashion Studio")
    st.info("1️⃣ Upload photo\n2️⃣ Set preferences\n3️⃣ Get results")
    st.markdown("---")
    st.caption("v3.1 – UI refresh")

    try:
        app = SimpleVTONApp()
        ok = requests.get(f"{app.api_base_url}/health", timeout=3).ok
        st.success("✅ Backend Connected") if ok else st.warning("⚠️ Backend Issues")
    except Exception:
        st.error("❌ Backend Offline")

# ---------- Main ----------
if __name__ == "__main__":
    try:
        SimpleVTONApp().run()
    except Exception as ex:
        st.error(f"Application error: {ex}")
