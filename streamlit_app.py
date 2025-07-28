"""
Simple VTON Frontend – UI refresh (no logic changes)
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Fashion Studio",
    page_icon="👗",
    layout="wide"
)

# ---------- CSS (only cosmetic updates) ----------
st.markdown(
    """
<style>
    /* High-contrast palette */
    :root {
        --primary: #5B21B6;      /* WCAG 7.8:1 on white */
        --primary-dark: #4C1D95; /* hover / active */
        --white: #FFFFFF;
        --text: #111827;         /* near-black for 12.6:1 */
        --muted: #4B5563;        /* 7:1 on white */
        --border: #D1D5DB;
    }

    /* Header stays identical gradient, just thicker */
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--primary) 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Cards */
    .result-card {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        background-color: var(--white);
        box-shadow: 0 2px 6px rgba(0,0,0,.08);
    }

    /* Ensure buttons meet contrast */
    .stButton > button {
        border-radius: 20px;
        font-weight: 600;
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        color: var(--white) !important;
        background: var(--primary) !important;
        border: none !important;
    }
    .stButton > button:hover,
    .stButton > button:active {
        background: var(--primary-dark) !important;
    }

    /* Buy Now link button */
    a.buy-now {
        display: inline-block;
        background: var(--primary);
        color: var(--white);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
    }
    a.buy-now:hover {
        background: var(--primary-dark);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- App Code – unchanged ----------
class SimpleVTONApp:
    def __init__(self):
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        if not self.api_base_url.startswith(("http://", "https://")):
            self.api_base_url = f"http://{self.api_base_url}"
        if self.api_base_url.endswith("/"):
            self.api_base_url = self.api_base_url[:-1]

        if "user_image" not in st.session_state:
            st.session_state.user_image = None
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "results" not in st.session_state:
            st.session_state.results = None
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False

    def run(self):
        st.markdown("<h1 class='main-header'>👗 AI Fashion Studio</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; font-size: 1.2rem; color: #666;'>"
            "Upload your photo, set preferences, get instant recommendations + try-on results!"
            "</p>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Upload Your Photo")
            uploaded_file = st.file_uploader("Choose your photo", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    st.image(image, caption="Your Photo", use_container_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    st.session_state.user_image = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.session_state.user_image = None

            if not st.session_state.user_image:
                st.info("👆 Please upload your photo to continue")

        with col2:
            st.subheader("⚙️ Style Preferences")
            gender = st.selectbox("Gender", ["male", "female", "unisex"], index=0)
            style = st.selectbox("Style", ["casual", "formal", "business", "sporty", "trendy"], index=0)
            season = st.selectbox("Season", ["spring", "summer", "fall", "winter"], index=1)
            occasion = st.selectbox("Occasion", ["everyday", "work", "party", "date", "formal"], index=0)

            st.markdown("---")

            process_disabled = st.session_state.processing or st.session_state.user_image is None
            if st.button("🪄 Get Recommendations & Try-On", disabled=process_disabled, type="primary"):
                if st.session_state.user_image:
                    st.session_state.processing = True
                else:
                    st.error("Please upload an image first!")

            if st.session_state.processing and st.session_state.user_image:
                self.process_recommendations_and_tryon(gender, style, season, occasion)

        if st.session_state.results:
            self.show_results()

    # ---- unchanged helper methods below ----
    def process_recommendations_and_tryon(self, gender, style, season, occasion):
        if not st.session_state.user_image:
            st.error("Please upload an image first!")
            st.session_state.processing = False
            return

        with st.container():
            st.info("🔄 Processing your request... This may take several minutes...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Step 1/3: Analyzing your photo and style...")
                progress_bar.progress(25)

                request_data = {
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

                status_text.text("Step 2/3: Getting personalized recommendations...")
                progress_bar.progress(50)

                response = self.api_call("/recommend-and-tryon", request_data)

                status_text.text("Step 3/3: Processing virtual try-on results...")
                progress_bar.progress(90)

                if response.get("success"):
                    st.session_state.results = {
                        "recommendations": response.get("recommendations", []),
                        "vton_result": response.get("vton_result", {}),
                        "user_analysis": response.get("user_analysis", {}),
                        "processing_time": response.get("processing_time", 0),
                        "correlation_id": response.get("correlation_id", ""),
                    }
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    st.success("✅ Complete! Check your results below!")
                else:
                    st.error(f"❌ Error: {response.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                st.session_state.processing = False
                st.rerun()

    def show_results(self):
        st.markdown("---")
        st.subheader("🎯 Your Results")

        results = st.session_state.results
        recommendations = results.get("recommendations", [])
        vton_result = results.get("vton_result", {})

        user_analysis = results.get("user_analysis", {})
        if user_analysis:
            st.markdown("### 👤 Your Style Analysis")
            analysis_cols = st.columns(3)

            with analysis_cols[0]:
                if "dominant_colors" in user_analysis:
                    st.write("**Your Color Palette:**")
                    colors = user_analysis["dominant_colors"][:5]
                    color_html = '<div style="display: flex; flex-direction: row; gap: 5px;">'
                    for color in colors:
                        color_html += f'<div style="width: 25px; height: 25px; background-color: {color}; border-radius: 50%; border: 1px solid #ddd;"></div>'
                    color_html += "</div>"
                    st.markdown(color_html, unsafe_allow_html=True)

            with analysis_cols[1]:
                if "body_shape" in user_analysis:
                    st.write(f"**Body Type:** {user_analysis['body_shape'].title()}")

            with analysis_cols[2]:
                if "season_compatibility" in user_analysis:
                    st.write(f"**Season:** {user_analysis['season_compatibility'].title()}")

            st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 👔 Recommended Clothing")
            for rec in recommendations:
                with st.container():
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    confidence = int(rec.get("confidence", 0) * 100)
                    st.markdown(f"**{rec['type'].replace('_', ' ').title()}** - {confidence}% match")
                    st.image(rec["image"], use_container_width=True)

                    metadata = rec.get("metadata", {})
                    st.markdown(f"**Brand:** {metadata.get('brand', 'Unknown')}")
                    if metadata.get("price", 0) > 0:
                        st.markdown(f"**Price:** ${metadata['price']:.2f}")

                    reasons = metadata.get("reasons", [])
                    if reasons:
                        with st.expander("Why recommended?"):
                            for reason in reasons[:3]:
                                st.markdown(f"• {reason}")

                    color = metadata.get("color", "")
                    if color and color.startswith("#"):
                        st.markdown(
                            f"**Color:** <span style='color: {color}; font-weight: bold;'>{color}</span>",
                            unsafe_allow_html=True,
                        )

                    product_link = metadata.get("product_link", "")
                    if product_link:
                        st.markdown(
                            f'<a class="buy-now" href="{product_link}" target="_blank">🛒 Buy Now</a>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No recommendations generated")

        with col2:
            st.markdown("### 🪞 Virtual Try-On Result")
            if vton_result and vton_result.get("success"):
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)

                final_image_url = vton_result.get("final_image_url")
                final_image_base64 = vton_result.get("final_image")
                image_displayed = False

                if final_image_url:
                    st.image(final_image_url, caption="Your Virtual Try-On", use_container_width=True)
                    image_displayed = True
                elif final_image_base64:
                    try:
                        if final_image_base64.startswith("data:image"):
                            final_image_base64 = final_image_base64.split(",")[1]
                        st.image(
                            io.BytesIO(base64.b64decode(final_image_base64)),
                            caption="Your Virtual Try-On",
                            use_container_width=True,
                        )
                        image_displayed = True
                    except Exception as e:
                        st.error(f"Error displaying try-on image: {e}")

                if not image_displayed:
                    try_on_images = vton_result.get("try_on_images", [])
                    for image_data in try_on_images[:1]:
                        image_url = image_data.get("image_url")
                        image_data_b64 = image_data.get("image_data")
                        if image_url:
                            st.image(image_url, caption="Your Virtual Try-On", use_container_width=True)
                            image_displayed = True
                            break
                        elif image_data_b64:
                            try:
                                if image_data_b64.startswith("data:image"):
                                    image_data_b64 = image_data_b64.split(",")[1]
                                st.image(
                                    io.BytesIO(base64.b64decode(image_data_b64)),
                                    caption="Your Virtual Try-On",
                                    use_container_width=True,
                                )
                                image_displayed = True
                                break
                            except Exception as e:
                                st.error(f"Error displaying try-on image: {e}")

                if not image_displayed:
                    st.info("Try-on image processing completed but image not available for display.")

                processing_time = vton_result.get("processing_time", 0)
                st.markdown(f"**Processing Time:** {processing_time:.1f} seconds")

                if vton_result.get("debug_info"):
                    with st.expander("Processing Details"):
                        debug_info = vton_result["debug_info"]
                        if "correlation_id" in debug_info:
                            st.write(f"**Correlation ID:** {debug_info['correlation_id']}")
                        if "items_processed" in debug_info:
                            st.write(f"**Items Processed:** {debug_info['items_processed']}")
                        if "final_image_size" in debug_info:
                            st.write(f"**Image Size:** {debug_info['final_image_size']} bytes")

                st.markdown("</div>", unsafe_allow_html=True)
            elif vton_result and not vton_result.get("success"):
                st.error(f"Virtual try-on failed: {vton_result.get('error', 'Unknown error')}")
                if vton_result.get("message"):
                    st.info(vton_result["message"])
            else:
                st.info("Virtual try-on result not available")

        st.markdown("---")
        _, c, _ = st.columns([1, 1, 1])
        with c:
            if st.button("🔄 Start Over", type="secondary", use_container_width=True):
                st.session_state.results = None
                st.session_state.user_image = None
                st.session_state.processing = False
                st.rerun()

    def api_call(self, endpoint, data):
        url = f"{self.api_base_url}{endpoint}"
        try:
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=400,
            )
            if response.status_code == 200:
                return response.json()
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise Exception(f"{response.status_code}: {detail}")
        except requests.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.ConnectionError:
            raise Exception("Could not connect to API. Please check if the backend is running.")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")


if __name__ == "__main__":
    # Sidebar – no logs, no doc dumps
    with st.sidebar:
        st.title("AI Fashion Studio")
        st.info("1️⃣ Upload photo\n2️⃣ Set preferences\n3️⃣ Get results")

    # Run app
    try:
        SimpleVTONApp().run()
    except Exception as e:
        st.error(f"Application error: {e}")
