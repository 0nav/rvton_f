"""
Simple VTON Frontend - One Page, Direct Results
Upload photo, set preferences, get recommendations + try-on results instantly.
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Fashion Studio",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .result-card {
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

class SimpleVTONApp:
    def __init__(self):
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        
        # Ensure URL format
        if not self.api_base_url.startswith(("http://", "https://")):
            self.api_base_url = f"http://{self.api_base_url}"
        if self.api_base_url.endswith("/"):
            self.api_base_url = self.api_base_url[:-1]
        
        # Initialize session state
        if "user_image" not in st.session_state:
            st.session_state.user_image = None
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "results" not in st.session_state:
            st.session_state.results = None
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
    
    def run(self):
        # Header
        st.markdown("<h1 class='main-header'>👗 AI Fashion Studio</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>Upload your photo, set preferences, get instant recommendations + try-on results!</p>", unsafe_allow_html=True)
        
        # Main content in two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Upload Your Photo")
            uploaded_file = st.file_uploader("Choose your photo", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    # Display uploaded image
                    st.image(image, caption="Your Photo", use_container_width=True)
                    
                    # Store as base64
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
            
            # Simple preferences
            gender = st.selectbox("Gender", ["male", "female", "unisex"], index=0)
            style = st.selectbox("Style", ["casual", "formal", "business", "sporty", "trendy"], index=0)
            season = st.selectbox("Season", ["spring", "summer", "fall", "winter"], index=1)
            occasion = st.selectbox("Occasion", ["everyday", "work", "party", "date", "formal"], index=0)
            
            st.markdown("---")
            
            # Process button
            process_disabled = st.session_state.processing or st.session_state.user_image is None
            
            if st.button("🪄 Get Recommendations & Try-On", 
                        disabled=process_disabled, 
                        type="primary"):
                if st.session_state.user_image:
                    st.session_state.processing = True
                    # Don't call st.rerun() here - let the processing happen first
                else:
                    st.error("Please upload an image first!")
            
            # Process when processing state is True
            if st.session_state.processing and st.session_state.user_image:
                self.process_recommendations_and_tryon(gender, style, season, occasion)
        
        # Show results if available
        if st.session_state.results:
            self.show_results()
    
    def process_recommendations_and_tryon(self, gender, style, season, occasion):
        """Process recommendations and try-on in one go"""
        if not st.session_state.user_image:
            st.error("Please upload an image first!")
            st.session_state.processing = False
            return
        
        # Create progress container
        with st.container():
            st.info("🔄 Processing your request... This may take several minutes...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Get recommendations + try-on
                status_text.text("Step 1/3: Analyzing your photo and style...")
                progress_bar.progress(25)
                
                request_data = {
                    "user_image": st.session_state.user_image,
                    "preferences": {
                        "style": style,
                        "gender": gender,
                        "season": season,
                        "occasion": occasion
                    },
                    "max_recommendations": 3,  # Fixed to 3 for simplicity
                    "include_vton": True  # Always include try-on
                }
                
                status_text.text("Step 2/3: Getting personalized recommendations...")
                progress_bar.progress(50)
                
                # Make API call
                response = self.api_call("/recommend-and-tryon", request_data)
                
                status_text.text("Step 3/3: Processing virtual try-on results...")
                progress_bar.progress(90)
                
                if response.get("success"):
                    st.session_state.results = {
                        "recommendations": response.get("recommendations", []),
                        "vton_result": response.get("vton_result", {}),
                        "user_analysis": response.get("user_analysis", {}),
                        "processing_time": response.get("processing_time", 0),
                        "correlation_id": response.get("correlation_id", "")
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
                # Rerun to update button state and show results
                st.rerun()
    
    def show_results(self):
        """Display recommendations and try-on results"""
        st.markdown("---")
        st.subheader("🎯 Your Results")
        
        results = st.session_state.results
        recommendations = results.get("recommendations", [])
        vton_result = results.get("vton_result", {})
        
        # Results in two columns
        col1, col2 = st.columns([1, 1])
        
        # Show user analysis if available
        user_analysis = results.get("user_analysis", {})
        if user_analysis:
            st.markdown("### 👤 Your Style Analysis")
            analysis_cols = st.columns(3)
            
            with analysis_cols[0]:
                if "dominant_colors" in user_analysis:
                    st.write("**Your Color Palette:**")
                    colors = user_analysis["dominant_colors"][:5]  # Top 5 colors
                    color_html = '<div style="display: flex; flex-direction: row; gap: 5px;">'
                    for color in colors:
                        color_html += f'<div style="width: 25px; height: 25px; background-color: {color}; border-radius: 50%; border: 1px solid #ddd;"></div>'
                    color_html += '</div>'
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
            
            if recommendations:
                for i, rec in enumerate(recommendations):
                    with st.container():
                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        
                        # Item info
                        confidence = int(rec["confidence"] * 100)
                        st.markdown(f"**{rec['type'].replace('_', ' ').title()}** - {confidence}% match")
                        
                        # Image
                        st.image(rec["image"], use_container_width=True)
                        
                        # Metadata
                        metadata = rec.get("metadata", {})
                        st.markdown(f"**Brand:** {metadata.get('brand', 'Unknown')}")
                        if metadata.get('price', 0) > 0:
                            st.markdown(f"**Price:** ${metadata['price']:.2f}")
                        
                        # Show reasons if available
                        reasons = metadata.get('reasons', [])
                        if reasons:
                            with st.expander("Why recommended?"):
                                for reason in reasons[:3]:  # Show top 3 reasons
                                    st.markdown(f"• {reason}")
                        
                        # Color info if available
                        color = metadata.get('color', '')
                        if color and color.startswith('#'):
                            st.markdown(f"**Color:** <span style='color: {color}; font-weight: bold;'>{color}</span>", unsafe_allow_html=True)
                        
                        # Buy button
                        product_link = metadata.get('product_link', '')
                        if product_link:
                            st.markdown(f"[🛒 Buy Now]({product_link})", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No recommendations generated")
        
        with col2:
            st.markdown("### 🪞 Virtual Try-On Result")
            
            if vton_result and vton_result.get("success"):
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                # Try to get the final image - handle different response formats
                final_image_url = vton_result.get("final_image_url")
                final_image_base64 = vton_result.get("final_image")
                
                image_displayed = False
                
                if final_image_url:
                    st.image(final_image_url, caption="Your Virtual Try-On", use_container_width=True)
                    image_displayed = True
                elif final_image_base64:
                    # Handle base64 image
                    try:
                        if final_image_base64.startswith("data:image"):
                            final_image_base64 = final_image_base64.split(',')[1]
                        st.image(io.BytesIO(base64.b64decode(final_image_base64)), 
                                caption="Your Virtual Try-On", use_container_width=True)
                        image_displayed = True
                    except Exception as e:
                        st.error(f"Error displaying try-on image: {e}")
                
                # Fallback: check for try_on_images array (legacy format)
                if not image_displayed:
                    try_on_images = vton_result.get("try_on_images", [])
                    if try_on_images:
                        for image_data in try_on_images[:1]:  # Show first image only
                            image_url = image_data.get("image_url")
                            image_data_b64 = image_data.get("image_data")
                            
                            if image_url:
                                st.image(image_url, caption="Your Virtual Try-On", use_container_width=True)
                                image_displayed = True
                                break
                            elif image_data_b64:
                                try:
                                    if image_data_b64.startswith("data:image"):
                                        image_data_b64 = image_data_b64.split(',')[1]
                                    st.image(io.BytesIO(base64.b64decode(image_data_b64)), 
                                            caption="Your Virtual Try-On", use_container_width=True)
                                    image_displayed = True
                                    break
                                except Exception as e:
                                    st.error(f"Error displaying try-on image: {e}")
                
                if not image_displayed:
                    st.info("Try-on image processing completed but image not available for display.")
                
                # Processing info
                processing_time = vton_result.get("processing_time", 0)
                st.markdown(f"**Processing Time:** {processing_time:.1f} seconds")
                
                # Show additional debug info if available
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
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start Over", type="secondary", use_container_width=True):
                st.session_state.results = None
                st.session_state.user_image = None
                st.session_state.processing = False
                st.rerun()
    
    def api_call(self, endpoint, data):
        """Make API call to backend"""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=400  # 400 seconds timeout as requested
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API Error ({response.status_code})"
                try:
                    error_detail = response.json().get("detail", response.text)
                    error_msg += f": {error_detail}"
                except Exception:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)
        
        except requests.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.ConnectionError:
            raise Exception("Could not connect to API. Please check if the backend is running.")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")

if __name__ == "__main__":
    # Sidebar info
    st.sidebar.title("AI Fashion Studio")
    st.sidebar.info("Simple workflow: Upload → Set preferences → Get results!")
    st.sidebar.markdown("---")
    st.sidebar.caption("v3.0 - Simplified")
    
    # API status check
    try:
        app = SimpleVTONApp()
        response = requests.get(f"{app.api_base_url}/health", timeout=3)
        if response.status_code == 200:
            st.sidebar.success("✅ Backend Connected")
        else:
            st.sidebar.warning("⚠️ Backend Issues")
    except Exception:
        st.sidebar.error("❌ Backend Offline")
    
    # Run app
    try:
        app = SimpleVTONApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
