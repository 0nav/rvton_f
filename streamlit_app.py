"""
VTON + Recommendation Engine - Minimal Streamlit Frontend
A clean, focused application for virtual try-on with AI recommendations.
"""

import streamlit as st
import requests
import base64
import time
import json
from PIL import Image
import io
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="AI Fashion Studio",
    page_icon="�",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .recommendation-card {
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .buy-button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        border-radius: 4px;
        margin-top: 10px;
    }
    .sub-header {
        font-size: 1.75rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #333;
    }
    .error-message {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-message {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class VTONFrontend:
    """VTON and Recommendation Frontend Application"""
    
    def __init__(self):
        """Initialize the application with configuration and session state"""
        # API Configuration
        self.api_base_url = st.sidebar.text_input(
            "API URL", 
            value="http://localhost:8000",
            help="URL of the VTON API service"
        )
        
        # Initialize session state for storing uploaded images and results
        if "user_image" not in st.session_state:
            st.session_state.user_image = None
        if "recommendations" not in st.session_state:
            st.session_state.recommendations = None
        if "vton_results" not in st.session_state:
            st.session_state.vton_results = None
        if "processing" not in st.session_state:
            st.session_state.processing = False
    
    def run(self):
        """Run the main application"""
        st.markdown("<h1 class='main-header'>AI Fashion Studio</h1>", unsafe_allow_html=True)
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Try On Clothing", "Your Results"])
        
        with tab1:
            self._render_upload_section()
            self._render_preferences_section()
            
            # Process button
            if st.button("Get Recommendations & Try On", type="primary", disabled=st.session_state.processing):
                self._process_request()
        
        with tab2:
            self._render_results_section()
    
    def _render_upload_section(self):
        """Render the user image upload section"""
        st.subheader("Upload Your Photo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded_file = st.file_uploader("Choose a photo of yourself", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    # Store the image in session state
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    st.session_state.user_image = base64.b64encode(buf.getvalue()).decode("utf-8")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.session_state.user_image = None
            
            if st.session_state.user_image is None:
                st.info("Please upload an image to continue")
        
        with col2:
            if st.session_state.user_image:
                st.image(
                    io.BytesIO(base64.b64decode(st.session_state.user_image)),
                    caption="Your Photo",
                    use_column_width=True
                )
    
    def _render_preferences_section(self):
        """Render the user preferences section"""
        st.subheader("Style Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            style = st.selectbox(
                "Style Preference",
                options=["casual", "formal", "business", "sporty"],
                index=0
            )
            
            gender = st.selectbox(
                "Gender",
                options=["male", "female", "unisex"],
                index=0
            )
        
        with col2:
            season = st.selectbox(
                "Season",
                options=["spring", "summer", "fall", "winter"],
                index=1
            )
            
            occasion = st.selectbox(
                "Occasion",
                options=["everyday", "work", "party", "date"],
                index=0
            )
        
        max_price = st.slider(
            "Maximum Price ($)",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )
        
        # Store preferences in session state
        st.session_state.preferences = {
            "style": style,
            "gender": gender,
            "season": season,
            "occasion": occasion,
            "price_max": max_price
        }
        
        # Number of recommendations
        st.session_state.max_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=5,
            value=3
        )
    
    def _process_request(self):
        """Process the recommendation and try-on request"""
        if not st.session_state.user_image:
            st.error("Please upload an image first")
            return
        
        st.session_state.processing = True
        
        try:
            with st.spinner("Processing your request... This may take a minute..."):
                # Prepare the request data
                request_data = {
                    "user_image": st.session_state.user_image,
                    "preferences": st.session_state.preferences,
                    "max_recommendations": st.session_state.max_recommendations,
                    "include_vton": True
                }
                
                # Make API request
                response = self._api_call("/recommend-and-tryon", request_data)
                
                if response.get("success"):
                    st.session_state.recommendations = response.get("recommendations", [])
                    st.session_state.vton_results = response.get("vton_result", {})
                    st.session_state.user_analysis = response.get("user_analysis", {})
                    st.success("Successfully processed your request!")
                else:
                    st.error(f"Error: {response.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.processing = False
    
    def _render_results_section(self):
        """Render the recommendations and try-on results"""
        if not st.session_state.recommendations:
            st.info("No results yet. Please submit a request first.")
            return
        
        st.subheader("Your Personalized Recommendations")
        
        # Display recommendations in a grid
        cols = st.columns(min(3, len(st.session_state.recommendations)))
        
        for i, recommendation in enumerate(st.session_state.recommendations):
            with cols[i % len(cols)]:
                st.markdown(f"<div class='recommendation-card'>", unsafe_allow_html=True)
                
                # Display clothing image
                st.image(
                    recommendation["image"],
                    caption=f"{recommendation['type'].replace('_', ' ').title()}",
                    use_column_width=True
                )
                
                # Display recommendation details
                st.markdown(f"**Brand**: {recommendation['metadata']['brand']}")
                st.markdown(f"**Price**: ${recommendation['metadata']['price']:.2f}")
                st.markdown(f"**Match Score**: {int(recommendation['confidence'] * 100)}%")
                
                # Display reasons for recommendation
                if "reasons" in recommendation["metadata"]:
                    with st.expander("Why This Item?"):
                        for reason in recommendation["metadata"]["reasons"]:
                            st.markdown(f"• {reason}")
                
                # Buy button
                if "product_link" in recommendation["metadata"] and recommendation["metadata"]["product_link"]:
                    st.markdown(
                        f"<a href='{recommendation['metadata']['product_link']}' target='_blank' class='buy-button'>Buy Now</a>",
                        unsafe_allow_html=True
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Display VTON results if available
        if st.session_state.vton_results and st.session_state.vton_results.get("success"):
            st.subheader("Virtual Try-On Results")
            
            try_on_images = st.session_state.vton_results.get("try_on_images", [])
            if try_on_images:
                # Display try-on images in a grid
                vton_cols = st.columns(min(3, len(try_on_images)))
                
                for i, image_data in enumerate(try_on_images):
                    with vton_cols[i % len(vton_cols)]:
                        st.image(
                            image_data["image_url"] if "image_url" in image_data else image_data["image_data"],
                            caption=f"Try-On Result {i+1}",
                            use_column_width=True
                        )
            else:
                st.info("No try-on images were generated.")
        elif st.session_state.vton_results:
            st.warning(f"Virtual try-on processing failed: {st.session_state.vton_results.get('message', 'Unknown error')}")
    
    def _api_call(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API call to the backend service"""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=120  # Extended timeout for VTON processing
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", str(error_response))
                except:
                    error_detail = response.text
                
                raise Exception(f"API Error ({response.status_code}): {error_detail}")
        
        except requests.RequestException as e:
            logger.error(f"API Request Error: {e}")
            raise Exception(f"API Connection Error: {e}")


if __name__ == "__main__":
    app = VTONFrontend()
    app.run()
