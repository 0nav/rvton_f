"""
VTON + Recommendation Engine - Minimal Streamlit Frontend
A clean, focused application for virtual try-on with AI recommendations.
"""

import streamlit as st
import requests
import base64
import time
import json
import os
from PIL import Image
import io
import logging
from typing import Dict, Any

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
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .recommendation-card {
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
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
        transition: background-color 0.3s ease;
        cursor: pointer;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    .buy-button:hover {
        background-color: #45a049;
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
    .color-swatch {
        display: inline-block;
        width: 25px;
        height: 25px;
        margin-right: 5px;
        border-radius: 50%;
        border: 1px solid #ddd;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    /* Additional styling for better UI */
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .stCheckbox > label {
        font-weight: 600;
    }
    /* Loading animation enhancements */
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* Section dividers */
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0));
        margin: 2rem 0;
    }
    /* Image container styling */
    img {
        border-radius: 8px;
        transition: transform 0.3s ease;
    }
    img:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

class VTONFrontend:
    """VTON and Recommendation Frontend Application"""
    
    def __init__(self):
        """Initialize the application with configuration and session state"""
        # API Configuration from Streamlit secrets or from environment variables as fallback
        try:
            self.api_base_url = st.secrets.get("API_BASE_URL") or os.environ.get("API_BASE_URL")
            self.debug_mode = st.secrets.get("DEBUG", "false").lower() == "true" or os.environ.get("DEBUG", "false").lower() == "true"
            
            if not self.api_base_url:
                # For local development - fallback to localhost if no config
                self.api_base_url = "http://localhost:8000"
                logger.info(f"No API URL configured, using default: {self.api_base_url}")
        except Exception as e:
            # Set to localhost if secrets not available
            self.api_base_url = "http://localhost:8000"
            self.debug_mode = False
            logger.error(f"Failed to load API configuration, using default: {e}")
            st.warning("""
            ⚠️ **Configuration Notice**: Using default API URL. 
            
            For production deployment, please set up your `.streamlit/secrets.toml` file with:
            ```toml
            API_BASE_URL = "https://your-backend-api-url"
            ```
            """)
        
        # Check health endpoint
        api_health = self._check_api_health()
        if api_health.get("status") == "healthy":
            st.sidebar.success(f"✅ Connected to API: {self.api_base_url}")
            # Show additional API info
            with st.sidebar.expander("API Details"):
                st.write(f"Status: {api_health.get('status', 'unknown')}")
                st.write(f"ComfyUI Connected: {api_health.get('comfyui_connected', False)}")
                st.write(f"Cache Entries: {api_health.get('cache_stats', {}).get('entries', 0)}")
        else:
            if self.api_base_url is None:
                st.sidebar.error("❌ API URL not configured. Please set API_BASE_URL in Streamlit secrets.")
            else:
                st.sidebar.error(f"❌ Cannot connect to API: {self.api_base_url}")
                if api_health.get("error"):
                    st.sidebar.error(f"Error: {api_health.get('error')}")
        
        # Initialize session state for storing uploaded images and results
        if "user_image" not in st.session_state:
            st.session_state.user_image = None
        if "recommendations" not in st.session_state:
            st.session_state.recommendations = None
        if "vton_results" not in st.session_state:
            st.session_state.vton_results = None
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "selected_items" not in st.session_state:
            st.session_state.selected_items = []
        if "current_step" not in st.session_state:
            st.session_state.current_step = "upload"  # Options: "upload", "select", "results"
    
    def run(self):
        """Run the main application"""
        # Enhanced header with logo and title
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <span style="font-size: 3rem; color: #4CAF50;">👗</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("<h1 class='main-header'>AI Fashion Studio</h1>", unsafe_allow_html=True)
        
        # Show different content based on current step
        if st.session_state.current_step == "upload":
            self._render_upload_section()
            self._render_preferences_section()
            
            # Process button to get recommendations
            if st.button("Get Recommendations", type="primary", disabled=st.session_state.processing):
                self._get_recommendations()
                
        elif st.session_state.current_step == "select":
            # Display recommendations and let user select items
            self._render_recommendation_selection()
            
            # Try-on button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back", type="secondary"):
                    st.session_state.current_step = "upload"
                    st.rerun()
            
            with col2:
                # Only enable button if there are selected items and they're compatible
                disable_tryon = not st.session_state.selected_items or (
                    st.session_state.selected_items and 
                    not self._check_clothing_compatibility(st.session_state.selected_items)["valid"]
                )
                if st.button("Try On Selected Items", type="primary", disabled=disable_tryon):
                    self._process_tryon()
        
        elif st.session_state.current_step == "results":
            # Display try-on results
            self._render_results_section()
            
            # Back button
            if st.button("← Back to Selection", type="secondary"):
                st.session_state.current_step = "select"
                st.rerun()
    
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
                    use_container_width=True
                )
    
    def _render_preferences_section(self):
        """Render the user preferences section"""
        st.subheader("Style Preferences")
        st.write("Tell us your preferences to get better recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            style = st.selectbox(
                "Style Preference",
                options=["casual", "formal", "business", "sporty", "trendy", "classic", "bohemian"],
                index=0,
                help="Choose your preferred clothing style"
            )
            
            gender = st.selectbox(
                "Gender",
                options=["male", "female", "unisex"],
                index=0,
                help="Select your gender for better fitting recommendations"
            )
        
        with col2:
            season = st.selectbox(
                "Season",
                options=["spring", "summer", "fall", "winter"],
                index=1,
                help="Season for which you need clothing recommendations"
            )
            
            occasion = st.selectbox(
                "Occasion",
                options=["everyday", "work", "party", "date", "formal", "vacation"],
                index=0,
                help="The occasion you're shopping for"
            )
        
        # Store preferences in session state
        st.session_state.preferences = {
            "style": style,
            "gender": gender,
            "season": season,
            "occasion": occasion
        }
        
        # Number of recommendations
        st.session_state.max_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5,
            help="How many recommendations would you like to see"
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
        """Render the try-on results"""
        if not st.session_state.vton_results:
            st.info("No try-on results yet. Please select items and try them on first.")
            return
        
        st.subheader("Your Virtual Try-On Results")
        
        # Display the selected items in a nice grid
        st.write("Items you tried on:")
        num_selected = len(st.session_state.selected_items)
        cols_per_row = min(3, num_selected)
        
        if cols_per_row > 0:
            selected_items_cols = st.columns(cols_per_row)
            
            for i, idx in enumerate(st.session_state.selected_items):
                with selected_items_cols[i % cols_per_row]:
                    recommendation = st.session_state.recommendations[idx]
                    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                    st.image(
                        recommendation["image"],
                        caption=f"{recommendation['type'].replace('_', ' ').title()}",
                        use_container_width=True
                    )
                    st.markdown(f"**Brand**: {recommendation['metadata']['brand']}")
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Add a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Display VTON results with nice styling
        st.subheader("Try-On Images")
        try_on_images = st.session_state.vton_results.get("try_on_images", [])
        if try_on_images:
            # Display try-on images in a grid
            vton_cols = st.columns(min(2, len(try_on_images)))
            
            for i, image_data in enumerate(try_on_images):
                with vton_cols[i % len(vton_cols)]:
                    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                    
                    image_url = image_data.get("image_url")
                    image_data_b64 = image_data.get("image_data")
                    
                    if image_url:
                        st.image(
                            image_url,
                            caption=f"Try-On Result {i+1}",
                            use_container_width=True
                        )
                    elif image_data_b64:
                        if image_data_b64.startswith("data:image"):
                            # Remove data URL prefix
                            image_data_b64 = image_data_b64.split(',')[1]
                        
                        st.image(
                            io.BytesIO(base64.b64decode(image_data_b64)),
                            caption=f"Try-On Result {i+1}",
                            use_container_width=True
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No try-on images were generated.")
            
        # Display any additional processing information
        with st.expander("Processing Details"):
            processing_time = st.session_state.vton_results.get("processing_time", 0)
            st.write(f"Processing Time: {processing_time:.2f} seconds")
            if "correlation_id" in st.session_state.vton_results:
                st.write(f"Correlation ID: {st.session_state.vton_results['correlation_id']}")
            
        # Start over button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start Over", type="primary", use_container_width=True):
                # Reset session state and go back to upload step
                st.session_state.current_step = "upload"
                st.session_state.selected_items = []
                st.session_state.vton_results = None
                st.rerun()
    
    def _get_recommendations(self):
        """Get recommendations from the API without try-on"""
        if not st.session_state.user_image:
            st.error("Please upload an image first")
            return
        
        st.session_state.processing = True
        
        try:
            # Create a container for the loading animation
            progress_container = st.container()
            with progress_container:
                # Show a more visible loading animation
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                    <div style="display: flex; justify-content: center; margin: 2rem 0;">
                        <div class="stSpinner">
                            <div class="st-cw">
                                <div class="st-cw-circle" style="width: 5rem; height: 5rem;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Analyzing your style and generating recommendations...</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;'>This may take a minute. Please wait...</p>", unsafe_allow_html=True)
            
            # Prepare the request data - without try-on
            request_data = {
                "user_image": st.session_state.user_image,
                "preferences": st.session_state.preferences,
                "max_recommendations": st.session_state.max_recommendations,
                "include_vton": False  # Don't include try-on yet
            }
            
            # Make API request
            response = self._api_call("/recommend-and-tryon", request_data)
            
            # Clear the loading animation
            progress_container.empty()            
            if response.get("success"):
                st.session_state.recommendations = response.get("recommendations", [])
                st.session_state.user_analysis = response.get("user_analysis", {})
                
                # Display user analysis summary
                st.subheader("Your Style Analysis")
                cols = st.columns(3)
                
                # Display dominant colors
                with cols[0]:
                    if "dominant_colors" in st.session_state.user_analysis:
                        st.write("**Your Color Palette:**")
                        colors = st.session_state.user_analysis["dominant_colors"]
                        color_html = '<div style="display: flex; flex-direction: row;">'
                        for color in colors[:5]:  # Show up to 5 colors
                            color_html += f'<div class="color-swatch" style="background-color: {color};"></div>'
                        color_html += '</div>'
                        st.markdown(color_html, unsafe_allow_html=True)
                
                # Display body shape
                with cols[1]:
                    if "body_shape" in st.session_state.user_analysis:
                        st.write(f"**Body Type:** {st.session_state.user_analysis['body_shape'].title()}")
                
                # Display season compatibility
                with cols[2]:
                    if "season_compatibility" in st.session_state.user_analysis:
                        st.write(f"**Season:** {st.session_state.user_analysis['season_compatibility'].title()}")
                
                st.session_state.current_step = "select"  # Move to selection step
                st.session_state.selected_items = []  # Clear any previous selections
                st.success("Recommendations generated successfully based on your style!")
            else:
                st.error(f"Error: {response.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.processing = False
    
    def _render_recommendation_selection(self):
        """Render the recommendations and let user select items to try on"""
        st.subheader("Your Personalized Recommendations")
        st.write("Our AI has analyzed your photo and style preferences to recommend these items.")
        st.write("Select items below that you'd like to try on:")
        
        if not st.session_state.recommendations:
            st.info("No recommendations available. Please go back and get recommendations first.")
            return
        
        # Show all recommendations in a grid layout
        all_recommendations = st.session_state.recommendations
        
        # Group by clothing type for better organization
        clothing_types = {
            "tops": ["shirt", "t_shirt", "blouse", "top", "tank_top", "crop_top"],
            "outerwear": ["jacket", "blazer", "coat", "cardigan", "sweater", "hoodie", "vest"],
            "bottoms": ["pants", "jeans", "shorts", "skirt", "leggings"],
            "dresses": ["dress", "maxi_dress", "mini_dress", "cocktail_dress"],
            "footwear": ["shoes", "sneakers", "boots", "heels"],
            "accessories": ["hat", "bag", "belt", "scarf", "watch", "gloves"]
        }
        
        # Create a separator between clothing categories
        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        
        # Display all recommendations in a grid
        num_items = len(all_recommendations)
        
        # Initialize selected_items if empty and there are recommendations
        if not st.session_state.selected_items and num_items > 0:
            # Auto-select up to 3 items (tops, bottoms, outerwear in that order)
            tops = [i for i, r in enumerate(all_recommendations) if r["type"] in clothing_types["tops"]]
            bottoms = [i for i, r in enumerate(all_recommendations) if r["type"] in clothing_types["bottoms"]]
            outerwear = [i for i, r in enumerate(all_recommendations) if r["type"] in clothing_types["outerwear"]]
            dresses = [i for i, r in enumerate(all_recommendations) if r["type"] in clothing_types["dresses"]]
            
            # Try to get a balanced outfit selection
            if dresses:
                # If we have a dress, select that and maybe an outerwear piece
                st.session_state.selected_items = [dresses[0]]
                if outerwear:
                    st.session_state.selected_items.append(outerwear[0])
            else:
                # Otherwise try for a top + bottom + outerwear
                if tops:
                    st.session_state.selected_items.append(tops[0])
                if bottoms:
                    st.session_state.selected_items.append(bottoms[0])
                if outerwear and len(st.session_state.selected_items) < 3:
                    st.session_state.selected_items.append(outerwear[0])
            
            # Ensure we don't have more than 3 selected
            st.session_state.selected_items = st.session_state.selected_items[:3]
        
        # Create columns based on number of items (max 3 per row)
        cols_per_row = min(3, num_items)
        
        # Show each clothing category that has items
        for category_name, category_types in clothing_types.items():
            # Get items in this category
            category_items = [(i, rec) for i, rec in enumerate(all_recommendations) 
                              if rec["type"] in category_types]
            
            if category_items:
                st.subheader(f"{category_name.title()}")
                
                # Create a row of items
                cols = st.columns(min(3, len(category_items)))
                
                for idx, (item_idx, item) in enumerate(category_items):
                    with cols[idx % len(cols)]:
                        st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                        
                        # Display clothing image
                        st.image(
                            item["image"],
                            caption=f"{item['type'].replace('_', ' ').title()}",
                            use_container_width=True
                        )
                        
                        # Show match score with progress bar
                        confidence = int(item["confidence"] * 100)
                        st.progress(item["confidence"], text=f"Match: {confidence}%")
                        
                        # Brand and metadata
                        st.markdown(f"**Brand**: {item['metadata']['brand']}")
                        if "price" in item["metadata"] and item["metadata"]["price"] > 0:
                            st.markdown(f"**Price**: ${item['metadata']['price']:.2f}")
                        
                        # Show recommendation reasons
                        if "reasons" in item["metadata"] and item["metadata"]["reasons"]:
                            with st.expander("Why we recommend this"):
                                for reason in item["metadata"]["reasons"][:3]:  # Show up to 3 reasons
                                    st.markdown(f"• {reason}")
                        
                        # Selection and Buy buttons in same row
                        col1, col2 = st.columns(2)
                        with col1:
                            # Checkbox for selection with key based on item index
                            is_selected = item_idx in st.session_state.selected_items
                            if st.checkbox("Select", value=is_selected, key=f"select_{item_idx}"):
                                if item_idx not in st.session_state.selected_items:
                                    st.session_state.selected_items.append(item_idx)
                            else:
                                if item_idx in st.session_state.selected_items:
                                    st.session_state.selected_items.remove(item_idx)
                        
                        with col2:
                            # Buy button if product link exists
                            product_link = item["metadata"].get("product_link", "")
                            if product_link:
                                st.markdown(
                                    f"<a href='{product_link}' target='_blank' class='buy-button'>Buy Now</a>",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    "<span class='buy-button' style='background-color: #cccccc;'>Not Available</span>",
                                    unsafe_allow_html=True
                                )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Add spacing between categories
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Display selection summary in a fixed section at the bottom
        st.markdown("<div style='position: sticky; bottom: 0; background: white; padding: 1rem; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
        
        # Selection summary
        st.subheader("Your Selection")
        if st.session_state.selected_items:
            # Check clothing compatibility
            compatibility = self._check_clothing_compatibility(st.session_state.selected_items)
            
            # Show selected items count
            selected_count = len(st.session_state.selected_items)
            selected_items_text = "item" if selected_count == 1 else "items"
            st.success(f"You've selected {selected_count} {selected_items_text} for try-on")
            
            # Show what's selected
            selected_types = [all_recommendations[i]["type"].replace("_", " ").title() 
                             for i in st.session_state.selected_items]
            st.write(f"Selected: {', '.join(selected_types)}")
            
            # Show compatibility warnings or conflicts
            if not compatibility["valid"]:
                st.error("⚠️ Incompatible combination detected:")
                for conflict in compatibility["conflicts"]:
                    st.error(f"- {conflict}")
                st.warning("Please adjust your selection to continue.")
            elif compatibility["warnings"]:
                st.warning("⚠️ Potential issues with this combination:")
                for warning in compatibility["warnings"]:
                    st.warning(f"- {warning}")
        else:
            st.warning("Please select at least one item to try on")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add help section for valid combinations as an expander
        with st.expander("Help: Valid clothing combinations"):
            st.markdown("""
            ### Valid clothing combinations:
            
            ✅ **Single Items:**
            - One top (shirt, t-shirt, blouse, etc.)
            - One bottom (pants, jeans, shorts, skirt)
            - One dress
            - One outerwear piece (jacket, coat, etc.)
            
            ✅ **Valid Combinations:**
            - Top + Bottom (e.g., shirt + pants)
            - Top + Bottom + Outerwear (e.g., t-shirt + jeans + jacket)
            - Dress + Outerwear (e.g., dress + cardigan)
            - Bottom + Leggings (specific combination allowed)
            
            ❌ **Invalid Combinations:**
            - Multiple tops (e.g., shirt + t-shirt)
            - Multiple bottoms (except with leggings)
            - Multiple dresses
            - Dress + Top or Dress + Bottom
            
            For best results, choose 1-3 compatible items.
            """)
    
    def _process_tryon(self):
        """Process the try-on request with only the selected items"""
        st.session_state.processing = True
        
        try:
            # Create a container for the loading animation
            progress_container = st.container()
            with progress_container:
                # Show a more visible loading animation
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                    <div style="display: flex; justify-content: center; margin: 2rem 0;">
                        <div class="stSpinner">
                            <div class="st-cw">
                                <div class="st-cw-circle" style="width: 5rem; height: 5rem;"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Processing your virtual try-on...</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;'>This may take up to 9 minutes. Please don't close this window while we generate your outfits.</p>", unsafe_allow_html=True)
                    
                    # Add a countdown timer that updates
                    countdown_placeholder = st.empty()
                    start_time = time.time()
                    
                    def update_timer():
                        elapsed = int(time.time() - start_time)
                        remaining = max(0, 540 - elapsed)
                        minutes = remaining // 60
                        seconds = remaining % 60
                        countdown_placeholder.markdown(f"<p style='text-align: center;'>Time remaining: {minutes}m {seconds}s</p>", unsafe_allow_html=True)
                    
                    # Initial timer display
                    update_timer()
            
            # Get selected clothing items
            selected_recommendations = [
                st.session_state.recommendations[i] for i in st.session_state.selected_items
            ]
            
            # Double-check clothing compatibility
            compatibility = self._check_clothing_compatibility(st.session_state.selected_items)
            if not compatibility["valid"]:
                progress_container.empty()  # Clear the loading animation
                st.error("Cannot process: incompatible clothing combination")
                for conflict in compatibility["conflicts"]:
                    st.error(f"- {conflict}")
                st.session_state.processing = False
                return
            
            # Create clothing items for VTON in the required format
            clothing_items = []
            for rec in selected_recommendations:
                clothing_items.append({
                    "image": {
                        "data": rec["image"]
                    },
                    "type": rec["type"],
                    "context": {
                        "brand": rec["metadata"]["brand"],
                        "style": st.session_state.preferences["style"]
                    }
                })
            
            # Prepare try-on request
            request_data = {
                "user_photo": {
                    "data": f"data:image/jpeg;base64,{st.session_state.user_image}"
                },
                "clothing_items": clothing_items,
                "scene_context": {
                    "style": st.session_state.preferences["style"],
                    "season": st.session_state.preferences["season"],
                    "occasion": st.session_state.preferences["occasion"]
                }
            }
            
            # Make API request to VTON endpoint
            response = self._api_call("/vton", request_data)
            
            # Clear the loading animation
            progress_container.empty()
            
            if response.get("success"):
                st.session_state.vton_results = response
                st.session_state.current_step = "results"  # Move to results step
                st.success("Try-on completed successfully!")
            else:
                st.error(f"Error: {response.get('message', 'Unknown error')}")
                if self.debug_mode:
                    st.expander("Debug Info").json(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if self.debug_mode:
                st.exception(e)
        finally:
            st.session_state.processing = False
    
    def _check_clothing_compatibility(self, selected_items):
        """Check if the selected clothing items are compatible for VTON processing
        
        This implements client-side validation similar to what the backend does,
        to prevent sending invalid combinations that would fail."""
        clothing_types = [st.session_state.recommendations[i]["type"] for i in selected_items]
        
        # Check for direct conflicts based on common rules
        conflicts = []
        warnings = []
        
        # Check for multiple items of same type or category
        top_types = ["shirt", "t_shirt", "blouse", "top", "tank_top", "crop_top"]
        outerwear_types = ["jacket", "blazer", "coat", "cardigan", "sweater", "hoodie", "vest"]
        bottom_types = ["pants", "jeans", "shorts", "skirt", "leggings"]
        dress_types = ["dress", "maxi_dress", "mini_dress", "cocktail_dress"]
        
        # Count categories
        tops = [t for t in clothing_types if t in top_types]
        bottoms = [t for t in clothing_types if t in bottom_types]
        dresses = [t for t in clothing_types if t in dress_types]
        
        # Note: outerwear check remains available in warnings logic if needed
        
        # Check for conflicts
        if len(dresses) > 0 and (len(tops) > 0 or len(bottoms) > 0):
            conflicts.append("Cannot combine dresses with tops or bottoms")
        
        if len(dresses) > 1:
            conflicts.append("Cannot use multiple dresses")
        
        if len(tops) > 1:
            conflicts.append("Cannot use multiple tops (shirts, t-shirts, etc.)")
        
        if len(bottoms) > 1 and not ("leggings" in bottoms and len(bottoms) == 2):
            warnings.append("Using multiple bottom garments may cause issues")
        
        # Return validation results
        return {
            "valid": len(conflicts) == 0,
            "conflicts": conflicts,
            "warnings": warnings
        }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API health endpoint and return status information."""
        # Return error information if api_base_url is None
        if self.api_base_url is None:
            return {"status": "error", "error": "API URL not configured"}
            
        try:
            # Try to access the health endpoint
            response = requests.get(
                f"{self.api_base_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error", 
                    "error": f"API returned status code: {response.status_code}"
                }
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}
    
    def _check_api_connection(self):
        """Check if the API is reachable - LEGACY METHOD"""
        # Return False immediately if api_base_url is None
        if self.api_base_url is None:
            return False
            
        try:
            # Try a simple HEAD request
            response = requests.head(
                self.api_base_url,
                timeout=10
            )
            return response.status_code < 400
        except Exception as e:
            logger.warning(f"API connection check failed: {e}")
            return False
                
    def _api_call(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API call to the backend service"""
        # Check if API URL is configured
        if self.api_base_url is None:
            st.error("⚠️ API is not configured. Please set up API_BASE_URL in Streamlit secrets.")
            raise ValueError("API_BASE_URL is not configured in Streamlit secrets")
            
        url = f"{self.api_base_url}{endpoint}"
        progress_placeholder = st.empty()
        
        try:
            # Show loading indicator while waiting for API response
            progress_placeholder.progress(0, "Connecting to API...")
            
            # Make the API call with 540 seconds timeout
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=540  # 9 minutes timeout
            )
            
            # Clear the progress indicator
            progress_placeholder.empty()
                
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", str(error_response))
                except Exception as json_error:
                    logger.error(f"Error parsing API error response: {json_error}")
                    error_detail = response.text
                
                error_msg = f"API Error ({response.status_code}): {error_detail}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        except requests.Timeout:
            # Specific handling for timeout errors
            progress_placeholder.empty()
            raise Exception("API request timed out. The server took too long to respond. Please try again later.")
            
        except requests.RequestException as e:
            # Clear any progress indicators
            progress_placeholder.empty()
            raise Exception(f"API Connection Error: {e}")


if __name__ == "__main__":
    # Display app information in sidebar
    st.sidebar.title("AI Fashion Studio")
    st.sidebar.info(
        "This application lets you view personalized clothing recommendations "
        "and try them on virtually using AI technology."
    )
    
    # Display workflow steps in sidebar
    st.sidebar.subheader("Workflow")
    st.sidebar.markdown(
        "1. Upload your photo\n"
        "2. Get clothing recommendations\n"
        "3. Select items to try on\n"
        "4. View try-on results"
    )
    
    try:
        # Initialize and run the app
        app = VTONFrontend()
        app.run()
    except ValueError as e:
        if "API_BASE_URL is not configured" in str(e):
            st.error("⚠️ Application cannot start: API configuration is missing.")
            st.info("Please configure the app by setting API_BASE_URL in Streamlit secrets.")
        else:
            st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
