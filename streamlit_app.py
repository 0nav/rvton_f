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
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

class VTONFrontend:
    """VTON and Recommendation Frontend Application"""
    
    def __init__(self):
        """Initialize the application with configuration and session state"""
        # API Configuration from environment variable
        default_api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_base_url = st.sidebar.text_input(
            "API URL", 
            value=default_api_url,
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
        if "selected_items" not in st.session_state:
            st.session_state.selected_items = []
        if "current_step" not in st.session_state:
            st.session_state.current_step = "upload"  # Options: "upload", "select", "results"
    
    def run(self):
        """Run the main application"""
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
        
        max_price = st.slider(
            "Maximum Price ($)",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Maximum price for recommended items"
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
        
        # Display the selected items
        st.write("Items you tried on:")
        selected_items_cols = st.columns(min(3, len(st.session_state.selected_items)))
        
        for i, idx in enumerate(st.session_state.selected_items):
            with selected_items_cols[i % len(selected_items_cols)]:
                recommendation = st.session_state.recommendations[idx]
                st.image(
                    recommendation["image"],
                    caption=f"{recommendation['type'].replace('_', ' ').title()}",
                    use_container_width=True
                )
        
        # Display VTON results
        st.subheader("Try-On Images")
        try_on_images = st.session_state.vton_results.get("try_on_images", [])
        if try_on_images:
            # Display try-on images in a grid
            vton_cols = st.columns(min(2, len(try_on_images)))
            
            for i, image_data in enumerate(try_on_images):
                with vton_cols[i % len(vton_cols)]:
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
        else:
            st.info("No try-on images were generated.")
            
        # Display any additional processing information
        with st.expander("Processing Details"):
            processing_time = st.session_state.vton_results.get("processing_time", 0)
            st.write(f"Processing Time: {processing_time:.2f} seconds")
            if "correlation_id" in st.session_state.vton_results:
                st.write(f"Correlation ID: {st.session_state.vton_results['correlation_id']}")
            
        # Start over button
        if st.button("Start Over", type="primary"):
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
            with st.spinner("Analyzing your style and generating recommendations..."):
                # Prepare the request data - without try-on
                request_data = {
                    "user_image": st.session_state.user_image,
                    "preferences": st.session_state.preferences,
                    "max_recommendations": st.session_state.max_recommendations,
                    "include_vton": False  # Don't include try-on yet
                }
                
                # Make API request
                response = self._api_call("/recommend-and-tryon", request_data)
                
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
        st.write("Select one or more items below that you'd like to try on:")
        
        if not st.session_state.recommendations:
            st.info("No recommendations available. Please go back and get recommendations first.")
            return
        
        # Group recommendations by clothing type
        clothing_categories = {
            "tops": ["shirt", "t_shirt", "blouse", "top", "tank_top", "crop_top"],
            "outerwear": ["jacket", "blazer", "coat", "cardigan", "sweater", "hoodie", "vest"],
            "bottoms": ["pants", "jeans", "shorts", "skirt", "leggings"],
            "dresses": ["dress", "maxi_dress", "mini_dress", "cocktail_dress"],
            "footwear": ["shoes", "sneakers", "boots", "heels"],
            "accessories": ["hat", "bag", "belt", "scarf", "watch", "gloves"]
        }
        
        grouped_recommendations = {}
        for category, types in clothing_categories.items():
            grouped_recommendations[category] = []
            for i, rec in enumerate(st.session_state.recommendations):
                if rec["type"] in types:
                    grouped_recommendations[category].append((i, rec))
        
        # Display recommendations by category with tabs
        categories_with_items = [cat for cat, items in grouped_recommendations.items() if items]
        if categories_with_items:
            tabs = st.tabs(categories_with_items)
            
            for i, category in enumerate(categories_with_items):
                with tabs[i]:
                    items = grouped_recommendations[category]
                    
                    # Create columns for items in this category
                    num_cols = min(3, len(items))
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        
                        for j, (index, recommendation) in enumerate(items):
                            with cols[j % num_cols]:
                                st.markdown(f"<div class='recommendation-card'>", unsafe_allow_html=True)
                                
                                # Display clothing image
                                st.image(
                                    recommendation["image"],
                                    caption=f"{recommendation['type'].replace('_', ' ').title()}",
                                    use_container_width=True
                                )
                                
                                # Show match score with progress bar
                                confidence = int(recommendation["confidence"] * 100)
                                st.progress(recommendation["confidence"], text=f"Match: {confidence}%")
                                
                                # Checkbox for selection
                                item_key = f"item_{index}"
                                if st.checkbox("Select for try-on", key=item_key):
                                    if index not in st.session_state.selected_items:
                                        st.session_state.selected_items.append(index)
                                else:
                                    if index in st.session_state.selected_items:
                                        st.session_state.selected_items.remove(index)
                                
                                # Display recommendation details
                                st.markdown(f"**Brand**: {recommendation['metadata']['brand']}")
                                
                                if "price" in recommendation["metadata"] and recommendation["metadata"]["price"] > 0:
                                    st.markdown(f"**Price**: ${recommendation['metadata']['price']:.2f}")
                                
                                # Display reasons for recommendation
                                if "reasons" in recommendation["metadata"] and recommendation["metadata"]["reasons"]:
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
        else:
            st.warning("No recommendations found in any category.")
        
        # Display selection summary
        st.subheader("Your Selection")
        if st.session_state.selected_items:
            # Check clothing compatibility
            compatibility = self._check_clothing_compatibility(st.session_state.selected_items)
            
            # Show selected items count
            st.success(f"You've selected {len(st.session_state.selected_items)} item(s) for try-on")
            
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
        
        # Add help section for valid combinations
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
            with st.spinner("Processing try-on... This may take a minute..."):
                # Get selected clothing items
                selected_recommendations = [
                    st.session_state.recommendations[i] for i in st.session_state.selected_items
                ]
                
                # Double-check clothing compatibility
                compatibility = self._check_clothing_compatibility(st.session_state.selected_items)
                if not compatibility["valid"]:
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
                
                # Format for VTON API request:
                # {
                #   "user_photo": {                      # User photo to try clothes on
                #     "data": "data:image/jpeg;base64,..." or "https://image-url..."  # Base64 image or URL
                #   },
                #   "clothing_items": [                  # List of 1-5 clothing items to try on
                #     {
                #       "image": {
                #         "data": "https://image-url..." or "base64-data"  # Clothing image URL or base64
                #       },
                #       "type": "t_shirt",               # Type matching ClothingType enum
                #       "context": {                     # Optional metadata used for processing
                #         "brand": "Brand name",
                #         "style": "casual"
                #       }
                #     }
                #   ],
                #   "scene_context": {                   # Optional scene context
                #     "style": "casual",
                #     "season": "summer",
                #     "occasion": "everyday"
                #   }
                # }
                
                # Make API request to VTON endpoint
                response = self._api_call("/vton", request_data)
                
                if response.get("success"):
                    st.session_state.vton_results = response
                    st.session_state.current_step = "results"  # Move to results step
                    st.success("Try-on completed successfully!")
                else:
                    st.error(f"Error: {response.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
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
        outerwear = [t for t in clothing_types if t in outerwear_types]
        bottoms = [t for t in clothing_types if t in bottom_types]
        dresses = [t for t in clothing_types if t in dress_types]
        
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
    
    # Initialize and run the app
    app = VTONFrontend()
    app.run()
