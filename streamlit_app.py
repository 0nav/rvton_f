"""
Virtual Try-On API Service (v1.0)
Production-ready FastAPI service with modular architecture and enhanced error handling.
"""

import aiohttp
import base64
import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Import our modular components
from config.settings import settings
from core.exceptions import (
    VTONException, ValidationError, global_exception_handler, log_error, create_error_response
)
from core.image_processing import handle_image_input, get_cache_stats, clear_cache
from core.masking import ClothingType, masking_system
from core.comfyui import comfyui_client
from core.storage import storage

# Import recommendation engine components
from recommendation.analyzer import UserAnalyzer
from recommendation.engine import RecommendationEngine, RecommendationRequest

logger = logging.getLogger(__name__)

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting Virtual Try-On API Service v1.0")
    yield
    logger.info("Shutting down Virtual Try-On API Service")
    await comfyui_client.close()

# Initialize recommendation components
user_analyzer = UserAnalyzer()
recommendation_engine = RecommendationEngine()

# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Add global exception handler
app.add_exception_handler(Exception, global_exception_handler)

# Pydantic models
class ImageInput(BaseModel):
    """Image input model supporting URLs and base64 data."""
    data: str = Field(..., description="Image URL or base64 string", min_length=10)
    
    @field_validator('data')
    @classmethod
    def validate_image_input(cls, v):
        """Validate image input format."""
        if not v or not v.strip():
            raise ValueError("Image data cannot be empty")
        
        v = v.strip()
        
        # Check if it's a URL
        if v.startswith(("http://", "https://")):
            from urllib.parse import urlparse
            try:
                parsed = urlparse(v)
                if not all([parsed.scheme, parsed.netloc]):
                    raise ValueError("Invalid URL format")
                return v
            except Exception:
                raise ValueError("Invalid URL format")
        
        # Check if it's a data URI
        if v.startswith("data:image/"):
            try:
                header, data = v.split(",", 1)
                base64.b64decode(data, validate=True)
                return v
            except Exception:
                raise ValueError("Invalid base64 image format")
        
        # Check if it's raw base64
        try:
            base64.b64decode(v, validate=True)
            return v
        except Exception:
            raise ValueError("Invalid image input: must be URL or base64")

class ClothingItem(BaseModel):
    """Individual clothing item model."""
    image: ImageInput
    type: ClothingType
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VTONRequest(BaseModel):
    """Virtual try-on request model."""
    user_photo: ImageInput
    clothing_items: List[ClothingItem] = Field(..., min_length=1, max_length=5)
    scene_context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VTONResponse(BaseModel):
    """Virtual try-on response model."""
    success: bool
    final_image_url: Optional[str] = Field(None, description="Public URL of the generated image (preferred)")
    final_image: Optional[str] = Field(None, description="Base64 encoded image (fallback if URL unavailable)")
    processing_time: float
    clothing_analysis: Dict[str, Any]
    masking_applied: List[Dict[str, Any]]
    debug_info: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    comfyui_connected: bool
    timestamp: float
    cache_stats: Dict[str, Any]
    system_info: Dict[str, Any]

class ClothingTypesResponse(BaseModel):
    """Clothing types information response model."""
    total_supported_types: int
    clothing_types: List[str]
    categories: Dict[str, List[str]]
    example_requests: Dict[str, Any]

# New models for recommendation integration
class UserPreferences(BaseModel):
    """User style preferences model."""
    style: str = Field(..., description="Style preference: casual, formal, business, sporty")
    gender: str = Field(..., description="Gender: male, female, unisex")
    season: str = Field(..., description="Season: spring, summer, fall, winter")
    occasion: Optional[str] = Field(None, description="Occasion: everyday, work, party, date")
    price_max: Optional[float] = Field(None, description="Maximum price preference")

class RecommendationVTONRequest(BaseModel):
    """Complete recommendation + VTON request model."""
    user_image: str = Field(..., description="Base64 encoded user image")
    preferences: UserPreferences
    max_recommendations: int = Field(default=3, ge=1, le=5, description="Number of recommendations")
    include_vton: bool = Field(default=True, description="Include virtual try-on processing")

class RecommendationVTONResponse(BaseModel):
    """Complete recommendation + VTON response model."""
    success: bool
    recommendations: List[Dict[str, Any]]
    user_analysis: Dict[str, Any]
    vton_result: Optional[Dict[str, Any]] = None
    processing_time: float
    correlation_id: str

# API Routes

@app.post("/vton", response_model=VTONResponse)
async def virtual_try_on(request: VTONRequest, http_request: Request):
    """
    Virtual Try-On endpoint with intelligent layering and masking.
    
    Processes a user photo with 1-5 clothing items to create a realistic try-on result.
    Features intelligent layering, context-aware masking, and optimized processing.
    """
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    logger.info(f"[{correlation_id}] Starting VTON request with {len(request.clothing_items)} items")
    
    try:
        # Validate clothing combination
        clothing_types = [item.type for item in request.clothing_items]
        validation_result = masking_system.validate_clothing_combination(clothing_types)
        
        if not validation_result["valid"]:
            raise ValidationError(
                "Invalid clothing combination",
                details={
                    "conflicts": validation_result["conflicts"],
                    "clothing_items": [t.value for t in clothing_types]
                },
                correlation_id=correlation_id
            )
        
        # Analyze clothing items for intelligent processing
        clothing_analysis = masking_system.analyze_clothing_items(
            clothing_types, 
            request.scene_context
        )
        
        logger.info(f"[{correlation_id}] Clothing analysis: {len(clothing_analysis['processing_order'])} items, order: {clothing_analysis['processing_order']}")
        
        # Get aiohttp session from ComfyUI client
        session = await comfyui_client.get_session()
        
        # Process and upload user image
        user_image_bytes = await handle_image_input(session, request.user_photo.data, correlation_id)
        current_user_filename = await comfyui_client.upload_image(
            user_image_bytes, 
            f"user_{correlation_id}.jpg", 
            correlation_id
        )
        
        # Process clothing items in optimal order
        masking_applied = []
        
        # Group clothing items by workflow nodes
        top_items = []  # Node 84: tops, dresses, outerwear
        bottom_items = []  # Node 83: bottoms
        
        for item in request.clothing_items:
            node_id = masking_system.get_workflow_node(item.type)
            if node_id == 84:
                top_items.append(item)
            elif node_id == 83:
                bottom_items.append(item)
            else:
                raise ValidationError(
                    f"Invalid workflow node {node_id} for clothing type {item.type.value}",
                    details={"clothing_type": item.type.value, "node_id": node_id}
                )
        
        # Validate that we don't have conflicting items in the same node
        if len(top_items) > 1:
            top_types = [item.type.value for item in top_items]
            logger.warning(f"[{correlation_id}] Multiple top items provided: {top_types}. Using last item.")
            top_items = [top_items[-1]]  # Use the last item if multiple
            
        if len(bottom_items) > 1:
            bottom_types = [item.type.value for item in bottom_items]
            logger.warning(f"[{correlation_id}] Multiple bottom items provided: {bottom_types}. Using last item.")
            bottom_items = [bottom_items[-1]]  # Use the last item if multiple
        
        logger.info(f"[{correlation_id}] Grouped items: {len(top_items)} top, {len(bottom_items)} bottom")
        
        # Process top clothing item (node 84)
        top_filename = None
        top_mask_config = {}
        if top_items:
            top_item = top_items[0]
            logger.info(f"[{correlation_id}] Processing top item: {top_item.type.value}")
            top_image_bytes = await handle_image_input(session, top_item.image.data, correlation_id)
            top_filename = await comfyui_client.upload_image(
                top_image_bytes, 
                f"top_{correlation_id}.jpg", 
                correlation_id
            )
            # Generate mask configuration for top item
            combined_context = {**(request.scene_context or {}), **(top_item.context or {})}
            top_mask_config = masking_system.generate_mask_config(
                top_item.type, 
                combined_context, 
                [top_item.type]
            )
            # Record masking information for top item
            masking_applied.append({
                "clothing_type": top_item.type.value,
                "mask_config": {k: v for k, v in top_mask_config.items() if v},
                "masks_enabled": sum(top_mask_config.values())
            })
        
        # Process bottom clothing item (node 83)
        bottom_filename = None
        bottom_mask_config = {}
        if bottom_items:
            bottom_item = bottom_items[0]
            logger.info(f"[{correlation_id}] Processing bottom item: {bottom_item.type.value}")
            bottom_image_bytes = await handle_image_input(session, bottom_item.image.data, correlation_id)
            bottom_filename = await comfyui_client.upload_image(
                bottom_image_bytes, 
                f"bottom_{correlation_id}.jpg", 
                correlation_id
            )
            # Generate mask configuration for bottom item
            combined_context = {**(request.scene_context or {}), **(bottom_item.context or {})}
            bottom_mask_config = masking_system.generate_mask_config(
                bottom_item.type, 
                combined_context, 
                [bottom_item.type]
            )
            # Record masking information for bottom item
            masking_applied.append({
                "clothing_type": bottom_item.type.value,
                "mask_config": {k: v for k, v in bottom_mask_config.items() if v},
                "masks_enabled": sum(bottom_mask_config.values())
            })
        
        # Combine mask configurations
        combined_mask_config = {**top_mask_config, **bottom_mask_config}
        
        # Create and execute workflow with all items
        logger.info(f"[{correlation_id}] Creating workflow with combined items")
        workflow = comfyui_client.update_workflow(
            current_user_filename, 
            top_filename, 
            bottom_filename, 
            combined_mask_config
        )
        
        # Submit and process workflow
        prompt_id = await comfyui_client.submit_workflow(workflow, correlation_id)
        outputs = await comfyui_client.poll_workflow(prompt_id, correlation_id)
        output_image_bytes = await comfyui_client.extract_image(outputs, correlation_id)
        
        # Try to upload to Supabase first, fallback to base64 if failed
        final_image_url = None
        final_image_base64 = None
        
        if storage.is_available():
            try:
                final_image_url = await storage.upload_image(output_image_bytes, correlation_id, "result")
                logger.info(f"[{correlation_id}] Image uploaded to Supabase: {final_image_url}")
            except Exception as e:
                logger.error(f"[{correlation_id}] Failed to upload to Supabase: {e}")
        
        # If Supabase upload failed or is not available, use base64 as fallback
        if not final_image_url:
            final_image_base64 = base64.b64encode(output_image_bytes).decode("utf-8")
            logger.info(f"[{correlation_id}] Using base64 fallback for image response")
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"[{correlation_id}] VTON completed successfully in {processing_time:.1f}s. "
            f"Final image: {len(output_image_bytes)} bytes"
        )
        
        return VTONResponse(
            success=True,
            final_image_url=final_image_url,
            final_image=final_image_base64,
            processing_time=processing_time,
            clothing_analysis=clothing_analysis,
            masking_applied=masking_applied,
            debug_info={
                "correlation_id": correlation_id,
                "items_processed": len(request.clothing_items),
                "final_image_size": len(output_image_bytes),
                "cache_hits": get_cache_stats().get("entries", 0)
            }
        )
        
    except VTONException as e:
        log_error(e, http_request, {"correlation_id": correlation_id})
        raise HTTPException(
            status_code=400 if isinstance(e, ValidationError) else 500,
            detail=create_error_response(e, http_request)["error"]
        )
    except Exception as e:
        vton_exc = VTONException(
            f"Unexpected error during VTON processing: {str(e)}",
            error_code="VTON_PROCESSING_ERROR",
            correlation_id=correlation_id,
            details={"original_error": str(e)}
        )
        log_error(vton_exc, http_request, {"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=create_error_response(vton_exc, http_request)["error"])

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns service status, ComfyUI connectivity, cache statistics, and system information.
    """
    try:
        # Test ComfyUI connectivity
        session = await comfyui_client.get_session()
        timeout = aiohttp.ClientTimeout(total=5)
        
        try:
            async with session.get(f"{settings.comfyui_base_url}/", timeout=timeout) as response:
                comfyui_status = response.status == 200
        except Exception as e:
            logger.warning(f"ComfyUI health check failed: {str(e)}")
            comfyui_status = False
        
        # Get cache statistics
        cache_stats = get_cache_stats()
        
        # Determine overall status
        overall_status = "healthy" if comfyui_status else "degraded"
        
        return HealthResponse(
            status=overall_status,
            comfyui_connected=comfyui_status,
            timestamp=time.time(),
            cache_stats=cache_stats,
            system_info={
                "total_clothing_types": len(masking_system.get_supported_types()),
                "max_clothing_items": 5,
                "caching_enabled": settings.cache_enabled,
                "storage_available": storage.is_available()
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Health check failed", "details": str(e)}
        )

@app.get("/clothing-types", response_model=ClothingTypesResponse)
async def get_clothing_types():
    """Get supported clothing types and categories."""
    try:
        supported_types = masking_system.get_supported_types()
        categories = masking_system.get_categories()
        
        # Example requests
        example_requests = {
            "single_item": {
                "description": "Try on a single clothing item",
                "request": {
                    "user_photo": {"data": "https://example.com/person.jpg"},
                    "clothing_items": [
                        {"image": {"data": "https://example.com/shirt.jpg"}, "type": "shirt"}
                    ]
                }
            },
            "jacket_over_shirt": {
                "description": "Intelligent layering - jacket over shirt",
                "request": {
                    "user_photo": {"data": "https://example.com/person.jpg"},
                    "clothing_items": [
                        {"image": {"data": "https://example.com/shirt.jpg"}, "type": "shirt"},
                        {"image": {"data": "https://example.com/jacket.jpg"}, "type": "jacket"}
                    ],
                    "scene_context": {"season": "winter", "style": "casual"}
                }
            },
            "summer_outfit": {
                "description": "Complete summer outfit with context",
                "request": {
                    "user_photo": {"data": "https://example.com/person.jpg"},
                    "clothing_items": [
                        {"image": {"data": "https://example.com/tank_top.jpg"}, "type": "tank_top"},
                        {"image": {"data": "https://example.com/shorts.jpg"}, "type": "shorts"},
                        {"image": {"data": "https://example.com/sneakers.jpg"}, "type": "sneakers"}
                    ],
                    "scene_context": {"season": "summer", "style": "casual", "occasion": "everyday"}
                }
            },
            "formal_outfit": {
                "description": "Formal business outfit",
                "request": {
                    "user_photo": {"data": "https://example.com/person.jpg"},
                    "clothing_items": [
                        {"image": {"data": "https://example.com/dress_shirt.jpg"}, "type": "shirt"},
                        {"image": {"data": "https://example.com/blazer.jpg"}, "type": "blazer"},
                        {"image": {"data": "https://example.com/dress_pants.jpg"}, "type": "pants"}
                    ],
                    "scene_context": {"style": "formal", "occasion": "work"}
                }
            }
        }
        
        return ClothingTypesResponse(
            total_supported_types=len(supported_types),
            clothing_types=supported_types,
            categories=categories,
            example_requests=example_requests
        )
        
    except Exception as e:
        logger.error(f"Error getting clothing types: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get clothing types", "details": str(e)}
        )


@app.post("/cache/clear")
async def clear_image_cache():
    """
    Clear the image cache.
    
    Useful for debugging or freeing memory. Returns cache statistics before and after clearing.
    """
    try:
        stats_before = get_cache_stats()
        clear_cache()
        stats_after = get_cache_stats()
        
        return {
            "success": True,
            "message": "Image cache cleared successfully",
            "stats_before": stats_before,
            "stats_after": stats_after
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to clear cache", "details": str(e)}
        )

@app.get("/cache/stats")
async def get_image_cache_stats():
    """Get detailed image cache statistics."""
    try:
        return {
            "success": True,
            "cache_stats": get_cache_stats(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get cache stats", "details": str(e)}
        )

@app.post("/recommend-and-tryon", response_model=RecommendationVTONResponse)
async def recommend_and_tryon(request: RecommendationVTONRequest, http_request: Request):
    """
    Complete pipeline: User analysis → Recommendations → VTON processing.
    
    This endpoint combines intelligent clothing recommendations with virtual try-on:
    1. Analyzes user uploaded image for style preferences
    2. Generates personalized clothing recommendations
    3. Processes VTON with recommended items
    """
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    logger.info(f"[{correlation_id}] Starting recommendation + VTON pipeline")
    
    try:
        # Step 1: Analyze user image
        logger.info(f"[{correlation_id}] Step 1: Analyzing user image")
        user_image_bytes = base64.b64decode(request.user_image)
        user_analysis = user_analyzer.analyze_image(user_image_bytes)
        
        logger.info(f"[{correlation_id}] User analysis completed: {len(user_analysis['dominant_colors'])} colors detected")
        
        # Step 2: Generate recommendations
        logger.info(f"[{correlation_id}] Step 2: Generating recommendations")
        recommendation_request = RecommendationRequest(
            user_analysis=user_analysis,
            occasion=request.preferences.occasion or request.preferences.style,
            max_items=request.max_recommendations,
            price_range=(0, request.preferences.price_max) if request.preferences.price_max else None
        )
        
        recommendations = recommendation_engine.get_recommendations(recommendation_request)
        
        if not recommendations:
            raise ValidationError(
                "No suitable recommendations found",
                details={
                    "user_preferences": request.preferences.dict(),
                    "analysis_summary": {
                        "colors": len(user_analysis.get("dominant_colors", [])),
                        "style": user_analysis.get("style_indicators", {}),
                        "season": user_analysis.get("season_compatibility", "unknown")
                    }
                },
                correlation_id=correlation_id
            )
        
        logger.info(f"[{correlation_id}] Generated {len(recommendations)} recommendations")
        
        # Convert recommendations to VTON-compatible format
        recommendations_data = []
        clothing_items_for_vton = []
        
        for rec in recommendations:
            rec_data = {
                "image": rec.item["image"],
                "type": rec.item["type"],
                "confidence": rec.compatibility_score,
                "style_compatibility": rec.style_match_score,
                "color_harmony": rec.color_match_score,
                "metadata": {
                    "brand": rec.item.get("brand", "Unknown"),
                    "price": rec.item.get("price", 0),
                    "product_link": rec.item.get("product_link", ""),
                    "color": rec.item.get("color", "#808080"),
                    "reasons": rec.reasons
                }
            }
            recommendations_data.append(rec_data)
            
            # Prepare for VTON if enabled
            if request.include_vton:
                clothing_items_for_vton.append(ClothingItem(
                    image=ImageInput(data=rec.item["image"]),
                    type=ClothingType(rec.item["type"]),
                    context={
                        "brand": rec.item.get("brand", ""),
                        "style": request.preferences.style
                    }
                ))
        
        # Step 3: Process VTON if requested
        vton_result = None
        if request.include_vton and clothing_items_for_vton:
            logger.info(f"[{correlation_id}] Step 3: Processing VTON with {len(clothing_items_for_vton)} items")
            
            # Create scene context from preferences
            scene_context = {
                "season": request.preferences.season.lower(),
                "style": request.preferences.style.lower(),
                "occasion": request.preferences.occasion or "everyday"
            }
            
            # Create VTON request
            vton_request = VTONRequest(
                user_photo=ImageInput(data=f"data:image/jpeg;base64,{request.user_image}"),
                clothing_items=clothing_items_for_vton,
                scene_context=scene_context
            )
            
            # Process VTON (call the existing function directly)
            try:
                vton_response = await virtual_try_on(vton_request, http_request)
                vton_result = vton_response.dict()
                logger.info(f"[{correlation_id}] VTON processing successful")
            except Exception as vton_error:
                logger.error(f"[{correlation_id}] VTON processing failed: {vton_error}")
                # Continue without VTON result - recommendations are still valuable
                vton_result = {
                    "success": False,
                    "error": str(vton_error),
                    "message": "Virtual try-on failed, but recommendations are available"
                }
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"[{correlation_id}] Complete pipeline finished in {processing_time:.1f}s. "
            f"Recommendations: {len(recommendations_data)}, VTON: {'success' if vton_result and vton_result.get('success') else 'skipped/failed'}"
        )
        
        return RecommendationVTONResponse(
            success=True,
            recommendations=recommendations_data,
            user_analysis=user_analysis,
            vton_result=vton_result,
            processing_time=processing_time,
            correlation_id=correlation_id
        )
        
    except VTONException as e:
        log_error(e, http_request, {"correlation_id": correlation_id})
        raise HTTPException(
            status_code=400 if isinstance(e, ValidationError) else 500,
            detail=create_error_response(e, http_request)["error"]
        )
    except Exception as e:
        vton_exc = VTONException(
            f"Unexpected error during recommendation + VTON pipeline: {str(e)}",
            error_code="RECOMMENDATION_PIPELINE_ERROR",
            correlation_id=correlation_id,
            details={"original_error": str(e)}
        )
        log_error(vton_exc, http_request, {"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=create_error_response(vton_exc, http_request)["error"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port, 
        log_level=settings.log_level.lower(),
        access_log=True
    )
