import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from detection_service import DetectionService
from analysis_service import AnalysisService

# Initialize services - will be done dynamically using session state
# detector = None
# analyzer = None

# Global settings storage
settings = {
    "yolo_endpoint": os.getenv("YOLO_ENDPOINT", "https://your-yolo-endpoint.com/predict"),
    "yolo_api_key": os.getenv("YOLO_API_KEY", "your-yolo-api-key-here"),
    "qwen_endpoint": os.getenv("QWEN_ENDPOINT", "https://your-qwen-endpoint.com/v1/chat/completions"),
    "qwen_api_key": os.getenv("QWEN_API_KEY", "your-qwen-api-key-here")
}

def update_settings(yolo_endpoint, yolo_api_key, qwen_endpoint, qwen_api_key):
    """Update API settings and reinitialize services"""
    # Update settings
    settings["yolo_endpoint"] = yolo_endpoint
    settings["yolo_api_key"] = yolo_api_key
    settings["qwen_endpoint"] = qwen_endpoint
    settings["qwen_api_key"] = qwen_api_key
    
    # Update environment variables
    os.environ["YOLO_ENDPOINT"] = yolo_endpoint.replace("/predict", "") if yolo_endpoint else ""
    os.environ["YOLO_API_KEY"] = yolo_api_key if yolo_api_key else ""
    os.environ["QWEN_ENDPOINT"] = qwen_endpoint if qwen_endpoint else ""
    os.environ["QWEN_API_KEY"] = qwen_api_key if qwen_api_key else ""
    
    # Force reinitialize services
    if 'detector' in st.session_state:
        del st.session_state.detector
    if 'analyzer' in st.session_state:
        del st.session_state.analyzer
    
    return "Settings updated successfully! You can now use the analysis features."

def get_detector():
    """Get detector instance with current environment variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = DetectionService()
    return st.session_state.detector

def get_analyzer():
    """Get analyzer instance with current environment variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AnalysisService()
    return st.session_state.analyzer

def process_image(image):
    """Process single image for traffic analysis"""
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Get detections
        detections = get_detector().detect_objects(pil_image)
        
        # Draw detections on image
        annotated_image = get_detector().draw_detections(pil_image, detections)
        
        # Get analysis only if we have detections or if API keys are configured
        if detections or (os.getenv("QWEN_ENDPOINT") and os.getenv("QWEN_API_KEY")):
            analysis = get_analyzer().analyze_traffic_scene(pil_image, detections)
        else:
            analysis = {"content": "No analysis available - configure API keys to enable AI analysis"}
        
        # Format results
        results_text = format_analysis_results(detections, analysis)
        
        return annotated_image, results_text
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def process_video(video_path):
    """Process video for traffic analysis (sample frames)"""
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames (every 30 frames or 1 second)
        sample_interval = max(1, int(fps))
        results = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                
                # Process frame
                detections = get_detector().detect_objects(pil_frame)
                analysis = get_analyzer().analyze_traffic_scene(pil_frame, detections)
                
                # Store result
                results.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "detections": detections,
                    "analysis": analysis
                })
                
                # Limit to 5 samples for demo
                if len(results) >= 5:
                    break
                    
            frame_count += 1
        
        cap.release()
        
        # Get a representative frame for display
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)  # Middle frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            detections = get_detector().detect_objects(pil_frame)
            annotated_frame = get_detector().draw_detections(pil_frame, detections)
        else:
            annotated_frame = None
        
        # Format video results
        video_results = format_video_results(results)
        
        return annotated_frame, video_results
        
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

def format_analysis_results(detections, analysis):
    """Format analysis results for display"""
    result_text = "## Traffic Analysis Results\n\n"
    
    # Detection summary - count objects by type
    result_text += "### Number of detected vehicles:\n"
    if detections:
        object_counts = {}
        for detection in detections:
            obj_type = detection['class'].title()
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        for obj_type, count in object_counts.items():
            result_text += f"- {obj_type}: {count}\n"
    else:
        result_text += "- No traffic objects detected\n"
        result_text += "- *Note: Configure YOLO API keys in Settings to enable real-time detection*\n"
    
    # Analysis results
    result_text += "\n### AI-powered traffic report:\n"
    analysis_content = analysis.get('content', 'Analysis not available')
    result_text += analysis_content
    
    return result_text

def format_video_results(results):
    """Format video analysis results"""
    if not results:
        return "No analysis results available"
    
    result_text = "## Video Traffic Analysis Results\n\n"
    
    for i, result in enumerate(results, 1):
        result_text += f"### Frame {i} (t={result['timestamp']:.1f}s)\n"
        
        # Detection count
        detections = result['detections']
        if detections:
            detection_counts = {}
            for detection in detections:
                cls = detection['class']
                detection_counts[cls] = detection_counts.get(cls, 0) + 1
            
            for cls, count in detection_counts.items():
                result_text += f"- {cls.title()}: {count}\n"
        else:
            result_text += "- No objects detected\n"
        
        # Analysis
        analysis = result['analysis']
        result_text += f"- **Flow:** {analysis.get('traffic_flow', 'Unknown')}\n"
        result_text += f"- **Safety:** {analysis.get('safety_concerns', 'OK')}\n\n"
    
    return result_text

# Create Streamlit interface
def create_interface():
    # Initialize session state if needed
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    st.set_page_config(
        page_title="Smart Traffic Report",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚦 Smart Traffic Report")
    st.markdown("Upload an image or video to analyze traffic conditions using YOLO detection and Qwen2.5-VL analysis.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Video Analysis", "Settings"])
    
    with tab1:
        st.header("Image Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Traffic Image")
            image_input = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png'],
                key="image_upload"
            )
            
            if image_input and st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Convert uploaded file to PIL Image
                    pil_image = Image.open(image_input)
                    
                    # Process the image
                    annotated_image, analysis_results = process_image(pil_image)
                    
                    # Store results in session state
                    st.session_state.image_results = (annotated_image, analysis_results)
        
        with col2:
            st.subheader("Results")
            
            # Display results if available
            if hasattr(st.session_state, 'image_results') and st.session_state.image_results:
                annotated_image, analysis_results = st.session_state.image_results
                
                if annotated_image:
                    st.image(annotated_image, caption="Detection Results", use_column_width=True)
                
                if analysis_results:
                    st.markdown("### Analysis Results")
                    st.markdown(analysis_results)
    
    with tab2:
        st.header("Video Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Traffic Video")
            video_input = st.file_uploader(
                "Choose a video...",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="video_upload"
            )
            
            if video_input and st.button("Analyze Video", type="primary"):
                with st.spinner("Analyzing video..."):
                    # Save uploaded video to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_input.read())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process the video
                        annotated_frame, video_results = process_video(tmp_path)
                        
                        # Store results in session state
                        st.session_state.video_results = (annotated_frame, video_results)
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
        
        with col2:
            st.subheader("Results")
            
            # Display results if available
            if hasattr(st.session_state, 'video_results') and st.session_state.video_results:
                annotated_frame, video_results = st.session_state.video_results
                
                if annotated_frame:
                    st.image(annotated_frame, caption="Sample Frame with Detections", use_column_width=True)
                
                if video_results:
                    st.markdown("### Video Analysis Results")
                    st.markdown(video_results)
    
    with tab3:
        st.header("API Configuration")
        st.markdown("Configure your YOLO and Qwen2.5-VL model endpoints and API keys.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("YOLO Detection Settings")
            yolo_endpoint_input = st.text_input(
                "YOLO Endpoint",
                value=settings["yolo_endpoint"],
                placeholder="https://your-yolo-endpoint.com/predict",
                help="Full endpoint URL including /predict"
            )
            yolo_key_input = st.text_input(
                "YOLO API Key",
                value=settings["yolo_api_key"],
                type="password",
                placeholder="your-yolo-api-key-here",
                help="Your YOLO API authentication key"
            )
        
        with col2:
            st.subheader("Qwen2.5-VL Analysis Settings")
            qwen_endpoint_input = st.text_input(
                "Qwen Endpoint",
                value=settings["qwen_endpoint"],
                placeholder="https://your-qwen-endpoint.com/v1/chat/completions",
                help="Full endpoint URL including /v1/chat/completions"
            )
            qwen_key_input = st.text_input(
                "Qwen API Key",
                value=settings["qwen_api_key"],
                type="password",
                placeholder="your-qwen-api-key-here",
                help="Your Qwen API authentication key"
            )
        
        if st.button("Save Settings", type="primary"):
            result = update_settings(yolo_endpoint_input, yolo_key_input, qwen_endpoint_input, qwen_key_input)
            st.success(result)
        
        st.markdown("### Instructions:")
        st.markdown("1. **YOLO Endpoint**: Enter your YOLOv8 detection service URL ending with `/predict`")
        st.markdown("2. **YOLO API Key**: Enter your authentication key for the YOLO service")
        st.markdown("3. **Qwen Endpoint**: Enter your Qwen2.5-VL service URL ending with `/v1/chat/completions`")
        st.markdown("4. **Qwen API Key**: Enter your authentication key for the Qwen service")
        st.markdown("5. **Save Settings**: Click to apply the new configuration")
        
        st.info("Settings are applied immediately and will be used for all subsequent analyses. The demo includes fallback detection data when APIs are unavailable.")
    
    # Footer
    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown("1. **Configure APIs**: Go to Settings tab to set up your model endpoints")
    st.markdown("2. **For Images**: Upload a traffic scene image to get real-time analysis")
    st.markdown("3. **For Videos**: Upload a traffic video to analyze multiple frames")
    st.markdown("4. **Results**: View detected objects and AI-generated traffic insights")
    
    st.markdown("*Powered by YOLOv8 for object detection and Qwen2.5-VL for scene analysis*")

# Check if API keys are set (only show warning in console for debugging)
if not os.getenv("YOLO_API_KEY") or not os.getenv("QWEN_API_KEY"):
    print("Warning: API keys not found. Please set environment variables:")
    print("- YOLO_ENDPOINT and YOLO_API_KEY")
    print("- QWEN_ENDPOINT and QWEN_API_KEY")

# Launch the Streamlit app
create_interface()