# Automated Traffic Report

Gen AI-powered smart traffic management system using YOLOv8 and Qwen2.5-VL for real-time traffic analysis.

This application leverages the complementary strengths of two specialized AI models to provide comprehensive traffic scene understanding:

**NOTE** The app should work in a localhost. However, the helm chart is WIP due to a pending fix for Gradio compatibility issue.

**YOLOv8 (Object Detection)** excels at:
- **Real-time detection** with minimal latency for immediate response
- **Precise localization** of vehicles, pedestrians, and traffic objects with accurate bounding boxes
- **High throughput** processing suitable for continuous monitoring
- **Consistent performance** across varying lighting and weather conditions
- **Quantitative analysis** providing exact counts and positions of detected objects

**Qwen2.5-VL (Vision-Language Model)** provides:
- **Contextual understanding** of complex traffic scenarios beyond simple object detection
- **Semantic analysis** interpreting traffic flow patterns, congestion levels, and safety conditions
- **Natural language insights** generating human-readable reports and recommendations
- **Scene comprehension** understanding relationships between objects and environmental factors
- **Qualitative assessment** providing traffic safety evaluations and incident analysis

Together, these models create a robust traffic management system where YOLOv8 provides the foundational object detection layer for precise, real-time identification, while Qwen2.5-VL adds the intelligence layer for contextual analysis and actionable insights. This dual-model approach ensures both accuracy in detection and depth in understanding, making it suitable for comprehensive traffic monitoring and management applications.

## Features

- **Object Detection**: YOLOv8 detects vehicles, pedestrians, and traffic violations
- **Scene Analysis**: Qwen2.5-VL provides detailed traffic flow analysis and safety insights
- **User Interface**: Gradio web interface for easy image/video upload and analysis
- **Real-time Processing**: Supports both static images and video analysis

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   - Copy `.env.example` to `.env`
   - Add your YOLOv8 and Qwen2.5-VL API endpoints and keys
   - **Important**: Don't forget to also update the endpoint configurations in the UI Settings tab after starting the app

3. **Run the Application**:
   ```bash
   ./run.sh
   ```
   Or manually:
   ```bash
   python app.py
   ```

## Usage

1. **Configure APIs**: Go to the Settings tab and enter your YOLO and Qwen API endpoints and keys
2. **Image Analysis**: Upload a traffic scene image for instant analysis
3. **Video Analysis**: Upload a traffic video for frame-by-frame analysis
4. **Results**: View detected objects with bounding boxes and AI-generated insights

**Important**: Make sure to configure the API endpoints in the Settings tab of the web interface, even if you've already set them in the `.env` file.

## Architecture

- `detection_service.py`: YOLOv8 integration for object detection
- `analysis_service.py`: Qwen2.5-VL integration for scene analysis
- `app.py`: Gradio web interface
- `config.py`: Configuration and API settings

## API Requirements

- YOLOv8 endpoint with object detection capabilities
- Qwen2.5-VL endpoint with vision-language understanding

## Kubernetes Deployment

This application includes a complete Helm chart for Kubernetes deployment, compatible with EZUA BYOApp platform.

### Helm Chart Features

- **EZUA BYOApp Compatible**: Includes Istio VirtualService for proper application endpoint exposure
- **Complete Kubernetes Resources**: Deployment, Service, Ingress, ConfigMaps, Secrets
- **Auto-scaling**: HorizontalPodAutoscaler support for dynamic scaling
- **Security**: ServiceAccount and proper secret management for API keys
- **Configurable**: Comprehensive values.yaml for customization

### Directory Structure

```
helm-chart/
├── Chart.yaml                 # Chart metadata
├── values.yaml               # Configuration values
├── charts/                   # Dependencies
├── templates/
│   ├── deployment.yaml       # Application deployment
│   ├── service.yaml         # Kubernetes service
│   ├── ingress.yaml         # External access (optional)
│   ├── secret.yaml          # API credentials
│   ├── serviceaccount.yaml  # Service account
│   ├── hpa.yaml             # Auto-scaling
│   ├── tests/               # Test templates
│   └── ezua/
│       └── virtualService.yaml  # EZUA Istio integration
├── traffic-report-0.0.1.tar.gz  # Packaged chart
└── traffic-report.png          # Application logo
```

### Docker Image

The application is containerized and available on Docker Hub:
- **Repository**: `caovd/traffic-report:latest`
- **URL**: https://hub.docker.com/r/caovd/traffic-report

### Quick Deployment

1. **Install the Helm chart**:
   ```bash
   helm install traffic-report ./helm-chart/
   ```

2. **Or use the packaged chart**:
   ```bash
   helm install traffic-report helm-chart/traffic-report-0.0.1.tar.gz
   ```

3. **Configure API endpoints** in `values.yaml`:
   ```yaml
   app:
     yolo:
       endpoint: "https://your-yolo-endpoint.com/predict"
       apiKey: "your-yolo-api-key"
     qwen:
       endpoint: "https://your-qwen-endpoint.com/v1/chat/completions"
       apiKey: "your-qwen-api-key"
   ```

4. **For EZUA BYOApp deployment**, the chart includes:
   - Istio VirtualService at `traffic-report.${DOMAIN_NAME}`
   - Gateway integration via `istio-system/ezaf-gateway`
   - Proper endpoint configuration for platform integration

### Customization

Key configuration options in `values.yaml`:

- **Scaling**: Adjust `replicaCount` and `autoscaling` settings
- **Resources**: Configure CPU/memory limits and requests
- **Networking**: Modify service ports and ingress settings
- **API Configuration**: Set YOLO and Qwen endpoints and credentials
- **EZUA Integration**: Configure virtualService endpoint and gateway

### Health Checks

The deployment includes:
- **Liveness Probe**: Ensures container health
- **Readiness Probe**: Confirms application readiness
- **Health Check Endpoint**: Available at `/` (port 7860)
