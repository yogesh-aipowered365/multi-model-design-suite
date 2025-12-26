# Multimodal AI Design Analysis Suite

**AI-powered Streamlit app that analyzes uploaded marketing creatives or product/app screens using multiple intelligent agents, retrieval-augmented generation (RAG), and vision models.**

Supports single-design analysis and multi-design comparison with rich visualizations, actionable recommendations, and downloadable reports.

## ✨ Features

- **Multi-agent analysis**: Visual, UX, Market, Conversion/CTA, and Brand agents (toggleable via sidebar)
- **BYOK (Bring Your Own Key)**: User-provided OpenRouter API key—no hardcoded credentials
- **Vision + RAG**: OpenRouter vision models with FAISS design-pattern retrieval; adjustable `top_k` slider for context depth
- **Dual modes**: Single design analysis or compare 2–5 designs side-by-side with ranking and scores
- **Smart controls**: Creative type selection (Marketing vs Product UI), agent toggles, and RAG depth adjustment
- **Rich visuals**: Performance gauges, radar charts, priority matrices, timelines, impact projections, annotated designs, before/after mockups
- **Robust**: Agent retries with exponential backoff, per-agent error surfacing, downloadable JSON reports
- **Branding**: Professional logo with link to aipowered365.com

## 🚀 Quickstart

### Prerequisites
- Python 3.9+
- Free OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/mm.git
cd mm

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**BYOK Workflow**:
1. Launch the app
2. In the sidebar, paste your OpenRouter API key in the **"Enter your OpenRouter API Key"** field
3. Upload a design image
4. Click **"Analyze Design"** or **"Compare Designs"** to start analysis

### Docker

```bash
# Build and start containers
docker-compose up --build

# Or use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Stop containers**:
```bash
docker-compose down
```

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

Configure these variables:
- `OPENROUTER_API_KEY` – Your OpenRouter API key (can also be provided via UI)
- `VISION_MODEL` – Vision model to use (default: `openai/gpt-4o`)
- `OPENROUTER_BASE_URL` – OpenRouter base URL (default: `https://openrouter.ai/api/v1`)

**Note**: The `OPENROUTER_API_KEY` in `.env` is optional. If not set, users must provide it via the sidebar input field in the app (BYOK model). If both are provided, the UI input takes precedence.

### BYOK (Bring Your Own Key)

The app uses a **BYOK** security model:
- **No hardcoded credentials**: API keys are never stored in the code
- **User-provided keys**: Each user enters their OpenRouter API key in the sidebar
- **Per-session**: Key is only used for the current session; not persisted
- **Validation**: Key is validated before any API calls

This ensures your API key remains private and secure.

## 📋 How to Use

1. **Launch the App**
   ```bash
   streamlit run app.py
   ```

2. **Provide Your API Key**
   - In the left sidebar, enter your OpenRouter API key in the password field
   - Get a free key from [openrouter.ai/keys](https://openrouter.ai/keys)

3. **Choose Analysis Mode**
   - **Single Design**: Analyze a single uploaded image
   - **Compare Designs**: Compare 2–5 uploaded images side-by-side

4. **Configure Analysis**
   - Select **Platform** (Mobile, Desktop, Web, etc.)
   - Select **Creative Type** (Marketing Creative or Product UI)
   - Choose which agents to run (Visual, UX, Market, Conversion, Brand)
   - Adjust **RAG top_k** to control context depth (higher = more design patterns, slower)

5. **Run Analysis**
   - Click **"Analyze Design"** or **"Compare Designs"**
   - View results across 5 tabs:
     - **Overview**: Key metrics and performance scores
     - **Recommendations**: Detailed suggestions per agent
     - **Impact Analysis**: Priority matrix and projected improvements
     - **Visual Feedback**: Annotated design and improvement mockups
     - **Detailed Data**: Complete JSON report for export

## 🏗️ Architecture

```
User Input (Image + API Key)
    ↓
Image Processing (RGB conversion, CLIP embeddings)
    ↓
RAG Retrieval (FAISS pattern search)
    ↓
Multi-Agent Analysis (5 agents in parallel)
    ├─ Visual Agent (colors, layout, typography)
    ├─ UX Agent (usability, accessibility)
    ├─ Market Agent (trends, audience fit)
    ├─ Conversion Agent (CTA effectiveness)
    └─ Brand Agent (consistency, identity)
    ↓
Result Aggregation (score blending, recommendations)
    ↓
Streamlit UI (visualizations, charts, exports)
```

**Key Components**:
- `app.py` – Main Streamlit interface with BYOK sidebar, image upload, agent controls
- `components/agents.py` – Visual, UX, Market, Conversion, Brand agents with OpenRouter vision API
- `components/orchestration.py` – LangGraph workflow orchestration and state management
- `components/rag_system.py` – FAISS index and design pattern retrieval
- `components/image_processing.py` – Image preprocessing and CLIP embeddings
- `components/enhanced_output.py` – Plotly visualizations
- `components/visual_feedback.py` – Design annotations and mockup generation
- `components/design_comparison.py` – Multi-design comparison logic
- `data/design_patterns.json` – 30+ design pattern database

## 🔒 Security & Privacy

- **API Keys**: Your OpenRouter API key is entered via the Streamlit UI and is only stored in memory for the current session
- **No Persistence**: Keys are never saved to disk or logged
- **BYOK Model**: You control your own credentials—no credentials hardcoded in the application
- **.env Protection**: The `.env` file is automatically excluded from git (see `.gitignore`)
- **Image Processing**: Uploaded images are processed locally and sent only to OpenRouter for vision analysis

## 🛠️ Troubleshooting

### "Please enter your OpenRouter API key to continue"
- Ensure you've entered a valid OpenRouter API key in the sidebar
- Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)

### "Error: OpenRouter API call failed"
- Check your API key is valid and has sufficient quota
- Ensure you have an active internet connection
- Verify the `VISION_MODEL` setting in `.env` (default: `openai/gpt-4o`)

### "Image processing failed"
- Ensure the uploaded image is in a standard format (PNG, JPG, WebP)
- Image is automatically resized to max 1024px for efficiency
- Check that PIL/Pillow is properly installed: `pip install pillow`

### App runs slowly
- Reduce the **RAG top_k** slider to retrieve fewer design patterns
- Use a faster vision model (check OpenRouter pricing/latency)
- Enable Docker for consistent environment

### FAISS Index Error
- The index is built from `data/design_patterns.json` on first run
- If corrupted, delete `data/.faiss_cache/` and restart the app
- Ensure FAISS is installed: `pip install faiss-cpu`

## 📦 Project Structure

```
mm/
├── app.py                          # Main Streamlit application
├── components/
│   ├── agents.py                  # Agent implementations
│   ├── orchestration.py            # LangGraph workflow
│   ├── rag_system.py              # FAISS retrieval
│   ├── image_processing.py        # Image preprocessing
│   ├── enhanced_output.py         # Plotly visualizations
│   ├── visual_feedback.py         # Design annotations
│   └── design_comparison.py       # Multi-design comparison
├── data/
│   └── design_patterns.json       # Design pattern database
├── assets/
│   └── logo.png                   # Application logo
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── Makefile                       # Build automation
└── README.md                      # This file
```

## 📚 Module Reference

- **app.py** – Main Streamlit interface with BYOK sidebar, image upload, agent controls
- **components/agents.py** – Visual, UX, Market, Conversion, Brand agents with OpenRouter vision API
- **components/orchestration.py** – LangGraph workflow orchestration and state management
- **components/rag_system.py** – FAISS index and design pattern retrieval
- **components/image_processing.py** – Image preprocessing and CLIP embeddings
- **components/enhanced_output.py** – Plotly charts and visualization helpers
- **components/visual_feedback.py** – Design annotations and mockup generation
- **components/design_comparison.py** – Multi-design comparison and ranking
- **data/design_patterns.json** – 30+ design pattern reference database

## 📄 License

MIT License – see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Website**: [aipowered365.com](https://www.aipowered365.com)
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions on GitHub

---

Made with ❤️ for designers and product teams
