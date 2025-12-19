# HuggingFace Space Configuration

This directory contains configuration for uploading to HuggingFace Spaces.

## Setup Instructions

### 1. Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Create account
3. Generate access token: Settings → Access Tokens → New Token (write)

### 2. Create Private Space

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create private space
huggingface-cli repo create arrhythmia-prediction --type space --private --space_sdk gradio
```

### 3. Upload Files

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/arrhythmia-prediction
cd arrhythmia-prediction

# Copy files
cp -r ../handover_package/* .

# Add and commit
git add .
git commit -m "Initial upload"
git push
```

## Files to Upload

✅ **Include**:
- `README.md`
- `requirements.txt`
- `models/*.joblib` (trained models)
- `app.py` (Gradio interface)
- `LICENSE`

❌ **NEVER Upload**:
- `旧的/*.xlsx` (patient data)
- Any CSV with patient information
- Raw data files

## Gradio App Template

See `app.py` for a simple Gradio interface.

## Model Card Template

Create `README.md` with:
- Model description
- Performance metrics
- Intended use
- Limitations
- Citation

Example: `model_card_template.md`
