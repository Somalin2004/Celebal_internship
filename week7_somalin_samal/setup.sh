#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. Copy .env.example to .env and add your GROQ_API_KEY"
echo "  2. Put a PDF file inside the docs/ folder"
echo "  3. Run: python ingest.py"
echo "  4. Run: streamlit run app.py"
