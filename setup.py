
import sys
import subprocess
import importlib


def check_python_version():
    print("─" * 50)
    print("Checking Python version...")
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required. You have {version.major}.{version.minor}")
        print("   Download Python from https://www.python.org/downloads/")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def install_requirements():
    print("─" * 50)
    print("Installing dependencies from requirements.txt...")
    print("   This may take 5-15 minutes on first run (models download later)")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ pip install failed. Check your internet connection and try again.")
        sys.exit(1)
    print("✅ Dependencies installed")


def download_spacy_model():
    print("─" * 50)
    print("Downloading spaCy English model (en_core_web_sm, ~12MB)...")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ spaCy model download failed.")
        sys.exit(1)
    print("✅ spaCy model downloaded")


def verify_imports():
    print("─" * 50)
    print("Verifying critical imports...")

    checks = [
        ("streamlit", "Streamlit UI"),
        ("transformers", "HuggingFace Transformers (FinBERT)"),
        ("torch", "PyTorch"),
        ("bertopic", "BERTopic"),
        ("sentence_transformers", "Sentence Transformers"),
        ("spacy", "spaCy"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pdfplumber", "pdfplumber (PDF extraction)"),
        ("plotly", "Plotly (charts)"),
        ("umap", "UMAP"),
        ("hdbscan", "HDBSCAN"),
    ]

    all_ok = True
    for module, label in checks:
        try:
            importlib.import_module(module)
            print(f"✅ {label}")
        except ImportError as e:
            print(f"❌ {label}: {e}")
            all_ok = False

    return all_ok


def print_next_steps():
    print("─" * 50)
    print("""
✅ SETUP COMPLETE!

Next steps:
  1. Start the app:
     streamlit run app.py

  2. Open your browser to:
     http://localhost:8501

  3. Upload a transcript:
     Use data/samples/sample_transcript.txt to test immediately.

  4. First-run note:
     FinBERT (~440MB) downloads automatically on first analysis.
     This takes 1-3 minutes depending on your internet speed.
     Subsequent runs are instant (cached locally).

  5. Run tests:
     python -m pytest tests/ -v

IMPORTANT: The first time you click "Analyze" in the app,
  models will download. Be patient — it only happens once!
""")


if __name__ == "__main__":
    print("=" * 50)
    print("EARNINGS CALL INTELLIGENCE PLATFORM — SETUP")
    print("=" * 50)

    check_python_version()
    install_requirements()
    download_spacy_model()
    ok = verify_imports()

    if ok:
        print_next_steps()
    else:
        print("\n❌ Some imports failed. Re-run: pip install -r requirements.txt")
        sys.exit(1)
