"""
Simple test to verify Gemini SDK works
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("=" * 50)
print("GEMINI SDK TEST")
print("=" * 50)

# Check if Gemini API key is set
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("❌ GEMINI_API_KEY not set!")
    print("\nTo set it:")
    print('  PowerShell: $env:GEMINI_API_KEY = "your-key-here"')
    print("\nGet your key at: https://makersuite.google.com/app/apikey")
    exit(1)

print("✅ GEMINI_API_KEY is set")

# Try importing google.generativeai
try:
    import google.generativeai as genai
    print("✅ google.generativeai imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nInstall with: pip install google-generativeai")
    exit(1)

# Try configuring and creating a model
try:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✅ Gemini model initialized successfully")
    print(f"   Model: gemini-2.5-flash")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    exit(1)

# Try a simple generation
try:
    print("\n🧪 Testing simple generation...")
    response = model.generate_content("Say 'Hello from Gemini!' in JSON format with a 'message' field.")
    print("✅ Generation successful!")
    print(f"\nResponse:\n{response.text}")
except Exception as e:
    print(f"❌ Generation error: {e}")
    exit(1)

print("\n" + "=" * 50)
print("🎉 GEMINI SDK TEST PASSED!")
print("=" * 50)
print("\nGemini is working correctly!")
print("The Flow Builder Agent can now use Gemini for flow generation.")
