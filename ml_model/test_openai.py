#!/usr/bin/env python3
"""
Test script to debug OpenAI API issues
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 OpenAI Setup Debugging")
print("=" * 50)

# Check if .env file exists
env_file_exists = os.path.exists('.env')
print(f"📁 .env file exists: {env_file_exists}")

# Check environment variable
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"🔑 OPENAI_API_KEY found: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else 'short'}")
else:
    print("❌ OPENAI_API_KEY not found in environment")

# Test OpenAI import
try:
    from openai import OpenAI
    print("✅ OpenAI library imported successfully")
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            print("✅ OpenAI client created successfully")
            
            # Test a simple API call
            print("\n🧪 Testing API call...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'API test successful' in exactly 3 words."}],
                max_tokens=10
            )
            print(f"✅ API Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ OpenAI client error: {e}")
    else:
        print("⚠️ Cannot test API without key")
        
except ImportError as e:
    print(f"❌ OpenAI import failed: {e}")

print("\n💡 Next Steps:")
if not env_file_exists:
    print("1. Create a .env file in this directory")
    print("2. Add: OPENAI_API_KEY=your_actual_api_key_here")
elif not api_key:
    print("1. Add OPENAI_API_KEY to your .env file")
    print("2. Get your key from: https://platform.openai.com/api-keys")
else:
    print("1. API key is set - check if it's valid at https://platform.openai.com/")
    print("2. Ensure you have API credits available")
