import streamlit as st
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Prompt for image-based diagnosis
image_prompt = """
You are a medical expert analyzing images of human body parts for potential diseases or health issues. You are working in a reputable hospital's telemedicine unit. 

Given an image, provide a detailed analysis of:
1. Observed findings (symptoms/anomalies)
2. Possible medical conditions
3. Recommended next steps (tests, consultation, self-care)
4. Urgency level (e.g., Immediate, Moderate, Routine)
5. Disclaimer: Always consult a licensed physician before taking action.

If the image quality is poor or unidentifiable, state that clearly.
"""

# Prompt for text-based medical description
text_prompt_template = """
You are a virtual medical assistant working in a hospital. A user has submitted the following description of their medical concern:

"{description}"

Provide a structured response that includes:
1. Possible conditions or explanations
2. Recommended actions (self-care or consult type)
3. Red flags (if any)
4. Disclaimer: This is not a medical diagnosis. Always consult a healthcare professional.
"""

# Encode image for GPT-4
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Handle GPT-4 analysis for image
def call_gpt4_model_for_image(filename: str):
    base64_image = encode_image(filename)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": image_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1500
    )
    return response.choices[0].message.content

# Handle GPT-4 analysis for text
def call_gpt4_model_for_text(description: str):
    prompt = text_prompt_template.format(description=description)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

# ELI5 explanation
def chat_eli(query):
    eli5_prompt = "Explain the following in very simple terms. :\n" + query
    messages = [{"role": "user", "content": eli5_prompt}]
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

# Streamlit app layout
st.set_page_config(page_title="AI Medical Assistant", page_icon="🩺", layout="centered")
st.title("🩺 AI Medical Diagnosis Assistant")

st.markdown("""
Welcome to the AI-powered Medical Assistant. You can:
- Upload a **photo** of a medical issue *(e.g., skin rash, swelling)*.
- Or describe your symptoms in **text**.

> ⚠️ This is not a substitute for a professional medical diagnosis. Always consult your doctor.
""")

input_mode = st.radio("Choose Input Method:", ["Upload Image", "Describe Symptoms"])

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                result = call_gpt4_model_for_image(image_path)
                st.markdown(result, unsafe_allow_html=True)
                if st.toggle("Explain in simpler terms"):
                    st.markdown(chat_eli(result), unsafe_allow_html=True)

elif input_mode == "Describe Symptoms":
    user_input = st.text_area("📝 Describe your issue:", placeholder="e.g., I have a sore throat and mild fever for the past 3 days.")
    if st.button("🧠 Analyze Description"):
        if user_input.strip():
            with st.spinner("Analyzing description..."):
                result = call_gpt4_model_for_text(user_input)
                st.markdown(result, unsafe_allow_html=True)
                if st.toggle("Explain in simpler terms"):
                    st.markdown(chat_eli(result), unsafe_allow_html=True)
        else:
            st.warning("Please enter a description before proceeding.")
