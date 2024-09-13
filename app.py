import streamlit as st
import os
from dotenv import load_dotenv
import groq
import json

# Load environment variables
load_dotenv()

# Existing functions from the original script
def create_groq_client():
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return groq.Groq(api_key=api_key)

def generate_chain_of_thought_prompt(question):
    return f"""Please approach this question step-by-step:

1) First, let's clarify the key elements of the question.
2) Then, let's break down the problem into smaller, manageable parts.
3) For each part, let's consider relevant information and potential approaches.
4) Let's reason through each part logically, considering cause and effect.
5) Finally, let's synthesize our findings to form a comprehensive answer.

Question: {question}

Let's begin our step-by-step analysis:"""

def generate_few_shot_prompt(question):
    examples = [
        {"question": "What would happen if the Moon disappeared?",
         "answer": "1. Tides would be affected: The Moon's gravitational pull causes ocean tides. Without it, tides would be much weaker, driven only by the Sun's gravity.\n2. Earth's axial tilt would become unstable: The Moon helps stabilize Earth's axial tilt. Its disappearance could lead to more extreme and unpredictable seasons.\n3. Day length would change: The Moon's gravity gradually slows Earth's rotation. Without this effect, our days might become shorter over time.\n4. Nocturnal animals would be impacted: Many species rely on moonlight for navigation and timing of activities. Their behavior and survival could be affected.\n5. Cultural and psychological effects: The Moon has significant cultural and emotional importance for humans. Its loss could have profound psychological impacts."},
        {"question": "How might we colonize Mars?",
         "answer": "1. Develop advanced spacecraft: We need reliable, efficient spacecraft for transporting people and supplies to Mars.\n2. Create life support systems: Mars has a thin atmosphere and no magnetic field, so we need to develop systems for air, water, and radiation protection.\n3. Establish power sources: Solar panels and nuclear reactors could provide energy for a Mars colony.\n4. Build habitats: We need to construct pressurized, insulated living spaces that can withstand Mars' harsh environment.\n5. Grow food: Developing methods for Martian agriculture, possibly in greenhouse structures, is crucial for long-term survival.\n6. Extract local resources: Learning to extract water from Martian ice and produce fuel from the Martian atmosphere will be key to sustainability.\n7. Address psychological challenges: We must prepare for the mental health impacts of isolation and confinement in an alien environment.\n8. Develop a Mars-based economy: To be sustainable, a colony needs to produce goods or services of value to Earth or other space endeavors."}
    ]
    
    few_shot_prompt = "Here are a couple of examples of how to approach complex questions:\n\n"
    for example in examples:
        few_shot_prompt += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
    
    few_shot_prompt += f"Now, let's apply this approach to the following question:\n\nQuestion: {question}\nAnswer:"
    
    return few_shot_prompt

def query_mixtral(client, prompt, temperature=0.7, max_tokens=2000):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except groq.error.GroqError as e:
        st.error(f"An error occurred while querying the Groq API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="Enhanced Reasoning with Mixtral", layout="wide")

# Enhanced custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap');
    body {
        background-color: black;
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    .output-box {
        border: 3px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #ff00ff, #00ffff);
        background-size: 200% 200%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .emphasis {
        font-weight: 900;
        color: #ff00ff;
    }
    .enhanced {
        font-style: italic;
        text-decoration: underline;
        color: #00ffff;
    }
    .stApp {
        background-color: black;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Enhanced Reasoning with Mixtral")
st.markdown("Explore complex questions using advanced prompting techniques!")

# Initialize Groq client
client = create_groq_client()

# Input for user's question
question = st.text_input("Enter your question:", key="user_question")

if st.button("Analyze"):
    if question:
        with st.spinner("Analyzing your question..."):
            # Chain of thought prompting
            cot_prompt = generate_chain_of_thought_prompt(question)
            cot_response = query_mixtral(client, cot_prompt, temperature=0.7)
            
            # Few-shot learning
            few_shot_prompt = generate_few_shot_prompt(question)
            few_shot_response = query_mixtral(client, few_shot_prompt, temperature=0.5)
            
            # Summary of outcomes
            summary_prompt = f"Summarize the potential outcomes or implications of the following question: {question}"
            summary_response = query_mixtral(client, summary_prompt, temperature=0.5)
        
        # Display results
        st.subheader("Chain of Thought Analysis")
        st.markdown(f'<div class="output-box">{cot_response}</div>', unsafe_allow_html=True)
        
        st.subheader("Few-Shot Learning Response")
        st.markdown(f'<div class="output-box">{few_shot_response}</div>', unsafe_allow_html=True)
        
        st.subheader("Summary of Outcomes")
        st.markdown(f'<div class="output-box">{summary_response}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a question to analyze.")

st.markdown("---")
st.markdown("Powered by Groq API and Mixtral-8x7B model")
