from flask import Flask, request, render_template, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import time
import re
from dotenv import load_dotenv

app = Flask(__name__)

# Environment Setup
load_dotenv() 

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY not set in environment variables.")
os.environ["OPENAI_API_KEY"] = groq_api_key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# LangChain setup
llm = ChatOpenAI(model="gemma2-9b-it", temperature=0.3)

# Enhanced Chain of Thought prompt with clearer structure
cot_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert problem solver. Solve this step by step with clear reasoning.

Question: {question}

Please solve this systematically:

REASONING:
Step 1: [Analyze what the question is asking]
Step 2: [Identify the given information]
Step 3: [Determine what needs to be calculated]
Step 4: [Apply the appropriate method or formula]
Step 5: [Perform the calculation]
Step 6: [Verify the result makes sense]

FINAL_ANSWER: [State your final answer clearly]
"""
)

# Direct answer prompt
direct_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Answer this question directly and concisely.

Question: {question}

Answer:
"""
)

# Create chains
cot_chain = LLMChain(llm=llm, prompt=cot_prompt)
direct_chain = LLMChain(llm=llm, prompt=direct_prompt)

# Sample questions that showcase CoT benefits
SAMPLE_QUESTIONS = [
    {
        "question": "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "category": "Math - Distance",
        "icon": "🚂"
    },
    {
        "question": "A restaurant bill is $48. If you want to leave an 18% tip, what's the total amount you'll pay?",
        "category": "Math - Percentage",
        "icon": "🍽️"
    },
    {
        "question": "Sarah has twice as many apples as Tom. Tom has 8 apples. How many apples do they have together?",
        "category": "Math - Word Problem",
        "icon": "🍎"
    },
    {
        "question": "If I have 3 red balls, 5 blue balls, and 2 green balls in a bag, what fraction of the balls are blue?",
        "category": "Math - Fractions",
        "icon": "🔵"
    },
    {
        "question": "A parking meter accepts quarters ($0.25) and gives 15 minutes per quarter. If I want to park for 2 hours, how much money do I need?",
        "category": "Math - Time & Money",
        "icon": "🅿️"
    },
    {
        "question": "In a class of 30 students, 18 like pizza, 12 like burgers, and 6 like both. How many students like neither pizza nor burgers?",
        "category": "Logic - Set Theory",
        "icon": "🧮"
    },
    {
        "question": "If 7 pencils cost $3.50, how much does 1 pencil cost?",
        "category": "Math - Unit Price",
        "icon": "✏️"
    },
    {
        "question": "A bus travels 120 km at a speed of 40 km/h. How long does it take?",
        "category": "Math - Speed & Time",
        "icon": "🚌"
    },
    {
        "question": "Lisa has twice as many stickers as Jack. Together they have 30. How many does each have?",
        "category": "Math - Word Problem",
        "icon": "🎟️"
    },
    {
        "question": "A rectangle’s length is 3 m more than its width. Its area is 70 m². Find its dimensions.",
        "category": "Math - Geometry",
        "icon": "📐"
    },
    {
        "question": "Emily saves $20 every week. How many weeks will it take her to save $600?",
        "category": "Math - Saving & Budgeting",
        "icon": "💰"
    },
    {
        "question": "Which of these is a quality assurance activity: verifying processes or inspecting final products?",
        "category": "CSQA - Concepts",
        "icon": "✅"
    },
    {
        "question": "Interoperability in software testing refers to: (A) ease of integration (B) user satisfaction (C) speed?",
        "category": "CSQA - Definitions",
        "icon": "🧩"
    },
    {
        "question": "If a person is born in 2000, can they celebrate their 21st birthday in 2021?",
        "category": "Logic - Reasoning",
        "icon": "🎂"
    },
    {
        "question": "Can a square have four acute angles?",
        "category": "Math - Geometry Logic",
        "icon": "🔷"
    },
    {
        "question": "If January 1, 2025 is a Wednesday, what day is March 1, 2025?",
        "category": "Date Understanding",
        "icon": "📅"
    },
    {
        "question": "How many Sundays were there in February 2024?",
        "category": "Date Understanding",
        "icon": "📆"
    },
    {
        "question": "A cricket bowler concedes 48 runs in 8 overs. What is their economy rate?",
        "category": "Sports - Cricket",
        "icon": "🏏"
    },
    {
        "question": "A basketball team won 18 out of 30 games. What is their win percentage?",
        "category": "Sports - Basketball",
        "icon": "🏀"
    },
    {
        "question": "A train departs at 3:15 PM and arrives at 5:45 PM. How long is the journey?",
        "category": "Math - Time Calculation",
        "icon": "⏱️"
    },
    {
        "question": "A shirt costs $40 and is discounted by 25%. What is the sale price?",
        "category": "Math - Discount",
        "icon": "👕"
    },
    {
        "question": "You roll a 6-sided die. What’s the probability of rolling an even number?",
        "category": "Math - Probability",
        "icon": "🎲"
    },
    {
        "question": "There are 12 marbles: 4 red, 5 blue, 3 green. What fraction are not blue?",
        "category": "Math - Fractions",
        "icon": "⚪"
    },
    {
        "question": "Which costs more: 3 pens at $2.50 each or 4 notebooks at $1.80 each?",
        "category": "Math - Cost Comparison",
        "icon": "📚"
    },
    {
        "question": "You’re driving 180 miles at 60 mph. How long will it take?",
        "category": "Math - Distance & Time",
        "icon": "🚗"
    },
    {
        "question": "An exam has 40 questions. You answer 32 correctly. What’s your score percentage?",
        "category": "Math - Percentage",
        "icon": "📝"
    }
]


def parse_cot_response(response):
    """Parse CoT response to extract reasoning and final answer"""
    # Clean up response
    clean_response = response.replace('**', '').strip()
    
    # Try to extract reasoning section
    reasoning_match = re.search(r'REASONING:\s*(.*?)(?=FINAL_ANSWER:|$)', clean_response, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    
    # Try to extract final answer
    answer_match = re.search(r'FINAL_ANSWER:\s*(.*?)(?:\n|$)', clean_response, re.IGNORECASE)
    final_answer = answer_match.group(1).strip() if answer_match else ""
    
    # Fallback parsing if structured format isn't found
    if not reasoning and not final_answer:
        # Try alternative patterns
        step_pattern = r'(Step \d+:.*?)(?=Step \d+:|Final|Answer:|$)'
        steps = re.findall(step_pattern, clean_response, re.DOTALL | re.IGNORECASE)
        if steps:
            reasoning = '\n'.join(step.strip() for step in steps)
        
        # Look for answer at the end
        lines = clean_response.split('\n')
        for line in reversed(lines):
            if line.strip() and not line.lower().startswith('step'):
                final_answer = line.strip()
                break
    
    # If still no clear structure, use the whole response
    if not reasoning:
        reasoning = clean_response
    if not final_answer:
        final_answer = "Please see the reasoning above for the complete answer."
    
    return reasoning, final_answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-samples', methods=['GET'])
def get_samples():
    """Return sample questions for the frontend"""
    return jsonify({'samples': SAMPLE_QUESTIONS})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        mode = data.get('mode', 'cot')
        
        if not question:
            return jsonify({'error': 'No question provided.'}), 400

        start_time = time.time()
        
        if mode == 'cot':
            response = cot_chain.run(question)
            reasoning, final_answer = parse_cot_response(response)
            
            return jsonify({
                'response': response,
                'reasoning': reasoning,
                'final_answer': final_answer,
                'mode': mode,
                'processing_time': round(time.time() - start_time, 2),
                'word_count': len(response.split())
            })
            
        elif mode == 'direct':
            response = direct_chain.run(question)
            
            return jsonify({
                'response': response.strip(),
                'mode': mode,
                'processing_time': round(time.time() - start_time, 2),
                'word_count': len(response.split())
            })
        else:
            return jsonify({'error': 'Invalid mode. Use "cot" or "direct".'}), 400

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Get both CoT and direct responses for comparison"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided.'}), 400

        # Get CoT response
        start_time = time.time()
        cot_response = cot_chain.run(question)
        cot_time = time.time() - start_time
        cot_reasoning, cot_final_answer = parse_cot_response(cot_response)
        
        # Get direct response
        start_time = time.time()
        direct_response = direct_chain.run(question)
        direct_time = time.time() - start_time
        
        return jsonify({
            'cot': {
                'response': cot_response,
                'reasoning': cot_reasoning,
                'final_answer': cot_final_answer,
                'processing_time': round(cot_time, 2),
                'word_count': len(cot_response.split())
            },
            'direct': {
                'response': direct_response.strip(),
                'processing_time': round(direct_time, 2),
                'word_count': len(direct_response.split())
            }
        })

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)