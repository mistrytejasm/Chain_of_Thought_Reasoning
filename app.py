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

# cot_chain = llm | cot_prompt
# direct_chain = llm | direct_prompt

# Sample questions that showcase CoT benefits
SAMPLE_QUESTIONS = [
    {
        "question": "If a plane crashes on the border of two countries, where do they bury the survivors?",
        "category": "Commonsense - Reasoning",
        "icon": "✈️"
    },
    {
        "question": "A man has 3 hats: red, blue, and green. He randomly picks one each day. If he wears the red hat on Monday, what is the chance he wears it again on Wednesday?",
        "category": "Commonsense - Probability",
        "icon": "🧢"
    },
    {
        "question": "If you’re running a race and overtake the person in second place, what position are you in now?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "🏃"
    },
    {
        "question": "A candle burns down completely in 6 hours. How long will it take for 3 candles to burn down if they are lit at the same time?",
        "category": "Commonsense - Reasoning",
        "icon": "🕯️"
    },
    {
        "question": "A farmer needs to cross a river with a wolf, a goat, and a cabbage. He can only take one at a time in his boat. How can he get all three across without the wolf eating the goat or the goat eating the cabbage?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "🚣"
    },
    {
        "question": "A clock is ticking backward and shows 4:00 PM now. What time will it show in 5 hours?",
        "category": "Commonsense - Time Reasoning",
        "icon": "⏰"
    },
    {
        "question": "If all roses are flowers and some flowers are red, must some roses be red?",
        "category": "Commonsense - Logical Reasoning",
        "icon": "🌹"
    },
    {
        "question": "A person says, 'I am lying right now.' Is this statement true or false?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "🧠"
    },
    {
        "question": "You have a cake and want to cut it into exactly 8 equal pieces using only 3 straight cuts. Is this possible, and if so, how?",
        "category": "Commonsense - Spatial Reasoning",
        "icon": "🍰"
    },
    {
        "question": "If a frog climbs a 10-meter well, rising 3 meters each day but slipping back 2 meters each night, how many days does it take to reach the top?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "🐸"
    },
    {
        "question": "If a town’s population doubles every 10 years and is 8,000 now, what was it 15 years ago?",
        "category": "Commonsense - Arithmetic Reasoning",
        "icon": "🏘️"
    },
    {
        "question": "A light bulb is on for 10 minutes, off for 5 minutes, then on for 10 minutes again. How long does this cycle take to complete?",
        "category": "Commonsense - Time Reasoning",
        "icon": "💡"
    },
    {
        "question": "If a ladder is leaning against a wall and slides down without moving its base, does the top of the ladder get closer to or farther from the wall?",
        "category": "Commonsense - Spatial Reasoning",
        "icon": "🪜"
    },
    {
        "question": "A man walks 3 miles south, 3 miles east, and 3 miles north, ending up at his starting point. Where could he be standing?",
        "category": "Commonsense - Spatial Reasoning",
        "icon": "🌍"
    },
    {
        "question": "If you have 4 identical boxes and 10 identical coins, can you place all coins in the boxes so each box has a different number of coins?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "📦"
    },
    {
        "question": "A woman buys a turkey that weighs 16 pounds. After cooking, it loses 20% of its weight, and she serves 1/4 of the cooked turkey. How much turkey is left?",
        "category": "Commonsense - Arithmetic Reasoning",
        "icon": "🦃"
    },
    {
        "question": "If a train leaves at 8:00 AM traveling 50 mph and another leaves at 9:00 AM traveling 70 mph in the same direction, when will they meet?",
        "category": "Commonsense - Relative Speed",
        "icon": "🚄"
    },
    {
        "question": "You have 3 switches controlling 3 lights, but you can’t see the lights from the switches. How can you determine which switch controls which light with only one trip to the lights?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "💡"
    },
    {
        "question": "If a rope is cut into 3 pieces, with the second piece 1 meter longer than the first, and the third piece twice as long as the first, and the total length is 14 meters, how long is each piece?",
        "category": "Commonsense - Arithmetic Reasoning",
        "icon": "🪢"
    },
    {
        "question": "A clock strikes once at 1:00, twice at 2:00, and so on up to twelve times at 12:00. How many times does it strike from 1:00 AM to 12:00 PM, inclusive?",
        "category": "Commonsense - Arithmetic Reasoning",
        "icon": "🕰️"
    },
    {
        "question": "If you’re in a room with 3 doors—one leads to freedom, one to a lion, and one to a dead end where you return to try again—how many doors do you expect to try before finding freedom?",
        "category": "Commonsense - Probability",
        "icon": "🚪"
    },
    {
        "question": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "category": "Commonsense - Arithmetic Reasoning",
        "icon": "⚾"
    },
    {
        "question": "If a car’s odometer reads 999 miles and you drive 2 more miles, what will the odometer read, assuming it has 3 digits?",
        "category": "Commonsense - Logical Reasoning",
        "icon": "🚗"
    },
    {
        "question": "A puzzle has 5 switches, each either on or off. Flipping a switch changes its state and its neighbors’. How many flips are needed to turn all switches on, starting from all off?",
        "category": "Commonsense - Logic Puzzle",
        "icon": "🔧"
    },
    {
        "question": "If a recipe for 6 servings requires 2 cups of flour and you have 5 cups of flour, how many full servings can you make, and how much flour is left?",
        "category": "Commonsense - Arithmetic Reasoning",
        "icon": "🥐"
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