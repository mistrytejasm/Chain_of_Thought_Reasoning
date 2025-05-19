from flask import Flask, request, render_template, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
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

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Answer the following question using step-by-step reasoning.

Question: {question}

First, think step by step to solve the problem.
Then, provide a final short answer separately in the format:

Final Answer: <your short answer here>
Reasoning: <step-by-step reasoning here>
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'response': 'No question provided.'}), 400

        response = chain.run(question)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

