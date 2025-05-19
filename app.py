from flask import Flask, request, render_template, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Environment Setup
load_dotenv() 

os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
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
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'response': 'No question provided.'})

    response = chain.run(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
