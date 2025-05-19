# 🧠 Chain of Thought Reasoning App

A simple and interactive web application that leverages **LangChain** and **Flask** to answer user questions using **step-by-step reasoning** powered by a large language model like **Gemma 2 (via Groq)**.

Try the app [Chain of Thought Reasoning App](https://chain-of-thought-reasoning.onrender.com)

![image](https://github.com/user-attachments/assets/1b1a50d2-7597-4a40-a16e-868f7348640a)


---

## 🚀 Features

* 🔍 Accepts any natural language question.
* 🧩 Generates detailed, step-by-step reasoning before producing a final answer.
* ⚡ Fast inference using Groq's LLMs (e.g., `gemma2-9b-it`).
* 🎨 Simple, responsive UI using TailwindCSS.
* 🧠 Built with LangChain, Flask, and OpenAI-compatible API.

---

## 🛠️ Tech Stack

| Layer        | Technology                        |
| ------------ | --------------------------------- |
| Backend      | Flask, LangChain                  |
| LLM Provider | Groq (OpenAI-compatible endpoint) |
| UI           | HTML, TailwindCSS, Vanilla JS     |
| Deployment   | Localhost (development)           |

---

## 📂 Project Structure

```
CHAIN-THOUGHT-APP/
├── templates/
│   └── index.html           # Frontend template
├── app.py                   # Main Flask application
├── .env                     # Environment variables (not committed)
├── .gitignore               # Ignore .env and other files
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mistrytejasm/Chain_of_Thought_Reasoning.git
cd chain-thought-app
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # For Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Ensure you have a Groq account and API access. This uses Groq’s OpenAI-compatible endpoint.

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

---

## 🧪 How It Works

1. You enter a question in natural language.
2. Flask sends the question to the `/ask` endpoint.
3. LangChain builds a prompt using a custom template:

   * **Step-by-step reasoning**
   * **Final answer format**
4. The Groq-hosted LLM (e.g., Gemma 2) responds.
5. The answer and reasoning are extracted and displayed on the page.

---

## 🖼️ Example Output

**Input:**

> What is the capital of France?

**Output:**

```text
Final Answer: Paris
Reasoning: France is a country in Europe. The capital city of France is Paris, known for its culture and landmarks like the Eiffel Tower.
```

---

## 📦 Dependencies

* `flask`
* `langchain`
* `langchain-openai`
* `langchain_community`
* `python-dotenv`

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧰 Troubleshooting

* ❗ **API Errors:** Ensure your `.env` file contains a valid `GROQ_API_KEY`.
* ⚠️ **LLM Model Errors:** Check if your chosen model (e.g., `gemma2-9b-it`) is supported by Groq and accessible.
* 💡 **No Answer Returned?** Make sure the prompt format is being respected and the model returns structured output.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 🙌 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [Groq API](https://console.groq.com/)
* [TailwindCSS](https://tailwindcss.com/)
* [Flask](https://flask.palletsprojects.com/)
