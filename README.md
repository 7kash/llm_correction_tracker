# 🤖 LLM Learning Visualizer

An interactive web application that helps non-technical users understand how Large Language Models (LLMs) work by visualizing how they adapt their responses when corrected.

![LLM Learning Visualizer](https://img.shields.io/badge/Status-Ready-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![React](https://img.shields.io/badge/React-18-blue)

## 🎯 What Does This Do?

This app lets you:
- **Ask questions** to an AI (powered by OpenAI's GPT)
- **Correct the AI** when it's wrong
- **See visual graphs** showing how the AI's response changes:
  - Response length (word count)
  - Sentiment (positive/negative tone)
  - Confidence level
  - And more!

Perfect for teachers, students, or anyone curious about how AI learns and adapts!

## ✨ Features

### 📊 Beautiful Visualizations
- **Response Length Chart**: See how the AI adjusts the detail level
- **Sentiment Analysis**: Track emotional tone changes
- **Confidence Tracking**: Monitor how certain the AI seems
- Real-time updates with smooth animations

### 🎓 Educational
- Clear explanations for every section
- Non-technical language throughout
- Learn by doing - interact with real AI!

### 🎨 Beautiful UI
- Modern, clean design with Tailwind CSS
- Smooth animations and transitions
- Responsive layout (works on phones, tablets, desktops)
- Professional gradient styling

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **OpenAI API Key** (get one at [platform.openai.com](https://platform.openai.com/api-keys))
- A web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd llm_correction_tracker
   ```

2. **Set up the backend**
   ```bash
   cd backend

   # Create a virtual environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure your API key**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_actual_key_here
   ```

4. **Run the application**
   ```bash
   # Start the backend (from the backend directory)
  PORT=5001 python app.py  
   ```

5. **Open the frontend**
   - Open `frontend/index.html` in your web browser
   - Or use a local server:
     ```bash
     # From the frontend directory
     python -m http.server 8000
     # Then visit http://localhost:8000
     ```

## 📖 How to Use

### Step 1: Ask a Question
Type any question in the input box. Examples:
- "What is the capital of France?"
- "How many planets are in our solar system?"
- "When did World War 2 end?"

### Step 2: Get AI Response
Click "Ask the AI" and watch it respond!

### Step 3: Provide Corrections
Tell the AI what's wrong or what needs improvement:
- "That's not quite right, it actually ended in 1945"
- "You're missing information about..."
- "Can you be more specific about..."

### Step 4: Watch the Magic! ✨
See the graphs update in real-time showing:
- How the response length changes
- Whether the tone becomes more positive or negative
- How the AI's confidence shifts

## 🏗️ Project Structure

```
llm_correction_tracker/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variables template
│   └── .env                # Your API keys (create this!)
├── frontend/
│   └── index.html          # React app (single file!)
├── .gitignore
└── README.md
```

## 🎨 Visualizations Explained

### 📏 Response Length Chart
Shows how many words the AI uses in each response.
- **Going up?** The AI is adding more detail
- **Going down?** The AI is being more concise

### 😊 Sentiment Analysis
Measures the emotional tone of the response.
- **Positive (green)**: Optimistic, affirming language
- **Negative (red)**: Cautious or corrective language

### 🎯 Confidence Level
Based on uncertainty words (maybe, perhaps, might, etc.)
- **Higher scores**: More definitive statements
- **Lower scores**: More hedging and uncertainty

## 🔧 API Endpoints

### `POST /api/start-session`
Start a new conversation session
```json
{
  "question": "What is the capital of France?"
}
```

### `POST /api/correct`
Submit a correction
```json
{
  "session_id": "123456",
  "correction": "Actually, that's not quite right..."
}
```

### `GET /api/session/:id`
Get complete session data with analytics

### `GET /api/health`
Health check endpoint

## 🛠️ Customization

### Using Different AI Models

Edit `backend/app.py` to change the model:
```python
# Line ~60 - change the model parameter
response = openai.chat.completions.create(
    model="gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
    messages=messages,
    temperature=0.7
)
```

### Adjusting Analysis Metrics

The `analyze_text()` function in `app.py` contains all the text analysis logic. You can:
- Add new sentiment words
- Change confidence scoring
- Add custom metrics

### Styling

The frontend uses Tailwind CSS. Edit the classes in `frontend/index.html` to customize:
- Colors: `bg-purple-500`, `text-blue-600`, etc.
- Spacing: `p-4`, `m-6`, `gap-4`, etc.
- Animations: Add custom CSS in the `<style>` section

## 📚 Educational Use Cases

### For Teachers
- Demonstrate AI concepts without technical jargon
- Show students how AI "learns" from feedback
- Discuss AI limitations and adaptability

### For Students
- Experiment with different types of questions
- See how feedback affects AI behavior
- Learn about natural language processing

### For Curious Minds
- Understand how chatbots work
- See the connection between input and output
- Explore AI decision-making patterns

## 🤔 Common Questions

**Q: Does the AI permanently learn from my corrections?**
A: No! The AI only uses your corrections within this conversation. Each new session starts fresh.

**Q: Why do I need an API key?**
A: The app uses OpenAI's API to generate responses. The API key authenticates your requests.

**Q: Can I use this without an API key?**
A: Not currently, but you could modify the code to use a local LLM or different API.

**Q: Is my data private?**
A: Conversations are stored temporarily in memory and cleared when you restart the server. However, they are sent to OpenAI's API (see their privacy policy).

## 🐛 Troubleshooting

### Backend won't start
- Check that Python 3.8+ is installed: `python --version`
- Verify your virtual environment is activated
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### "API Key Error"
- Verify your `.env` file exists in the `backend/` directory
- Check that your API key is correctly formatted
- Test your API key at platform.openai.com

### CORS Errors
- Ensure the backend is running on port 5000
- Check that the `CORS(app)` line is present in `app.py`
- Try opening the frontend with a local server instead of file://

### Graphs not showing
- Open browser console (F12) to check for errors
- Ensure Chart.js is loading (check internet connection)
- Try refreshing the page

## 🚀 Future Enhancements

Ideas for extending this project:
- [ ] Add database persistence for sessions
- [ ] Support for multiple AI models (Claude, Llama, etc.)
- [ ] Export conversation data as PDF/JSON
- [ ] More advanced visualizations (word clouds, topic modeling)
- [ ] User accounts and saved sessions
- [ ] Side-by-side model comparison
- [ ] Real-time collaboration features

## 📄 License

MIT License - feel free to use this for educational purposes!

## 🤝 Contributing

Found a bug or have an idea? Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 💬 Support

Need help? Check:
- OpenAI API Documentation: https://platform.openai.com/docs
- Flask Documentation: https://flask.palletsprojects.com/
- Chart.js Documentation: https://www.chartjs.org/docs/

---

**Built with ❤️ to help people understand AI**

Happy exploring! 🚀
