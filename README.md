# YouTube Voice-to-Question Analyzer

A Streamlit application that analyzes YouTube videos by transcribing speech and extracting questions with political context analysis. The app uses advanced AI models for transcription and fact-checking capabilities.

## Features

- 🎥 YouTube video audio extraction
- 🗣️ Speech-to-text transcription
- ❓ Question extraction and analysis
- 🏛️ Political context classification
- ✅ Fact-checking system with confidence scoring
- ⏱️ Time range selection for specific video segments

## Requirements

- Python 3.13+
- ffmpeg
- yt-dlp
- Ollama with deepseek-r1:8b model

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd politicalsystem
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install system dependencies:
```bash
# On macOS using Homebrew
brew install ffmpeg yt-dlp

# On Ubuntu/Debian
sudo apt-get install ffmpeg
```

5. Start Ollama server and pull the required model:
```bash
ollama serve
ollama pull deepseek-r1:8b
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run indian.py
```

2. Enter a YouTube URL in the input field

3. (Optional) Specify time range for analysis

4. Click "Analyze" to process the video

5. Use the Fact Checker tab to verify statements from the transcript

## Features in Detail

### Video Analysis
- Extracts audio from YouTube videos
- Transcribes speech to text using Google's Speech Recognition
- Processes audio in chunks for better reliability
- Supports time range selection for partial video analysis

### Question Analysis
- Identifies questions from the transcript
- Classifies political alignment (left/right/neutral)
- Determines question intent (criticism/support/inquiry)
- Displays results in an organized dataframe

### Fact Checking
- Vector database storage of transcript segments
- Semantic search for relevant context
- AI-powered fact verification
- Confidence scoring for answers
- Quote extraction for reference

## Technical Architecture

- **Frontend**: Streamlit
- **Speech Processing**: SpeechRecognition, pydub
- **Video Processing**: yt-dlp, ffmpeg
- **AI Models**: 
  - Ollama (deepseek-r1:8b) for analysis
  - Sentence Transformers for embeddings
- **Vector Database**: Qdrant for fact storage and retrieval

## Error Handling

The application includes comprehensive error handling for:
- Invalid YouTube URLs
- Network connectivity issues
- Video accessibility problems
- Audio processing errors
- Transcription failures

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]
