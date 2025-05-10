# app.py

import streamlit as st
import subprocess
import os
import tempfile
import pandas as pd
import ollama
import json
import speech_recognition as sr
from pydub import AudioSegment
import math
from urllib.parse import urlparse
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Initialize Qdrant client and encoder
@st.cache_resource
def init_qdrant():
    client = QdrantClient(":memory:")  # In-memory storage for demonstration
    # Create collection for storing text embeddings
    client.recreate_collection(
        collection_name="facts",
        vectors_config={
            "size": 384,  # Default size for all-MiniLM-L6-v2
            "distance": "Cosine"
        }
    )
    return client

@st.cache_resource
def get_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize resources
qdrant_client = init_qdrant()
encoder = get_encoder()

def is_valid_youtube_url(url):
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc]) and ('youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc)
    except:
        return False

def transcribe_audio_chunk(recognizer, audio_chunk):
    try:
        # Add timeout and retry logic
        for attempt in range(3):  # Try 3 times
            try:
                return recognizer.recognize_google(audio_chunk, language='en-US')
            except sr.RequestError:
                if attempt == 2:  # Last attempt
                    raise
                continue
    except sr.RequestError:
        raise Exception("Could not connect to Google Speech Recognition service. Please check your internet connection.")
    except sr.UnknownValueError:
        raise Exception("Speech Recognition could not understand the audio")

def store_transcript_facts(transcript: str):
    try:
        # Split transcript into sentences
        sentences = [s.strip() for s in transcript.split('.') if s.strip()]
        
        # Generate embeddings for sentences
        embeddings = encoder.encode(sentences)
        
        # Create points list manually with proper structure
        points = []
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            point = {
                "id": str(i),  # Convert id to string to avoid potential int/dict confusion
                "vector": embedding.tolist(),
                "payload": {"text": sentence}
            }
            points.append(point)
        
        # Store in Qdrant with explicit point format
        qdrant_client.upsert(
            collection_name="facts",
            points=points
        )
        return sentences
    except Exception as e:
        st.error(f"Error storing facts: {str(e)}")
        return []

def verify_fact(question: str, context: List[str]) -> Dict:
    try:
        # Generate embedding for the question
        question_embedding = encoder.encode(question).tolist()
        
        # Search for similar facts in Qdrant
        search_results = qdrant_client.search(
            collection_name="facts",
            query_vector=question_embedding,
            limit=3
        )
        
        # Get relevant context with proper error handling
        relevant_facts = []
        for result in search_results:
            try:
                # Handle both dictionary and ScoredPoint object formats
                if isinstance(result, dict):
                    if 'payload' in result and 'text' in result['payload']:
                        relevant_facts.append(result['payload']['text'])
                else:
                    if hasattr(result, 'payload') and hasattr(result.payload, 'text'):
                        relevant_facts.append(result.payload.text)
            except Exception as e:
                continue  # Skip problematic results
        
        # Prepare prompt for fact verification
        verification_prompt = f"""
        Question: {question}
        
        Relevant context from transcript:
        {'. '.join(relevant_facts) if relevant_facts else 'No relevant context found'}
        
        Based on the context above, please:
        1. Determine if the question can be answered from the context
        2. Calculate a confidence score (0-100%) for the answer
        3. Provide reasoning for your assessment
        
        Respond in JSON format:
        {{
            "can_answer": true/false,
            "confidence_score": percentage,
            "reasoning": "explanation",
            "relevant_quotes": ["quote1", "quote2"]
        }}
        """
        
        # Get verification from Ollama with proper error handling
        try:
            response = ollama.chat(
                model="deepseek-r1:8b",
                messages=[{"role": "user", "content": verification_prompt}]
            )
            
            if isinstance(response, dict) and 'message' in response:
                content = response['message'].get('content', '')
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {
                        "can_answer": False,
                        "confidence_score": 0,
                        "reasoning": "Failed to parse LLM response as JSON",
                        "relevant_quotes": []
                    }
            else:
                return {
                    "can_answer": False,
                    "confidence_score": 0,
                    "reasoning": "Invalid response format from LLM",
                    "relevant_quotes": []
                }
        except Exception as e:
            return {
                "can_answer": False,
                "confidence_score": 0,
                "reasoning": f"Error getting LLM response: {str(e)}",
                "relevant_quotes": []
            }
    except Exception as e:
        return {
            "can_answer": False,
            "confidence_score": 0,
            "reasoning": f"Error during fact verification: {str(e)}",
            "relevant_quotes": []
        }

def parse_time(time_str: str) -> int:
    """Convert time string (HH:MM:SS or MM:SS or SS) to seconds"""
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    else:
        return int(parts[0])

st.title("🎙️ YouTube Voice-to-Question Analyzer")

# Add tabs for different functionalities
tab1, tab2 = st.tabs(["Analyze Video", "Fact Checker"])

with tab1:
    # Step 1: Get YouTube URL and time range
    url = st.text_input("Enter YouTube URL", "")
    
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input("Start Time (HH:MM:SS or MM:SS)", "")
        st.caption("Example: 1:30 for 1 minute 30 seconds")
    with col2:
        end_time = st.text_input("End Time (HH:MM:SS or MM:SS)", "")
        st.caption("Leave empty to process until the end")

    if url:
        # Validate URL before proceeding
        if not is_valid_youtube_url(url):
            st.error("Please enter a valid YouTube URL")
            st.stop()
        
        # Validate time format
        try:
            start_seconds = parse_time(start_time) if start_time else 0
            end_seconds = parse_time(end_time) if end_time else None
            
            if end_seconds and start_seconds >= end_seconds:
                st.error("End time must be after start time")
                st.stop()
        except ValueError:
            st.error("Invalid time format. Please use HH:MM:SS or MM:SS")
            st.stop()
        
        # Check internet connection
        try:
            requests.get("https://www.google.com", timeout=5)
        except requests.ConnectionError:
            st.error("No internet connection. Please check your network and try again.")
            st.stop()

    if url and st.button("Analyze"):
        with st.spinner("Downloading and processing video..."):
            # Create temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download audio using yt-dlp with time range
                audio_path = os.path.join(tmpdir, "audio")
                progress_text = st.empty()
                progress_text.info("Downloading audio from YouTube...")
                
                try:
                    # Prepare yt-dlp command with time range
                    yt_dlp_cmd = [
                        "yt-dlp",
                        "-x",  # Extract audio
                        "--audio-format", "wav",  # Convert to WAV
                        "--audio-quality", "0",  # Best quality
                    ]
                    
                    # Add time range parameters if specified
                    if start_time:
                        yt_dlp_cmd.extend(["--download-sections", f"*{start_seconds}-{end_seconds or ''}"]) 
                    
                    yt_dlp_cmd.extend(["--output", f"{audio_path}.%(ext)s", url])
                    
                    # Download the audio
                    result = subprocess.run(yt_dlp_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        if "Video unavailable" in result.stderr:
                            st.error("This video is unavailable or private.")
                        else:
                            st.error(f"Failed to download audio: {result.stderr}")
                        st.stop()

                    # Initialize recognizer
                    recognizer = sr.Recognizer()
                    recognizer.operation_timeout = 30  # Set timeout for operations
                    
                    # Load the audio file
                    progress_text.info("Transcribing audio...")
                    wav_path = f"{audio_path}.wav"
                    
                    if not os.path.exists(wav_path):
                        st.error("Failed to convert video to audio. Please try another video.")
                        st.stop()

                    # Process audio in chunks
                    audio_segment = AudioSegment.from_wav(wav_path)
                    chunk_length_ms = 30000  # 30 seconds
                    chunks = math.ceil(len(audio_segment) / chunk_length_ms)
                    
                    progress_bar = st.progress(0)
                    transcript_parts = []

                    for i, chunk_start in enumerate(range(0, len(audio_segment), chunk_length_ms)):
                        chunk = audio_segment[chunk_start:chunk_start + chunk_length_ms]
                        chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
                        chunk.export(chunk_path, format="wav")
                        
                        with sr.AudioFile(chunk_path) as source:
                            audio_chunk = recognizer.record(source)
                            try:
                                chunk_transcript = transcribe_audio_chunk(recognizer, audio_chunk)
                                transcript_parts.append(chunk_transcript)
                                progress_bar.progress((i + 1) / chunks)
                                progress_text.info(f"Transcribing... ({i + 1}/{chunks} chunks)")
                            except Exception as e:
                                st.error(f"Error transcribing chunk {i + 1}: {str(e)}")
                                continue

                    if not transcript_parts:
                        st.error("Could not transcribe any part of the audio. Please try another video.")
                        st.stop()

                    transcript = " ".join(transcript_parts)
                    progress_text.empty()
                    progress_bar.empty()

                    st.subheader("📝 Transcript")
                    st.write(transcript)
                    
                    # Store facts in Qdrant
                    sentences = store_transcript_facts(transcript)
                    st.session_state['transcript_sentences'] = sentences

                    # Step 2: Ask Ollama to extract and analyze questions
                    prompt = f"""
                    From this transcript, extract all the questions asked. For each question, return:
                    1. The question text
                    2. The political side (left, right, neutral)
                    3. The intent of the question (criticism, support, inquiry, etc.)

                    Transcript:
                    {transcript}

                    Respond as a JSON list of objects like:
                    [
                        {{
                            "question": "...",
                            "political_side": "...",
                            "intent": "..."
                        }}
                    ]
                    """

                    with st.spinner("Analyzing questions using Ollama..."):
                        response = ollama.chat(
                            model="deepseek-r1:8b",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        try:
                            questions = json.loads(response['message']['content'])
                            df = pd.DataFrame(questions)
                            st.subheader("🔍 Analyzed Questions")
                            st.dataframe(df)
                        except Exception as e:
                            st.error("Failed to parse Ollama response.")
                            st.text(response['message']['content'])
                            
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    st.error("Please make sure:")
                    st.error("1. The YouTube URL is valid")
                    st.error("2. You have a working internet connection")
                    st.error("3. The video is accessible")
                    st.stop()

with tab2:
    if 'transcript_sentences' not in st.session_state:
        st.warning("Please analyze a video first to enable fact checking.")
    else:
        st.subheader("🔍 Fact Checker")
        question = st.text_input("Enter your question about the transcript:")
        
        if question and st.button("Verify Fact"):
            with st.spinner("Checking facts..."):
                verification_result = verify_fact(
                    question, 
                    st.session_state['transcript_sentences']
                )
                
                # Display results
                st.subheader("Verification Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Confidence Score", 
                        f"{verification_result['confidence_score']}%"
                    )
                
                with col2:
                    st.metric(
                        "Can Answer", 
                        "Yes" if verification_result['can_answer'] else "No"
                    )
                
                st.subheader("Reasoning")
                st.write(verification_result['reasoning'])
                
                if verification_result['relevant_quotes']:
                    st.subheader("Relevant Quotes")
                    for quote in verification_result['relevant_quotes']:
                        st.quote(quote)