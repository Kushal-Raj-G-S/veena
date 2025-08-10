import os
import tempfile
import streamlit as st
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI  # Changed from groq to openai client for compatibility
from TTS.api import TTS
import pygame
import time

# -----------------------------------------
# ENV SETUP
# -----------------------------------------
load_dotenv()
api_key = os.getenv("PROVIDER3_API_KEY")  # Changed from GROQ_API_KEY to API_KEY
if not api_key:
    st.error("❌ Missing PROVIDER3_API_KEY in .env")
    st.stop()

# Initialize components
@st.cache_resource
def initialize_components():
    # Configure client for provider-3/llama-4-maverick
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.a4f.co/v1"
    )
    tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    return client, tts_model

client, tts_model = initialize_components()

# Your custom speaker mapping
speakers_map = {
    "p236": "Victor Mature", "p225": "Ava Gentle", "p226": "Mark Warm",
    "p228": "Ryan Calm", "p230": "James Deep", "p233": "Daniel Bold",
    "p234": "Sam Rich", "p238": "Josh Clear", "p240": "Laura Bright",
    "p241": "Eric Warm", "p243": "Sophie Friendly", "p244": "Grace Soft",
    "p245": "Emma Bright", "p246": "Clara Clear", "p247": "Hannah Smooth",
    "p248": "Bella Calm", "p249": "Mia Gentle", "p250": "Olivia Rich",
    "p251": "Steve Gentle", "p256": "Noah Calm", "p259": "Isla Soft",
    "p260": "Ruby Smooth", "p263": "Lily Clear", "p264": "John Deep",
    "p270": "Sarah Mature (Best)", "p271": "Kate Warm",
    "p273": "Emily Swift", "p274": "Anna Swift", "p275": "Rachel Gentle",
    "p277": "Sophie Gentle", "p278": "Chloe Gentle", "p280": "Zoe Sharp",
    "p283": "Faith Calm", "p284": "Alice Arrogant", "p293": "Nina Sharp",
    "p295": "Paula Rich", "p297": "Daisy Soft", "p299": "Jack Bold",
    "p306": "Eva Calm", "p308": "Laura Fast", "p310": "Megan Sharp",
    "p314": "Julia Gentle", "p316": "Sophia Sweet", "p318": "Alan Clear",
    "p329": "Clara Smooth", "p334": "Hazel Soft", "p335": "Rosa Gentle",
    "p339": "Fiona Calm", "p340": "Max Warm", "p360": "Grace Mature"
}

# -----------------------------------------
# CORE FUNCTIONS
# -----------------------------------------

def listen_for_speech():
    """Intelligent speech detection with natural pause handling"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Enhanced recognition settings for natural speech
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.2  # Longer pause tolerance
    recognizer.operation_timeout = None
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8
    
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
        # Multiple attempts to capture complete thoughts
        attempts = 0
        collected_speech = []
        
        while attempts < 3:  # Max 3 listening cycles
            try:
                with mic as source:
                    # Listen with generous timeout for thinking pauses
                    audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
                
                # Transcribe the audio segment
                text_segment = recognizer.recognize_google(audio)
                
                if text_segment.strip():
                    collected_speech.append(text_segment.strip())
                    
                    # Analyze if this seems like a complete thought
                    if is_complete_statement(text_segment, attempts):
                        break
                    else:
                        # Brief pause to see if user continues
                        time.sleep(0.8)
                        attempts += 1
                        continue
                
            except sr.WaitTimeoutError:
                # If we have some speech collected, analyze it
                if collected_speech:
                    break
                # If first attempt and no speech, return None
                if attempts == 0:
                    return None
                # Otherwise try again
                attempts += 1
                continue
                
            except sr.UnknownValueError:
                if collected_speech:
                    break
                attempts += 1
                continue
        
        # Combine all collected speech segments
        if collected_speech:
            full_text = " ".join(collected_speech)
            return full_text if len(full_text.strip()) > 2 else None
        
        return None
        
    except Exception:
        return None

def is_complete_statement(text, attempt_number):
    """Analyze if the speech segment seems like a complete thought"""
    text = text.strip().lower()
    
    # Always assume first long statement is complete if it's substantial
    if attempt_number == 0 and len(text.split()) >= 5:
        return True
    
    # Check for clear ending indicators
    ending_indicators = [
        # Questions
        text.endswith('?'),
        # Statements with clear conclusions
        any(text.endswith(ending) for ending in ['.', '!', 'thanks', 'please', 'okay', 'ok']),
        # Command/request patterns
        any(text.startswith(start) for start in ['can you', 'could you', 'please', 'tell me', 'what is', 'how do', 'where is']),
        # Greeting/social patterns
        any(word in text for word in ['hello', 'hi', 'goodbye', 'bye', 'thank you', 'thanks']),
        # Complete sentence structures
        text.count(' ') >= 4 and any(text.endswith(end) for end in ['it', 'that', 'this', 'now', 'here', 'there'])
    ]
    
    # Check for incomplete/continuing indicators
    continuing_indicators = [
        text.endswith(('and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'also', 'then')),
        text.endswith(('the', 'a', 'an', 'my', 'your', 'his', 'her', 'their', 'our')),
        text.endswith(('is', 'are', 'was', 'were', 'will', 'would', 'could', 'should', 'might')),
        len(text.split()) <= 3,  # Very short phrases usually incomplete
        text.endswith(('uh', 'um', 'er', 'ah')),  # Filler words
    ]
    
    # If there are clear continuing indicators, don't treat as complete
    if any(continuing_indicators):
        return False
    
    # If there are clear ending indicators, treat as complete
    if any(ending_indicators):
        return True
    
    # For longer statements without clear indicators, assume complete after first attempt
    if len(text.split()) >= 6:
        return True
    
    # Default: not complete, continue listening
    return False

def process_with_llama4_maverick(text):
    """Process text with Llama-4-Maverick - silent operation"""
    try:
        resp = client.chat.completions.create(
            model="provider-3/llama-4-maverick",  # Updated model name
            messages=[
                {"role": "system", "content": "You are a helpful, friendly, and conversational assistant. Keep responses concise and natural, as if speaking to a friend. Be innovative in your thinking and provide creative solutions."},
                {"role": "user", "content": text}
            ],
            max_tokens=150,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")  # Debug info
        return "I didn't catch that. Could you repeat?"

def text_to_speech_with_interruption(text, speaker):
    """Convert text to speech with interruption detection"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
        
        tts_model.tts_to_file(text=text, file_path=wav_path, speaker=speaker, speed=1.3)
        
        sound = pygame.mixer.Sound(wav_path)
        sound.play()
        
        # Monitor for interruption while playing
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        # Set sensitive settings for interruption detection
        recognizer.energy_threshold = 200  # Lower threshold for quick detection
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.3   # Very quick detection
        
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
        except:
            pass
        
        # Check for interruption every 0.5 seconds
        while pygame.mixer.get_busy():
            try:
                with mic as source:
                    # Quick listen for any speech
                    audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                
                # If we detect any speech, stop the AI and process interruption
                try:
                    interrupt_text = recognizer.recognize_google(audio)
                    if interrupt_text.strip() and len(interrupt_text.strip()) > 1:
                        # Stop the current playback immediately
                        pygame.mixer.stop()
                        os.unlink(wav_path)
                        
                        # Return the interruption text to be processed
                        return interrupt_text.strip()
                
                except sr.UnknownValueError:
                    # Unclear speech, but still might be interruption
                    if len(audio.frame_data) > 1000:  # Sufficient audio data
                        pygame.mixer.stop()
                        os.unlink(wav_path)
                        return "INTERRUPTED"  # Signal that user interrupted
                    
            except sr.WaitTimeoutError:
                # No interruption detected, continue playing
                continue
            except:
                # Any other error, continue playing
                continue
        
        # Clean up if playback completed normally
        os.unlink(wav_path)
        return True
        
    except Exception:
        return False

def listen_for_speech_quick():
    """Quick speech detection for interruptions"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Quick settings for immediate response
    recognizer.energy_threshold = 250
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        
        text = recognizer.recognize_google(audio)
        return text
    except:
        return None

# -----------------------------------------
# STREAMLIT APP
# -----------------------------------------

st.set_page_config(page_title="Voice Assistant - Llama-4-Maverick", layout="centered")

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5em;
        margin-bottom: 30px;
    }
    .status {
        text-align: center;
        font-size: 1.5em;
        color: #28B463;
        margin: 20px 0;
    }
    .model-info {
        text-align: center;
        color: #8E44AD;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎙️ Voice Assistant</h1>', unsafe_allow_html=True)
st.markdown('<div class="model-info">🚀 Powered by Llama-4-Maverick</div>', unsafe_allow_html=True)

# Voice selection
selected_name = st.selectbox("🎵 Choose Voice:", options=list(speakers_map.values()))
selected_speaker = [k for k, v in speakers_map.items() if v == selected_name][0]

st.markdown("---")

# Initialize session state
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = False
if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = st.empty()

# Main interface - Start/Stop conversation
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if not st.session_state.conversation_mode:
        if st.button("▶️ Start Conversation", use_container_width=True, type="primary"):
            st.session_state.conversation_mode = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Conversation", use_container_width=True):
            st.session_state.conversation_mode = False
            st.session_state.status_placeholder.empty()
            st.rerun()

# Continuous conversation loop
if st.session_state.conversation_mode:
    # Show listening status
    st.session_state.status_placeholder.markdown('<div class="status">🎤 Listening intelligently...</div>', unsafe_allow_html=True)
    
    # Listen for complete speech with intelligence
    user_input = listen_for_speech()
    
    if user_input and len(user_input.strip()) > 2:
        # Update status
        st.session_state.status_placeholder.markdown('<div class="status">🧠 Maverick thinking...</div>', unsafe_allow_html=True)
        
        # Get AI response using Llama-4-Maverick
        ai_response = process_with_llama4_maverick(user_input)
        
        if ai_response:
            # Update status
            st.session_state.status_placeholder.markdown('<div class="status">🔊 Responding... (you can interrupt)</div>', unsafe_allow_html=True)
            
            # Convert to speech with interruption detection
            result = text_to_speech_with_interruption(ai_response, selected_speaker)
            
            # Check if user interrupted
            if isinstance(result, str) and result != "True":
                if result == "INTERRUPTED":
                    # User interrupted but we couldn't catch what they said
                    st.session_state.status_placeholder.markdown('<div class="status">🎤 You interrupted - listening...</div>', unsafe_allow_html=True)
                    # Quick listen for what they want to say
                    interrupt_input = listen_for_speech_quick()
                    if interrupt_input:
                        # Process the interruption immediately
                        st.session_state.status_placeholder.markdown('<div class="status">🧠 Got it...</div>', unsafe_allow_html=True)
                        new_response = process_with_llama4_maverick(interrupt_input)
                        if new_response:
                            st.session_state.status_placeholder.markdown('<div class="status">🔊 New response...</div>', unsafe_allow_html=True)
                            text_to_speech_with_interruption(new_response, selected_speaker)
                else:
                    # We caught what they said during interruption
                    st.session_state.status_placeholder.markdown('<div class="status">🧠 Processing interruption...</div>', unsafe_allow_html=True)
                    interrupt_response = process_with_llama4_maverick(result)
                    if interrupt_response:
                        st.session_state.status_placeholder.markdown('<div class="status">🔊 Responding to interruption...</div>', unsafe_allow_html=True)
                        text_to_speech_with_interruption(interrupt_response, selected_speaker)
            
            # Brief pause before next listening cycle
            time.sleep(1)
        else:
            # Error - speak error message
            text_to_speech_with_interruption("I didn't catch that completely. Could you repeat?", selected_speaker)
            time.sleep(1)
    else:
        # No complete speech detected - continue listening with shorter pause
        time.sleep(0.3)
    
    # Continue the conversation loop
    st.rerun()

# Instructions (minimal)
st.markdown("---")
st.markdown("**💡 Smart Listening: Take your time, breathe, think - Maverick knows when you're finished speaking!**")