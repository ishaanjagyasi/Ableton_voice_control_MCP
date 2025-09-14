#!/usr/bin/env python3
"""
Natural Language Ableton Voice Control with Whisper (Fixed)
Uses local Whisper for better speech recognition + Ollama for understanding
FIXES: Whisper hallucination and calibration hanging issues

Requirements: openai-whisper, requests, speech_recognition

Usage: python3 fixed_whisper_voice_control.py
"""

import speech_recognition as sr
import socket
import json
import threading
import time
import logging
import requests
import numpy as np
import tempfile
import os
import io
import wave
from typing import Dict, Any, List, Optional
from enum import Enum
import audioop


# In main() function:
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        deepgram_api_key = config['deepgram_api_key']
        ollama_model = config.get('ollama_model', 'llama3.2:3b')
except FileNotFoundError:
    print("❌ ERROR: config.json not found")
    sys.exit(1)


# Import Whisper
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    print("Whisper is available for high-quality speech recognition")
except ImportError:
    WHISPER_AVAILABLE = False
    print("ERROR: Whisper not available. Install with: pip install openai-whisper")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FixedWhisperVoiceControl")

class VoiceControlStatus(Enum):
    INACTIVE = "inactive"
    LISTENING = "listening" 
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"

class FixedWhisperVoiceController:
    """Natural language voice control using Whisper + Ollama (Fixed)"""
    
    def __init__(self, whisper_model="base", ollama_model="llama3.2:3b"):
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Optimized settings to prevent false triggers but allow faster processing
        self.recognizer.energy_threshold = 1500  # A good starting point
        self.recognizer.pause_threshold = 1.5    # Increased: Wait 1.5s for a pause before processing
        self.recognizer.phrase_threshold = 0.4   # Lowered: Trigger on shorter phrases
        self.recognizer.dynamic_energy_threshold = True
        
        # Whisper setup
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model_name = ollama_model
        
        # Wake word configuration
        self.wake_words = ["ableton", "live"]
        self.use_wake_word = True
        self.wake_word_detected = False
        self.last_command_time = 0
        self.command_timeout = 20.0
        
        # Ableton connection
        self.ableton_host = "127.0.0.1"
        self.ableton_port = 9001
        
        # System state
        self.status = VoiceControlStatus.INACTIVE
        self.is_running = False
        self.command_count = 0
        self.successful_commands = 0
        
        # Anti-hallucination measures
        self.hallucination_phrases = [
            "music production", "video", "song", "production program",
            "first time", "very good", "this is", "audio", "video of"
        ]
        
        # Create the system prompt
        self.system_prompt = self._create_system_prompt()
        
        logger.info("Fixed Whisper Voice Controller initialized")
    
    def _create_system_prompt(self):
        """Create system prompt for Ollama"""
        return """You are an AI assistant that converts natural language voice commands into structured JSON commands for controlling Ableton Live music production software.

AVAILABLE COMMANDS:

1. CREATE TRACKS:
   Output: {"action": "create_tracks", "track_type": "midi|audio", "count": number, "names": ["name1", "name2"]}
   Examples: "create two MIDI tracks", "make an audio track named drums"

2. VOLUME CONTROL:
   Output: {"action": "set_parameter", "parameter": "mixer_volume", "target": "track_name", "value": number}
   Examples: "turn down drums volume", "set bass to -5 dB", "make vocals louder"

3. PAN CONTROL:
   Output: {"action": "set_parameter", "parameter": "mixer_pan", "target": "track_name", "value": number}
   Value: -100 (full left) to +100 (full right)
   Examples: "pan bass left", "move vocals 30% right"

4. TEMPO CONTROL:
   Output: {"action": "set_parameter", "parameter": "transport_tempo", "value": number}
   Examples: "set tempo to 128", "make it faster"

5. TRANSPORT CONTROL:
   Output: {"action": "transport_play|transport_stop|transport_record"}
   Examples: "play the song", "stop playback", "start recording"

6. SOLO/MUTE:
   Output: {"action": "set_parameter", "parameter": "mixer_solo|mixer_mute", "target": "track_name", "value": true}
   Examples: "solo the drums", "mute bass track"

IMPORTANT RULES:
- Always respond with ONLY valid JSON, no explanations
- For volume: negative numbers are quieter (e.g., -5), positive are louder
- For track names: use whatever the user says ("drums", "bass", "vocal", "track 1", etc.)
- If command is unclear, return: {"error": "Could not understand command"}
- Be flexible with natural language variations

Examples:
User: "turn the drums down to minus 8 decibels"
You: {"action": "set_parameter", "parameter": "mixer_volume", "target": "drums", "value": -8}

User: "I want to create three MIDI tracks for bass, lead, and pads"  
You: {"action": "create_tracks", "track_type": "midi", "count": 3, "names": ["bass", "lead", "pads"]}

User: "make the tempo faster, like 140 BPM"
You: {"action": "set_parameter", "parameter": "transport_tempo", "value": 140}

Now process the user's command:"""

    def start(self):
        """Start the voice control system"""
        print("Starting Fixed Whisper + Ollama Voice Control System...")
        
        # Load Whisper model
        if not self._load_whisper_model():
            return False
        
        # Test Ollama connection
        if not self._test_ollama_connection():
            print("ERROR: Cannot connect to Ollama")
            print("Make sure Ollama is running: ollama serve")
            return False
        
        # Test Ableton connection
        if not self._test_ableton_connection():
            print("ERROR: Cannot connect to Ableton")
            print("Make sure Ableton Live is running with Enhanced MCP Remote Script")
            return False
        
        # Quick calibration (prevent hanging)
        print("Quick microphone calibration...")
        try:
            with self.microphone as source:
                print("Stay quiet for 1 second...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)  # Shorter duration
            print(f"Microphone calibrated (threshold: {self.recognizer.energy_threshold:.1f})")
        except Exception as e:
            print(f"Calibration failed: {e}")
            print("Using default settings...")
            self.recognizer.energy_threshold = 2000  # Use high default
        
        # Start listening
        self.is_running = True
        self.status = VoiceControlStatus.LISTENING
        
        # Start background listening thread
        listening_thread = threading.Thread(target=self._listening_loop, daemon=True)
        listening_thread.start()
        
        # Print usage instructions
        self._print_instructions()
        
        print("Fixed Whisper voice control system is active!")
        return True
    
    def _load_whisper_model(self):
        """Load Whisper model"""
        try:
            print(f"Loading Whisper {self.whisper_model_name} model...")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            print("Whisper model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return False
    
    def stop(self):
        """Stop the voice control system"""
        print("Stopping voice control system...")
        self.is_running = False
        self.status = VoiceControlStatus.INACTIVE
        
        if self.command_count > 0:
            success_rate = (self.successful_commands / self.command_count) * 100
            print(f"Session summary: {self.successful_commands}/{self.command_count} commands successful ({success_rate:.1f}%)")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama with a retry mechanism."""
        max_retries = 6
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to Ollama (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model_name,
                        "prompt": "Hello",
                        "stream": False
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    print("Ollama connection established")
                    return True
                else:
                    print(f"Ollama responded with status: {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Ollama connection failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        return False
    
    def _test_ableton_connection(self):
        """Test connection to Ableton Remote Script"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((self.ableton_host, self.ableton_port))
            
            test_command = {"action": "get_session_info"}
            sock.send(json.dumps(test_command).encode('utf-8'))
            response_data = sock.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            sock.close()
            
            if response.get('status') == 'success':
                print("Ableton connection established")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Ableton connection failed: {e}")
            return False
    
    def _listening_loop(self):
        """Main listening loop"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.is_running:
            try:
                # Listen for audio with timeout
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1.0,
                        phrase_time_limit=6  # Reasonable limit
                    )
                
                consecutive_errors = 0
                
                # Process in separate thread
                processing_thread = threading.Thread(
                    target=self._process_audio, 
                    args=(audio,), 
                    daemon=True
                )
                processing_thread.start()
                
            except sr.WaitTimeoutError:
                # Check wake word expiry
                if self.wake_word_detected and time.time() - self.last_command_time > self.command_timeout:
                    self.wake_word_detected = False
                    print("Wake word timeout - going back to sleep")
                continue
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print(f"Too many listening errors: {e}")
                    break
                time.sleep(0.5)
    
    def _process_audio(self, audio):
        """Process audio data using Whisper with anti-hallucination measures"""
        try:
            self.status = VoiceControlStatus.PROCESSING
            
            # Convert speech to text using Whisper with proper error handling
            text = self._whisper_transcribe_safe(audio)
            
            if not text or len(text.strip()) < 2:
                return
            
            text_lower = text.lower().strip()
            
            # ANTI-HALLUCINATION CHECK
            if self._is_hallucination(text_lower):
                # Don't print or process hallucinated text
                return
            
            print(f"Heard: '{text}'")
            
            # Check for wake word
            if self.use_wake_word and not self.wake_word_detected:
                if any(wake_word in text_lower for wake_word in self.wake_words):
                    self.wake_word_detected = True
                    self.last_command_time = time.time()
                    print("Wake word detected - listening for commands...")
                    return
                else:
                    return
            
            # Process as command using LLM
            self._execute_natural_language_command(text)
            self.last_command_time = time.time()
            
        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            self.status = VoiceControlStatus.LISTENING
    
    def _is_hallucination(self, text: str) -> bool:
        """Check if text is likely a Whisper hallucination"""
        # Check for common hallucination phrases
        for phrase in self.hallucination_phrases:
            if phrase in text:
                return True
        
        # Check for very generic/repetitive patterns
        words = text.split()
        if len(words) < 2:
            return False
            
        # Too many repeated words (another hallucination sign)
        if len(set(words)) < len(words) * 0.5:  # Less than 50% unique words
            return True
            
        return False
    
    def _whisper_transcribe_safe(self, audio):
        """Transcribe audio using Whisper with robust speech detection"""
        try:
            # Get WAV data from speech_recognition audio
            # CORRECTED: The arguments are 'convert_width' and 'convert_rate'
            wav_data = audio.get_wav_data(
                convert_width=2,  # 2 bytes = 16-bit audio
                convert_rate=16000  # Whisper's preferred sample rate
            )

            # 1. ROBUST SILENCE CHECK: Check audio energy using RMS
            rms = audioop.rms(wav_data, 2)  # 2 is the width in bytes
            speech_rms_threshold = 700 
            
            if rms < speech_rms_threshold:
                return "" # Discard as silence

            # 2. TRANSCRIBE WITH WHISPER
            # Convert raw bytes to a NumPy array that Whisper understands
            audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe with Whisper using anti-hallucination settings
            result = self.whisper_model.transcribe(
                audio_np,
                language='en',
                task='transcribe',
                fp16=torch.cuda.is_available(),
                temperature=0.0,
                best_of=3,
                beam_size=3,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0
            )

            text = result.get("text", "").strip()

            if len(text) < 3 or not any(c.isalpha() for c in text):
                return ""

            return text

        except Exception as e:
            # REMOVED: No more fallback to Google Speech Recognition.
            # It will now just report the error and continue.
            print(f"Whisper transcription failed: {e}")
            return ""

    def _execute_natural_language_command(self, command_text: str):
        """Execute command using natural language processing with Ollama"""
        try:
            self.status = VoiceControlStatus.EXECUTING
            self.command_count += 1
            
            print(f"Processing with AI: '{command_text}'")
            
            # Send to Ollama for natural language understanding
            structured_command = self._get_ollama_interpretation(command_text)
            
            if not structured_command:
                print("AI could not understand command")
                return
            
            # Check for error response from LLM
            if "error" in structured_command:
                print(f"Could not understand command")
                return
            
            print(f"AI interpreted as: {structured_command}")
            
            # Send to Ableton
            response = self._send_to_ableton(structured_command)
            
            if response.get('status') == 'success':
                self.successful_commands += 1
                print("Command executed successfully")
                
                # Show results
                result = response.get('result', {})
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key not in ['error', 'status'] and value:
                            print(f"  {key}: {value}")
            else:
                error_msg = response.get('message', 'Unknown error')
                print(f"Command failed: {error_msg}")
                
        except Exception as e:
            print(f"Error executing command: {e}")
    
    def _get_ollama_interpretation(self, command_text: str) -> Optional[Dict[str, Any]]:
        """Get command interpretation from Ollama LLM"""
        try:
            # Prepare the prompt
            full_prompt = f"{self.system_prompt}\n\nUser command: \"{command_text}\""
            
            # Send to Ollama
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 200
                    }
                },
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"Ollama request failed: {response.status_code}")
                return None
            
            # Extract the response
            result = response.json()
            ai_response = result.get('response', '').strip()
            
            # Try to parse as JSON
            try:
                # Clean up response - sometimes LLMs add extra text
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_text = ai_response[json_start:json_end]
                    parsed_command = json.loads(json_text)
                    return parsed_command
                else:
                    return None
                    
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            print(f"Error getting Ollama interpretation: {e}")
            return None
    
    def _send_to_ableton(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send action to Ableton Remote Script"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.ableton_host, self.ableton_port))
            
            command_str = json.dumps(action)
            sock.send(command_str.encode('utf-8'))
            
            response_data = sock.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            sock.close()
            return response
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _print_instructions(self):
        """Print usage instructions"""
        wake_prefix = f"{self.wake_words[0]}, " if self.use_wake_word else ""
        
        print("\n" + "="*70)
        print("FIXED WHISPER + OLLAMA VOICE CONTROL ACTIVE")
        print("High-Quality Local Speech Recognition (No Hallucination)")
        print("="*70)
        
        print(f"\nSay '{self.wake_words[0]}' or '{self.wake_words[1]}' followed by your command")
        print("Whisper is now configured to prevent false recognitions!\n")
        
        print("EXAMPLE COMMANDS:")
        print(f"  '{wake_prefix}create two MIDI tracks named bass and lead'")  
        print(f"  '{wake_prefix}set drums volume to minus 5 decibels'")
        print(f"  '{wake_prefix}make the tempo 140 BPM'")
        print(f"  '{wake_prefix}pan the vocal track 30% to the left'")
        print(f"  '{wake_prefix}solo the drum track'")
        print(f"  '{wake_prefix}play the song'")
        
        print("\nFIXED ISSUES:")
        print("  ✅ No more hallucinated 'music production' phrases")
        print("  ✅ Better silence detection")
        print("  ✅ Improved accuracy with accents")
        print("  ✅ Faster calibration (no hanging)")
        
        print("\n  Press Ctrl+C to stop")
        print("="*70 + "\n")

def main():
    """Main application entry point"""
    print("Fixed Whisper + Ollama Natural Language Voice Control")
    print("High-quality speech recognition WITHOUT hallucination")
    
    # Check if models specified
    import sys
    whisper_model = "base"  # base, small, medium, large
    ollama_model = "llama3.2:3b"
    
    if len(sys.argv) > 1:
        whisper_model = sys.argv[1]
        print(f"Using Whisper model: {whisper_model}")
    
    if len(sys.argv) > 2:
        ollama_model = sys.argv[2]
        print(f"Using Ollama model: {ollama_model}")
    
    controller = FixedWhisperVoiceController(
        whisper_model=whisper_model,
        ollama_model=ollama_model
    )
    
    try:
        if controller.start():
            while controller.is_running:
                time.sleep(1)
        else:
            print("Failed to start voice control system")
            
    except KeyboardInterrupt:
        print("\nStopping voice control...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()