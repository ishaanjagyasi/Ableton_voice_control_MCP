#!/usr/bin/env python3
"""
Deepgram + Ollama Natural Language Ableton Voice Control
High-quality real-time speech recognition with Deepgram SDK v4+

Requirements: 
- pip install "deepgram-sdk>=4.0" pyaudio requests
- Deepgram API key (get free $200 credits at https://deepgram.com)
- Ollama running locally
- Ableton with Enhanced MCP Remote Script

Usage: python3 Voice_command_script_new_13Sept25.py
"""

import os
import sys
import json
import socket
import threading
import time
import logging
import requests
import pyaudio
import asyncio
from typing import Dict, Any, Optional
from enum import Enum

# Import Deepgram SDK v4+
from deepgram import DeepgramClient
from deepgram.clients.live.v1 import LiveClient, LiveTranscriptionEvents

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepgramAbletonControl")

class VoiceControlStatus(Enum):
    INACTIVE = "inactive"
    LISTENING = "listening" 
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"

class DeepgramAbletonController:
    """Natural language voice control using Deepgram + Ollama (Modern SDK Version)"""
    
    def __init__(self, deepgram_api_key: str, ollama_model: str = "llama3.2:3b"):
        self.deepgram_api_key = deepgram_api_key
        self.deepgram_client = DeepgramClient(deepgram_api_key)
        self.dg_connection: LiveClient = None
        
        self.sample_rate = 16000
        self.chunk_size = 8000
        self.channels = 1
        self.format = pyaudio.paInt16
        
        self.audio_queue = asyncio.Queue()
        self.loop = None
        
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model_name = ollama_model
        
        self.wake_words = ["ableton", "live", "hey ableton"]
        self.use_wake_word = True
        self.wake_word_detected = False
        self.last_command_time = 0
        self.command_timeout = 20.0
        
        self.ableton_host = "127.0.0.1"
        self.ableton_port = 9001
        
        self.status = VoiceControlStatus.INACTIVE
        self.is_running = False
        self.command_count = 0
        self.successful_commands = 0
        
        self.pyaudio = pyaudio.PyAudio()
        self.audio_stream = None
        
        self.system_prompt = self._create_system_prompt()
        
        logger.info("Deepgram Ableton Controller initialized")

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

Now process the user's command:"""

    async def start(self):
        """Start the voice control system"""
        print("\n🎙️  Starting Deepgram + Ollama Voice Control System...")
        
        self.loop = asyncio.get_running_loop()
        
        if not self._test_ollama_connection() or not self._test_ableton_connection():
            return False

        try:
            print("🔗 Connecting to Deepgram...")
            self.dg_connection = self.deepgram_client.listen.asyncwebsocket.v("1")
            
            self.dg_connection.on(LiveTranscriptionEvents.Open, self.on_open)
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Close, self.on_close)

            options = {
                "model": "nova-2", "encoding": "linear16",
                "sample_rate": self.sample_rate, "channels": self.channels,
                "punctuate": True, "smart_format": True
            }
            await self.dg_connection.start(options)
        
        except Exception as e:
            print(f"❌ Deepgram connection failed: {e}")
            return False

        self.audio_stream = self.pyaudio.open(
            format=self.format, channels=self.channels, rate=self.sample_rate,
            input=True, frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.audio_stream.start_stream()
        print("🎤 Microphone activated")
        
        asyncio.create_task(self._process_audio_queue())
        
        self.is_running = True
        self.status = VoiceControlStatus.LISTENING
        self._print_instructions()
        print("\n✨ Voice control system is active and listening!")
        return True

    async def _process_audio_queue(self):
        while self.is_running:
            try:
                data = await self.audio_queue.get()
                if data is None: break
                if self.dg_connection:
                    await self.dg_connection.send(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio queue: {e}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Puts microphone audio into the asyncio queue."""
        try:
            if self.loop:
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, in_data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        return (None, pyaudio.paContinue)

    # MODIFIED: Event handlers now use a universal signature (*args, **kwargs) to prevent TypeErrors
    async def on_open(self, *args, **kwargs):
        logger.info(f"✅ Deepgram connection opened.")
    
    async def on_error(self, *args, **kwargs):
        error = kwargs.get('error')
        logger.error(f"❌ Deepgram error: {error}")

    async def on_close(self, *args, **kwargs):
        logger.info("📡 Deepgram connection closed.")

    async def on_transcript(self, *args, **kwargs):
        result = kwargs.get('result')
        try:
            if result and result.is_final and len(result.channel.alternatives) > 0:
                transcript = result.channel.alternatives[0].transcript.strip()
                if transcript:
                    print(f"\n🎤 Heard: '{transcript}'")
                    self._process_transcript(transcript)
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
    
    def _process_transcript(self, text: str):
        text_lower = text.lower().strip()
        if self.use_wake_word and not self.wake_word_detected:
            if any(word in text_lower for word in self.wake_words):
                self.wake_word_detected = True
                self.last_command_time = time.time()
                print("✅ Wake word detected - listening for commands...")
            return
        if self.wake_word_detected and (time.time() - self.last_command_time > self.command_timeout):
            self.wake_word_detected = False
            print("⏰ Wake word timeout - say wake word again")
            return
        self._execute_natural_language_command(text)
        self.last_command_time = time.time()
    
    def _test_ollama_connection(self):
        try:
            print("🔗 Testing Ollama connection...")
            response = requests.post(
                self.ollama_url, json={"model": self.ollama_model_name, "prompt": "Hello", "stream": False}, timeout=10)
            if response.status_code == 200:
                print("✅ Ollama connection established")
                return True
            else:
                print(f"❌ Ollama responded with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            return False
    
    def _test_ableton_connection(self):
        try:
            print("🔗 Testing Ableton connection...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((self.ableton_host, self.ableton_port))
            sock.send(json.dumps({"action": "get_session_info"}).encode('utf-8'))
            response = json.loads(sock.recv(4096).decode('utf-8'))
            sock.close()
            if response.get('status') == 'success':
                print("✅ Ableton connection established")
                return True
            else: return False
        except Exception as e:
            print(f"❌ Ableton connection failed: {e}")
            return False
    
    def _execute_natural_language_command(self, command_text: str):
        try:
            self.status = VoiceControlStatus.EXECUTING
            self.command_count += 1
            print(f"🤖 Processing with AI: '{command_text}'")
            structured_command = self._get_ollama_interpretation(command_text)
            if not structured_command or "error" in structured_command:
                print(f"❌ Could not understand command: {structured_command.get('error', '')}")
                return
            print(f"📋 AI interpreted as: {structured_command}")
            response = self._send_to_ableton(structured_command)
            if response.get('status') == 'success':
                self.successful_commands += 1
                print("✅ Command executed successfully")
                result = response.get('result', {})
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key not in ['error', 'status'] and value:
                            print(f"   {key}: {value}")
            else:
                print(f"❌ Command failed: {response.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Error executing command: {e}")
        finally:
            self.status = VoiceControlStatus.LISTENING
    
    def _get_ollama_interpretation(self, command_text: str) -> Optional[Dict[str, Any]]:
        try:
            full_prompt = f"{self.system_prompt}\n\nUser command: \"{command_text}\""
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model_name, "prompt": full_prompt, "stream": False,
                      "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 200}},
                timeout=15)
            if response.status_code != 200: return None
            ai_response = response.json().get('response', '').strip()
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(ai_response[json_start:json_end])
            return None
        except Exception as e:
            print(f"❌ Error getting Ollama interpretation: {e}")
            return None
    
    def _send_to_ableton(self, action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.ableton_host, self.ableton_port))
            sock.send(json.dumps(action).encode('utf-8'))
            response = json.loads(sock.recv(4096).decode('utf-8'))
            sock.close()
            return response
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _print_instructions(self):
        wake_prefix = f"{self.wake_words[0]}, " if self.use_wake_word else ""
        print("\n" + "="*70)
        print("🎵 DEEPGRAM + OLLAMA VOICE CONTROL FOR ABLETON LIVE")
        print("="*70 + f"\n\n🎤 Say '{self.wake_words[0]}' or '{self.wake_words[1]}' followed by your command")
        print("   Powered by Deepgram's real-time speech recognition\n")
        print("📝 EXAMPLE COMMANDS:")
        print(f"  '{wake_prefix}create two MIDI tracks named bass and lead'")  
        print(f"  '{wake_prefix}set drums volume to minus 5 decibels'")
        print(f"  '{wake_prefix}make the tempo 140 BPM'")
        print(f"  '{wake_prefix}pan the vocal track 30% to the left'")
        print(f"  '{wake_prefix}solo the drum track'")
        print(f"  '{wake_prefix}play the song'")
        print("\n✨ DEEPGRAM ADVANTAGES:\n  • Ultra-low latency (< 300ms)\n  • Superior accuracy with music production terms")
        print("  • No hallucinations or false triggers\n  • Works great with background noise")
        print("\n⌨️  Press Ctrl+C to stop\n" + "="*70 + "\n")
    
    async def stop(self):
        print("\n🛑 Stopping voice control system...")
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.pyaudio.terminate()
        if self.dg_connection:
            await self.dg_connection.finish()
        if self.command_count > 0:
            success_rate = (self.successful_commands / self.command_count) * 100
            print(f"\n📊 Session summary:\n   Commands processed: {self.command_count}\n   Successful: {self.successful_commands}\n   Success rate: {success_rate:.1f}%")

async def main():
    print("🎵 Deepgram + Ollama Natural Language Voice Control for Ableton")
    print("   High-quality real-time speech recognition\n")
    deepgram_api_key = None
    ollama_model = "llama3.2:3b"
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            deepgram_api_key = config.get('deepgram_api_key')
            ollama_model = config.get('ollama_model', 'llama3.2:3b')
        print(f"✅ Loaded configuration from config.json\n📦 Using Ollama model: {ollama_model}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if not deepgram_api_key and len(sys.argv) > 1:
        deepgram_api_key = sys.argv[1]
        if len(sys.argv) > 2: ollama_model = sys.argv[2]
    if not deepgram_api_key:
        print("❌ ERROR: Deepgram API key required\n\n📝 Either:\n   1. Create config.json with your API key\n   2. Run: python3 script.py YOUR_DEEPGRAM_API_KEY\n\n💡 Get your free API key ($200 credits) at:\n   https://console.deepgram.com/signup\n")
        sys.exit(1)
    controller = DeepgramAbletonController(deepgram_api_key=deepgram_api_key, ollama_model=ollama_model)
    try:
        if await controller.start():
            while controller.is_running:
                await asyncio.sleep(0.1)
        else:
            print("❌ Failed to start voice control system")
    except KeyboardInterrupt:
        print("\n⌨️  Keyboard interrupt received")
    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())