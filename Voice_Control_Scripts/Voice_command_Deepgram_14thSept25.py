#!/usr/bin/env python3
"""
Context-Aware Deepgram + Ollama Voice Control with Real-Time Session State
AI is now aware of the complete Ableton session state for intelligent command processing

Requirements: 
- pip install "deepgram-sdk>=4.0" pyaudio requests
- Deepgram API key
- Ollama running locally with llama3.2:3b model
- Ableton with Context-Aware MCP Remote Script
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
logger = logging.getLogger("ContextAwareAbletonControl")

class VoiceControlStatus(Enum):
    INACTIVE = "inactive"
    LISTENING = "listening" 
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"

class ContextAwareDeepgramAbletonController:
    """Context-aware voice control with real-time session state awareness"""
    
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
        self.command_timeout = 25.0
        
        self.ableton_host = "127.0.0.1"
        self.ableton_port = 9001
        
        self.status = VoiceControlStatus.INACTIVE
        self.is_running = False
        self.command_count = 0
        self.successful_commands = 0
        
        self.pyaudio = pyaudio.PyAudio()
        self.audio_stream = None
        
        # Session state tracking
        self.current_session_state = {}
        self.last_state_update = 0
        self.state_update_interval = 3.0  # Update every 3 seconds
        
        self.base_system_prompt = self._create_base_system_prompt()
        
        logger.info("Context-Aware Deepgram Ableton Controller initialized")

    def _create_base_system_prompt(self):
        """Create the base system prompt without session context"""
        return """You are an AI assistant that converts natural language voice commands into structured JSON commands for controlling Ableton Live. You have FULL AWARENESS of the current session state.

AVAILABLE COMMANDS:

1. CREATE TRACKS:
   {"action": "create_tracks", "track_type": "midi" OR "audio", "count": number, "names": ["name1"]}

2. VOLUME CONTROL:
   {"action": "set_parameter", "parameter": "mixer_volume", "target": "track_name", "value": dB_value}
   Range: -70 to +6 dB

3. PAN CONTROL:
   {"action": "set_parameter", "parameter": "mixer_pan", "target": "track_name", "value": pan_value}
   Range: -100 (full left) to +100 (full right)

4. AUDIO EFFECTS:
   Add: {"action": "add_audio_effect", "track": "track_name", "effect": "effect_name"}
   Remove: {"action": "remove_audio_effect", "track": "track_name", "effect_index": -1}
   
   Supported effects: reverb, compressor, eq8, equalizer, eq3, delay, chorus, saturator, limiter, gate

5. TRANSPORT:
   {"action": "transport_play"} / {"action": "transport_stop"} / {"action": "transport_record"}
   {"action": "set_parameter", "parameter": "transport_tempo", "value": BPM}

6. SOLO/MUTE:
   {"action": "set_parameter", "parameter": "mixer_solo|mixer_mute", "target": "track_name", "value": true/false}

7. GET INFORMATION:
   {"action": "get_session_info"} / {"action": "get_track_effects", "track": "track_name"}

INTELLIGENT BEHAVIOR RULES:
- Use EXACT track names from the current session
- If user says "drums" but there's only "Drums Track", use "Drums Track"
- If user doesn't specify a track, use the currently selected track
- Prevent duplicate effects (check if effect already exists)
- Suggest alternatives when appropriate
- Be context-aware about current session state

CRITICAL RULES:
- Always respond with ONLY valid JSON
- Use exact track names from session state
- Pan: -100 (left) to +100 (right)
- Volume: -70 to +6 dB
- If unclear, return: {"error": "Could not understand command"}

Now process the user's command using the current session context:"""

    def get_current_session_state(self) -> Dict[str, Any]:
        """Get complete session state from Ableton"""
        try:
            response = self._send_to_ableton({"action": "get_detailed_session_info"})
            if response.get('status') == 'success':
                return response.get('result', {})
            return {}
        except Exception as e:
            logger.error(f"Failed to get session state: {e}")
            return {}

    def _create_context_aware_prompt(self, command_text: str) -> str:
        """Create system prompt with current session context"""
        # Get fresh session state if needed
        current_time = time.time()
        if current_time - self.last_state_update > self.state_update_interval:
            self.current_session_state = self.get_current_session_state()
            self.last_state_update = current_time

        # Build session context
        session_context = "\n=== CURRENT ABLETON SESSION STATE ===\n"
        
        if self.current_session_state:
            session_context += f"Tempo: {self.current_session_state.get('tempo', 'Unknown')} BPM\n"
            session_context += f"Playing: {self.current_session_state.get('is_playing', False)}\n"
            session_context += f"Selected Track: {self.current_session_state.get('selected_track', 'None')}\n"
            session_context += f"Total Tracks: {len(self.current_session_state.get('tracks', []))}\n\n"
            
            session_context += "AVAILABLE TRACKS:\n"
            for track in self.current_session_state.get('tracks', []):
                status_flags = []
                if track.get('selected'): status_flags.append("SELECTED")
                if track.get('muted'): status_flags.append("MUTED")
                if track.get('soloed'): status_flags.append("SOLO")
                
                status_str = f" [{', '.join(status_flags)}]" if status_flags else ""
                
                session_context += f"- \"{track['name']}\" ({track['type']}){status_str}\n"
                session_context += f"  Volume: {track.get('volume_db', 0):.1f} dB, Pan: {track.get('pan_percent', 0)}%\n"
                
                effects = track.get('effects', [])
                if effects:
                    effect_names = [effect['name'] for effect in effects if effect.get('active', True)]
                    session_context += f"  Effects: {', '.join(effect_names)}\n"
                else:
                    session_context += f"  Effects: None\n"
                session_context += "\n"
        else:
            session_context += "Session state unavailable\n\n"
        
        session_context += "=== END SESSION STATE ===\n\n"
        
        # Combine with base prompt
        full_prompt = session_context + self.base_system_prompt + f"\n\nUser command: \"{command_text}\""
        
        return full_prompt

    async def start(self):
        """Start the context-aware voice control system"""
        print("\nContext-Aware Deepgram + Ollama Voice Control System")
        print("AI now has full awareness of your Ableton session state")
        
        self.loop = asyncio.get_running_loop()
        
        if not self._test_ollama_connection() or not self._test_ableton_connection():
            return False

        # Get initial session state
        print("Getting initial session state...")
        self.current_session_state = self.get_current_session_state()
        self.last_state_update = time.time()
        
        if self.current_session_state:
            print(f"Session state loaded: {len(self.current_session_state.get('tracks', []))} tracks")
        else:
            print("Warning: Could not load session state")

        try:
            print("Connecting to Deepgram...")
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
            print(f"Deepgram connection failed: {e}")
            return False

        self.audio_stream = self.pyaudio.open(
            format=self.format, channels=self.channels, rate=self.sample_rate,
            input=True, frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.audio_stream.start_stream()
        print("Microphone activated")
        
        asyncio.create_task(self._process_audio_queue())
        asyncio.create_task(self._periodic_state_update())
        
        self.is_running = True
        self.status = VoiceControlStatus.LISTENING
        self._print_context_aware_instructions()
        print("\nContext-aware voice control system is active!")
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

    async def _periodic_state_update(self):
        """Periodically update session state in background"""
        await asyncio.sleep(5)  # Wait 5 seconds before first update
        
        while self.is_running:
            try:
                # Update session state every few seconds
                self.current_session_state = self.get_current_session_state()
                self.last_state_update = time.time()
                
                # Log state changes for debugging
                track_count = len(self.current_session_state.get('tracks', []))
                if track_count > 0:
                    logger.debug(f"Session state updated: {track_count} tracks")
                
                await asyncio.sleep(self.state_update_interval)
                
            except Exception as e:
                logger.error(f"Periodic state update failed: {e}")
                await asyncio.sleep(self.state_update_interval)

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Puts microphone audio into the asyncio queue."""
        try:
            if self.loop:
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, in_data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        return (None, pyaudio.paContinue)

    # Event handlers
    async def on_open(self, *args, **kwargs):
        logger.info("Deepgram connection opened.")
    
    async def on_error(self, *args, **kwargs):
        error = kwargs.get('error')
        logger.error(f"Deepgram error: {error}")

    async def on_close(self, *args, **kwargs):
        logger.info("Deepgram connection closed.")

    async def on_transcript(self, *args, **kwargs):
        result = kwargs.get('result')
        try:
            if result and result.is_final and len(result.channel.alternatives) > 0:
                transcript = result.channel.alternatives[0].transcript.strip()
                if transcript:
                    print(f"\nHeard: '{transcript}'")
                    self._process_transcript(transcript)
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
    
    def _process_transcript(self, text: str):
        text_lower = text.lower().strip()
        if self.use_wake_word and not self.wake_word_detected:
            if any(word in text_lower for word in self.wake_words):
                self.wake_word_detected = True
                self.last_command_time = time.time()
                print("Wake word detected - AI is now aware of your session...")
            return
        if self.wake_word_detected and (time.time() - self.last_command_time > self.command_timeout):
            self.wake_word_detected = False
            print("Wake word timeout - say wake word again")
            return
        self._execute_context_aware_command(text)
        self.last_command_time = time.time()
    
    def _test_ollama_connection(self):
        try:
            print("Testing Ollama connection...")
            response = requests.post(
                self.ollama_url, 
                json={"model": self.ollama_model_name, "prompt": "Hello", "stream": False}, 
                timeout=10
            )
            if response.status_code == 200:
                print("Ollama connection established")
                return True
            else:
                print(f"Ollama responded with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False
    
    def _test_ableton_connection(self):
        try:
            print("Testing Ableton connection...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((self.ableton_host, self.ableton_port))
            sock.send(json.dumps({"action": "get_session_info"}).encode('utf-8'))
            response = json.loads(sock.recv(4096).decode('utf-8'))
            sock.close()
            if response.get('status') == 'success':
                print("Ableton connection established")
                return True
            else: 
                return False
        except Exception as e:
            print(f"Ableton connection failed: {e}")
            return False
    
    def _execute_context_aware_command(self, command_text: str):
        try:
            self.status = VoiceControlStatus.EXECUTING
            self.command_count += 1
            print(f"Processing with Context-Aware AI: '{command_text}'")
            
            # Create context-aware prompt with current session state
            context_prompt = self._create_context_aware_prompt(command_text)
            
            # Get AI interpretation with full session context
            structured_command = self._get_ollama_interpretation_with_context(context_prompt)
            
            if not structured_command or "error" in structured_command:
                print(f"Could not understand command: {structured_command.get('error', 'Unknown error')}")
                return
            
            print(f"Context-Aware AI interpreted as: {structured_command}")
            
            # Validate the command
            validated_command = self._validate_command_with_context(structured_command)
            if "error" in validated_command:
                print(f"Command validation failed: {validated_command['error']}")
                return
                
            response = self._send_to_ableton(validated_command)
            
            if response.get('status') == 'success':
                self.successful_commands += 1
                print("Command executed successfully")
                result = response.get('result', {})
                
                # Enhanced result display with context
                if isinstance(result, dict):
                    if 'effects' in result:
                        print(f"   Effects on {result.get('track', 'track')}:")
                        for effect in result['effects']:
                            print(f"     • {effect}")
                    else:
                        for key, value in result.items():
                            if key not in ['error', 'status'] and value is not None:
                                print(f"   {key}: {value}")
                
                # Update session state after command execution
                asyncio.create_task(self._delayed_state_update())
                
            else:
                print(f"Command failed: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error executing context-aware command: {e}")
        finally:
            self.status = VoiceControlStatus.LISTENING

    async def _delayed_state_update(self):
        """Update session state after a short delay to capture changes"""
        await asyncio.sleep(1)
        self.current_session_state = self.get_current_session_state()
        self.last_state_update = time.time()
    
    def _validate_command_with_context(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Validate command using session context"""
        try:
            # Fix track targeting using session state
            if command.get("action") == "add_audio_effect":
                track_name = command.get("track", "")
                
                # If no track specified, use selected track
                if not track_name or track_name == "selected":
                    selected_track = self.current_session_state.get('selected_track')
                    if selected_track and selected_track != "None":
                        command["track"] = selected_track
                    else:
                        # Fallback to first track
                        tracks = self.current_session_state.get('tracks', [])
                        if tracks:
                            command["track"] = tracks[0]['name']
                        else:
                            return {"error": "No tracks available"}
                
                # Check if effect already exists
                effect_name = command.get("effect", "").lower()
                target_track_name = command.get("track")
                
                for track in self.current_session_state.get('tracks', []):
                    if track['name'].lower() == target_track_name.lower():
                        existing_effects = [effect['name'].lower() for effect in track.get('effects', [])]
                        if any(effect_name in existing for existing in existing_effects):
                            print(f"   Note: {track['name']} already has a {effect_name}-like effect")
            
            return command
            
        except Exception as e:
            return {"error": f"Context validation failed: {str(e)}"}
    
    def _get_ollama_interpretation_with_context(self, context_prompt: str) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model_name, 
                    "prompt": context_prompt, 
                    "stream": False,
                    "options": {
                        "temperature": 0.05,
                        "top_p": 0.8, 
                        "num_predict": 300
                    }
                },
                timeout=20  # Longer timeout for context processing
            )
            
            if response.status_code != 200: 
                return None
                
            ai_response = response.json().get('response', '').strip()
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                return json.loads(ai_response[json_start:json_end])
            return None
            
        except Exception as e:
            print(f"Error getting context-aware interpretation: {e}")
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
    
    def _print_context_aware_instructions(self):
        wake_prefix = f"{self.wake_words[0]}, " if self.use_wake_word else ""
        print("\n" + "="*80)
        print("CONTEXT-AWARE DEEPGRAM + OLLAMA VOICE CONTROL FOR ABLETON LIVE")
        print("="*80)
        print(f"\nSay '{self.wake_words[0]}' followed by your command")
        print("The AI now knows your complete session state in real-time!")
        
        if self.current_session_state:
            tracks = self.current_session_state.get('tracks', [])
            if tracks:
                print(f"\nYour current tracks:")
                for track in tracks[:5]:  # Show first 5 tracks
                    effects_count = len(track.get('effects', []))
                    status = " (SELECTED)" if track.get('selected') else ""
                    print(f"  - {track['name']}{status} - {effects_count} effects")
                if len(tracks) > 5:
                    print(f"  ... and {len(tracks) - 5} more tracks")
        
        print("\nCONTEXT-AWARE COMMANDS:")
        print(f"  '{wake_prefix}add reverb' (targets selected track)")
        print(f"  '{wake_prefix}add compressor to drums' (exact track name)")
        print(f"  '{wake_prefix}what effects are on bass'")
        print(f"  '{wake_prefix}pan the selected track left'")
        print(f"  '{wake_prefix}remove last effect from vocals'")
        
        print("\nINTELLIGENT FEATURES:")
        print("  • AI knows all track names and current effects")
        print("  • Prevents duplicate effects")
        print("  • Auto-completes partial track names")
        print("  • Defaults to selected track when not specified")
        print("  • Real-time session state updates")
        
        print("\nPress Ctrl+C to stop\n" + "="*80 + "\n")
    
    async def stop(self):
        print("\nStopping context-aware voice control system...")
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.pyaudio.terminate()
        if self.dg_connection:
            await self.dg_connection.finish()
        if self.command_count > 0:
            success_rate = (self.successful_commands / self.command_count) * 100
            print(f"\nContext-Aware Session Summary:")
            print(f"   Commands processed: {self.command_count}")
            print(f"   Successful: {self.successful_commands}")
            print(f"   Success rate: {success_rate:.1f}%")

async def main():
    print("Context-Aware Deepgram + Ollama Voice Control for Ableton")
    print("AI has full real-time awareness of your session state\n")
    
    deepgram_api_key = None
    ollama_model = "llama3.2:3b"
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            deepgram_api_key = config.get('deepgram_api_key')
            ollama_model = config.get('ollama_model', 'llama3.2:3b')
        print(f"Loaded configuration from config.json")
        print(f"Using Ollama model: {ollama_model}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    if not deepgram_api_key and len(sys.argv) > 1:
        deepgram_api_key = sys.argv[1]
        if len(sys.argv) > 2: 
            ollama_model = sys.argv[2]
    
    if not deepgram_api_key:
        print("ERROR: Deepgram API key required\n")
        print("Either:")
        print("   1. Create config.json with your API key")
        print("   2. Run: python3 script.py YOUR_DEEPGRAM_API_KEY\n")
        print("Get your free API key ($200 credits) at:")
        print("   https://console.deepgram.com/signup\n")
        sys.exit(1)
    
    controller = ContextAwareDeepgramAbletonController(
        deepgram_api_key=deepgram_api_key, 
        ollama_model=ollama_model
    )
    
    try:
        if await controller.start():
            while controller.is_running:
                await asyncio.sleep(0.1)
        else:
            print("Failed to start context-aware voice control system")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())