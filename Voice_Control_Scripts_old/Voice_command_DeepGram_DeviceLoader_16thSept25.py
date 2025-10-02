#!/usr/bin/env python3
"""
Enhanced Voice Control with Tool Manager Integration
Integrates the Ableton Tool Manager for better command understanding and execution
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
import re
from typing import Dict, Any, Optional, List
from enum import Enum

# Import Deepgram SDK v4+
from deepgram import DeepgramClient
from deepgram.clients.live.v1 import LiveClient, LiveTranscriptionEvents

# Import our tool manager
from ableton_tool_manager import AbletonToolManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedVoiceControl")

class VoiceControlStatus(Enum):
    INACTIVE = "inactive"
    LISTENING = "listening" 
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"

class EnhancedDeepgramAbletonController:
    """Enhanced voice control with comprehensive tool management"""
    
    def __init__(self, deepgram_api_key: str, ollama_model: str = "qwen2.5-coder:7b"):
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
        
        # Initialize the Tool Manager - this is our MCP-like layer
        self.tool_manager = AbletonToolManager(self.ableton_host, self.ableton_port)
        
        logger.info("Enhanced Voice Control with Tool Manager initialized")

    async def start(self):
        """Start the enhanced voice control system"""
        print("\nEnhanced Deepgram + Ollama + Tool Manager Voice Control")
        print("Now with structured tool discovery and session state awareness")
        
        self.loop = asyncio.get_running_loop()
        
        if not self._test_ollama_connection() or not self._test_ableton_connection():
            return False

        # Load initial session state through tool manager
        print("Loading session state...")
        session_info = self.tool_manager.get_cached_session_info()
        
        if session_info:
            tracks_count = len(session_info.get('tracks', []))
            tempo = session_info.get('tempo', 120)
            print(f"Session loaded: {tracks_count} tracks, tempo: {tempo} BPM")
            
            # Show available tools
            tools_count = len(self.tool_manager.get_available_tools())
            print(f"Tool Manager loaded: {tools_count} available tools")
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
        
        self.is_running = True
        self.status = VoiceControlStatus.LISTENING
        self._print_enhanced_instructions()
        print("\nEnhanced voice control system is active!")
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
        """Audio callback for microphone input"""
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
                print("Wake word detected - processing with enhanced tool system...")
            return
        if self.wake_word_detected and (time.time() - self.last_command_time > self.command_timeout):
            self.wake_word_detected = False
            print("Wake word timeout - say wake word again")
            return
        self._execute_enhanced_command(text)
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
            # Test through tool manager
            session_info = self.tool_manager.get_cached_session_info()
            if session_info:
                print("Ableton connection established")
                return True
            else: 
                return False
        except Exception as e:
            print(f"Ableton connection failed: {e}")
            return False
    
    def _execute_enhanced_command(self, command_text: str):
        """Execute command using the enhanced tool management system"""
        try:
            self.status = VoiceControlStatus.PROCESSING
            self.command_count += 1
            print(f"Processing with Enhanced Tool System: '{command_text}'")
            
            # Get context-aware prompt from tool manager
            context_prompt = self.tool_manager.get_context_for_ollama(command_text)
            
            # Get AI interpretation
            ollama_response = self._get_ollama_interpretation(context_prompt)
            
            if not ollama_response:
                print("Could not get response from Ollama")
                return
            
            print(f"Ollama interpreted as: {ollama_response}")
            
            # Process through tool manager
            self.status = VoiceControlStatus.EXECUTING
            result = self.tool_manager.process_voice_command(command_text, ollama_response)
            
            if result.get('status') == 'success':
                self.successful_commands += 1
                print("✓ Command executed successfully")
                
                # Enhanced result display
                self._display_enhanced_result(result)
                
            else:
                print(f"✗ Command failed: {result.get('message', 'Unknown error')}")
                
                # Show what tool was attempted
                if 'interpreted_as' in result:
                    print(f"   Attempted: {result['interpreted_as']}")
                
        except Exception as e:
            print(f"Error executing enhanced command: {e}")
        finally:
            self.status = VoiceControlStatus.LISTENING

    def _display_enhanced_result(self, result: Dict[str, Any]):
        """Enhanced result display with tool information"""
        try:
            interpreted_command = result.get('interpreted_as', {})
            action = interpreted_command.get('action', 'unknown')
            
            # Show what tool was used
            tool_info = result.get('tool_used')
            if tool_info:
                print(f"   Tool used: {tool_info.name} ({tool_info.category})")
            
            # Show specific results based on action type
            actual_result = result.get('result', {})
            
            if action == "set_tempo":
                tempo = actual_result.get("tempo", interpreted_command.get("tempo"))
                print(f"   Tempo set to: {tempo} BPM")
                
            elif action == "set_master_volume":
                volume = actual_result.get("volume", interpreted_command.get("value"))
                print(f"   Master volume: {volume}")
                
            elif action == "create_tracks":
                count = interpreted_command.get("count", 1)
                track_type = interpreted_command.get("track_type", "midi")
                names = interpreted_command.get("names", [])
                if names:
                    print(f"   Created {count} {track_type} track(s): {', '.join(names)}")
                else:
                    print(f"   Created {count} {track_type} track(s)")
                    
            elif action == "add_audio_effect":
                track = interpreted_command.get("track", "unknown")
                effect = interpreted_command.get("effect", "unknown")
                if 'device_name' in actual_result:
                    print(f"   Loaded {actual_result['device_name']} on {track}")
                else:
                    print(f"   Added {effect} to {track}")
                    
            elif action == "arm_track":
                track = interpreted_command.get("track", "unknown")
                print(f"   Armed track: {track}")
                
            elif action == "set_parameter":
                parameter = interpreted_command.get("parameter", "")
                target = interpreted_command.get("target", "")
                value = interpreted_command.get("value", "")
                param_type = parameter.replace("mixer_", "")
                print(f"   Set {param_type} on {target}: {value}")
                
            elif action in ["transport_play", "transport_stop"]:
                status = "started" if action == "transport_play" else "stopped"
                print(f"   Playback {status}")
                
            else:
                # Generic result display
                if isinstance(actual_result, dict):
                    for key, value in actual_result.items():
                        if key not in ['error', 'status'] and value is not None:
                            print(f"   {key}: {value}")
                            
        except Exception as e:
            logger.error(f"Error displaying enhanced result: {e}")
    
    def _get_ollama_interpretation(self, context_prompt: str) -> Optional[Dict[str, Any]]:
        """Get JSON interpretation from Ollama with enhanced error handling"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model_name, 
                    "prompt": context_prompt, 
                    "stream": False,
                    "options": {
                        "temperature": 0.01,
                        "top_p": 0.5, 
                        "num_predict": 500,  # Increase from 150 to 500
                        "stop": ["\n\n", "User:", "Command:"]
                    }
                },
                timeout=20
            )
            
            if response.status_code != 200: 
                print(f"Ollama error: Status {response.status_code}")
                return None
                
            ai_response = response.json().get('response', '').strip()
            print(f"Ollama raw response: {ai_response}")
            
            # Enhanced JSON extraction
            return self._extract_json_from_response(ai_response)
            
        except Exception as e:
            print(f"Error getting Ollama interpretation: {e}")
            return None
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Enhanced JSON extraction with multiple strategies"""
        
        # Strategy 1: Find complete JSON objects
        json_objects = []
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_str = response[start_pos:i+1]
                    json_objects.append(json_str)
        
        # Try parsing each JSON object (prefer the last/most complete one)
        for json_str in reversed(json_objects):
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and 'action' in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Clean up common AI patterns
        cleaned = response
        
        # Remove common prefixes
        prefixes = [
            "Here's the JSON command:", "The command would be:", "Based on the request:",
            "Here is the JSON:", "Command:", "JSON:", "Response:"
        ]
        
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Extract JSON portion
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = cleaned[json_start:json_end]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        print(f"No valid JSON found in response: {response}")
        return None
    
    def _print_enhanced_instructions(self):
        """Print enhanced instructions showing available tools and current state"""
        wake_prefix = f"{self.wake_words[0]}, " if self.use_wake_word else ""
        print("\n" + "="*80)
        print("ENHANCED DEEPGRAM + OLLAMA + TOOL MANAGER VOICE CONTROL")
        print("="*80)
        print(f"\nSay '{self.wake_words[0]}' followed by your command")
        
        # Show current session state
        session_info = self.tool_manager.get_cached_session_info()
        if session_info:
            tempo = session_info.get('tempo', 120)
            playing = session_info.get('is_playing', False)
            tracks = session_info.get('tracks', [])
            
            print(f"\nCurrent Session State:")
            print(f"  Tempo: {tempo} BPM")
            print(f"  Status: {'Playing' if playing else 'Stopped'}")
            print(f"  Tracks: {len(tracks)} total")
            
            if tracks:
                print(f"\nYour Tracks:")
                for track in tracks[:8]:  # Show first 8 tracks
                    flags = []
                    if track.get('soloed'): flags.append("SOLO")
                    if track.get('muted'): flags.append("MUTE")
                    if track.get('armed'): flags.append("ARMED")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    
                    effects_count = len(track.get('effects', []))
                    effects_str = f" ({effects_count} FX)" if effects_count > 0 else ""
                    
                    print(f"  - {track['name']} ({track['type']}){flag_str}{effects_str}")
        
        # Show available tool categories
        categories = {}
        for tool in self.tool_manager.get_available_tools():
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool.name)
        
        print(f"\nAvailable Tool Categories:")
        for category, tools in sorted(categories.items()):
            print(f"  {category.title()}: {len(tools)} tools")
        
        print(f"\nExample Commands:")
        print(f"  '{wake_prefix}change tempo to 135'")
        print(f"  '{wake_prefix}create midi track called bass'")
        print(f"  '{wake_prefix}add reverb to vocals'")
        print(f"  '{wake_prefix}set drums volume to -10 dB'")
        print(f"  '{wake_prefix}solo vocal track'")
        print(f"  '{wake_prefix}arm bass track for recording'")
        print(f"  '{wake_prefix}play' / '{wake_prefix}stop'")
        print(f"  '{wake_prefix}get session info'")
        
        print(f"\nEnhanced Features:")
        print("  ✓ Real-time session state awareness")
        print("  ✓ Structured tool discovery and validation")
        print("  ✓ Enhanced command interpretation") 
        print("  ✓ Automatic track name resolution")
        print("  ✓ Parameter validation and error handling")
        print("  ✓ Session cache for performance")
        
        print("\nPress Ctrl+C to stop\n" + "="*80 + "\n")
    
    async def stop(self):
        print("\nStopping enhanced voice control system...")
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.pyaudio.terminate()
        if self.dg_connection:
            await self.dg_connection.finish()
        if self.command_count > 0:
            success_rate = (self.successful_commands / self.command_count) * 100
            print(f"\nEnhanced Session Summary:")
            print(f"   Commands processed: {self.command_count}")
            print(f"   Successful: {self.successful_commands}")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Tools available: {len(self.tool_manager.get_available_tools())}")

async def main():
    print("Enhanced Deepgram + Ollama + Tool Manager for Ableton")
    print("MCP-like tool discovery and session state management\n")
    
    deepgram_api_key = None
    ollama_model = "qwen2.5-coder:7b"
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            deepgram_api_key = config.get('deepgram_api_key')
            ollama_model = config.get('ollama_model', 'qwen2.5-coder:7b')
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
    
    controller = EnhancedDeepgramAbletonController(
        deepgram_api_key=deepgram_api_key, 
        ollama_model=ollama_model
    )
    
    try:
        if await controller.start():
            while controller.is_running:
                await asyncio.sleep(0.1)
        else:
            print("Failed to start enhanced voice control system")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())