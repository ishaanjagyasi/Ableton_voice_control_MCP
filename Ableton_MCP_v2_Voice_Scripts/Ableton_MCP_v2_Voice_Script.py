#!/usr/bin/env python3
"""
Updated Deepgram + Ollama Voice Control
Fixed missing parameters: tempo, master volume, track arming, device search
Enhanced system prompt to handle all Ableton parameters correctly

Requirements: 
- pip install "deepgram-sdk>=4.0" pyaudio requests
- Deepgram API key
- Ollama running locally with llama3.2:3b model
- Ableton with Enhanced Remote Script
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UpdatedAbletonControl")

class VoiceControlStatus(Enum):
    INACTIVE = "inactive"
    LISTENING = "listening" 
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"

class UpdatedDeepgramAbletonController:
    """Updated voice control with comprehensive parameter support"""
    
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
        self.state_update_interval = 5.0
        
        # Enhanced system prompt with comprehensive parameter support
        self.base_system_prompt = self._create_comprehensive_system_prompt()
        
        logger.info("Updated Deepgram Ableton Controller initialized")

    def _create_comprehensive_system_prompt(self):
        """Create comprehensive system prompt covering all parameters"""
        return """Convert voice commands to JSON for Ableton Live control. Respond with ONLY valid JSON, no explanations.

COMMANDS:

1. CREATE TRACKS:
   {"action": "create_tracks", "track_type": "audio" OR "midi", "count": number, "names": ["name1"]}
   ONLY create tracks for explicit commands like: "create track", "add track", "new track", "make track"
   DO NOT create tracks when "track" is used to reference existing tracks (e.g., "drum track", "bass track")

2. TRACK MANAGEMENT:
   {"action": "rename_track", "track": "old_track_name", "new_name": "new_track_name"}
   {"action": "delete_track", "track": "track_name"}
   {"action": "delete_tracks", "tracks": ["track1", "track2"]}  # For multiple deletions
   {"action": "duplicate_track", "track": "track_name", "new_name": "optional_new_name"}
   {"action": "group_tracks", "tracks": ["track1", "track2"], "group_name": "Group Name"}
   {"action": "batch_commands", "commands": [{"action": "...", ...}, {"action": "...", ...}]}  # For complex multi-step operations

3. ADD EFFECTS:
   {"action": "add_audio_effect", "track": "track_name", "effect": "effect_name"}
   Available effects: reverb, delay, echo, compressor, limiter, gate, eq, equalizer, saturator, overdrive, filter, drum bus, glue compressor, multiband dynamics, chorus, flanger, phaser

3. TRANSPORT CONTROLS:
   {"action": "transport_play"} / {"action": "transport_stop"} / {"action": "transport_record"}
   {"action": "set_tempo", "tempo": BPM_NUMBER}
   {"action": "set_time_signature", "numerator": 4, "denominator": 4}
   {"action": "set_loop", "enabled": true/false}
   {"action": "toggle_metronome", "enabled": true/false}

4. MASTER/GLOBAL CONTROLS:
   {"action": "set_master_volume", "value": VOLUME_LEVEL}
   {"action": "set_cue_volume", "value": VOLUME_LEVEL}
   {"action": "set_crossfader", "value": POSITION}
   {"action": "set_groove_amount", "value": AMOUNT}

5. MIXER CONTROLS:
   {"action": "set_parameter", "parameter": "mixer_solo", "target": "track_name", "value": true}
   {"action": "set_parameter", "parameter": "mixer_mute", "target": "track_name", "value": true}
   {"action": "set_parameter", "parameter": "mixer_volume", "target": "track_name", "value": -5}
   {"action": "set_parameter", "parameter": "mixer_pan", "target": "track_name", "value": -100}
   {"action": "set_parameter", "parameter": "mixer_arm", "target": "track_name", "value": true}
   {"action": "set_parameter", "parameter": "mixer_send_a", "target": "track_name", "value": 50}

6. TRACK CONTROLS:
   {"action": "arm_track", "track": "track_name"}
   {"action": "disarm_track", "track": "track_name"}
   {"action": "set_track_monitor", "track": "track_name", "mode": "auto"}

7. CLIP CONTROLS:
   {"action": "launch_clip", "track": "track_name", "clip": 0}
   {"action": "stop_clip", "track": "track_name"}
   {"action": "launch_scene", "scene": 0}
   {"action": "stop_all_clips"}

COMMAND PARSING RULES:
- "create a new track" -> {"action": "create_tracks", "track_type": "audio", "count": 1}
- "add audio track" -> {"action": "create_tracks", "track_type": "audio", "count": 1}
- "make midi track" -> {"action": "create_tracks", "track_type": "midi", "count": 1}
- "new track called vocals" -> {"action": "create_tracks", "track_type": "audio", "count": 1, "names": ["vocals"]}
- "delete the drum and bass tracks" -> {"action": "delete_tracks", "tracks": ["drum", "bass"]}
- "remove drums and bass" -> {"action": "delete_tracks", "tracks": ["drums", "bass"]}
- "create three audio tracks and name them vocals, harmony, and lead" -> {"action": "batch_commands", "commands": [{"action": "create_tracks", "track_type": "audio", "count": 3, "names": ["vocals", "harmony", "lead"]}]}
- "duplicate bass, add reverb to it, and rename it to bass reverb" -> {"action": "batch_commands", "commands": [{"action": "duplicate_track", "track": "bass", "new_name": "bass reverb"}, {"action": "add_audio_effect", "track": "bass reverb", "effect": "reverb"}]}
- "rename drums to percussion" -> {"action": "rename_track", "track": "drums", "new_name": "percussion"}
- "rename bass track to synth bass" -> {"action": "rename_track", "track": "bass", "new_name": "synth bass"}
- "delete the drums track" -> {"action": "delete_track", "track": "drums"}
- "remove bass track" -> {"action": "delete_track", "track": "bass"}
- "duplicate drums track" -> {"action": "duplicate_track", "track": "drums"}
- "copy bass track as synth bass" -> {"action": "duplicate_track", "track": "bass", "new_name": "synth bass"}
- "group drums and bass" -> {"action": "group_tracks", "tracks": ["drums", "bass"], "group_name": "Rhythm"}
- "group vocals and harmony tracks" -> {"action": "group_tracks", "tracks": ["vocals", "harmony"], "group_name": "Vocals"}
- "change tempo to 135" -> {"action": "set_tempo", "tempo": 135}
- "reduce master volume" -> {"action": "set_master_volume", "value": -10}
- "turn down master" -> {"action": "set_master_volume", "value": -15}
- "set vocals to -23 dB" -> {"action": "set_parameter", "parameter": "mixer_volume", "target": "vocals", "value": -23}
- "adjust vocals volume to minus 23 dB" -> {"action": "set_parameter", "parameter": "mixer_volume", "target": "vocals", "value": -23}
- "change volume on drum track by 20 percent" -> {"action": "set_parameter", "parameter": "mixer_volume", "target": "drum", "value": "increase_20%"}
- "decrease bass track volume by 20 percent" -> {"action": "set_parameter", "parameter": "mixer_volume", "target": "bass", "value": "decrease_20%"}
- "increase synth track volume by 5 dB" -> {"action": "set_parameter", "parameter": "mixer_volume", "target": "synth", "value": "increase_5db"}
- "set drums to 80 percent" -> {"action": "set_parameter", "parameter": "mixer_volume", "target": "drums", "value": "80%"}
- "put drum bus on drums" -> {"action": "add_audio_effect", "track": "drums", "effect": "drum bus"}
- "add drum buss to drums" -> {"action": "add_audio_effect", "track": "drums", "effect": "drum bus"}
- "record enable bass" -> {"action": "arm_track", "track": "bass"}
- "arm the bass track" -> {"action": "arm_track", "track": "bass"}
- "solo drum track" -> {"action": "set_parameter", "parameter": "mixer_solo", "target": "drum", "value": true}
- "play" -> {"action": "transport_play"}
- "stop" -> {"action": "transport_stop"}

MULTI-COMMAND PLANNING RULES:
- For commands affecting multiple items, analyze and create appropriate batch operations
- "delete drums and bass" -> {"action": "delete_tracks", "tracks": ["drums", "bass"]}
- For complex sequences, use batch_commands with ordered command list
- Always break down high-level requests into executable steps
- Consider dependencies between commands (create before rename, duplicate before modify)

VOLUME PARSING RULES:
- For exact dB values: use the exact number (e.g., -23 for "-23 dB")
- For percentage values: use "XX%" format (e.g., "50%" for 50 percent)
- For relative changes: use "increase_XdbB" or "decrease_X%" format
- When user says "20 percent", interpret as "20%" not 0.2
- When user says "minus 23 dB" or "-23 dB", use -23 as the value

DEVICE NAME MAPPING:
- "drum bus", "drum buss" -> "drum bus"
- "glue comp", "glue compressor" -> "glue compressor"
- "multiband" -> "multiband dynamics"
- "eq" -> "eq eight"
- "comp", "compressor" -> "compressor"

RULES:
- Use EXACT track names from session
- Match closest track name if not exact
- For tempo commands, use "set_tempo" action with "tempo" parameter
- For master volume, use "set_master_volume" action with "value" parameter
- For track arming, use "arm_track" or "disarm_track" actions
- RESPOND WITH ONLY JSON, NO OTHER TEXT

Current session tracks:"""

    def get_current_session_state(self) -> Dict[str, Any]:
        """Get session state from Ableton"""
        try:
            response = self._send_to_ableton({"action": "get_detailed_session_info"})
            if response.get('status') == 'success':
                return response.get('result', {})
            return {}
        except Exception as e:
            logger.error(f"Failed to get session state: {e}")
            return {}

    def _create_context_prompt(self, command_text: str) -> str:
        """Create context prompt with better command understanding"""
        # Get fresh session state
        current_time = time.time()
        if current_time - self.last_state_update > self.state_update_interval:
            self.current_session_state = self.get_current_session_state()
            self.last_state_update = current_time

        # Build context with current state
        context = "\nCURRENT SESSION:\n"
        
        if self.current_session_state:
            context += f"Tempo: {self.current_session_state.get('tempo', 120)} BPM\n"
            context += f"Playing: {self.current_session_state.get('is_playing', False)}\n"
            context += f"Recording: {self.current_session_state.get('is_recording', False)}\n"
            context += f"Loop: {self.current_session_state.get('loop_enabled', False)}\n"
            
            selected_track = self.current_session_state.get('selected_track', 'None')
            context += f"Selected: {selected_track}\n"
            
            context += "Tracks:\n"
            for track in self.current_session_state.get('tracks', []):
                solo_flag = " (SOLO)" if track.get('soloed') else ""
                mute_flag = " (MUTE)" if track.get('muted') else ""
                arm_flag = " (ARMED)" if track.get('armed') else ""
                context += f"- {track['name']}{solo_flag}{mute_flag}{arm_flag}\n"
        
        # Enhanced prompt with better command understanding
        full_prompt = context + "\n" + self.base_system_prompt + f"\n\nCommand: \"{command_text}\""
        
        return full_prompt

    async def start(self):
        """Start the updated voice control system"""
        print("\nUpdated Deepgram + Ollama Voice Control System")
        print("Comprehensive parameter support including tempo, master volume, and device search")
        
        self.loop = asyncio.get_running_loop()
        
        if not self._test_ollama_connection() or not self._test_ableton_connection():
            return False

        # Get initial session state
        print("Loading session state...")
        self.current_session_state = self.get_current_session_state()
        self.last_state_update = time.time()
        
        if self.current_session_state:
            tracks_count = len(self.current_session_state.get('tracks', []))
            tempo = self.current_session_state.get('tempo', 120)
            print(f"Session loaded: {tracks_count} tracks, tempo: {tempo} BPM")
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
        self._print_updated_instructions()
        print("\nUpdated voice control system is active!")
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
        """Update session state periodically"""
        await asyncio.sleep(5)
        
        while self.is_running:
            try:
                self.current_session_state = self.get_current_session_state()
                self.last_state_update = time.time()
                await asyncio.sleep(self.state_update_interval)
            except Exception as e:
                logger.error(f"Periodic state update failed: {e}")
                await asyncio.sleep(self.state_update_interval)

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
                print("Wake word detected - processing with updated AI...")
                # Don't process wake word only as a command
                if text_lower.strip() in self.wake_words:
                    return  # Just wake word, wait for actual command
            return
        if self.wake_word_detected and (time.time() - self.last_command_time > self.command_timeout):
            self.wake_word_detected = False
            print("Wake word timeout - say wake word again")
            return
        # Only process if there's actual content beyond wake word
        if len(text.strip()) > len(max(self.wake_words, key=len)):
            self._execute_updated_command(text)
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
    
    def _execute_updated_command(self, command_text: str):
        try:
            self.status = VoiceControlStatus.EXECUTING
            self.command_count += 1
            self._last_command_text = command_text  # Store for validation
            print(f"Processing with Updated AI: '{command_text}'")
            
            # Create enhanced context prompt
            context_prompt = self._create_context_prompt(command_text)
            
            # Get AI interpretation with updated prompt
            structured_command = self._get_ollama_interpretation(context_prompt)
            
            if not structured_command or "error" in structured_command:
                print(f"Could not understand command: {structured_command.get('error', 'Unknown error') if structured_command else 'No response'}")
                return
            
            print(f"Updated AI interpreted as: {structured_command}")
            
            # Validate and enhance the command
            validated_command = self._validate_and_enhance_command(structured_command)
            if "error" in validated_command:
                print(f"Command validation failed: {validated_command['error']}")
                return
            
            # Check if command was intercepted and should be handled specially
            if validated_command.get("_intercepted"):
                print("Command intercepted - handling in voice script...")
                success = self._execute_single_command(validated_command)
                if success:
                    self.successful_commands += 1
                    asyncio.create_task(self._delayed_state_update())
            else:
                # Handle batch commands or single commands normally
                if validated_command.get("action") == "batch_commands":
                    success = self._execute_batch_commands(validated_command)
                else:
                    success = self._execute_single_command(validated_command)
                
                if success:
                    self.successful_commands += 1
                    asyncio.create_task(self._delayed_state_update())
                
        except Exception as e:
            print(f"Error executing updated command: {e}")
        finally:
            self.status = VoiceControlStatus.LISTENING
            
    

    def _execute_single_command(self, command: Dict[str, Any]) -> bool:
        """Execute a single command with improved timeout handling"""
        try:
            # FIRST: Check if this is an intercepted command - handle it without sending to Ableton
            if command.get("_intercepted"):
                track_names = command.get("_track_list", [])
                print(f"Executing {len(track_names)} individual deletions...")
                
                successful = 0
                failed = 0
                
                for track_name in track_names:
                    print(f"  Deleting track: {track_name}")
                    # Create a clean single delete command
                    single_cmd = {"action": "delete_track", "track": track_name}
                    
                    try:
                        response = self._send_to_ableton(single_cmd, timeout=10)
                        
                        if response.get('status') == 'success':
                            print(f"    ✓ Deleted: {track_name}")
                            successful += 1
                        else:
                            error_msg = response.get('message', 'Unknown error')
                            print(f"    ✗ Failed: {error_msg}")
                            failed += 1
                    except Exception as e:
                        print(f"    ✗ Exception: {str(e)}")
                        failed += 1
                    
                    # Wait between deletions to let Ableton process
                    time.sleep(0.5)
                
                print(f"Batch deletion completed: {successful} successful, {failed} failed")
                return successful > 0
            
            # SECOND: Normal command execution (not intercepted)
            response = self._send_to_ableton(command, timeout=10)
            
            if response.get('status') == 'success':
                print("Command executed successfully")
                result = response.get('result', {})
                self._display_command_result(result, command)
                return True
            else:
                print(f"Command failed: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"Error executing single command: {e}")
            return False

    def _execute_batch_commands(self, batch_command: Dict[str, Any]) -> bool:
        """Execute multiple commands in sequence with error handling"""
        try:
            commands = batch_command.get("commands", [])
            if not commands:
                print("No commands in batch")
                return False
            
            print(f"Executing batch of {len(commands)} commands...")
            successful_count = 0
            
            for i, command in enumerate(commands):
                print(f"  Step {i+1}/{len(commands)}: {command.get('action', 'unknown')}")
                
                try:
                    response = self._send_to_ableton(command, timeout=8)
                    
                    if response.get('status') == 'success':
                        result = response.get('result', {})
                        self._display_command_result(result, command)
                        successful_count += 1
                        time.sleep(0.5)
                    else:
                        print(f"    Step {i+1} failed: {response.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"    Step {i+1} error: {e}")
            
            print(f"Batch completed: {successful_count}/{len(commands)} commands successful")
            return successful_count > 0
            
        except Exception as e:
            print(f"Error executing batch commands: {e}")
            return False


    def _display_command_result(self, result: Dict[str, Any], command: Dict[str, Any]):
        """Enhanced result display for different command types"""
        try:
            action = command.get("action", "unknown")
            
            if action == "set_tempo":
                tempo = result.get("tempo", command.get("tempo"))
                print(f"   Tempo set to: {tempo} BPM")
                
            elif action == "set_master_volume":
                volume = result.get("volume", command.get("value"))
                print(f"   Master volume set to: {volume}")
                
            elif action == "arm_track":
                track = result.get("track", command.get("track"))
                print(f"   Armed track: {track}")
                
            elif action == "add_audio_effect":
                if 'device_name' in result:
                    print(f"   Loaded: {result['device_name']}")
                    print(f"   Target: {result.get('track', 'unknown track')}")
                    
            elif isinstance(result, dict):
                if 'set' in result:
                    print(f"   {result['set']}: {result.get('value', '')} on {result.get('track', 'track')}")
                else:
                    for key, value in result.items():
                        if key not in ['error', 'status'] and value is not None:
                            print(f"   {key}: {value}")
                            
        except Exception as e:
            logger.error(f"Error displaying result: {e}")

    async def _delayed_state_update(self):
        """Update session state after delay"""
        await asyncio.sleep(1.5)
        self.current_session_state = self.get_current_session_state()
        self.last_state_update = time.time()
    
    def _validate_and_enhance_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced command validation and preprocessing"""
        try:
            action = command.get("action", "")
            
            # Intercept delete_tracks and handle it in the voice script
            if action == "delete_tracks":
                track_names = command.get("tracks", [])
                if track_names:
                    print(f"   Intercepting delete_tracks - will execute {len(track_names)} individual deletions")
                    # Return a special intercepted command that won't be validated further
                    return {
                        "action": "intercepted_delete_tracks",  # Changed action name
                        "_intercepted": True,
                        "_track_list": track_names,
                        "tracks": track_names  # Keep for display
                    }
                else:
                    return {"error": "No tracks specified for deletion"}
            
            # Prevent unwanted track creation when "track" is used as reference
            if action == "create_tracks":
                # Check if this is actually a track reference, not creation
                original_text = getattr(self, '_last_command_text', '').lower()
                
                # Look for actual creation keywords
                creation_keywords = ['create', 'add', 'new', 'make']
                track_keywords = ['track', 'tracks']
                
                has_creation_word = any(word in original_text for word in creation_keywords)
                has_track_word = any(word in original_text for word in track_keywords)
                
                # Check for reference patterns that should NOT create tracks
                reference_patterns = [
                    'volume on', 'adjust', 'change', 'increase', 'decrease',
                    'solo', 'mute', 'arm', 'record enable', 'effect on',
                    'reverb on', 'delay on', 'compressor on'
                ]
                
                is_reference = any(pattern in original_text for pattern in reference_patterns)
                
                # If it's clearly a reference and not creation, change to parameter action
                if is_reference and not has_creation_word:
                    print(f"   Detected track reference, not creation. Original: '{original_text}'")
                    return {"error": "Track reference detected, not creating track"}
                
                # If no clear creation intent, reject
                if has_track_word and not has_creation_word:
                    print(f"   No clear track creation intent. Original: '{original_text}'")
                    return {"error": "No clear track creation intent"}
            
            # Fix track targeting for commands that need it
            if action in ["add_audio_effect", "arm_track", "disarm_track", "rename_track", "delete_track", "duplicate_track"]:
                track_name = command.get("track", "")
                
                # Clean up track names that include "track" suffix
                if track_name.endswith(" track"):
                    track_name = track_name.replace(" track", "")
                    command["track"] = track_name
                
                if not track_name or track_name == "selected":
                    selected_track = self.current_session_state.get('selected_track')
                    if selected_track and selected_track != "None":
                        command["track"] = selected_track
                    else:
                        tracks = self.current_session_state.get('tracks', [])
                        if tracks:
                            command["track"] = tracks[0]['name']
                        else:
                            return {"error": "No tracks available"}
            
            # Handle group tracks command
            if action == "group_tracks":
                track_names = command.get("tracks", [])
                if not track_names:
                    return {"error": "No tracks specified for grouping"}
                
                # Clean up track names
                cleaned_names = []
                for track_name in track_names:
                    if track_name.endswith(" track"):
                        track_name = track_name.replace(" track", "")
                    cleaned_names.append(track_name)
                command["tracks"] = cleaned_names
                
                # Default group name if not provided
                if not command.get("group_name"):
                    command["group_name"] = "Group"
            
            # Validate rename command
            if action == "rename_track":
                new_name = command.get("new_name", "")
                if not new_name:
                    return {"error": "No new name provided for rename"}
            
            # Clean up targets for parameter commands
            if action == "set_parameter":
                target = command.get("target", "")
                if target.endswith(" track"):
                    target = target.replace(" track", "")
                    command["target"] = target
                if target.startswith("the "):
                    target = target.replace("the ", "")
                    command["target"] = target
            
            # Enhanced device name mapping
            if action == "add_audio_effect":
                effect = command.get("effect", "").lower()
                device_mappings = {
                    "drum buss": "drum bus",
                    "glue comp": "glue compressor",
                    "multiband": "multiband dynamics",
                    "eq": "eq eight",
                    "comp": "compressor"
                }
                
                if effect in device_mappings:
                    command["effect"] = device_mappings[effect]
            
            # Validate tempo range
            if action == "set_tempo":
                tempo = command.get("tempo")
                if tempo is None or tempo < 20 or tempo > 999:
                    return {"error": f"Invalid tempo: {tempo}. Must be between 20-999 BPM"}
            
            # Validate count for create_tracks
            if action == "create_tracks":
                count = command.get("count")
                if count is None or count <= 0:
                    command["count"] = 1
                # Ensure default is audio
                if not command.get("track_type"):
                    command["track_type"] = "audio"
            
            return command
            
        except Exception as e:
            return {"error": f"Command validation failed: {str(e)}"}
    
    def _get_ollama_interpretation(self, context_prompt: str) -> Optional[Dict[str, Any]]:
        """Enhanced JSON extraction with better parsing"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model_name, 
                    "prompt": context_prompt, 
                    "stream": False,
                    "options": {
                        "temperature": 0.01,  # Very low for consistent JSON
                        "top_p": 0.5, 
                        "num_predict": 200
                    }
                },
                timeout=20
            )
            
            if response.status_code != 200: 
                return None
                
            ai_response = response.json().get('response', '').strip()
            print(f"AI raw response: {ai_response}")
            
            # Enhanced JSON extraction strategies
            
            # Strategy 1: Find the most complete JSON object
            json_objects = []
            brace_count = 0
            start_pos = -1
            
            for i, char in enumerate(ai_response):
                if char == '{':
                    if brace_count == 0:
                        start_pos = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        json_str = ai_response[start_pos:i+1]
                        json_objects.append(json_str)
            
            # Try parsing each JSON object found (prioritize the last/most complete one)
            for json_str in reversed(json_objects):
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and 'action' in parsed:
                        print(f"Successfully parsed JSON: {parsed}")
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            # Strategy 2: Clean up common AI response patterns
            cleaned_response = ai_response
            
            # Remove common prefixes
            prefixes_to_remove = [
                "Here's the JSON command:",
                "The command would be:",
                "Based on the request:",
                "Here is the JSON:",
                "Command:",
                "JSON:"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned_response.lower().startswith(prefix.lower()):
                    cleaned_response = cleaned_response[len(prefix):].strip()
            
            # Try to extract JSON from cleaned response
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        print(f"Cleaned parsing successful: {parsed}")
                        return parsed
                except json.JSONDecodeError as e:
                    print(f"Cleaned JSON parse error: {e}")
            
            print(f"No valid JSON found in response: {ai_response}")
            return None
            
        except Exception as e:
            print(f"Error getting interpretation: {e}")
            return None
    
    def _send_to_ableton(self, action: Dict[str, Any], timeout: int = 5) -> Dict[str, Any]:
        """Send command to Ableton with configurable timeout"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.ableton_host, self.ableton_port))
            
            # Send command
            command_json = json.dumps(action)
            sock.send(command_json.encode('utf-8'))
            
            # Receive response with timeout
            response_data = sock.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            sock.close()
            return response
            
        except socket.timeout:
            return {"status": "error", "message": f"Command timed out after {timeout} seconds"}
        except ConnectionRefusedError:
            return {"status": "error", "message": "Could not connect to Ableton - is the remote script running?"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _print_updated_instructions(self):
        wake_prefix = f"{self.wake_words[0]}, " if self.use_wake_word else ""
        print("\n" + "="*70)
        print("UPDATED DEEPGRAM + OLLAMA VOICE CONTROL FOR ABLETON LIVE")
        print("="*70)
        print(f"\nSay '{self.wake_words[0]}' followed by your command")
        print("Comprehensive parameter support including missing functionalities")
        
        if self.current_session_state:
            tempo = self.current_session_state.get('tempo', 120)
            playing = self.current_session_state.get('is_playing', False)
            print(f"\nCurrent State:")
            print(f"  Tempo: {tempo} BPM")
            print(f"  Playing: {playing}")
            
            tracks = self.current_session_state.get('tracks', [])
            if tracks:
                print(f"\nYour tracks:")
                for track in tracks[:6]:
                    solo_flag = " (SOLO)" if track.get('soloed') else ""
                    mute_flag = " (MUTE)" if track.get('muted') else ""
                    arm_flag = " (ARMED)" if track.get('armed') else ""
                    print(f"  - {track['name']}{solo_flag}{mute_flag}{arm_flag}")
        
        print("\nUPDATED COMMANDS:")
        print(f"  '{wake_prefix}change tempo to 135'")
        print(f"  '{wake_prefix}reduce master volume'")
        print(f"  '{wake_prefix}put drum bus on drums'")
        print(f"  '{wake_prefix}add drum buss to drums'")  # Alternative spelling
        print(f"  '{wake_prefix}record enable bass track'")
        print(f"  '{wake_prefix}arm the drums'")
        print(f"  '{wake_prefix}solo drum track'")
        print(f"  '{wake_prefix}mute bass'")
        print(f"  '{wake_prefix}add reverb to vocals'")
        print(f"  '{wake_prefix}play' / '{wake_prefix}stop'")
        
        print("\nFIXED ISSUES:")
        print("  ✓ Tempo control (set_tempo action)")
        print("  ✓ Master volume control (set_master_volume action)")
        print("  ✓ Track arming (arm_track/disarm_track actions)")
        print("  ✓ Enhanced device search including 'Drum Bus'")
        print("  ✓ Better command parsing and validation")
        print("  ✓ Comprehensive parameter support")
        
        print("\nPress Ctrl+C to stop\n" + "="*70 + "\n")
    
    async def stop(self):
        print("\nStopping updated voice control system...")
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.pyaudio.terminate()
        if self.dg_connection:
            await self.dg_connection.finish()
        if self.command_count > 0:
            success_rate = (self.successful_commands / self.command_count) * 100
            print(f"\nUpdated Session Summary:")
            print(f"   Commands processed: {self.command_count}")
            print(f"   Successful: {self.successful_commands}")
            print(f"   Success rate: {success_rate:.1f}%")

async def main():
    print("Updated Deepgram + Ollama Voice Control for Ableton")
    print("Fixed tempo, master volume, track arming, and device search\n")
    
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
    
    controller = UpdatedDeepgramAbletonController(
        deepgram_api_key=deepgram_api_key, 
        ollama_model=ollama_model
    )
    
    try:
        if await controller.start():
            while controller.is_running:
                await asyncio.sleep(0.1)
        else:
            print("Failed to start updated voice control system")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())