#!/usr/bin/env python3
"""
Ableton Tool Manager - MCP-like layer for Ollama
Provides tool discovery, session state management, and structured command handling
with support for sequential command execution
"""

import json
import socket
import time
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger("AbletonToolManager")

@dataclass
class ToolParameter:
    """Define a tool parameter with type and description"""
    name: str
    type: str  # "string", "number", "boolean", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None

@dataclass
class AbletonTool:
    """Define an Ableton tool with structured parameters"""
    name: str
    description: str
    category: str
    parameters: List[ToolParameter]
    examples: List[str]
    
    def to_prompt_format(self) -> str:
        """Convert tool to prompt-friendly format for Ollama"""
        param_list = []
        for param in self.parameters:
            req_marker = " (required)" if param.required else " (optional)"
            enum_info = f" Options: {param.enum}" if param.enum else ""
            default_info = f" Default: {param.default}" if param.default is not None else ""
            param_list.append(f"  - {param.name} ({param.type}): {param.description}{req_marker}{enum_info}{default_info}")
        
        examples_text = "\n".join([f"  Example: {ex}" for ex in self.examples])
        
        return f"""
{self.name} ({self.category}):
  Description: {self.description}
  Parameters:
{chr(10).join(param_list)}
  Usage Examples:
{examples_text}
  JSON Format: {{"action": "{self.name}", "param1": "value1", "param2": "value2"}}
"""

class UniversalContextualProcessor:
    """Intelligently extract and provide relevant session context for ANY command"""
    
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager
        
        # Universal command pattern mapping
        self.command_patterns = {
            # Transport commands
            "transport": {
                "patterns": ["play", "stop", "record", "tempo", "bpm", "metronome", "loop"],
                "context_needed": ["transport_state", "tempo", "loop_state"]
            },
            
            # Track operations 
            "track_management": {
                "patterns": ["create", "make", "add", "delete", "remove", "rename", "duplicate", "copy", "group"],
                "context_needed": ["track_names", "track_types", "track_count"]
            },
            
            # Mixer operations
            "mixer": {
                "patterns": ["volume", "pan", "solo", "mute", "send", "level", "gain"],
                "context_needed": ["mixer_states", "current_levels", "solo_mute_states"]
            },
            
            # Device/Effect operations
            "devices": {
                "patterns": ["add", "load", "insert", "remove", "effect", "instrument", "plugin", "vst"],
                "context_needed": ["current_effects", "available_devices", "device_parameters"]
            },
            
            # Recording operations
            "recording": {
                "patterns": ["arm", "record", "enable", "monitor", "input"],
                "context_needed": ["arm_states", "monitor_states", "input_routing"]
            },
            
            # Clip operations
            "clips": {
                "patterns": ["launch", "trigger", "fire", "stop", "clip", "scene"],
                "context_needed": ["clip_states", "scene_info", "session_grid"]
            },
            
            # Parameter control (for any device parameter)
            "parameters": {
                "patterns": ["set", "change", "adjust", "increase", "decrease", "raise", "lower"],
                "context_needed": ["current_parameter_values", "parameter_ranges", "device_states"]
            },
            
            # Navigation/View operations
            "navigation": {
                "patterns": ["zoom", "navigate", "view", "focus", "select", "arrange"],
                "context_needed": ["current_view", "selection_state", "zoom_level"]
            },
            
            # Automation operations
            "automation": {
                "patterns": ["automate", "envelope", "curve", "keyframe"],
                "context_needed": ["automation_states", "parameter_automation", "envelope_data"]
            }
        }
    
    def get_universal_contextual_prompt(self, command_text: str) -> str:
        """Generate contextually intelligent prompt for ANY command"""
        
        # Analyze command to understand ALL relevant contexts
        command_analysis = self._universal_command_analysis(command_text)
        
        # Extract ALL relevant session data for this command
        relevant_context = self._extract_universal_context(command_analysis, command_text)
        
        # Build focused prompt with precisely targeted context
        return self._build_universal_prompt(command_text, command_analysis, relevant_context)
    
    def _universal_command_analysis(self, command_text: str) -> Dict[str, Any]:
        """Analyze ANY command to understand what session data is needed"""
        cmd_lower = command_text.lower()
        
        analysis = {
            "command_types": [],
            "target_entities": {
                "tracks": [],
                "devices": [],
                "parameters": [],
                "clips": [],
                "scenes": []
            },
            "operation_type": None,
            "context_scope": [],
            "session_data_needed": set()
        }
        
        # Identify ALL applicable command types (commands can be multi-faceted)
        for cmd_type, config in self.command_patterns.items():
            if any(pattern in cmd_lower for pattern in config["patterns"]):
                analysis["command_types"].append(cmd_type)
                analysis["session_data_needed"].update(config["context_needed"])
        
        # Identify ALL entities mentioned in command
        analysis["target_entities"] = self._identify_all_entities(cmd_lower)
        
        # Determine operation type for intelligent context prioritization
        operation_keywords = {
            "read": ["get", "show", "display", "info", "status", "current", "what"],
            "modify": ["set", "change", "adjust", "increase", "decrease", "raise", "lower"],
            "create": ["create", "make", "add", "new", "insert"],
            "delete": ["delete", "remove", "clear", "reset"],
            "control": ["play", "stop", "record", "launch", "trigger", "arm", "solo", "mute"]
        }
        
        for op_type, keywords in operation_keywords.items():
            if any(keyword in cmd_lower for keyword in keywords):
                analysis["operation_type"] = op_type
                break
        
        return analysis
    
    def _identify_all_entities(self, command_text: str) -> Dict[str, List[str]]:
        """Identify ALL entities (tracks, devices, parameters, etc.) mentioned"""
        session_info = self.tool_manager.get_cached_session_info()
        entities = {
            "tracks": [],
            "devices": [],
            "parameters": [],
            "clips": [],
            "scenes": []
        }
        
        if not session_info:
            return entities
        
        tracks = session_info.get('tracks', [])
        
        # Track identification (direct names + generic references)
        for track in tracks:
            if track['name'].lower() in command_text:
                entities["tracks"].append(track['name'])
        
        # Generic track references
        track_patterns = {
            r"track (\d+)": lambda m: tracks[int(m.group(1)) - 1]['name'] if int(m.group(1)) - 1 < len(tracks) else None,
            r"track (one|two|three|four|five)": lambda m: {
                "one": 0, "two": 1, "three": 2, "four": 3, "five": 4
            }.get(m.group(1)),
            r"(first|second|third|fourth|fifth) track": lambda m: {
                "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4
            }.get(m.group(1)),
            r"(selected|current) track": lambda m: "selected"
        }
        
        for pattern, resolver in track_patterns.items():
            match = re.search(pattern, command_text)
            if match:
                result = resolver(match)
                if isinstance(result, int) and result < len(tracks):
                    track_name = tracks[result]['name']
                    if track_name not in entities["tracks"]:
                        entities["tracks"].append(track_name)
                elif result == "selected":
                    entities["tracks"].append("selected")
        
        # Device/Effect identification
        device_keywords = [
            "reverb", "delay", "compressor", "eq", "saturator", "gate", "limiter",
            "chorus", "flanger", "phaser", "filter", "distortion", "overdrive",
            "bass", "analog", "operator", "wavetable", "drums", "impulse", "synth"
        ]
        for device in device_keywords:
            if device in command_text:
                entities["devices"].append(device)
        
        # Parameter identification
        parameter_keywords = [
            "volume", "pan", "send", "gain", "level", "cutoff", "resonance",
            "attack", "decay", "sustain", "release", "frequency", "pitch", "tempo"
        ]
        for param in parameter_keywords:
            if param in command_text:
                entities["parameters"].append(param)
        
        # Clip/Scene identification
        clip_patterns = {
            r"clip (\d+)": lambda m: int(m.group(1)),
            r"scene (\d+)": lambda m: int(m.group(1)),
            r"(first|second|third) clip": lambda m: {"first": 0, "second": 1, "third": 2}[m.group(1)]
        }
        
        for pattern, resolver in clip_patterns.items():
            match = re.search(pattern, command_text)
            if match:
                result = resolver(match)
                if "clip" in pattern:
                    entities["clips"].append(result)
                else:
                    entities["scenes"].append(result)
        
        return entities
    
    def _extract_universal_context(self, analysis: Dict[str, Any], command_text: str) -> Dict[str, Any]:
        """Extract ALL relevant session data based on comprehensive analysis"""
        session_info = self.tool_manager.get_cached_session_info()
        if not session_info:
            return {"error": "No session data available"}
        
        context = {}
        tracks = session_info.get('tracks', [])
        
        # Extract track context based on targets
        target_tracks = analysis["target_entities"]["tracks"]
        if target_tracks or "track_names" in analysis["session_data_needed"]:
            context["tracks"] = {}
            
            if target_tracks:
                # Specific track context
                for track_identifier in target_tracks:
                    if track_identifier == "selected":
                        selected_track = session_info.get('selected_track', 'None')
                        track_data = next((t for t in tracks if t['name'] == selected_track), None)
                    else:
                        track_data = next((t for t in tracks if t['name'] == track_identifier), None)
                    
                    if track_data:
                        context["tracks"][track_data['name']] = self._get_comprehensive_track_context(track_data, analysis)
            else:
                # All tracks context (for commands like "what tracks exist")
                context["tracks"]["all"] = [t['name'] for t in tracks]
        
        # Extract mixer context if needed
        if "mixer_states" in analysis["session_data_needed"] or "current_levels" in analysis["session_data_needed"]:
            context["mixer"] = {}
            for track in tracks:
                if not target_tracks or track['name'] in target_tracks:
                    context["mixer"][track['name']] = {
                        "volume_db": track.get('volume_db', 0),
                        "pan_percent": track.get('pan_percent', 0),
                        "muted": track.get('muted', False),
                        "soloed": track.get('soloed', False),
                        "sends": track.get('sends', [])
                    }
        
        return context
    
    def _get_comprehensive_track_context(self, track_data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Get complete context for a specific track based on command needs"""
        track_context = {
            "name": track_data['name'],
            "type": track_data.get('type', 'audio'),
            "index": track_data.get('index', 0)
        }
        
        # Add mixer info if command involves mixer operations
        if any(cmd_type in ["mixer", "parameters"] for cmd_type in analysis["command_types"]):
            track_context["mixer"] = {
                "volume_db": track_data.get('volume_db', 0),
                "pan_percent": track_data.get('pan_percent', 0),
                "muted": track_data.get('muted', False),
                "soloed": track_data.get('soloed', False)
            }
        
        return track_context
    
    def _build_universal_prompt(self, command_text: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build contextually intelligent prompt for ANY command"""
        
        # Format context in the most relevant way for this specific command
        context_section = self._format_universal_context(context, analysis)
        
        # Get appropriate action formats for ALL identified command types
        action_formats = self._get_universal_action_formats(analysis["command_types"])
        
        prompt = f'''{context_section}

Command: "{command_text}"

{action_formats}

Use EXACT names from context above. For sequential operations, return JSON array.

JSON:'''
        
        # DEBUG: Print what we're sending to Ollama
        print(f"DEBUG PROMPT:\n{prompt}\n" + "="*50)
        return prompt
    
    def _format_universal_context(self, context: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Format context in the most useful way for this specific command"""
        
        if not context or "error" in context:
            return "No session context available."
        
        context_lines = []
        
        # Track context
        if "tracks" in context:
            if "all" in context["tracks"]:
                track_list = ", ".join(context["tracks"]["all"])
                context_lines.append(f"Available tracks: {track_list}")
            else:
                context_lines.append("Target tracks:")
                for track_name, track_info in context["tracks"].items():
                    context_lines.append(f"  {track_name}:")
                    if "mixer" in track_info:
                        mixer = track_info["mixer"]
                        states = []
                        if mixer["muted"]: states.append("MUTED")
                        if mixer["soloed"]: states.append("SOLO")
                        state_str = f" [{', '.join(states)}]" if states else ""
                        context_lines.append(f"    Volume: {mixer['volume_db']}dB, Pan: {mixer['pan_percent']}%{state_str}")
        
        # Mixer context (if not already covered in tracks)
        if "mixer" in context and "tracks" not in context:
            context_lines.append("Track mixer states:")
            for track_name, mixer_info in context["mixer"].items():
                states = []
                if mixer_info["muted"]: states.append("MUTED")
                if mixer_info["soloed"]: states.append("SOLO")
                state_str = f" [{', '.join(states)}]" if states else ""
                context_lines.append(f"  {track_name}: {mixer_info['volume_db']}dB{state_str}")
        
        return "\n".join(context_lines) if context_lines else "Minimal context available."
    
    def _get_universal_action_formats(self, command_types: List[str]) -> str:
        """Get action formats for ALL applicable command types"""
        
        all_formats = {
            "transport": '''Transport actions:
- Play: {"action": "transport_play"}
- Stop: {"action": "transport_stop"}
- Tempo: {"action": "set_tempo", "tempo": number}''',
            
            "track_management": '''Track actions:
- Create: {"action": "create_tracks", "track_type": "midi/audio", "count": number, "names": ["name1"]}
- Rename: {"action": "rename_track", "track": "exact_name", "new_name": "new_name"}
- Delete: {"action": "delete_track", "track": "exact_name"}''',
            
            "mixer": '''Mixer actions:
- Volume: {"action": "set_parameter", "parameter": "mixer_volume", "target": "track_name", "value": "change"}
- Solo ON: {"action": "set_parameter", "parameter": "mixer_solo", "target": "track_name", "value": "true"}
- Solo OFF (unsolo): {"action": "set_parameter", "parameter": "mixer_solo", "target": "track_name", "value": "false"}
- Mute ON: {"action": "set_parameter", "parameter": "mixer_mute", "target": "track_name", "value": "true"}
- Mute OFF (unmute): {"action": "set_parameter", "parameter": "mixer_mute", "target": "track_name", "value": "false"}''',
            
            "devices": '''Device actions:
- Add: {"action": "add_audio_effect", "track": "track_name", "effect": "effect_name"}''',
            
            "recording": '''Recording actions:
- Arm: {"action": "arm_track", "track": "track_name"}''',
            
            "clips": '''Clip actions:
- Launch: {"action": "launch_clip", "track": "track_name", "clip": number}''',
            
            "parameters": '''Parameter actions:
- Set: {"action": "set_parameter", "parameter": "param_type", "target": "target_name", "value": "new_value"}'''
        }
        
        applicable_formats = []
        for cmd_type in command_types:
            if cmd_type in all_formats:
                applicable_formats.append(all_formats[cmd_type])
        
        return "\n\n".join(applicable_formats) if applicable_formats else "Use appropriate action format."


class AbletonToolManager:
    """Manages Ableton tools and provides MCP-like functionality for Ollama with sequential command support"""
    
    def __init__(self, ableton_host: str = "127.0.0.1", ableton_port: int = 9001, csv_file: str = None):
        self.ableton_host = ableton_host
        self.ableton_port = ableton_port
        self.session_cache = {}
        self.last_session_update = 0
        self.session_cache_timeout = 5.0
        
        # Initialize tool registry
        self.tools: Dict[str, AbletonTool] = {}
        
        # Load tools from CSV if provided, otherwise use manual registration
        if csv_file:
            self._load_tools_from_csv(csv_file)
        else:
            self._register_all_tools()
        
        logger.info(f"Ableton Tool Manager initialized with {len(self.tools)} tools")
    
    def _load_tools_from_csv(self, csv_file: str):
        """Load and generate tools from CSV parameter list"""
        try:
            from parameter_tool_generator import ParameterToolGenerator
            generator = ParameterToolGenerator(csv_file)
            generated_tools = generator.generate_parameter_tools()
            
            for tool in generated_tools:
                self.register_tool(tool)
                
            logger.info(f"Loaded {len(generated_tools)} tools from CSV: {csv_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load CSV tools: {e}, falling back to manual registration")
            self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all available Ableton tools"""
        
        # === TRANSPORT TOOLS ===
        self.register_tool(AbletonTool(
            name="transport_play",
            description="Start playback in Ableton Live",
            category="transport",
            parameters=[],
            examples=["play", "start playback", "hit play"]
        ))
        
        self.register_tool(AbletonTool(
            name="transport_stop", 
            description="Stop playback in Ableton Live",
            category="transport",
            parameters=[],
            examples=["stop", "stop playback", "hit stop"]
        ))
        
        self.register_tool(AbletonTool(
            name="set_tempo",
            description="Change the project tempo in BPM",
            category="transport",
            parameters=[
                ToolParameter("tempo", "number", "Tempo in BPM (20-999)", True)
            ],
            examples=["set tempo to 120", "change tempo to 140 bpm", "tempo 85"]
        ))
        
        # === TRACK CREATION AND MANAGEMENT ===
        self.register_tool(AbletonTool(
            name="create_tracks",
            description="Create new MIDI or audio tracks",
            category="tracks",
            parameters=[
                ToolParameter("track_type", "string", "Type of track", True, ["midi", "audio"]),
                ToolParameter("count", "number", "Number of tracks to create", False, default=1),
                ToolParameter("names", "array", "Names for the tracks", False)
            ],
            examples=["create midi track", "add audio track called vocals", "make 3 midi tracks"]
        ))
        
        self.register_tool(AbletonTool(
            name="rename_track",
            description="Rename an existing track",
            category="tracks", 
            parameters=[
                ToolParameter("track", "string", "Current track name", True),
                ToolParameter("new_name", "string", "New name for track", True)
            ],
            examples=["rename track 1 to drums", "rename bass track to synth bass"]
        ))

        self.register_tool(AbletonTool(
            name="delete_track",
            description="Delete an existing track",
            category="tracks",
            parameters=[
                ToolParameter("track", "string", "Track name to delete", True)
            ],
            examples=["delete drums track", "remove track 2"]
        ))

        self.register_tool(AbletonTool(
            name="duplicate_track", 
            description="Duplicate an existing track",
            category="tracks",
            parameters=[
                ToolParameter("track", "string", "Track to duplicate", True),
                ToolParameter("new_name", "string", "Name for the copy", False)
            ],
            examples=["duplicate bass track", "copy drums track as drums copy"]
        ))

        self.register_tool(AbletonTool(
            name="create_return_track",
            description="Create a return/send track",
            category="tracks",
            parameters=[
                ToolParameter("name", "string", "Name for return track", False, default="Return")
            ],
            examples=["create return track", "add return track called reverb send"]
        ))

        self.register_tool(AbletonTool(
            name="create_group_track",
            description="Create a group track from existing tracks",
            category="tracks",
            parameters=[
                ToolParameter("name", "string", "Name for group track", True),
                ToolParameter("tracks", "array", "Track names to group", True)
            ],
            examples=["group drums and percussion into rhythm section"]
        ))
        
        # === DEVICE/EFFECT TOOLS ===
        self.register_tool(AbletonTool(
            name="add_audio_effect",
            description="Add an audio effect or instrument to a track",
            category="devices",
            parameters=[
                ToolParameter("track", "string", "Track name or 'selected'", True),
                ToolParameter("effect", "string", "Effect name (reverb, delay, compressor, etc.)", True)
            ],
            examples=["add reverb to vocals", "put compressor on drums", "load bass instrument"]
        ))
        
        # === MIXER CONTROLS ===
        self.register_tool(AbletonTool(
            name="set_parameter",
            description="Set mixer parameters like volume, pan, solo, mute",
            category="mixer",
            parameters=[
                ToolParameter("parameter", "string", "Parameter to set", True, 
                            ["mixer_volume", "mixer_pan", "mixer_solo", "mixer_mute", "mixer_arm"]),
                ToolParameter("target", "string", "Track name or 'selected'", True),
                ToolParameter("value", "string", "Value (dB, %, true/false, etc.)", True)
            ],
            examples=["set drums volume to -10 dB", "solo vocal track", "mute bass", "pan left 50%"]
        ))
        
        self.register_tool(AbletonTool(
            name="arm_track",
            description="Arm a track for recording",
            category="recording",
            parameters=[
                ToolParameter("track", "string", "Track name or 'selected'", True)
            ],
            examples=["arm vocal track", "record enable drums", "arm for recording"]
        ))
        
        # === SESSION INFO ===
        self.register_tool(AbletonTool(
            name="get_session_info",
            description="Get current session information including tracks, tempo, playing status",
            category="info",
            parameters=[],
            examples=["get session info", "what tracks exist", "current status"]
        ))
        
        self.register_tool(AbletonTool(
            name="get_detailed_session_info",
            description="Get comprehensive session info with track details, effects, levels",
            category="info", 
            parameters=[],
            examples=["detailed session info", "full session status", "track details"]
        ))
        
        # === MASTER CONTROLS ===
        self.register_tool(AbletonTool(
            name="set_master_volume",
            description="Set the master volume level",
            category="master",
            parameters=[
                ToolParameter("value", "string", "Volume level (dB, % or relative)", True)
            ],
            examples=["set master volume to -10 dB", "reduce master volume", "master at 80%"]
        ))
        
        # === CLIP CONTROLS ===
        self.register_tool(AbletonTool(
            name="launch_clip",
            description="Launch a specific clip in a track",
            category="clips",
            parameters=[
                ToolParameter("track", "string", "Track name", True),
                ToolParameter("clip", "number", "Clip slot index (0-based)", True)
            ],
            examples=["launch clip 1 in drums", "play first clip on bass", "trigger clip"]
        ))
    
    def register_tool(self, tool: AbletonTool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_available_tools(self) -> List[AbletonTool]:
        """Get list of all available tools"""
        return list(self.tools.values())
    
    def get_tools_by_category(self, category: str) -> List[AbletonTool]:
        """Get tools filtered by category"""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def get_tool(self, name: str) -> Optional[AbletonTool]:
        """Get a specific tool by name"""
        return self.tools.get(name)
    
    def get_cached_session_info(self) -> Optional[Dict[str, Any]]:
        """Get session info with caching to avoid overloading Ableton"""
        current_time = time.time()
        
        if (current_time - self.last_session_update) > self.session_cache_timeout:
            try:
                response = self._send_to_ableton({"action": "get_detailed_session_info"})
                if response.get('status') == 'success':
                    self.session_cache = response.get('result', {})
                    self.last_session_update = current_time
                    logger.debug("Session cache updated")
                else:
                    logger.warning(f"Failed to update session cache: {response}")
            except Exception as e:
                logger.error(f"Error updating session cache: {e}")
        
        return self.session_cache
    
    def validate_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation with comprehensive parameter mapping"""
        action = command.get("action")
        
        if not action:
            return {"error": "No action specified"}
        
        # === COMPREHENSIVE PARAMETER MAPPING ===
        if action == "create_tracks":
            if "param1" in command: command["track_type"] = command.pop("param1")
            if "param2" in command: command["count"] = command.pop("param2") 
            if "param3" in command: command["names"] = command.pop("param3")
            if "type" in command: command["track_type"] = command.pop("type")
            if "number" in command: command["count"] = command.pop("number")
            
        elif action == "name":
            command["action"] = "rename_track"
            if "object" in command:
                command["track"] = command.pop("object") 
            if "value" in command:
                command["new_name"] = command.pop("value")
            
        elif action == "rename_track":
            if "param1" in command: command["track"] = command.pop("param1")
            if "param2" in command: command["new_name"] = command.pop("param2")
            if "current_name" in command: command["track"] = command.pop("current_name")
            
        elif action == "set_parameter":
            if "param1" in command: command["parameter"] = command.pop("param1")
            if "param2" in command: command["target"] = command.pop("param2")
            if "param3" in command: command["value"] = command.pop("param3")
            if "track_name" in command: command["target"] = command.pop("track_name")
        
        # Continue with existing validation...
        tool = self.get_tool(action)
        if not tool:
            return {"error": f"Unknown action: {action}"}
        
        # Validate required parameters
        for param in tool.parameters:
            if param.required and param.name not in command:
                return {"error": f"Missing required parameter: {param.name}"}
        
        # Apply defaults
        for param in tool.parameters:
            if not param.required and param.name not in command and param.default is not None:
                command[param.name] = param.default
        
        # Enhanced validation for specific tools
        if action == "set_tempo":
            tempo = command.get("tempo")
            if not isinstance(tempo, (int, float)) or not (20 <= tempo <= 999):
                return {"error": f"Invalid tempo: {tempo}. Must be between 20-999 BPM"}
        
        elif action == "create_tracks":
            track_type = command.get("track_type", "").lower()
            if track_type not in ["midi", "audio"]:
                return {"error": f"Invalid track_type: {track_type}. Must be 'midi' or 'audio'"}
        
        # Track name resolution
        if "track" in command:
            track_name = command["track"]
            if track_name != "selected":
                session_info = self.get_cached_session_info()
                if session_info:
                    tracks = session_info.get('tracks', [])
                    track_names = [t['name'].lower() for t in tracks]
                    
                    if track_name.lower() in track_names:
                        for track in tracks:
                            if track['name'].lower() == track_name.lower():
                                command["track"] = track['name']
                                break
                    else:
                        matches = [t['name'] for t in tracks if track_name.lower() in t['name'].lower()]
                        if matches:
                            command["track"] = matches[0]
        
        return command
    
    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validated command on Ableton"""
        try:
            response = self._send_to_ableton(command)
            
            # Clear session cache after state-changing commands
            if self._is_state_changing_command(command.get("action")):
                self.last_session_update = 0  # Force cache refresh
            
            return response
            
        except Exception as e:
            return {"status": "error", "message": f"Execution failed: {str(e)}"}
    
    def _is_state_changing_command(self, action: str) -> bool:
        """Check if command changes session state"""
        state_changing = {
            "create_tracks", "add_audio_effect", "set_parameter", 
            "arm_track", "disarm_track", "set_tempo", "set_master_volume",
            "launch_clip", "stop_clip", "rename_track", "delete_track",
            "duplicate_track", "create_return_track", "create_group_track"
        }
        return action in state_changing
    
    def _send_to_ableton(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to Ableton via socket"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.ableton_host, self.ableton_port))
            sock.send(json.dumps(command).encode('utf-8'))
            response = json.loads(sock.recv(4096).decode('utf-8'))
            sock.close()
            return response
        except Exception as e:
            raise Exception(f"Failed to communicate with Ableton: {str(e)}")
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Enhanced JSON extraction with array support for sequential commands"""
        
        # Strategy 1: Find complete JSON objects or arrays
        json_objects = []
        brace_count = 0
        bracket_count = 0
        start_pos = -1
        
        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0 and bracket_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0 and start_pos != -1:
                    json_str = response[start_pos:i+1]
                    json_objects.append(json_str)
            elif char == '[':
                if bracket_count == 0 and brace_count == 0:
                    start_pos = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0 and brace_count == 0 and start_pos != -1:
                    json_str = response[start_pos:i+1]
                    json_objects.append(json_str)
        
        # Try parsing each JSON object/array
        for json_str in reversed(json_objects):
            try:
                parsed = json.loads(json_str)
                print(f"DEBUG: Parsed JSON type: {type(parsed)}")
                
                if isinstance(parsed, list) and len(parsed) > 0:
                    print(f"DEBUG: Found array with {len(parsed)} commands")
                    valid_commands = []
                    for cmd in parsed:
                        if isinstance(cmd, dict) and 'action' in cmd:
                            valid_commands.append(cmd)
                    
                    if valid_commands:
                        print(f"DEBUG: Returning sequential format with {len(valid_commands)} commands")
                        return {"_sequential_commands": valid_commands}
                # ... rest of method
                # Handle single command
                elif isinstance(parsed, dict) and 'action' in parsed:
                    return parsed
                    
            except json.JSONDecodeError:
                continue
        
        # Fallback to existing single command extraction
        return self._extract_single_command_fallback(response)

    def _extract_single_command_fallback(self, response: str) -> Optional[Dict[str, Any]]:
        """Fallback for single command extraction"""
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
        
        return None
    
    def process_voice_command(self, command_text: str, ollama_response: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced command processing with sequential execution support"""
        
        print(f"DEBUG: ollama_response type: {type(ollama_response)}")
        print(f"DEBUG: ollama_response keys: {ollama_response.keys() if isinstance(ollama_response, dict) else 'Not a dict'}")
        
        # Check for sequential commands
        if isinstance(ollama_response, dict) and "_sequential_commands" in ollama_response:
            print(f"DEBUG: Found sequential commands: {len(ollama_response['_sequential_commands'])}")
            return self._execute_sequential_commands(command_text, ollama_response["_sequential_commands"])
        
        print("DEBUG: Processing as single command")
        
        # Check for sequential commands
        if isinstance(ollama_response, dict) and "_sequential_commands" in ollama_response:
            return self._execute_sequential_commands(command_text, ollama_response["_sequential_commands"])
        
        # Handle single command (existing logic)
        if not isinstance(ollama_response, dict) or "action" not in ollama_response:
            return {
                "status": "error", 
                "message": "Invalid command format from Ollama",
                "raw_response": ollama_response
            }
        
        # Validate and execute single command
        validated_command = self.validate_command(ollama_response)
        if "error" in validated_command:
            return {"status": "error", "message": validated_command["error"]}
        
        result = self.execute_command(validated_command)
        result["original_command"] = command_text
        result["interpreted_as"] = validated_command
        result["tool_used"] = self.get_tool(validated_command["action"])
        
        return result

    def _execute_sequential_commands(self, command_text: str, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple commands in sequence"""
        
        results = []
        successful_count = 0
        
        print(f"Executing {len(commands)} commands sequentially...")
        
        for i, command in enumerate(commands):
            print(f"  Command {i+1}/{len(commands)}: {command.get('action', 'unknown')}")
            
            # Validate command
            validated_command = self.validate_command(command.copy())
            if "error" in validated_command:
                result = {
                    "command_index": i,
                    "status": "error", 
                    "message": validated_command["error"],
                    "command": command
                }
                results.append(result)
                print(f"    ✗ Failed: {validated_command['error']}")
                continue
            
            # Execute command
            try:
                result = self.execute_command(validated_command)
                result["command_index"] = i
                result["command"] = validated_command
                result["tool_used"] = self.get_tool(validated_command["action"])
                results.append(result)
                
                if result.get("status") == "success":
                    successful_count += 1
                    print(f"    ✓ Success")
                else:
                    print(f"    ✗ Failed: {result.get('message', 'Unknown error')}")
                    
                # Small delay between commands to avoid overwhelming Ableton
                time.sleep(0.2)
                
            except Exception as e:
                result = {
                    "command_index": i,
                    "status": "error",
                    "message": f"Execution failed: {str(e)}",
                    "command": validated_command
                }
                results.append(result)
                print(f"    ✗ Exception: {str(e)}")
        
        # Return summary result
        return {
            "status": "sequential_execution",
            "original_command": command_text,
            "total_commands": len(commands),
            "successful_commands": successful_count,
            "failed_commands": len(commands) - successful_count,
            "results": results,
            "summary": f"Executed {successful_count}/{len(commands)} commands successfully"
        }
    
    def get_context_for_ollama(self, command_text: str) -> str:
        """Universal contextually intelligent prompt for ANY command"""
        processor = UniversalContextualProcessor(self)
        return processor.get_universal_contextual_prompt(command_text)


# Example usage integrating with existing voice control
if __name__ == "__main__":
    # Initialize tool manager
    tool_manager = AbletonToolManager()
    
    # Example: Get tools prompt for Ollama
    print("=== AVAILABLE TOOLS ===")
    prompt = tool_manager.generate_tools_prompt()
    print(prompt[:1000] + "...")  # Show first 1000 chars
    
    # Example: Process a voice command
    example_command = "set tempo to 120"
    context_prompt = tool_manager.get_context_for_ollama(example_command)
    
    # This would go to Ollama:
    print(f"\n=== PROMPT FOR OLLAMA ===")
    print(f"Context length: {len(context_prompt)} characters")
    
    # Simulated Ollama response
    ollama_response = {"action": "set_tempo", "tempo": 120}
    
    # Process through tool manager
    result = tool_manager.process_voice_command(example_command, ollama_response)
    print(f"\n=== RESULT ===")
    print(json.dumps(result, indent=2))