#!/usr/bin/env python3
"""
Ableton Parameter Tool Generator
Auto-generates Tool Manager tools from CSV parameter list
"""

import csv
import json
from typing import Dict, List, Any
from ableton_tool_manager import AbletonTool, ToolParameter

class ParameterToolGenerator:
    """Generate tools automatically from Ableton parameter CSV"""
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.parameters = []
        self.load_parameters()
        
    def load_parameters(self):
        """Load parameters from CSV"""
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.parameters = list(reader)
            print(f"Loaded {len(self.parameters)} parameters from CSV")
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    def generate_parameter_tools(self) -> List[AbletonTool]:
        """Generate tools for all parameters"""
        tools = []
        
        # Group parameters by category for better organization
        category_groups = self._group_by_category()
        
        for category, params in category_groups.items():
            # Create category-specific tools
            category_tools = self._create_category_tools(category, params)
            tools.extend(category_tools)
        
        return tools
    
    def _group_by_category(self) -> Dict[str, List[Dict]]:
        """Group parameters by category"""
        groups = {}
        for param in self.parameters:
            category = param['Category']
            if category not in groups:
                groups[category] = []
            groups[category].append(param)
        return groups
    
    def _create_category_tools(self, category: str, params: List[Dict]) -> List[AbletonTool]:
        """Create tools for a specific category"""
        tools = []
        
        if category == "Transport":
            tools.extend(self._create_transport_tools(params))
        elif category == "Mixer":
            tools.extend(self._create_mixer_tools(params))
        elif category == "Instruments":
            tools.extend(self._create_instrument_tools(params))
        elif category == "Audio Effects":
            tools.extend(self._create_audio_effect_tools(params))
        elif category == "MIDI Effects":
            tools.extend(self._create_midi_effect_tools(params))
        elif category == "Session View":
            tools.extend(self._create_session_tools(params))
        elif category == "Arrangement View":
            tools.extend(self._create_arrangement_tools(params))
        else:
            # Generic parameter control for other categories
            tools.extend(self._create_generic_parameter_tools(category, params))
        
        return tools
    
    def _create_transport_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create transport-specific tools"""
        tools = []
        
        # Group transport parameters into logical tools
        tempo_params = [p for p in params if 'tempo' in p['Parameter Name'].lower()]
        time_sig_params = [p for p in params if 'time' in p['Parameter Name'].lower() or 'signature' in p['Parameter Name'].lower()]
        
        if tempo_params:
            tools.append(AbletonTool(
                name="set_transport_tempo",
                description="Set the project tempo (BPM)",
                category="transport",
                parameters=[
                    ToolParameter("tempo", "number", "Tempo in BPM", True)
                ],
                examples=["set tempo to 120", "change tempo to 140 bpm"]
            ))
        
        # Add other transport tools...
        tools.append(AbletonTool(
            name="control_transport",
            description="Control transport playback state",
            category="transport", 
            parameters=[
                ToolParameter("action", "string", "Transport action", True, 
                            ["play", "stop", "record", "continue"])
            ],
            examples=["play", "stop playback", "start recording"]
        ))
        
        return tools
    
    def _create_mixer_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create mixer-specific tools"""
        tools = []
        
        # Volume control tool
        volume_params = [p for p in params if 'volume' in p['Parameter Name'].lower()]
        if volume_params:
            tools.append(AbletonTool(
                name="set_track_volume",
                description="Set track volume level",
                category="mixer",
                parameters=[
                    ToolParameter("track", "string", "Track name or index", True),
                    ToolParameter("volume", "string", "Volume level (dB, %, or 0-1)", True)
                ],
                examples=["set drums volume to -10 dB", "set track 1 volume to 80%"]
            ))
        
        # Pan control tool
        pan_params = [p for p in params if 'pan' in p['Parameter Name'].lower()]
        if pan_params:
            tools.append(AbletonTool(
                name="set_track_pan",
                description="Set track panning position",
                category="mixer",
                parameters=[
                    ToolParameter("track", "string", "Track name or index", True),
                    ToolParameter("pan", "number", "Pan position (-100 to 100)", True)
                ],
                examples=["pan drums left 50", "center the vocal track"]
            ))
        
        # Send control tool
        send_params = [p for p in params if 'send' in p['Parameter Name'].lower()]
        if send_params:
            tools.append(AbletonTool(
                name="set_track_send",
                description="Set track send level",
                category="mixer",
                parameters=[
                    ToolParameter("track", "string", "Track name or index", True),
                    ToolParameter("send", "string", "Send letter (A, B, C, D)", True),
                    ToolParameter("level", "number", "Send level (0-100)", True)
                ],
                examples=["set drums send A to 50", "send vocals to reverb bus"]
            ))
        
        return tools
    
    def _create_instrument_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create instrument-specific parameter tools"""
        tools = []
        
        # Group by instrument type
        instruments = {}
        for param in params:
            instrument = param['Sub-Category']
            if instrument not in instruments:
                instruments[instrument] = []
            instruments[instrument].append(param)
        
        # Create tools for major instruments
        for instrument_name, instrument_params in instruments.items():
            if len(instrument_params) > 5:  # Only create tools for instruments with many parameters
                tools.append(AbletonTool(
                    name=f"control_{instrument_name.lower().replace(' ', '_')}",
                    description=f"Control {instrument_name} instrument parameters",
                    category="instruments",
                    parameters=[
                        ToolParameter("track", "string", "Track with this instrument", True),
                        ToolParameter("parameter", "string", "Parameter to control", True),
                        ToolParameter("value", "string", "Parameter value", True)
                    ],
                    examples=[f"set {instrument_name.lower()} filter cutoff to 50"]
                ))
        
        return tools
    
    def _create_audio_effect_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create audio effect parameter tools"""
        tools = []
        
        # Group by effect type
        effects = {}
        for param in params:
            effect = param['Sub-Category']
            if effect not in effects:
                effects[effect] = []
            effects[effect].append(param)
        
        # Create tools for common effects
        common_effects = ['EQ Eight', 'Compressor', 'Reverb', 'Delay', 'Saturator']
        
        for effect_name in common_effects:
            if effect_name in effects:
                effect_params = effects[effect_name]
                tools.append(AbletonTool(
                    name=f"control_{effect_name.lower().replace(' ', '_')}",
                    description=f"Control {effect_name} effect parameters",
                    category="audio_effects",
                    parameters=[
                        ToolParameter("track", "string", "Track with this effect", True),
                        ToolParameter("parameter", "string", "Effect parameter to control", True),
                        ToolParameter("value", "string", "Parameter value", True)
                    ],
                    examples=[f"set {effect_name.lower()} on vocals"]
                ))
        
        return tools
    
    def _create_midi_effect_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create MIDI effect tools"""
        tools = []
        
        tools.append(AbletonTool(
            name="control_midi_effect",
            description="Control MIDI effect parameters",
            category="midi_effects", 
            parameters=[
                ToolParameter("track", "string", "Track with MIDI effect", True),
                ToolParameter("effect", "string", "MIDI effect name", True),
                ToolParameter("parameter", "string", "Parameter to control", True),
                ToolParameter("value", "string", "Parameter value", True)
            ],
            examples=["set arpeggiator rate to 1/8", "control scale on track 2"]
        ))
        
        return tools
    
    def _create_session_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create session view tools"""
        tools = []
        
        tools.extend([
            AbletonTool(
                name="launch_clip",
                description="Launch a clip in session view",
                category="session",
                parameters=[
                    ToolParameter("track", "string", "Track name or index", True),
                    ToolParameter("scene", "number", "Scene index", True)
                ],
                examples=["launch clip 1 in drums", "play scene 3"]
            ),
            AbletonTool(
                name="stop_clips",
                description="Stop clips in session view",
                category="session",
                parameters=[
                    ToolParameter("track", "string", "Track name or 'all'", False, default="all")
                ],
                examples=["stop all clips", "stop drums track"]
            )
        ])
        
        return tools
    
    def _create_arrangement_tools(self, params: List[Dict]) -> List[AbletonTool]:
        """Create arrangement view tools"""
        tools = []
        
        tools.extend([
            AbletonTool(
                name="set_loop_region",
                description="Set loop region in arrangement view",
                category="arrangement",
                parameters=[
                    ToolParameter("start", "string", "Loop start position", True),
                    ToolParameter("length", "string", "Loop length", True)
                ],
                examples=["loop from bar 1 to 5", "set loop region 8 bars"]
            ),
            AbletonTool(
                name="navigate_arrangement",
                description="Navigate in arrangement view",
                category="arrangement",
                parameters=[
                    ToolParameter("action", "string", "Navigation action", True,
                                ["zoom_to_fit", "zoom_in", "zoom_out", "go_to_start", "go_to_end"])
                ],
                examples=["zoom to fit", "go to beginning"]
            )
        ])
        
        return tools
    
    def _create_generic_parameter_tools(self, category: str, params: List[Dict]) -> List[AbletonTool]:
        """Create generic parameter control tools"""
        tools = []
        
        # Create a generic parameter control tool for this category
        tools.append(AbletonTool(
            name=f"control_{category.lower().replace(' ', '_')}_parameter",
            description=f"Control {category} parameters",
            category=category.lower().replace(' ', '_'),
            parameters=[
                ToolParameter("target", "string", "Target object (track, device, etc.)", True),
                ToolParameter("parameter", "string", "Parameter name", True),
                ToolParameter("value", "string", "Parameter value", True)
            ],
            examples=[f"control {category.lower()} parameter"]
        ))
        
        return tools
    
    def generate_parameter_mapping(self) -> Dict[str, Any]:
        """Generate a mapping of parameter names to their details"""
        mapping = {}
        
        for param in self.parameters:
            key = f"{param['Category']}.{param['Sub-Category']}.{param['Parameter Name']}"
            mapping[key] = {
                "type": param['Parameter Type'],
                "range": param['Range/Values'],
                "description": param['Description'],
                "category": param['Category'],
                "sub_category": param['Sub-Category']
            }
        
        return mapping
    
    def save_generated_tools(self, output_file: str = "generated_ableton_tools.json"):
        """Save generated tools to JSON file for inspection"""
        tools = self.generate_parameter_tools()
        tool_data = []
        
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "enum": p.enum,
                        "default": p.default
                    } for p in tool.parameters
                ],
                "examples": tool.examples
            }
            tool_data.append(tool_dict)
        
        with open(output_file, 'w') as f:
            json.dump(tool_data, f, indent=2)
        
        print(f"Generated {len(tools)} tools, saved to {output_file}")
        return tools

# Usage example
if __name__ == "__main__":
    # Generate tools from your CSV
    generator = ParameterToolGenerator("Ableton_parameter_list.csv")
    
    # Generate all tools
    tools = generator.generate_parameter_tools()
    print(f"Generated {len(tools)} tools")
    
    # Show tool categories
    categories = {}
    for tool in tools:
        if tool.category not in categories:
            categories[tool.category] = 0
        categories[tool.category] += 1
    
    print("\nTools by category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} tools")
    
    # Save for inspection
    generator.save_generated_tools()
    
    # Generate parameter mapping
    param_mapping = generator.generate_parameter_mapping()
    print(f"\nParameter mapping contains {len(param_mapping)} parameters")