#!/usr/bin/env python3
"""
Debug script to check what session state information is being sent to AI
"""

import socket
import json

def test_session_state():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", 9001))
        
        # Get detailed session info
        command = {"action": "get_detailed_session_info"}
        sock.send(json.dumps(command).encode('utf-8'))
        response = json.loads(sock.recv(4096).decode('utf-8'))
        sock.close()
        
        if response.get('status') == 'success':
            result = response.get('result', {})
            
            print("=== CURRENT SESSION STATE ===")
            print(f"Selected track: {result.get('selected_track', 'None')}")
            print(f"Total tracks: {len(result.get('tracks', []))}")
            
            print("\n=== ALL TRACKS ===")
            for track in result.get('tracks', []):
                selected_flag = " (SELECTED)" if track.get('selected') else ""
                effects_count = len(track.get('effects', []))
                print(f"- \"{track['name']}\"{selected_flag} - {effects_count} effects")
                
                if track.get('effects'):
                    for effect in track['effects']:
                        status = "active" if effect.get('active') else "inactive"
                        print(f"    • {effect['name']} ({status})")
            
            print(f"\n=== DEVICE CACHE ===")
            device_info = result.get('available_devices_sample', {})
            print(f"Total cached devices: {device_info.get('total_cached', 'unknown')}")
            
        else:
            print(f"Error: {response.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

def test_device_search():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", 9001))
        
        # Search for reverb devices
        command = {"action": "search_devices", "query": "reverb", "limit": 5}
        sock.send(json.dumps(command).encode('utf-8'))
        response = json.loads(sock.recv(4096).decode('utf-8'))
        sock.close()
        
        if response.get('status') == 'success':
            result = response.get('result', {})
            matches = result.get('matches', [])
            
            print(f"\n=== REVERB DEVICES FOUND ({len(matches)}) ===")
            for device in matches:
                print(f"- {device['name']} ({device['category']})")
                print(f"  Path: {device.get('path', 'Unknown')}")
        else:
            print(f"Device search error: {response.get('message')}")
            
    except Exception as e:
        print(f"Device search failed: {e}")

if __name__ == "__main__":
    test_session_state()
    test_device_search()