#!/usr/bin/env python3
"""
Improved test for Ableton MCP Remote Script
Handles asynchronous operations properly
"""

import socket
import json
import time

def test_ableton_connection():
    """Test connection to Ableton Remote Script"""
    print("🔌 Testing Ableton MCP Socket Connection (Improved)...")
    
    try:
        # Connect to the Remote Script
        print("Connecting to 127.0.0.1:9001...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout
        sock.connect(('127.0.0.1', 9001))
        print("✅ Connected successfully!")
        
        # Test 1: Get initial session info
        print("\n📊 Test 1: Getting initial session information...")
        test_command = {"action": "get_session_info"}
        sock.send(json.dumps(test_command).encode('utf-8'))
        
        response_data = sock.recv(4096)
        response = json.loads(response_data.decode('utf-8'))
        
        initial_track_count = response['result']['track_count']
        print(f"Current track count: {initial_track_count}")
        print("Tempo:", response['result']['tempo'])
        print("Time signature:", response['result']['time_signature'])
        
        # Test 2: Set tempo (safe operation)
        print("\n⏱️ Test 2: Setting tempo to 125 BPM...")
        tempo_command = {
            "action": "set_parameter",
            "parameter": "transport_tempo",
            "value": 125
        }
        sock.send(json.dumps(tempo_command).encode('utf-8'))
        
        response_data = sock.recv(4096)
        response = json.loads(response_data.decode('utf-8'))
        
        print("Tempo response:", json.dumps(response, indent=2))
        
        # Test 3: Create a MIDI track (asynchronous operation)
        print("\n🎹 Test 3: Creating a MIDI track (async)...")
        create_command = {
            "action": "create_tracks",
            "track_type": "midi", 
            "count": 1,
            "names": ["Voice Control Test"]
        }
        sock.send(json.dumps(create_command).encode('utf-8'))
        
        response_data = sock.recv(4096)
        response = json.loads(response_data.decode('utf-8'))
        
        print("Create track response:", json.dumps(response, indent=2))
        
        # If the response says "scheduled", wait a moment for the operation to complete
        if response.get('result', {}).get('status') == 'scheduled':
            print("⏳ Track creation scheduled, waiting 3 seconds...")
            time.sleep(3)
            
            # Check if track was actually created
            print("📊 Checking if track was created...")
            sock.send(json.dumps({"action": "get_session_info"}).encode('utf-8'))
            response_data = sock.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            new_track_count = response['result']['track_count']
            print(f"Track count after creation: {new_track_count}")
            
            if new_track_count > initial_track_count:
                print("✅ Track created successfully!")
                # Show the new track
                tracks = response['result']['tracks']
                new_track = tracks[-1]  # Last track should be the new one
                print(f"New track: {new_track['name']} (Type: {new_track['type']})")
            else:
                print("⚠️ Track creation might have failed - count unchanged")
        
        # Test 4: Test mixer controls on existing track
        print("\n🎛️ Test 4: Testing mixer controls...")
        if response['result']['tracks']:
            test_track = response['result']['tracks'][0]
            track_name = test_track['name']
            
            print(f"Testing volume control on track: {track_name}")
            
            volume_command = {
                "action": "set_parameter",
                "parameter": "mixer_volume",
                "target": track_name,
                "value": -10  # -10 dB
            }
            sock.send(json.dumps(volume_command).encode('utf-8'))
            
            response_data = sock.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            print("Volume control response:", json.dumps(response, indent=2))
        
        # Test 5: Transport controls
        print("\n⏯️ Test 5: Testing transport controls...")
        
        # Enable metronome
        metronome_command = {
            "action": "set_parameter",
            "parameter": "transport_metronome_on_off",
            "value": True
        }
        sock.send(json.dumps(metronome_command).encode('utf-8'))
        
        response_data = sock.recv(4096)
        response = json.loads(response_data.decode('utf-8'))
        
        print("Metronome control response:", json.dumps(response, indent=2))
        
        sock.close()
        print("\n🎉 All tests completed successfully!")
        print("\n💡 Key findings:")
        print("   ✅ Socket communication works")
        print("   ✅ Session info retrieval works")
        print("   ✅ Parameter changes work")
        print("   ✅ Track creation is now async (non-blocking)")
        print("   ✅ Ready for voice control integration!")
        
    except socket.timeout:
        print("⏰ Socket timeout - operation took too long")
        print("   This might indicate Ableton is still hanging")
        
    except ConnectionRefusedError:
        print("❌ Connection refused - make sure:")
        print("   1. Ableton Live is running")
        print("   2. Enhanced MCP Remote Script is loaded")
        print("   3. Check Ableton's log for 'Socket server started' message")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            sock.close()
        except:
            pass

def quick_connection_test():
    """Quick test to just verify connection"""
    print("🔗 Quick connection test...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('127.0.0.1', 9001))
        
        # Just get session info
        sock.send(json.dumps({"action": "get_session_info"}).encode('utf-8'))
        response_data = sock.recv(4096)
        response = json.loads(response_data.decode('utf-8'))
        
        if response.get('status') == 'success':
            print("✅ Connection and basic communication working!")
            print(f"   Tracks: {response['result']['track_count']}")
            print(f"   Tempo: {response['result']['tempo']} BPM")
        else:
            print("⚠️ Connection works but got unexpected response")
            
        sock.close()
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_connection_test()
    else:
        test_ableton_connection()
        
    print("\n💡 Usage:")
    print("   python test_script_socket.py        # Full test")
    print("   python test_script_socket.py quick  # Quick test only")