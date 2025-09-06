#!/usr/bin/env python3
"""
Test microphone and speech recognition before full voice control
"""

import speech_recognition as sr
import time

def test_microphone():
    """Test if microphone is working"""
    print("🎤 Testing Microphone & Speech Recognition...")
    
    # Initialize recognizer and microphone
    r = sr.Recognizer()
    
    # List available microphones
    print("\n📱 Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  {index}: {name}")
    
    # Use default microphone
    mic = sr.Microphone()
    
    # Get microphone name safely
    mic_names = sr.Microphone.list_microphone_names()
    if mic.device_index is not None:
        mic_name = mic_names[mic.device_index]
    else:
        mic_name = mic_names[0]  # Default to first microphone
    
    print(f"\n🔧 Using microphone: {mic_name}")
    
    # Calibrate for ambient noise
    print("\n🔇 Calibrating for ambient noise... (stay quiet for 2 seconds)")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=2)
    
    print(f"✅ Energy threshold set to: {r.energy_threshold}")
    
    # Test speech recognition
    for test_num in range(3):
        print(f"\n🎙️ TEST {test_num + 1}/3")
        print("Say something like: 'Hello, can you hear me?'")
        print("Listening...")
        
        try:
            with mic as source:
                # Listen for audio with timeout
                audio = r.listen(source, timeout=10, phrase_time_limit=5)
            
            print("🔄 Processing speech...")
            
            # Try Google Speech Recognition first
            try:
                text = r.recognize_google(audio)
                print(f"✅ Google recognized: '{text}'")
            except sr.UnknownValueError:
                print("❌ Google could not understand audio")
                continue
            except sr.RequestError as e:
                print(f"⚠️ Google service error: {e}")
                
                # Fallback to offline recognition
                try:
                    text = r.recognize_sphinx(audio)
                    print(f"✅ Sphinx recognized: '{text}'")
                except:
                    print("❌ Offline recognition also failed")
                    continue
            
            # Test Ableton-specific phrases
            text_lower = text.lower()
            if any(word in text_lower for word in ['ableton', 'live', 'music', 'track', 'volume']):
                print("🎵 Great! Detected music-related terms.")
            
            break
            
        except sr.WaitTimeoutError:
            print("⏰ Timeout - no speech detected")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    else:
        print("❌ All speech recognition tests failed")
        print("💡 Try:")
        print("   - Check microphone permissions")
        print("   - Speak louder and clearer")
        print("   - Try a different microphone")
        return False
    
    return True

def test_ableton_phrases():
    """Test recognition of Ableton-specific commands"""
    print("\n🎵 Testing Ableton Command Recognition...")
    
    r = sr.Recognizer()
    mic = sr.Microphone()
    
    test_phrases = [
        "create two MIDI tracks",
        "set volume to minus five",
        "set tempo to one twenty eight",
        "ableton create track"
    ]
    
    print("Try saying one of these test phrases:")
    for phrase in test_phrases:
        print(f"  - '{phrase}'")
    
    print("\n🎙️ Listening for Ableton commands...")
    
    try:
        with mic as source:
            audio = r.listen(source, timeout=15, phrase_time_limit=8)
        
        text = r.recognize_google(audio)
        print(f"🎤 You said: '{text}'")
        
        # Check if it matches our expected command patterns
        text_lower = text.lower()
        
        if 'create' in text_lower and ('track' in text_lower or 'midi' in text_lower):
            print("✅ Detected track creation command!")
        elif 'volume' in text_lower and ('set' in text_lower or 'minus' in text_lower):
            print("✅ Detected volume control command!")
        elif 'tempo' in text_lower:
            print("✅ Detected tempo control command!")
        elif 'ableton' in text_lower or 'live' in text_lower:
            print("✅ Detected wake word!")
        else:
            print("🤔 Command not recognized, but speech works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing phrases: {e}")
        return False

if __name__ == "__main__":
    print("🎤 ABLETON VOICE CONTROL - MICROPHONE TEST")
    print("=" * 50)
    
    # Test basic microphone functionality
    if test_microphone():
        print("\n" + "=" * 50)
        
        # Test Ableton-specific phrases
        test_ableton_phrases()
        
        print("\n🎉 Microphone testing complete!")
        print("✅ Ready to proceed with full voice control setup!")
    else:
        print("\n❌ Microphone setup needs attention before proceeding")
        print("💡 Make sure:")
        print("   - Microphone is connected and working")
        print("   - Python has microphone permissions") 
        print("   - You're in a quiet environment")