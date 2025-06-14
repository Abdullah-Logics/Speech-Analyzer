import speech_recognition as sr
import wave
import math
import numpy as np
import librosa
from datetime import timedelta
import os
from scipy.signal import find_peaks
import google.generativeai as genai

api_key = "AIzaSyDRMAt_0IAMqj_k-YycgHPzsC1Sg2okHFU"

def aiBot(prompt):
    try:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        
        reply = response.text
        reply = reply.replace("*", "")
        reply = reply.replace("#", "")
        return reply
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_wav_duration(audio_path):
    """Get duration of a WAV file in seconds."""
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def analyze_audio_features(audio_path, start_time, end_time):
    """Analyze pitch frequency and loudness for a specific segment."""
    try:
        # Load audio segment
        y, sr = librosa.load(audio_path, offset=start_time, duration=end_time-start_time)
        
        # Calculate pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        avg_pitch = np.mean(pitch_values) if pitch_values else 0
        
        # Calculate loudness (RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        avg_loudness = np.mean(rms)
        loudness_db = librosa.amplitude_to_db([avg_loudness])[0]
        
        # Calculate additional features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        return {
            'avg_pitch': avg_pitch,
            'avg_loudness': avg_loudness,
            'loudness_db': loudness_db,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zero_crossing_rate
        }
    except Exception as e:
        print(f"Error analyzing audio features: {e}")
        return {
            'avg_pitch': 0,
            'avg_loudness': 0,
            'loudness_db': -60,
            'spectral_centroid': 0,
            'zero_crossing_rate': 0
        }

def transcribe_audio_segment(audio_path, start_time, end_time):
    """Transcribe a segment of a WAV audio file from start_time to end_time (in seconds)."""
    recognizer = sr.Recognizer()
    duration = end_time - start_time
    
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source, offset=start_time, duration=duration)
            text = recognizer.recognize_sphinx(audio_data)
            return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Error processing segment: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""

def get_advice(metric_type, value, segment_values=None):
    """Generate advice based on audio metrics."""
    advice = {
        'pitch': {
            'low': "Consider speaking with more vocal variety. Try varying your pitch to make your speech more engaging and expressive.",
            'normal': "Your pitch variation is good. Continue maintaining natural vocal inflection to keep listeners engaged.",
            'high': "Your pitch is quite high. Try speaking from your chest voice for a more authoritative and calming tone."
        },
        'loudness': {
            'low': "Your voice is quite soft. Try projecting more from your diaphragm to ensure your audience can hear you clearly.",
            'normal': "Your volume level is appropriate. Maintain this consistent level for optimal audience engagement.",
            'high': "Your voice is quite loud. Consider moderating your volume slightly for better audience comfort and clarity."
        },
        'pace': {
            'slow': "Your speaking pace is slow. Consider increasing your speed slightly (aim for 150-160 WPM) to maintain audience engagement.",
            'normal': "Your speaking pace is well-balanced. Continue maintaining this natural rhythm of 150-170 WPM.",
            'fast': "You're speaking quite fast. Try slowing down to 150-170 WPM to ensure clarity and better comprehension."
        }
    }
    
    if metric_type == 'pitch':
        if value < 120:
            return advice['pitch']['low']
        elif value > 250:
            return advice['pitch']['high']
        else:
            return advice['pitch']['normal']
    elif metric_type == 'loudness':
        if value < -35:
            return advice['loudness']['low']
        elif value > -15:
            return advice['loudness']['high']
        else:
            return advice['loudness']['normal']
    elif metric_type == 'pace':
        if value < 130:
            return advice['pace']['slow']
        elif value > 180:
            return advice['pace']['fast']
        else:
            return advice['pace']['normal']
    
    return "No specific advice available for this metric."

def process_audio(audio_path, segment_duration=10):
    """Process WAV audio file with comprehensive analysis."""
    if not audio_path.lower().endswith('.wav'):
        print("Error: Input file must be in WAV format.")
        return
    
    duration_seconds = get_wav_duration(audio_path)
    segments = []
    total_words = 0
    all_text = ""
    
    print("Processing audio segments...")
    
    # Process each segment
    for start_time in range(0, int(duration_seconds), segment_duration):
        end_time = min(start_time + segment_duration, duration_seconds)
        
        print(f"Processing segment {start_time//segment_duration + 1}...")
        
        # Transcribe text
        text = transcribe_audio_segment(audio_path, start_time, end_time)
        words = text.split()
        word_count = len(words)
        total_words += word_count
        all_text += text + " "
        
        # Analyze audio features
        features = analyze_audio_features(audio_path, start_time, end_time)
        
        # Calculate WPM for this segment
        segment_duration_minutes = (end_time - start_time) / 60
        segment_wpm = word_count / segment_duration_minutes if segment_duration_minutes > 0 else 0
        
        segments.append({
            'start': start_time,
            'end': end_time,
            'text': text,
            'word_count': word_count,
            'wpm': segment_wpm,
            'pitch': features['avg_pitch'],
            'loudness_db': features['loudness_db'],
            'spectral_centroid': features['spectral_centroid']
        })
    
    print("Getting AI vocabulary analysis...")
    
    # Get AI analysis of vocabulary
    vocab_prompt = f"""Analyze the following transcribed speech text for vocabulary usage, complexity, and communication effectiveness:

Text: {all_text}

Please provide:
1. Vocabulary complexity level (beginner/intermediate/advanced)
2. Word variety and repetition analysis
3. Sentence structure assessment
4. Overall communication effectiveness
5. Specific recommendations for improvement
6. Strengths in the vocabulary usage

Keep the analysis concise but comprehensive."""
    
    vocab_analysis = aiBot(vocab_prompt)
    
    # Calculate overall statistics
    total_minutes = duration_seconds / 60
    overall_wpm = total_words / total_minutes if total_minutes > 0 else 0
    overall_pitch = np.mean([s['pitch'] for s in segments if s['pitch'] > 0]) if any(s['pitch'] > 0 for s in segments) else 0
    overall_loudness = np.mean([s['loudness_db'] for s in segments])
    
    # Generate comprehensive report
    generate_comprehensive_report(segments, vocab_analysis, duration_seconds, total_words, 
                                overall_wpm, overall_pitch, overall_loudness, all_text)
    
    print("Analysis complete. Comprehensive report written to 'comprehensive_speech_analysis.txt'.")

def generate_comprehensive_report(segments, vocab_analysis, duration_seconds, total_words, 
                                overall_wpm, overall_pitch, overall_loudness, full_text):
    """Generate a comprehensive structured report."""
    
    with open('comprehensive_speech_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SPEECH ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Duration: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes)\n")
        f.write(f"Total Words: {total_words}\n")
        f.write(f"Overall Speaking Rate: {overall_wpm:.2f} WPM\n")
        f.write(f"Average Pitch: {overall_pitch:.2f} Hz\n")
        f.write(f"Average Loudness: {overall_loudness:.2f} dB\n\n")
        
        # Overall Analysis & Advice
        f.write("OVERALL PERFORMANCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Pace Analysis: {get_advice('pace', overall_wpm)}\n\n")
        f.write(f"Pitch Analysis: {get_advice('pitch', overall_pitch)}\n\n")
        f.write(f"Volume Analysis: {get_advice('loudness', overall_loudness)}\n\n")
        
        # Vocabulary Analysis from AI
        f.write("VOCABULARY & COMMUNICATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(vocab_analysis + "\n\n")
        
        # Segment-by-Segment Analysis
        f.write("DETAILED SEGMENT ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        for i, segment in enumerate(segments):
            start_time = str(timedelta(seconds=segment['start']))
            end_time = str(timedelta(seconds=segment['end']))
            
            f.write(f"\nSegment {i+1} ({start_time} - {end_time})\n")
            f.write("~" * 40 + "\n")
            f.write(f"Text: {segment['text']}\n")
            f.write(f"Word Count: {segment['word_count']}\n")
            f.write(f"Speaking Rate: {segment['wpm']:.2f} WPM\n")
            f.write(f"Pitch: {segment['pitch']:.2f} Hz\n")
            f.write(f"Loudness: {segment['loudness_db']:.2f} dB\n")
            
            # Segment-specific advice
            f.write("\nSegment Advice:\n")
            f.write(f"• Pace: {get_advice('pace', segment['wpm'])}\n")
            f.write(f"• Pitch: {get_advice('pitch', segment['pitch'])}\n")
            f.write(f"• Volume: {get_advice('loudness', segment['loudness_db'])}\n")
        
        # Performance Metrics Summary
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE METRICS SUMMARY\n")
        f.write("="*80 + "\n")
        
        wpm_values = [s['wpm'] for s in segments]
        pitch_values = [s['pitch'] for s in segments if s['pitch'] > 0]
        loudness_values = [s['loudness_db'] for s in segments]
        
        f.write(f"Speaking Rate Statistics:\n")
        f.write(f"  Average: {np.mean(wpm_values):.2f} WPM\n")
        f.write(f"  Range: {np.min(wpm_values):.2f} - {np.max(wpm_values):.2f} WPM\n")
        f.write(f"  Consistency: {np.std(wpm_values):.2f} (lower is more consistent)\n\n")
        
        if pitch_values:
            f.write(f"Pitch Statistics:\n")
            f.write(f"  Average: {np.mean(pitch_values):.2f} Hz\n")
            f.write(f"  Range: {np.min(pitch_values):.2f} - {np.max(pitch_values):.2f} Hz\n")
            f.write(f"  Variation: {np.std(pitch_values):.2f} Hz\n\n")
        
        f.write(f"Loudness Statistics:\n")
        f.write(f"  Average: {np.mean(loudness_values):.2f} dB\n")
        f.write(f"  Range: {np.min(loudness_values):.2f} - {np.max(loudness_values):.2f} dB\n")
        f.write(f"  Consistency: {np.std(loudness_values):.2f}\n\n")
        
        # Key Recommendations
        f.write("KEY RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        f.write("1. PACE: Maintain consistent speaking rate around 150-160 WPM\n")
        f.write("2. PITCH: Use natural voice variation to maintain engagement\n")
        f.write("3. VOLUME: Ensure consistent audibility without being too loud\n")
        f.write("4. VOCABULARY: Focus on clear, varied word choice\n")
        f.write("5. PRACTICE: Record and analyze regularly to track improvement\n\n")
        
        # Full Transcription
        f.write("COMPLETE TRANSCRIPTION\n")
        f.write("-" * 25 + "\n")
        f.write(full_text + "\n")

if __name__ == "__main__":
    audio_file = input("Enter the path to your WAV audio file: ")
    if not os.path.exists(audio_file):
        print("Audio file not found.")
    else:
        try:
            process_audio(audio_file)
        except ImportError as e:
            print(f"Missing required library: {e}")
            print("Please install required packages: pip install librosa numpy scipy")
        except Exception as e:
            print(f"Error processing audio: {e}")