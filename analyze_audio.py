import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- CONFIGURATION ---
SILENCE_THRESHOLD_DB = 30  

def analyze_specific_segment(audio_path, start_min, start_sec, end_min, end_sec):
    # Convert input times to total seconds
    start_time = (start_min * 60) + start_sec
    end_time = (end_min * 60) + end_sec
    duration_to_analyze = end_time - start_time
    
    print(f"--- Loading Audio Segment: {audio_path} ---")
    print(f"Start: {start_min}m {start_sec}s ({start_time}s)")
    print(f"End:   {end_min}m {end_sec}s ({end_time}s)")
    
    # Load ONLY the specific segment
    y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=duration_to_analyze)

    if len(y) == 0:
        print("Error: No audio loaded.")
        return

    # --- 1. Detect Silence ---
    non_silent_intervals = librosa.effects.split(y, top_db=SILENCE_THRESHOLD_DB)
    silence_intervals = []
    
    if len(non_silent_intervals) > 0:
        if non_silent_intervals[0][0] > 0:
            silence_intervals.append((0, non_silent_intervals[0][0]))
        for i in range(len(non_silent_intervals) - 1):
            silence_intervals.append((non_silent_intervals[i][1], non_silent_intervals[i+1][0]))
        if non_silent_intervals[-1][1] < len(y):
            silence_intervals.append((non_silent_intervals[-1][1], len(y)))
    else:
        silence_intervals.append((0, len(y)))

    # --- 2. Plotting ---
    plt.figure(figsize=(14, 6))
    time_axis = np.linspace(start_time, end_time, num=len(y))
    plt.plot(time_axis, y, color='#3333cc', alpha=0.8, label='Speech')
    
    has_labeled_red = False
    for start_sample, end_sample in silence_intervals:
        dur_s = (end_sample - start_sample) / sr
        
        # --- THIS IS THE FILTER ---
        # Only show pauses BETWEEN 1.0s and 4.0s
        if dur_s > 1.0 and dur_s < 3.5:
            
            abs_start = start_time + (start_sample / sr)
            abs_end = start_time + (end_sample / sr)
            
            # Updated Label
            label = 'Pause (1.0s - 3.5s)' if not has_labeled_red else None
            plt.axvspan(abs_start, abs_end, color='#ffaaaa', alpha=0.5, label=label)
            has_labeled_red = True
            
            mid = abs_start + (dur_s / 2)
            max_y = np.max(np.abs(y)) if len(y) > 0 else 1.0
            plt.text(mid, 0.8 * max_y, f"{dur_s:.1f}s", color='black', fontsize=9, 
                     ha='center', fontweight='bold', 
                     bbox=dict(facecolor='white', edgecolor='red', alpha=0.8, pad=1))

    # --- Formatting ---
    plt.title(f"Segment Analysis For Participant 4: {start_min}:{start_sec:02d} Sec to {end_min}:{end_sec:02d} Sec", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim(start_time, end_time)
    
    plt.tight_layout()
    output_file = f"segment_{start_min}m{start_sec}s_to_{end_min}m{end_sec}s.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_segment.py <audio_file>")
    else:
        # Currently set to: 0m 31s to 2m 34s
        analyze_specific_segment(sys.argv[1], 0, 31, 2, 34)