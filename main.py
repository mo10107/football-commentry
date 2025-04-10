"""
Main entry point for Arabic Football Commentary System
"""

import os
import cv2
import time
import argparse
import numpy as np
from queue import Queue
from threading import Thread

# Import configuration
import config

# Import system components
from detection.detector import FootballDetector
from tracking.tracker import FootballTracker
from reid.player_identification import PlayerIdentifier
from events.event_recognition import EventRecognizer
from events.game_analyzer import GameStateAnalyzer
from commentary.arabic_generator import ArabicCommentaryGenerator
from speech.arabic_tts import ArabicTTS
from database.knowledge_base import FootballKnowledgeBase
from visualization.display import ResultVisualizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Arabic Football Commentary System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=config.VIDEO["output_path"], help="Path to output video file")
    parser.add_argument("--display", action="store_true", help="Display video with visualizations")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio commentary")
    return parser.parse_args()

def audio_player_thread(audio_queue):
    """Thread for playing audio files"""
    import pygame
    pygame.mixer.init()
    
    while True:
        audio_file = audio_queue.get()
        if audio_file == "STOP":
            break
            
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            audio_queue.task_done()

def main():
    args = parse_args()

    # Update config with args
    config.VIDEO["input_source"] = args.video
    config.VIDEO["output_path"] = args.output
    config.VIDEO["display"] = args.display
    
    # Initialize system components
    detector = FootballDetector(
        model_path=config.DETECTION["model_path"],
        conf_threshold=config.DETECTION["conf_threshold"],
        device=config.DETECTION["device"]
    )
    
    tracker = FootballTracker(
        tracker_type=config.TRACKING["tracker_type"],
        track_threshold=config.TRACKING["track_threshold"],
        track_buffer=config.TRACKING["track_buffer"],
        match_threshold=config.TRACKING["match_threshold"],
        frame_rate=config.TRACKING["frame_rate"]
    )
    
    player_identifier = PlayerIdentifier(
        model_path=config.REID["model_path"],
        embeddings_db=config.REID["embeddings_db"],
        similarity_threshold=config.REID["similarity_threshold"]
    )
    
    event_recognizer = EventRecognizer(
        model_path=config.EVENT["model_path"],
        frame_buffer_size=config.EVENT["frame_buffer_size"],
        event_classes=config.EVENT["event_classes"],
        confidence_threshold=config.EVENT["event_confidence_threshold"]
    )
    
    knowledge_base = FootballKnowledgeBase(
        players_db=config.KNOWLEDGE["players_db"],
        teams_db=config.KNOWLEDGE["teams_db"],
        phrases_db=config.KNOWLEDGE["phrases_db"]
    )
    
    game_analyzer = GameStateAnalyzer(knowledge_base)
    
    commentary_generator = ArabicCommentaryGenerator(
        api_endpoint=config.COMMENTARY["api_endpoint"],
        api_key=config.COMMENTARY["api_key"],
        templates_file=config.COMMENTARY["templates_file"],
        knowledge_base=knowledge_base
    )
    
    tts_engine = ArabicTTS(
        model_path=config.TTS["model_path"],
        output_dir=config.TTS["output_dir"]
    )
    
    visualizer = ResultVisualizer()
    
    # Start audio player thread if audio is enabled
    audio_queue = Queue()
    if not args.no_audio:
        audio_thread = Thread(target=audio_player_thread, args=(audio_queue,), daemon=True)
        audio_thread.start()
    
    # Set up video capture
    cap = cv2.VideoCapture(config.VIDEO["input_source"])
    if not cap.isOpened():
        print(f"Error: Could not open video source {config.VIDEO['input_source']}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize frame buffer for event recognition
    frame_buffer = []
    
    # Main processing loop
    frame_count = 0
    last_commentary_time = 0
    min_commentary_interval = 5  # Minimum seconds between commentaries
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detect football objects
            detections = detector.detect(frame)
            
            # Track players and ball
            player_tracks, ball_track = tracker.update(detections, frame)
            
            # Update frame buffer for event recognition
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > config.EVENT["frame_buffer_size"]:
                frame_buffer.pop(0)
            
            # Identify players
            player_info = {}
            for track in player_tracks:
                player_id = player_identifier.identify(frame, track['bbox'])
                if player_id:
                    player_details = knowledge_base.get_player_info(player_id)
                    player_info[track['track_id']] = {
                        'player_id': player_id,
                        'name': player_details.get('name', f'لاعب {player_id}'),
                        'team': player_details.get('team', 'فريق غير معروف'),
                        'position': player_details.get('position', 'مركز غير معروف'),
                        'track': track
                    }
            
            # Detect events (every few frames to reduce computation)
            current_event = None
            if frame_count % 10 == 0 and len(frame_buffer) == config.EVENT["frame_buffer_size"]:
                current_event = event_recognizer.recognize(frame_buffer)
            
            # Update game state
            game_state = game_analyzer.update_state(player_tracks, ball_track, current_event)
            
            # Generate commentary for significant events
            current_time = time.time()
            if current_event and current_event['confidence'] > config.EVENT["event_confidence_threshold"]:
                if current_time - last_commentary_time > min_commentary_interval:
                    # Generate Arabic commentary text
                    arabic_text = commentary_generator.generate_commentary(
                        game_state, 
                        current_event, 
                        player_info
                    )
                    
                    # Convert to speech
                    if not args.no_audio:
                        audio_file = tts_engine.generate_speech(
                            arabic_text,
                            emotion=current_event['event'] if current_event['event'] == 'goal' else 'neutral'
                        )
                        audio_queue.put(audio_file)
                    
                    last_commentary_time = current_time
            
            # Visualize results
            vis_frame = visualizer.draw_results(frame, player_tracks, ball_track, player_info, game_state, current_event)
            
            # Write to output video
            if args.output:
                out.write(vis_frame)
            
            # Display if requested
            if args.display:
                cv2.imshow("Arabic Football Commentary", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up
        cap.release()
        if args.output:
            out.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Stop audio thread
        if not args.no_audio:
            audio_queue.put("STOP")
            audio_thread.join(timeout=1)
        
        print("Processing complete!")

if __name__ == "__main__":
    main()