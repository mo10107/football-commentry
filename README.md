# Arabic Football Commentary System

A comprehensive system for generating real-time Arabic football commentary using computer vision and AI.

## Overview

This system combines state-of-the-art computer vision and natural language processing to create an engaging Arabic football commentary experience. It uses:

- **YOLOv12m** for ball and player detection
- **Supervision** for player tracking
- **OSNet** for player re-identification
- **TimeSformer** for football event recognition
- **Arabic Template System** for commentary generation
- **Arabic TTS** for speech synthesis

## System Architecture

![System Architecture Diagram](./docs/system_architecture.png)

## Features

- Real-time player and ball tracking
- Player identification with Arabic name pronunciation
- Football event detection (goals, passes, tackles, etc.)
- Culturally appropriate Arabic commentary generation
- Authentic Arabic speech synthesis with emotion
- Game state analysis (score, possession, field zone)
- Video visualization with overlays

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg for video processing

### Setup

1. Clone the repository:
```bash
git clone https://github.com/username/arabic-football-commentary.git
cd arabic-football-commentary
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
```bash
python download_models.py
```

## Usage

### Basic Usage

```bash
python m.py --video /teamspace/studios/this_studio/project/data/football_video.mp4 --output /teamspace/studios/this_studio/project/output/output_video.mp4
```

### Options

- `--video`: Path to input video file (required)
- `--output`: Path to output video file with visualization
- `--display`: Display video with visualizations in real-time
- `--no-audio`: Disable audio commentary

### Example

```bash
python main.py --video samples/match_clip.mp4 --output output/match_with_commentary.mp4 --display
```

## Data Preparation

### Player Database

Create a JSON file with player information in the following format:

```json
{
  "player_id1": {
    "name": "محمد صلاح",
    "arabic_name": "محمد صلاح",
    "team": "home",
    "position": "forward",
    "number": "10",
    "stats": {
      "goals": 24,
      "assists": 13
    }
  },
  "player_id2": {
    "name": "كريم بنزيما",
    "arabic_name": "كريم بنزيما",
    "team": "away",
    "position": "forward",
    "number": "9",
    "stats": {
      "goals": 22,
      "assists": 9
    }
  }
}
```

### Team Database

Create a JSON file with team information:

```json
{
  "home": {
    "name": "ليفربول",
    "arabic_name": "ليفربول",
    "color": "أحمر",
    "formation": "4-3-3"
  },
  "away": {
    "name": "ريال مدريد",
    "arabic_name": "ريال مدريد",
    "color": "أبيض",
    "formation": "4-4-2"
  }
}
```

## Custom Configuration

Edit `config.py` to customize system parameters:

- Detection confidence thresholds
- Tracking parameters
- Re-identification settings
- Event recognition settings
- TTS voice and emotion settings

## Components

### 1. Object Detection (YOLOv12m)

Uses YOLOv12m trained specifically for football scenes to detect players and the ball with high accuracy.

### 2. Object Tracking (Supervision)

Implements Supervision's ByteTrack algorithm to maintain consistent player tracking throughout the video.

### 3. Player Re-identification (OSNet)

Uses OSNet to generate player embeddings and match them against a database of known players.

### 4. Event Recognition (TimeSformer)

Detects football events like goals, passes, tackles, and fouls using a TimeSformer model trained on football action sequences.

### 5. Game State Analysis

Analyzes the game state including score, possession, field zone, and game phase to provide context for commentary.

### 6. Arabic Commentary Generation

Generates culturally appropriate Arabic football commentary using templates and contextual information.

### 7. Arabic Text-to-Speech

Converts generated Arabic text to natural-sounding speech with appropriate emotion and intonation.

## Extending the System

### Adding New Events

1. Update the `event_classes` dictionary in `config.py`
2. Add new templates in `data/commentary_templates.json`
3. Add event handling in `events/game_analyzer.py`

### Adding Custom TTS Voices

1. Place custom voice models in the `models` directory
2. Update the `model_path` in `config.py`

## Acknowledgements

- YOLOv12m: Ultralytics
- ByteTrack: ByteTrack authors
- Supervision: Roboflow
- OSNet: Kaiyang Zhou et al.
- TimeSformer: Facebook Research
- TTS: Mozilla TTS
