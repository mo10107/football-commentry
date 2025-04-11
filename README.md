# EchoPlay AI 🎙️⚽

**Smart Commentary System for the Visually Impaired**

EchoPlay AI is an intelligent, real-time sports commentary system powered by generative AI, tailored specifically for visually impaired individuals. It transforms live or recorded football matches into immersive auditory experiences through real-time audio descriptions of in-game events.

---

## 📌 Problem

Visually impaired audiences lack access to rich, descriptive sports commentary that translates the visual experience into sound. Current solutions do not provide detailed, real-time descriptions of player positions, goals, and match dynamics.

---

## ✅ Our Solution

EchoPlay AI leverages computer vision and natural language generation to:
- Analyze football match videos
- Identify players, passes, goals, and other events
- Generate real-time, Arabic-language audio commentary
- Describe player locations and actions accurately

---

## 💡 Idea Summary

An AI-powered system that processes sports video feeds and generates real-time descriptive commentary. Designed to support Arabic language output and visually impaired accessibility.

---

## 🛠️ Tech Stack

### Short-Term Technologies:
- **Qwen2.5-VL-32B-Instruct** – Vision-language understanding and generate commentary script
- **qTTS** – Text-to-speech engine (Arabic)

### Medium-Term Vision:
- **Full Ecosystem** 
- **End-to-End ML Design** – For scalable deployment and modularity

---

## 🧪 Testing & Evaluation

- Trained and tested **YOLOv11l**, **YOLOv12m**, and **RF-DETR** on 600 annotated football images (100 epochs)
- Evaluated **ByteTrack** and **DeepSort** for MOT
- Tested **OsNet** and **ResNet** for player identification
- Deployed **Qwen-VL-32B-Instruct** and **Qwen 2.5 Omni 7B** for scene and text generation
- Experimented with **qTTS** for real-time Arabic speech output

---

## 📂 Data Sources

- Football video datasets from **Kaggle**
- Annotated detection/tracking data from **Roboflow**

---

## 🚧 Challenges & Future Plans

### Key Challenges:
- Arabic speech synthesis quality
- Accurate Arabic player identification

### What We Need:
- Support in **TTS** and **ML system design**

### Roadmap:
- **Short Term:** Integrate APIs for Vision & Speech (Qwen, TTS)
- **Medium Term:** Build a full-fledged smart broadcasting ecosystem

---

## 🎯 Impact & Use Cases

- Inclusive AI experiences for the **visually impaired**
- AI-powered **sports broadcasting**
- Creative content production in **sports media**

---

## 🎥 How to run 

# For Short Term 
## 1. clone repo
    ```bash
   git clone https://github.com/mo10107/football-commentry.git
   ```
## 2. Get API Key from Alibaba Cloud (DASHSCOPE_API_KEY)

## 3. Run notebook

# For Medium Term

## 1. clone repo
    ```bash
   git clone https://github.com/mo10107/football-commentry.git
   ```
## 2. python main_with_reid.py --video \data\football_video.mp4 --output processed_video.mp4 

## 📌 Summary

EchoPlay AI is an assistive technology tool that uses cutting-edge AI to transform how visually impaired audiences experience football matches. Early tests on real match scenes are promising. We aim to improve voice quality and player detection in future iterations.

---

## 🙌 Team Members

- **Mohamed Abdelaziz**
- **Mahmoud El-Waleed**

---

