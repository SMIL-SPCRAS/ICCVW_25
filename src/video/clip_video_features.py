import cv2
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import pickle
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Текстовые описания эмоций
# EMOTION_TEXTS = [
#     "person / people in the video experiences Neutral emotion",
#     "person / people in the video experiences Anger emotion",
#     "person / people in the video experiences Disgust emotion",
#     "person / people in the video experiences Fear emotion",
#     "person / people in the video experiences Happiness emotion",
#     "person / people in the video experiences Sadness emotion",
#     "person / people in the video experiences Surprise emotion",
#     "person / people in the video experiences Other emotion",
#     "person / people in the video experiences Fearfully Surprised emotion",
#     "person / people in the video experiences Happily Surprised emotion",
#     "person / people in the video experiences Sadly Surprised emotion",
#     "person / people in the video experiences Disgustedly Surprised emotion",
#     "person / people in the video experiences Angrily Surprised emotion",
#     "person / people in the video experiences Sadly Fearful emotion",
#     "person / people in the video experiences Sadly Angry emotion"
# ]

EMOTION_TEXTS = [
    "neutral",
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "sad",
    "surprised",
    "other",
    "fearfully surprised",
    "happily surprised",
    "sadly surprised",
    "disgustedly surprised",
    "angrily surprised",
    "sadly fearful",
    "sadly angry"
]

# EMOTION_TEXTS = [
#     # Neutral
#     "A person with a neutral expression, relaxed facial muscles, straight closed lips, and neutral gaze. No strong emotion visible.",
    
#     # Anger
#     "An angry person with furrowed eyebrows, narrowed eyes, tense jaw, and tightened lips. The face may appear red or flushed.",
    
#     # Disgust
#     "A disgusted person with a wrinkled nose, raised upper lip, and a look of repulsion. The eyebrows may be lowered, and the head may be pulled back.",
    
#     # Fear
#     "A fearful person with wide-open eyes, raised eyebrows, and an open mouth. The body may appear tense or ready to flee.",
    
#     # Happiness
#     "A happy person with a genuine smile, raised cheeks (crow's feet around eyes), and relaxed eyebrows. The expression appears warm and friendly.",
    
#     # Sadness
#     "A sad person with downturned lips, drooping eyelids, and possibly tears. The eyebrows may be slightly raised in the inner corners (like a 'puppy dog' look).",
    
#     # Surprise
#     "A surprised person with wide-open eyes, raised eyebrows, and an open mouth. The body may be slightly recoiled.",
    
#     # Other (generic catch-all)
#     "A person showing an ambiguous or mixed emotional expression that doesn't clearly fit standard categories.",
    
#     # Fearfully Surprised
#     "A person with a mix of fear and surprise: wide eyes, raised eyebrows, open mouth, but with a tense or alarmed expression (like a jump scare).",
    
#     # Happily Surprised
#     "A positively surprised person with bright eyes, a wide smile, and raised eyebrows (like receiving a good gift).",
    
#     # Sadly Surprised
#     "A person showing surprise combined with sadness: widened eyes but with downturned lips and a somber expression (like hearing bad news unexpectedly).",
    
#     # Disgustedly Surprised
#     "A surprised person with a disgusted reaction: wrinkled nose, open mouth in revulsion, and recoiling (like smelling something foul unexpectedly).",
    
#     # Angrily Surprised
#     "A surprised person who quickly becomes angry: wide eyes transitioning into a scowl, tense mouth, and aggressive posture (like being suddenly provoked).",
    
#     # Sadly Fearful
#     "A person showing sadness mixed with fear: teary eyes, slightly raised inner eyebrows, and a tense, apprehensive expression (like mourning in a dangerous situation).",
    
#     # Sadly Angry
#     "A person blending sadness and anger: teary eyes, clenched jaw, and a pained expression (like crying in frustration or betrayal)."
# ]

def load_clip_model():
    """Загрузка модели CLIP"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

def get_text_embeddings(model, processor, device):
    """Предварительное вычисление текстовых эмбеддингов"""
    text_inputs = processor(text=EMOTION_TEXTS, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def process_video(video_path, model, processor, device, text_features, frame_rate=1, output_dir="results"):
    """Обработка одного видео и сохранение результатов в CSV"""
    video_name = Path(video_path).stem
    
    cap = cv2.VideoCapture(str(video_path))  # str() для совместимости с Windows
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_image_features = []
    all_similarity_scores = []
    
    # for frame_num in tqdm(range(0, total_frames, frame_rate), desc=f"Processing {video_name}"):
    for frame_num in range(0, total_frames, frame_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Конвертация BGR -> RGB и обработка CLIP
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            # print(image_features.shape)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Сравнение с текстовыми эмбеддингами
        all_similarity_scores.append((image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0].tolist())

        all_image_features.append(image_features.cpu().numpy().tolist()[0])
        
        # Сохранение результатов для кадра
        # frame_results = {
        #     "filename": str(video_path),
        #     "frame_number": frame_num,
        #     "image_features":image_features,
        #     **{emotion: float(score) for emotion, score in zip(EMOTION_TEXTS, similarity_scores)}
        # }
        # results.append(frame_results)

    all_similarity_scores = np.mean(np.array(all_similarity_scores), axis=0)
    all_image_features = np.mean(np.array(all_image_features), axis=0)
    all_text_features =np.mean(text_features.cpu().numpy(), axis=0)

    mean_basic_emo = all_similarity_scores[:8]/np.sum(all_similarity_scores[:8])
    mean_comp_emo = all_similarity_scores[-7:]/np.sum(all_similarity_scores[-7:])
    # final_prob = mean_basic_emo.tolist()+mean_comp_emo.tolist()
    
    cap.release()

    return {"video_name": video_name, "image_features": all_image_features, "text_features": all_text_features, "basic_predictions": mean_basic_emo, "compound_predictions": mean_comp_emo}
    
    # Сохранение в CSV
    # Path(output_dir).mkdir(exist_ok=True, parents=True)  # parents=True для создания вложенных папок
    # pd.DataFrame(results).to_csv(output_csv, index=False)
    # print(f"Results saved to {output_csv}")

def find_videos_in_folder(folder_path, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """Рекурсивный поиск видеофайлов в папке"""
    folder = Path(folder_path)
    videos = []
    for ext in extensions:
        videos.extend(folder.rglob(f"*{ext}"))  # Рекурсивный поиск по всем подпапкам
    return videos

def main():
    parser = argparse.ArgumentParser(description="Analyze emotions in videos using CLIP")
    parser.add_argument("input_path", help="Path to a video file or directory containing videos")
    parser.add_argument("--frame_rate", type=int, default=1, help="Extract every N-th frame (default: 1)")
    parser.add_argument("--output_dir", default="results", help="Directory to save CSV results")
    args = parser.parse_args()

    # Загрузка модели CLIP
    model, processor, device = load_clip_model()
    text_features = get_text_embeddings(model, processor, device)

    # Поиск видеофайлов
    input_path = Path(args.input_path)
    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = find_videos_in_folder(input_path)
    # else:
    #     raise ValueError(f"Invalid path: {input_path}")

    if not videos:
        print(f"No videos found in {input_path}")
        return
    
    results = []

    # Обработка каждого видео
    for video_path in tqdm(videos):
        results.append(process_video(video_path, model, processor, device, text_features, args.frame_rate, args.output_dir))

    with open(f"{args.output_dir.split('_')[1]}_{args.output_dir.split('_')[0]}_clip_predictions.pkl", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()