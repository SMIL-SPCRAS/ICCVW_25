# ICCVW 2025

## Prompts

``1``

> You are an expert in multimodal emotion recognition. Carefully analyze the provided video clip containing a person’s facial expressions, body posture, gestures, head movements, and the surrounding scene. Your task is to describe, in continuous, natural language, how the visible scene, environment, and the person’s nonverbal behavior reflect emotional states.
>
> Use the following emotion categories as reference: Joy, Sadness, Anger, Fear, Disgust, Surprise, and Neutral. You are encouraged to identify mixed emotions, ambiguous expressions, and emotion transitions if present.
>
> In your description:
>
> - Briefly describe the visible scene only in relation to how it may influence or reflect the person's emotional state (e.g., a dimly lit room enhancing a sad or anxious mood).
> - Observe and explain the person’s facial expressions (eyes, eyebrows, mouth, gaze), body posture, gestures, and head movements as emotional indicators.
> - Describe how these visual cues evolve over time and signify possible emotion transitions.
> - Mention any mixed or ambiguous emotions with a short explanation based on visual cues.
> - Avoid assumptions about context beyond what is visually present.
>
> Your final response must be a fluent, continuous natural language emotional interpretation of the scene and the person’s behavior, no longer than 100 tokens in total, written as a single coherent paragraph without any line breaks, bullet points, special characters, or formatting. The response must express a complete, finished thought and always end with a period. Focus entirely on emotional states and visual cues.

``2``

> You are an expert in multimodal emotion recognition. Carefully analyze the provided video clip containing a person’s facial expressions, body posture, gestures, head movements, and the surrounding scene. Your task is to describe, in continuous, natural language, how the visible scene, environment, and the person’s nonverbal behavior reflect emotional states.
>
> Use the following emotion categories as reference: Joy, Sadness, Anger, Fear, Disgust, Surprise, and Neutral. Identify mixed emotions if present, using exactly one of: Fearfully Surprised, Happily Surprised, Sadly Surprised, Disgustedly Surprised, Angrily Surprised, Sadly Fearful, or Sadly Angry. If no mixed emotion is present, use exactly one of the basic emotions listed above.
>
> In your description:
>
> - Briefly describe the visible scene only in relation to how it may influence or reflect the person's emotional state.
> - Observe and explain the person’s facial expressions (eyes, eyebrows, mouth, gaze), body posture, gestures, and head movements as emotional indicators.
> - Describe how these visual cues evolve over time and signify possible emotion transitions.
> - Mention any mixed or ambiguous emotions with a short explanation based on visual cues.
> - Avoid assumptions about context beyond what is visually present.
> - You must explicitly mention at least once one of the emotion labels from the list somewhere in the paragraph.
> - If an emotion is mentioned explicitly in the description text, do NOT add a final emotion label at the end.
> - If no emotion word is mentioned explicitly in the description, you MUST end the paragraph with exactly one final emotion label from the list, separated from the text by a period and a space.
> - The final emotion label must be one of these: Fearfully Surprised, Happily Surprised, Sadly Surprised, Disgustedly Surprised, Angrily Surprised, Sadly Fearful, Sadly Angry, Joy, Sadness, Anger, Fear, Disgust, Surprise, Neutral.
> - The entire response must be a single, fluent paragraph with no line breaks or bullet points.
> - The total length should not exceed 100 tokens.
> - The response must express a complete thought.

## AFEW

### Qwen2.5-VL-3B-Instruct

- [Train_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-3B-Instruct/train_segment_with_text_1.csv)
- [Dev_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-3B-Instruct/dev_segment_with_text_1.csv)
- [Train_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-3B-Instruct/train_segment_with_text_2.csv)
- [Dev_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-3B-Instruct/dev_segment_with_text_2.csv)

### Qwen2.5-VL-7B-Instruct

- [Train_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-7B-Instruct/train_segment_with_text_1.csv)
- [Dev_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-7B-Instruct/dev_segment_with_text_1.csv)
- [Train_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-7B-Instruct/train_segment_with_text_2.csv)
- [Dev_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AFEW/Qwen2.5-VL-7B-Instruct/dev_segment_with_text_2.csv)

## AffWild2

### Qwen2.5-VL-3B-Instruct

- [Train_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-3B-Instruct/train_segment_with_text_1.csv)
- [Dev_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-3B-Instruct/dev_segment_with_text_1.csv)
- [Train_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-3B-Instruct/train_segment_with_text_2.csv)
- [Dev_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-3B-Instruct/dev_segment_with_text_2.csv)

### Qwen2.5-VL-7B-Instruct

- [Train_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-7B-Instruct/train_segment_with_text_1.csv)
- [Dev_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-7B-Instruct/dev_segment_with_text_1.csv)
- [Train_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-7B-Instruct/train_segment_with_text_2.csv)
- [Dev_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/AffWild2/Qwen2.5-VL-7B-Instruct/dev_segment_with_text_2.csv)

## C-EXPR-DB

### Qwen2.5-VL-3B-Instruct

- [Test_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/C-EXPR-DB/Qwen2.5-VL-3B-Instruct/test_segment_with_text_1.csv)
- [Test_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/C-EXPR-DB/Qwen2.5-VL-3B-Instruct/test_segment_with_text_2.csv)

### Qwen2.5-VL-7B-Instruct

- [Test_1](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/C-EXPR-DB/Qwen2.5-VL-7B-Instruct/test_segment_with_text_1.csv)
- [Test_2](https://github.com/SMIL-SPCRAS/ICCVW_25/blob/LLMs/C-EXPR-DB/Qwen2.5-VL-7B-Instruct/test_segment_with_text_2.csv)
