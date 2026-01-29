import os
# import re
import json
import base64
import argparse
import threading
import time
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# meta instruction for evaluation
vlm_content_template = '''
### Task Description: 
You are tasked with carefully examining the provided image and answering the following 10 multiple-choice questions. You MUST ONLY rely on the provided image to answer the questions. DO NOT use any external resources like world knowledge or external information beyond the provided image.

### Multiple-Choice Questions:
##Multiple-Choice Questions##

### Instructions:
1. Answer these 10 questions on a separate 10 lines, beginning with the correct choice option (A/B/C/D/E/..., not the number) and followed by a detailed reason (in the same line as answer).
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Each answer must be on its own line.
5. Ensure the index of answers matches the index of questions.
6. Select the option 'E: None' when the image can not answer the question.

### Output Format (Example, 10 lines for 10 questions):
E: None - The image does not depict a log or any specific object categories clearly enough to match any listed options.
B: Large and brown bear, small and red fox - The bear is visibly larger and brown, while the fox is smaller and red.
C: The bear is on the left and the fox is on the right - The bear appears on the left and the fox on the right side of the image.
A: The bear is facing the fox - The bear is looking directly at the fox, indicating it is facing the fox.
B: They are positioned opposite each other on the left and right - They are facing each other from opposite sides of the image.
E: None - The image does not provide clear indication of height comparison that matches the provided statements.
B: They are positioned closely together - Bear and fox are seen near each other, interacting without any major distance or separation.
E: None - The image does not show any notable occlusion from logs or surrounding objects.
E: None - The image does not show the bear initiating any of the described motions.
E: None - No direct causal results of the bear's movement are depicted in the image.
'''

def format_questions_prompt(questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    return vlm_content_template.replace("##Multiple-Choice Questions##", formatted_questions)

def vllm_eval_api_call(client, vlm_prompt, image_path, api_name, temperature):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    messages = [
        {"role": "system", "content": "You are a professional image critic."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": vlm_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                }
            ]
        }
    ]

    completion = client.chat.completions.create(
        model=api_name,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content

def load_jsonl_lines(jsonl_file):
    lines = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Warning] Skip the lines that cannot be parsed: {line.strip()} | Error: {e}")
    return lines

def write_jsonl_lines(jsonl_file, data_list):
    # Write to jsonl file & Sort ID
    sorted_data = sorted(data_list, key=lambda x: x.get('id', ''))
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        for item in sorted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def check_qa_model_preds(answers, all_preds, min_count=3):
    """Check if the QA model predictions are correct"""
    num_questions = len(answers)
    results = []
    selected_options = []
    
    for q_idx in range(num_questions):
        option_count = {}
        # Confirm that all_preds has enough length
        valid_preds_for_q = [preds[q_idx] for preds in all_preds if len(preds) > q_idx]
        if not valid_preds_for_q:
            results.append(False)
            selected_options.append(["(No valid predictions)"])
            continue

        for option in valid_preds_for_q:
            option_count[option] = option_count.get(option, 0) + 1
        
        high_freq_options = [opt for opt, count in option_count.items() if count >= min_count]
        is_correct = len(high_freq_options) > 0 and answers[q_idx] in high_freq_options
        results.append(is_correct)
        
        if high_freq_options:
            selected_options.append(high_freq_options)
        else:
            max_count = max(option_count.values())
            most_frequent = [opt for opt, count in option_count.items() if count == max_count]
            selected_options.append(most_frequent)
            
    return results, selected_options


# --- Main function ---
def process_single_item(item_index, data, image_path, args):
    """
    Process a single data item's full workflow.
    continue to get enough successful rollout or max_attempts times
    """
    thread_id = threading.get_ident()
    client = OpenAI(base_url=args.base_url)

    id_text = data.get('id', '')
    scene_text = data.get('scene', '')
    prompt_text = data.get('prompt', '')
    questions = data.get('questions', [])
    answers = data.get('answers', [])

    if not all([prompt_text, questions, answers, os.path.exists(image_path)]):
        print(f"[Threading {thread_id}] Error: Skip ID {item_index+1:06d}, data not complete or image not exists.")
        return None
    
    vlm_prompt = format_questions_prompt(questions)
    
    model_preds_list = []
    model_preds_cot_list = []
    
    # Set max_attempts to avioid infinite loop.
    total_attempts = 0
    # max_attempts = args.rollout
    max_attempts = args.rollout * 5

    while len(model_preds_list) < args.rollout and total_attempts < max_attempts:
        total_attempts += 1
        try:
            raw_response = vllm_eval_api_call(client, vlm_prompt, image_path, args.api_name, args.temperature)
            if not raw_response:
                print(f"[Threading {thread_id}] Warning: ID {item_index+1:06d} API return empty response (Attempt {total_attempts}/{max_attempts})")
                time.sleep(5)
                continue

            preds_cot = [line.strip() for line in raw_response.strip().split('\n') if line.strip()]
            
            if len(preds_cot) == len(questions):
                preds = [cot[0] for cot in preds_cot]
                model_preds_list.append(preds)
                model_preds_cot_list.append(preds_cot)
            else:
                print(f"preds_cot ({len(preds_cot)}): {preds_cot}")
                print(f"[Threading {thread_id}] Warning: ID {item_index+1:06d} response format not match (Attempt {total_attempts}/{max_attempts})")

        except Exception as e:
            print(f"[Threading {thread_id}] Error: ID {item_index+1:06d} API call failed: {e} (Attempt {total_attempts}/{max_attempts})")
            time.sleep(5*total_attempts) # If API call continues to fail, wait longer

    # When the loop ends, check if the number of successful rollout is less than args.rollout
    if len(model_preds_list) < args.rollout:
        print(f"[Threading {thread_id}] Error: Failed to complete {args.rollout} valid calls for ID {item_index+1:06d} after {max_attempts} attempts. Task abandoned.")
        return None

    # If all rollout is completed, continue processing
    try:
        results, selected_options = check_qa_model_preds(answers, model_preds_list, args.count)
    except Exception as e:
        print(f"[Threading {thread_id}] Error: ID {item_index+1:06d} when checking answers: {e}")
        return None

    save_json_data = {
        "id": id_text,
        "scene": scene_text,
        "avg_acc": f"{sum(results)}/{len(results)}",
        "basic_acc": f"{sum(results[:2])}/{len(results[:2])}",
        "spatial_acc": f"{sum(results[2:])}/{len(results[2:])}",
        "image_path": image_path,
        "prompt": prompt_text,
        "questions": questions,
        "answers": answers,
        "model_preds": selected_options,
        "true-or-false": results,
        "model_preds_cot": model_preds_cot_list
    }
    return save_json_data

# --- Multi-threading ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_name", type=str, default="", help="api name")
    parser.add_argument("--base_url", type=str, default="")
    # parser.add_argument("--model_name", type=str, default="mymodel", help="T2I model name")
    parser.add_argument("--input_json", type=str, required=True, help="json file for evaluation")
    parser.add_argument("--image_pth", type=str, required=True, help="image path for evaluation")
    parser.add_argument("--output_json", type=str, default="./eval/eval_results/eval_results.jsonl", help="The evaluation results of SpatialGenEval")
    parser.add_argument("--scene", type=str, default="", help="scene name")
    parser.add_argument("--rollout", type=int, default=5, help="rollout times")
    parser.add_argument("--count", type=int, default=4, help="count threshold for correct")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature setting")
    parser.add_argument("--max_workers", type=int, default=5, help="Max number of concurrent threads")
    args = parser.parse_args()

    # Stage 1. Preparing data and tasks
    json_data_all = load_jsonl_lines(args.input_json)
    image_list_all = sorted(os.listdir(args.image_pth))

    if args.scene:
        json_data_list = [item for item in json_data_all if item['scene'] == args.scene]
        image_list = [image_list_all[idx] for idx, item in enumerate(json_data_all) if item['scene'] == args.scene]
    else:
        json_data_list = json_data_all
        image_list = image_list_all
    
    if len(json_data_list) != len(image_list):
        print(f"Error: Found {len(json_data_list)} JSONL entries but {len(image_list)} image files. The counts must be equal.")

    ## Create a list of tasks, each task contains all the information needed for processing
    tasks = [(idx, data, os.path.join(args.image_pth, img_name), args) 
             for idx, (data, img_name) in enumerate(zip(json_data_list, image_list))]

    # print(f"Data preparation finished. Total tasks: {len(tasks)}. Now starting multi-threaded processing...")
    
    # Stage 2. Process tasks concurrently using ThreadPoolExecutor
    all_results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_item, *task): task for task in tasks}
        # Using tqdm and as_completed to get results and display progress
        progress_bar = tqdm(as_completed(future_to_task), total=len(tasks), desc=f"{args.scene}, {args.max_workers} threads processing")
        
        for future in progress_bar:
            result = future.result()
            if result:
                all_results.append(result)

    end_time = time.time()
    
    # Stage 3. Once all tasks are complete, write the results to a file
    if all_results:
        output_json_file = args.output_json
        write_jsonl_lines(output_json_file, all_results)
    else:
        print("\nAll tasks processed, but none were successful.")