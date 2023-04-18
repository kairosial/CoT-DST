import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
import copy
import time
from collections import defaultdict
from tqdm import tqdm
from utils.helper import SpeedLimitTimer, PreviousStateRecorder
from utils.typo_fix import typo_fix
from config import CONFIG

# from codex_completion import codex_completion
from gpt35_completion import gpt35_completion
from utils.custom_parse import custom_pred_parse, sv_dict_to_string
from gpt35_prompting_base import get_custom_prompt, conversion, custom_prompt
from retriever.code.embed_based_retriever import EmbeddingRetriever
from evaluate_metrics import evaluate
from evaluate_FGA import printFGA

'''
default command
python run_GPT35_test.py \
      --train_fn data/mw21_5p_train_v2.json \
      --retriever_dir retriever/expts/mw21_5p_v2 \
      --output_dir expts/gpt35_5p_v2_ours  \
      --mwz_ver 2.4
'''

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, help="training data file (few-shot or full shot)", required=True)  # e.g. "./data/mw21_10p_train_v3.json"
parser.add_argument('--retriever_dir', type=str, required=True, help="sentence transformer saved path")  # "./retriever/expts/mw21_10p_v3_0304_400_20"
parser.add_argument('--output_cond', type=str, default="./expts/debug", help="directory to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.1", choices=['2.1', '2.4'], help="version of MultiWOZ")  
parser.add_argument('--test_fn', type=str, default='', help="file to evaluate on, empty means use the test set")
parser.add_argument('--test_size', type=int, default='10', help="")
args = parser.parse_args()

# current time
cur_time = time.strftime('%y%m%d_%H%M-')

# create the output folder
output_dir = 'expts/' + cur_time + args.output_cond + '_inst' + str(args.test_size)
os.makedirs(output_dir, exist_ok=True)

'''
"exp_config.json"
{
    "train_fn": "data/mw21_5p_train_v2.json",
    "retriever_dir": "retriever/expts/mw21_5p_v2_0304_200_10/",
    "output_dir": "expts/codex_5p_v2",
    "mwz_ver": "2.1",
    "test_fn": ""
}
'''
with open(os.path.join(output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

NUM_EXAMPLE=10

# read the selection pool
with open(args.train_fn) as f:
    train_set = json.load(f)

# read the ontology and the test set
if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
    if args.test_fn == "":
        test_set_path = "./data/mw21_100p_test.json"
else:
    ontology_path = CONFIG["ontology_24"]
    if args.test_fn == "":
        test_set_path = "./data/mw24_100p_test.json"

# evaluate on some other file
if args.test_fn:
    test_set_path = args.test_fn

with open(ontology_path) as f:
    ontology = json.load(f)
with open(test_set_path) as f:
    test_set = json.load(f)

# load the retriever
retriever = EmbeddingRetriever(datasets=[train_set],
                               model_path=args.retriever_dir,
                               search_index_filename=os.path.join(args.retriever_dir, "train_index.npy"), 
                               sampling_method="pre_assigned")


def run(test_set, turn=-1, use_gold=False):
    # turn and use_gold are for analysis purpose
    # turn = -1 means evalute all dialogues
    # turn = 0 means evaluate single-turn dialogues
    # turn = 1 means evalute two-turn dialogues... etc.
    # when use_gold = True, the context are gold context (for analysis purpose)

    timer = SpeedLimitTimer(second_per_step=3.1)  # openai limitation 20 queries/min

    result_dict = defaultdict(list)  # use to record the accuracy

    selected_set = test_set
    # if needed, only evaluate on particular turns (analysis purpose)
    if turn >= 0:
        if not use_gold:
            raise ValueError("can only evaluate particular turn when using gold context")
        selected_set = [d for d in test_set if len(d['dialog']['usr']) == turn + 1]
    
    prediction_recorder = PreviousStateRecorder()  # state recorder

    # start experiment
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    for idx, data_item in tqdm(enumerate(selected_set)):
        n_total += 1

        completion = ""
        if use_gold: # 이전 dialogue context 가 gold context 일 때 (분석 용도)
            prompt_text = get_custom_prompt(
                data_item, examples=retriever.item_to_nearest_examples(data_item, k=NUM_EXAMPLE))
        else:
            # 각 data_items에 대한 이전 턴의 state를 받아옴
            predicted_context = prediction_recorder.state_retrieval(data_item)
            modified_item = copy.deepcopy(data_item)
            modified_item['last_slot_values'] = predicted_context
            # 이전 턴의 state을 참조해 예시 검색
            examples = retriever.item_to_nearest_examples(
                modified_item, k=NUM_EXAMPLE)
            # prompt_생성
            # prompt_text = get_prompt(
            #     data_item, examples=examples, given_context=predicted_context)
            # prompt 바꾼 버전
            prompt_text = get_custom_prompt(
                data_item, examples=examples, given_context=predicted_context)
        
        # print the retrieved examples (without the sql table)
        # print(prompt_text.replace(conversion(table_prompt), ""))

        print(prompt_text.replace(conversion(custom_prompt), ""))

        # record the prompt
        data_item['prompt'] = prompt_text

        # gpt35 completion
        complete_flag = False
        parse_error_count = 0
        while not complete_flag:
            try:
                completion = gpt35_completion(prompt_text)
                # convert back the sql completion result
                completion = conversion(completion, reverse=True)
            except Exception as e:
                # example 개수 줄이기
                if e.user_message.startswith("This model's maximum context length"):
                    print("prompt overlength")
                    examples = examples[1:]
                    prompt_text = get_custom_prompt(
                        data_item, examples=examples, given_context=predicted_context)
                else:
                    # throughput too high
                    timer.sleep(10)
            else:
                try:
                    # check if CODEX is crazy 
                    temp_parse = custom_pred_parse(completion)
                except:
                    parse_error_count += 1
                    if parse_error_count >= 3:
                        complete_flag = True
                else:
                    complete_flag = True
            # limit query speed
            timer.step()

        # aggregate the prediction and the history states
        predicted_slot_values = {}
        try:
            predicted_slot_values = custom_pred_parse(completion) # a dictionary
        except:
            print("the output is not a valid result")
            data_item['not_valid'] = 1

        predicted_slot_values = typo_fix(predicted_slot_values, ontology=ontology, version=args.mwz_ver)

        context_slot_values = data_item['last_slot_values']  # a dictionary

        # merge context and prediction
        if use_gold:
            all_slot_values = context_slot_values.copy()
        else:
            # 이전 턴의 slot values를 all_slot_values로 저장
            all_slot_values = prediction_recorder.state_retrieval(
                data_item).copy()

        # slot changes를 참조해 dialogues states 만듬
        for s, v in predicted_slot_values.items():
            # 사라진 경우라면 없앰
            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            # 사라진 경우가 아니라면 업데이트
            elif v != "[DELETE]":
                all_slot_values[s] = v

        # some slots may contain multiple values (하나만 선택)
        all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}
        
        # record current turn prediction, 업데이트 된 값을 dialogue states의 형태로 recoder에 기록
        prediction_recorder.add_state(data_item, all_slot_values)

        # record the predictions
        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = ontology_path
        data_item['completion'] = completion

        # print the result
        print(completion)
        print(f"this is the {n_total - 1}th example. {data_item['ID']}_turn_{data_item['turn_id']}")
        print(f"pred turn change: {sv_dict_to_string(predicted_slot_values, sep='-')}")
        print(f"gold turn change: {sv_dict_to_string(data_item['turn_slot_values'], sep='-')}")
        print(f"pred states: {sv_dict_to_string(all_slot_values, sep='-')}")
        print(f"gold states: {sv_dict_to_string(data_item['slot_values'], sep='-')}")

        this_jga, this_acc, this_f1 = evaluate(all_slot_values,data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1

        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
            print("\n=====================correct!=======================")
        else:
            result_dict[data_item['turn_id']].append(0)
            print("\n======================wrong!========================")

        # Record Evaluation Result
        data_item['JGA'] = n_correct / n_total
        data_item['SA'] = total_acc / n_total
        data_item['Joint_F1'] = total_f1/n_total

        pred_status = ''
        if this_jga:
            data_item['pred_status'] = 'Correct'
        else:
            data_item['pred_status'] = '============================= Wrong ==============================='

        data_item['current_time'] = cur_time

        all_result.append(data_item)

        # Log Checkpoint
        if idx % 5 == 0:
            with open(os.path.join(output_dir, f'running_log.json'), 'w') as f:
                json.dump(all_result, f, indent=4)


    print(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}")
    print(f"Slot Acc {total_acc/n_total}")
    print(f"Joint F1 {total_f1/n_total}")
    print()

    # calculate the accuracy of each turn
    for k, v in result_dict.items():
        print(f"accuracy of turn {k} is {sum(v)}/{len(v)} = {sum(v) / len(v)}")

    return all_result


if __name__ == "__main__":
    all_results = run(test_set[:args.test_size])

    with open(os.path.join(output_dir, "running_log.json"), 'w') as f:
        json.dump(all_results, f, indent=4)

    printFGA(output_dir)

    print(f"종료 시간: {time.strftime('%y%m%d_%H%M')}")