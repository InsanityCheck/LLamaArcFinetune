import json 
import numpy as np

import random
import time 


import matplotlib.pyplot as plt

from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from loader.ArcDataLoader import ArcKaggleDataLoader, DataType, ArcProblem



def text_to_array(text):
    arr = text.split("\n")
    ress = []
    for a in arr:
        if a[0]=="[":
            res = json.loads(a)
            ress.append(res)
        else:
            break
    return ress




import os



def train(data_path, name, epochs, r, batch_size):

    start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit", # "unsloth/mistral-7b" for 16bit loading
        max_seq_length =8192,
        dtype = None,
        load_in_4bit = True,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )



    dataLoader = ArcKaggleDataLoader(data_path)

    #create text dataset
    dataset = dataLoader.create_full_text_training_dataset(type=DataType.EVALUATION,transpose_rotated=True)

    #append EOS Token for each example
    EOS_TOKEN = tokenizer.eos_token
    for i in range(len(dataset)):
        dataset[i] = dataset[i]+EOS_TOKEN


    #filter for token length, smaller than 7000, because we also need space to generate
    MAX_LENGTH=7000
    small_examples = [d for d in dataset if len(d)<MAX_LENGTH]

    #shuffle examples
    random.shuffle(small_examples)

    #convert into Dataset format
    dict_dataset = [{"text":val} for val in small_examples]
    dataset_model = Dataset.from_list(dict_dataset)



    #create PEFT LoRA model
    model = FastLanguageModel.get_peft_model(
        model,
        r = r, #Rank of the LoRA weights
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head",], #which modules we want to target
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", 
        random_state = 1212,
        use_rslora = True,  # Rank stabilized Lora
        loftq_config = None, 
    )



    
    #Create Trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_model,
        dataset_text_field = "text",
        max_seq_length = 8192,
        dataset_num_proc = 8,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = max(8//batch_size,1),

            warmup_steps = 5,
            num_train_epochs = epochs,

            learning_rate = 1e-4,
            embedding_learning_rate = 1e-5,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.00,
            lr_scheduler_type = "cosine",
            seed = 2222,
            output_dir = "outputs",
        ),
    )


    print(str(time.time()-start)+" s until training start")

    last_time = time.time()


    #start training
    stats = trainer.train()

    print(str(time.time()-last_time)+" s until training end")

    print(stats)

    #save results
    model.save_pretrained(name+"_model")
    tokenizer.save_pretrained(name+"_model")
    return model,tokenizer


def load(name):
    max_seq_length=8192
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = name+"_model", # "unsloth/mistral-7b" for 16bit loading
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    return model, tokenizer

def evaluate_model(model,tokenizer,data_path,name):
    try:
        os.mkdir(name)
    except:
        pass
    try:
        os.mkdir(name+"/imgs")
    except:
        pass

    print("============== Inference Start ================")
    inf_start = time.time()

    #Reload data
    dataLoaderTest = ArcKaggleDataLoader(data_path)

    #create evaluation dataset and change format
    x,y,problems = dataLoaderTest.create_full_text_evaluation_dataset(DataType.EVALUATION)
    examples = {}
    solutions = {}
    problem_dict = {}

    for i in range(0,len(problems),2):
        examples[problems[i].challenge_code]=[x[i],x[i+1]]
        solutions[problems[i].challenge_code]=y
        problem_dict[problems[i].challenge_code] = problems[i]






    #Put model in inference mode
    FastLanguageModel.for_inference(model)




    prediction_results = {}
    counter=0

    num_keys = len(examples.keys())
    for key in examples.keys():
        counter+=1
        print("%s/%s"%(counter,num_keys))

        prediction_results[key]=[{}]
        for k in range(2):

            #Get challenge string k=0 for normal k=1 for transposed
            input_text = examples[key][k]

            inputs = tokenizer(
            [
                input_text
            ]*1, return_tensors = "pt").to("cuda")


            #Ignore too long examples
            if len(input_text)>=7000:
                continue



            #Generate output
            outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True,do_sample=True, top_k=16, pad_token_id=tokenizer.eos_token_id)
            result = tokenizer.batch_decode(outputs)

            

            # Try to convert output to array, if not possible, ignore
            try:
                arrayP = np.array(text_to_array(result[0].split("OUTPUT: \n")[-1]))
                #arrayT = np.array(text_to_array(solutions[key]))
                if k==1:
                    arrayP = np.transpose(arrayP)
                
                prediction_results[key][0]["attempt%s"%(k+1)]=(problem_dict[key].reverse_color_map(arrayP).tolist())
                
                if counter%20==0:
                    with open(name+"/res%s.json"%counter, "w") as f:

                        json.dump(prediction_results,f)
            
            
                #Save image of prediction
                max_val = np.max(arrayP)
                plt.matshow(arrayP,vmin=0,vmax=max_val)    
                plt.savefig(name+"/imgs/img%s_%s.png"%(key,k))
                plt.close()

            except Exception as error:
                print(error, key)



    print("Inference End")
    print(str(time.time()-inf_start)+ " s seconds")


    #save final_results
    with open(name+"/res_fin.json", "w") as f:
        json.dump(prediction_results,f)



import argparse

def main():
    parser = argparse.ArgumentParser(description="Finetuning Llama on ARC Kaggle dataset")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the arc kaggle dataset')
    parser.add_argument('--training', required=True, action=argparse.BooleanOptionalAction, help='Whether to train the model')
    parser.add_argument('--evaluate', required=True, action=argparse.BooleanOptionalAction, help='Whether to evaluate the model')
    parser.add_argument('--save-name', type=str, default='llama3ArcFinetune', help='Name of savedata folders')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--r', type=int, default=16, help='Rank of LoRA weights')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')

    args = parser.parse_args()

    data_path = args.data_path
    name = args.save_name
    epochs = args.epochs
    r = args.r
    batch_size = args.batch_size
    training = args.training
    evaluate = args.evaluate
    print(training)

    if training:
        model,tokenizer = train(data_path, name, epochs, r, batch_size)
    else:
        model,tokenizer = load(name)

    if evaluate:
        evaluate_model(model, tokenizer, data_path, name)


if __name__ == "__main__":
    main()
