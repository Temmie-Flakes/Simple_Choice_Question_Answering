#time functions for evaluation of efficiency
import time
timeKeeper=time.perf_counter()

def howLong(start_time=None, name = "function", resetCount=False):
    global timeKeeper
    if 'timeKeeper' in globals():
        if not start_time:
            start_time=timeKeeper
        if resetCount:
            timeKeeper=time.perf_counter()
    elif not start_time:
        start_time=0.0
    print(f"{name} took: {time.perf_counter() - start_time}s")
from transformers import AutoModelForMultipleChoice, AutoTokenizer#, pipeline
import argparse
import os
import torch
howLong(name="import things",resetCount=True)
'''
#example of error handling
try:
    [code]
except Exception as err:
    print(err)
    pass

'''
#deepset/roberta-base-squad2
#model_name = "ncduy/bert-base-uncased-finetuned-swag"
#C:\\AI\\MyThings\\Question_Answering\\modelsCache\\bert-base-uncased-finetuned-swag
parser = argparse.ArgumentParser()
parser.add_argument('--repo-id','-m', default='LIAMF-USP/roberta-large-finetuned-race', help='Path to model repo (default: ncduy/bert-base-uncased-finetuned-swag)')
parser.add_argument('--device','-d', default='cuda', help='Device to use for inference (default: cuda)')
parser.add_argument('--cache-dir', default=None, help='Directory of the folder to download models. Ex: "models" will make/use a folder named models in the same directory as this program (~\\models\\). The default directory is C:\\Users\\[username]\\.cache\\huggingface\\hub\\')
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
args = parser.parse_args()

if args.cache_dir:
    cache_dir = (args.cache_dir+"/"+args.repo_id.split("/")[-1])
else:
    cache_dir = args.repo_id
  
  
if os.path.isdir(cache_dir):
    model_path = cache_dir
else:
    import huggingface_hub
    print("Downloading model...")
    kwargs = {}
    if cache_dir is not None:
        kwargs["local_dir"] = cache_dir
        kwargs["cache_dir"] = cache_dir
        # kwargs["local_dir"] = "C:/AI/MyThings/nimple Speech Recognition/ssd"
        # kwargs["cache_dir"] = "C:/AI/MyThings/nimple Speech Recognition/sfw"
        kwargs["local_dir_use_symlinks"] = False
    #minimum to run.
    #allow_patterns = ["config.json","pytorch_model.bin","merges.txt","vocab.json"]
    #allow_patterns = ["config.json","model.bin","tokenizer.json","vocabulary.txt",]
    model_path=huggingface_hub.snapshot_download(args.repo_id,**kwargs)#tqdm_class=disabled_tqdm,,
    #model_path=huggingface_hub.snapshot_download(args.repo_id,allow_patterns=allow_patterns,**kwargs)#tqdm_class=disabled_tqdm,,
#import tensorflow
#from_tf=True
howLong(name="download model",resetCount=True)
model = AutoModelForMultipleChoice.from_pretrained(model_path).to(args.device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
howLong(name="load model",resetCount=True)
print(model.config.max_position_embeddings)
#nlp = pipeline('question-answering', model=model, tokenizer=tokenizer, device=args.device)
        
def restartRecompile():
    import sys
    if '--autolaunch' in sys.argv:
        sys.argv.remove('--autolaunch')
    os.execl(sys.executable, 'python', __file__, *sys.argv[1:])

def findAnswerInText(question, context, answers):
    timeKeeper=time.perf_counter()
    labels = torch.tensor(0, device=args.device).unsqueeze(0)
    print(answers)
    logits_Sum=torch.zeros(1, len(answers),device=args.device)
    howLong(name="create tensors",resetCount=True)
    #for i in 
    #ANS=[[question] * len(answers), answers]
    #MAX_TOKEN_LENGTH=model.config.max_position_embeddings
    MAX_TOKEN_LENGTH=model.config.max_position_embeddings-4
    
    questionTokens=tokenizer.encode(question)[1:-1]
    contextTokens=tokenizer.encode(context)[1:-1]
    
    maxAns=0
    for x in answers:
        tokenLen=len(tokenizer.encode(x))-2
        if (tokenLen>maxAns):
            maxAns=tokenLen
    remainingTokens=len(contextTokens)
    howLong(name="create remainingTokens",resetCount=True)
    while (remainingTokens>0):
        print("_______________REMAINING TOKENS_______________-")
        print(remainingTokens)
        print("_______________REMAINING TOKENS_______________-")
        tokensToRemove=0
        totalMaxTokens=len(questionTokens)+len(contextTokens)+maxAns+3
        if (totalMaxTokens>MAX_TOKEN_LENGTH):
            tokensToRemove=totalMaxTokens-MAX_TOKEN_LENGTH
            #choppedQuestion=questionTokens[:-tokensToRemove]
            choppedContext=contextTokens[:-tokensToRemove]
        else:
            choppedContext=contextTokens
            #choppedQuestion=questionTokens
            
        
        
        #remainingTokens=remainingTokens-len(choppedQuestion)
        remainingTokens=remainingTokens-len(choppedContext)
        contextTokens=contextTokens[-tokensToRemove:]

        ANS = [[tokenizer.decode(choppedContext)+tokenizer.decode(questionTokens), candidate] for candidate in answers]  
        print(tokenizer.decode(choppedContext))       
        #print(ANS)   
        howLong(name="chopping text",resetCount=True)        
        #inputs = tokenizer(ANS, return_tensors="pt",max_length=MAX_TOKEN_LENGTH, padding=True).to(args.device)
        inputs = tokenizer(ANS, return_tensors="pt",padding='max_length').to(args.device)
        #print(inputs)
        howLong(name="tokenize text",resetCount=True)      
    #ANS=[([question] * len(answers), answers)]
    #print(ANS)
    #inputs = tokenizer(ANS, return_tensors="pt", padding='max_length',max_length=model.config.max_position_embeddings,truncation=True).to(args.device)
    #inputs = tokenizer(ANS, return_tensors="pt",max_length=model.config.max_position_embeddings,truncation=True, padding=True).to(args.device)
        ##print(inputs.items())
        #for x in inputs.items():
        #    print(x)
        length=len(inputs['input_ids'][0])
        #print(length)

        

        #print(len(labels))
        #print(labels)
        with torch.no_grad():
            outputs = (model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)).__dict__
        print(outputs)
        howLong(name="created output",resetCount=True)      
        #predicted_class=outputs.logits.argmax().item()
        print("________________________________________________________")
        #logits_Sum += outputs.logits.clone().detach()
        logits_Sum += outputs['logits']
        howLong(name="created logits_Sum",resetCount=True)
        #print(outputs.logits)
        howLong(name="print outputs.logits",resetCount=True)
        print(logits_Sum)
        howLong(name="print logits_Sum",resetCount=True)
        #print(inputs)
        #print(outputs)
        #del inputs
        #outputs.clear()
        howLong(name="outputs.clear()",resetCount=True)
        #torch.cuda.empty_cache()
        #print(outputs.logits[0][0].item())
        #print(outputs.logits.argmax())
        #print(outputs)
        howLong(name="empty cache",resetCount=True)
    #return res.get('answer'),res.get('score')
    predicted_class=logits_Sum.argmax().item()
    howLong(name="endblock",resetCount=True)
    return answers[predicted_class] ,predicted_class

import gradio as gr
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            #autoscroll=True
            question_box= gr.Textbox(label='Question',placeholder="Why is model conversion important?")
            context_box= gr.Textbox(label='Context',placeholder="The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.")
            answers= gr.Dropdown(multiselect=True,allow_custom_value=True)
            button = gr.Button("Do Stuff")
            restartButton = gr.Button("Restart Program",variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label='Answer',max_lines=30,interactive=True)
            confidence = gr.Textbox(label='Score',max_lines=30,interactive=True)
    button.click(findAnswerInText, 
    inputs=[
    question_box, 
    context_box, 
    answers,
    ], outputs=[output_text,confidence])
    restartButton.click(restartRecompile, None, None, queue=False)
demo.queue(max_size=30)
howLong(name="gradio setup",resetCount=True)
demo.launch(inbrowser=args.autolaunch, show_error=True, share=False)  




