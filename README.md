A Super simple, Super easy to install implementation of the multiple choice models from huggingface at https://huggingface.co/models?pipeline_tag=multiple-choice  
(note this tab is hidden from huggingface, you can only see it by direct link... why...)<br />
I tried to make this installer beginner friendly. 
There are dosens of things to improve but for right now it WORKS and I want to upload it before I break it again.  
Please ignore the terminal outputs, its alot of garbage left over from debugging. Will remove later

quick install (windows):<br />
----
1. download this repo as a zip or using Git.
2. download and install [python 3.10.6](https://www.python.org/downloads/release/python-3106/) (recommended) and add to PATH (untested on later versions)
3. run Install.bat and wait for it to finish completly, it will say press anything to continue
4. run RunBaseModel.bat

how to use:
----
1. put the question in the Question box, Put the any relivent context in the context, and put the questions one by one into the dropdown box.
2. hit "Do Stuff"

The text answer will appear in the right side. Ignore the score. <br />

you can translate the audio to many different languages 

you can edit the `--repo-id [hugging face repo name]` line in the batch file. ex: `venv\Scripts\python.exe Open_WebUI.py --autolaunch --cache-dir "modelsCache" --repo-id "LIAMF-USP/roberta-large-finetuned-race"`<br />

Notes:<br />
----
- If you don't have a cuda enabled GPU, edit RunBaseModel.bat and change:<br />
`venv\Scripts\python.exe Open_WebUI.py --cache_dir "modelsCache"` <br />
to:<br />
`venv\Scripts\python.exe Open_WebUI.py --cache_dir "modelsCache" --device "cpu"`<br />

- The model has to split the text into many chunks to feed to the ai then adds all the confidence scores together.
- the Restart button re-opens and reloads the program. This is for faster debugging. You probably wont need to use this. 

Todo:
----
- [x] make Todo.
- [ ] add option to compute text chunks in parrallel.
- [ ] add option for multiple choice questions.
- [ ] output scores in tables.
- [ ] clean up terminal output.
- [ ] test on different python versions
- [ ] test on different pytorch versions
- [ ] add multiple ways to compute the most correct answer.
