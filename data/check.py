import json
import pickle
with open('train_story_generation_all.json', 'rb') as f:
    data = json.load(f)

inputs = []
targets = []
for d in data:
    events = d['events']
    events = [' ; '.join(e) for e in events]
    input_string = ' <eoe> '.join(events) + ' <eoe>'
    story = d['story']
    inputs.append(input_string)
    targets.append(story)

with open('train.pkl', 'wb') as f:
    pickle.dump({'input':inputs, 'target':targets}, f)



with open('dev_story_generation_all.json', 'rb') as f:
    data = json.load(f)

inputs = []
targets = []
for d in data:
    events = d['events']
    events = [' ; '.join(e) for e in events]
    input_string = ' <eoe> '.join(events) + ' <eoe>'
    story = d['story']
    inputs.append(input_string)
    targets.append(story)

with open('dev.pkl', 'wb') as f:
    pickle.dump({'input':inputs, 'target':targets}, f)
