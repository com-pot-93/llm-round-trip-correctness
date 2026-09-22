import os
import re
import json

""" this function splits plain text into array of sentences """
def split_into_sentences(paragraph):
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s|[.?!](?=[A-Z]|\"|\')'
    sentences = re.split(sentence_endings, paragraph)
    return [s for s in sentences if s != "" and s != "\n" and len(s) > 1]

datasets = ["domain","mad","pet", "sapsam","realset"]
# sentences
for d in datasets:
    container = []
    descriptions = os.path.join(os.getcwd(),"data", d, "process_descriptions")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d, "process_descriptions",file)) as f:
            #print("--------------------------")
            sentences = split_into_sentences(f.read())
            container.append(len(sentences))
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)

print("---------iwords--------")

for d in datasets:
    container = []
    descriptions = os.path.join(os.getcwd(),"data", d, "process_descriptions")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d, "process_descriptions",file)) as f:
            sentences = split_into_sentences(f.read())
            words = []
            for s in sentences:
                count = len(re.findall("[a-zA-Z_]+", s))
                if count != 0:
                    words.append(count)
            container.append(sum(words))
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)

print("--------tasks------")

for d in datasets:
    container = []
    descriptions = os.path.join(os.getcwd(),"data", d, "ground_truth")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d,"ground_truth",file)) as f:
            data = json.load(f)
            container.append(len(data["tasks"]))
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)


print("----events----------")

for d in datasets:
    container = []
    types = []
    descriptions = os.path.join(os.getcwd(),"data", d, "ground_truth")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d,"ground_truth",file)) as f:
            data = json.load(f)
            container.append(len(data["events"]))
            #print(data["events"])
            for e in data["events"]:
                types.append(e["type"])
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)
    #print(types)
    unique = list(dict.fromkeys(types))
    #print(unique)

print("------gateways---------")

for d in datasets:
    container = []
    types = []
    descriptions = os.path.join(os.getcwd(),"data", d, "ground_truth")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d,"ground_truth",file)) as f:
            data = json.load(f)
            container.append(len(data["gateways"]))
            #print(data["events"])
            for e in data["gateways"]:
                types.append(e["type"])
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)
    #print(types)
    unique = list(dict.fromkeys(types))
    #print(unique)


print("-------pools--------")

for d in datasets:
    container = []
    types = []
    descriptions = os.path.join(os.getcwd(),"data", d, "ground_truth")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d,"ground_truth",file)) as f:
            data = json.load(f)
            #container.append(len(data["gateways"]))
            if data["pools"]:
                a = len(data["pools"])
            else:
                a = 0
            container.append(a)
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)


print("--lanes-------------")

for d in datasets:
    container = []
    types = []
    descriptions = os.path.join(os.getcwd(),"data", d, "ground_truth")
    dir_list = os.listdir(descriptions)
    for file in dir_list:
        with open(os.path.join(os.getcwd(),"data", d,"ground_truth",file)) as f:
            data = json.load(f)
            #container.append(len(data["gateways"]))
            if data["pools"]:
                a = sum(len(p["lanes"]) for p in data["pools"])
            else:
                a = 0
            container.append(a)
    average = round(sum(container) / len(container),1)
    #print(container)
    print(average)


