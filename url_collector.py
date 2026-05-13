import json
#Values searching for:
target_urls = {"yes":[], "no":[], "please":[], "thank you":[]}

#Opening file
with open ("WLASL_v0.3.json", 'r') as file:
    data = json.load(file)
    for record in data:        
        #Saves and loads a line, and saves what value its looking at
        gloss_value = record.get("gloss")

        if gloss_value in target_urls:
            #searches for url
            instances = record.get("instances", [])
            for instance in instances:
                url = instance.get("url")
                target_urls[gloss_value].append(url)
for target, urls in target_urls.items():
    print(f"URLS for values: '{target}':{urls}")
