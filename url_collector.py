import json
import requests
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
                if("youtube" in url or "youtu.be" in url):
                    continue
                
                else:
                    target_urls[gloss_value].append(url)
for target, urls in target_urls.items():
    print(f"URLS for values: '{target}':{urls}")

#Saving urls of each sign into a file
for key in target_urls:
    for i, value in enumerate(target_urls[key]):
        #Creating file
        with open (key + str(i) + ".mp4", "wb") as file:
            try:
                x = requests.get(value)
                #ensuring video works
                if x.status_code == 200:
                    file.write(x.content)
                else:
                    print(f"Skipping {value}: status {x.status_code}")
            except Exception as e:
                 print(f"Skipping {value}: {e}")
                
