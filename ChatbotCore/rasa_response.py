import requests
import json
import os

def RasaRespone(nameFileTxt, sentence):
    URL = "http://localhost:5005/model/parse"
    message = {"text": sentence}
    r = requests.post(url=URL, data=json.dumps(message))

    try:
        # Extracting required information from the response
        response_data = {
            "text": r.json().get("text", ""),
            "intent": {
                "name": r.json()["intent"]["name"],
                "confidence": r.json()["intent"]["confidence"]
            },
            "entities": r.json().get("entities", [])
        }

        # Writing extracted information to a text file in append mode ('a')
        with open(f"{nameFileTxt}.txt", "a", encoding="utf-8") as f:
            f.write(f"Text: {response_data['text']}\n\n")
            f.write(f"Intent: {response_data['intent']['name']} - Confidence: {response_data['intent']['confidence']}\n\n")
            f.write("Entities:\n")
            for entity in response_data['entities']:
                f.write(f"- Entity: {entity['entity']}, Value: {entity['value']}, Confidence: {entity['confidence_entity']}\n")

        return f"- {response_data['intent']['name']} - {response_data['intent']['confidence']}\n"

    except KeyError as e:
        print(f"KeyError encountered for sentence: {sentence}. Error: {e}")
        return ""