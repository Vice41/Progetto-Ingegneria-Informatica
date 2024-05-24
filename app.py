import os
from flask import Flask, request, render_template
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

print("API Key:", api_key)
#print("Endpoint:", azure_endpoint)

#token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=azure_endpoint
)

#client = AzureOpenAI(
#    azure_endpoint="https://fairnesschatbot.openai.azure.com/",
#    azure_ad_token_provider=token_provider,
#    api_version="2024-02-01",
#)

app = Flask(__name__)

messagesMemory = []

# Function to add a new message while maintaining the limit of 5 messages
def add_message(msgList, new_message):
    msgList.append(new_message)
    if len(msgList) > 5:
        del msgList[0]  # Delete the oldest message if the limit is exceeded

@app.route('/')
def home():
    # just render the HTML homepage
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def detect_intent():
    text=request.form["message"]
    add_message(messagesMemory, 
            {
                "role": "user",
                "content": text,
            }
        )
    currentMessages=[
            {
                "role": "system",
                "content": "be a nice and helpful assistant",
            },
        ]
    #add the up to the last 5 messages as history
    for msg in messagesMemory:
        currentMessages.append(msg)

    chat_completion = client.chat.completions.create(
        model="fairBot",
        messages=currentMessages,
        #model="gpt-35-turbo",
    )

    print(chat_completion)
    add_message(messagesMemory, 
            {
                "role": "assistant",
                "content": chat_completion.choices[0].message.content,
            }
        )
    
    

    return str(chat_completion.choices[0].message.content)


if __name__ == '__main__':
    app.run()