import os
from flask import Flask, request, render_template
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


initialPrompt = """
                You are a helpful assistant and should answer me clearly.
                Specifically you are a chatbot. Your responces should be short and undestandable (less than 800 characters!).
                You should guide the user into asking meaningful questions, and navigate the results.
                I have a dataset made of women that might have diabetes, with different data and the outcome (diabetes or not diabetes)
                the columns are: Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Outcome		AgeCategory
                I trained the 2 ML model one using bagging, the other with random forest. And I calculated the fairness metrics. the sensitive attribute is AgeCategory.
                age category = 1 means that the person is older than 25 years old
                age category = 0 means that the person i younger than 25 years old
                the discriminated group selected in the study is the one with value “0”. so the younger group
                Fairness Metrics List:
                Group Fairness: (d=1|G=priv) = (d=1|G=discr)
                (a.k.a. statistical parity, equal acceptance rate, benchmarking). A classifier satisfies this definition if subjects in both protected and unprotected groups have equal probability of being assigned to the positive predicted class.

                Predictive Parity: (Y=1|d=1,G=priv) = (Y=1|d=1,G=discr)
                (a.k.a. outcome test). A classifier satisfies this definition if both protected and unprotected groups have equal PPV : the probability of a subject with positive predictive value to truly belong to the positive class.

                Predictive Equality: (d=1|Y=0,G=priv) = (d=1|Y=0,G=discr)
                (a.k.a. False positive error rate balance). A classifier satisfies this definition if both protected and unprotected groups have equal FPR : the probability of a subject in the negative class to have a positive predictive value.

                Equal Opportunity: (d=0|Y=1,G=priv) = (d=0|Y=1,G=discr)
                (a.k.a. False negative error rate balance). A classifier satisfies this definition if both protected and unprotected groups have equal FNR : the probability of a subject in a positive class to have a negative predictive value.

                Equalized Odds: (d=1|Y=i,G=priv) = (d=1|Y=i,G=discr), i ∈ 0,1
                (a.k.a. conditional procedure accuracy equality and disparate mistreatment). This definition combines the previous two: a classifier satisfies the definition if protected and unprotected groups have equal TPR and equal FPR. Mathematically, it is equivalent to the conjunction of conditions for false positive error rate balance and false negative error rate balance definitions.

                ConditionalUseAccuracyEquality: (Y=1|d=1, G=priv) = (Y=1|d=1,G=discr) and (Y=0|d=0,G=priv) = (Y=0|d=0,G=discr)
                This definition conjuncts two conditions: equal PPV and NPV : the probability of subjects with positive predictive value to truly belong to the positive class and the probability of subjects with negative predictive value to truly belong to the negative class.

                Overall Accuracy Equality: (d=Y, G=priv) = (d=Y, G=priv)
                A classifier satisfies this definition if both protected and unprotected groups have equal prediction accuracy : the probability of a subject from either positive or negative class to be assigned to its respective class. The definition assumes that true negatives are as desirable as true positives.

                Treatment Equality: (Y=1, d=0, G=priv)/(Y=0, d=1, G=priv) = (Y=1, d=0, G=discr)/(Y=0, d=1, G=discr)
                This definition looks at the ratio of errors that the classifier makes rather than at its accuracy. A classifier satisfies this definition if both protected and unprotected groups have an equal ratio of false negatives and false positives.

                FOR Parity: (Y=1|d=0, G=priv) = (Y=1|d=0,G=discr)
                compares the False Omission Rates (FOR) across different demographic groups

                the metrics are calculated in 2 ways: with divison and subtraction. the
                the metric calculated with the division is fair when is close to 1, but in this case we always subtract 1 to the metric calculated with the division so that its centered in "0", so that a value close to 0 means that its fair.
                the metric calculated with the subtraction is fair when is close to 0.
                the metrics have values from -1 to 1.
                a value greater than zero means that the discriminated group is discriminated.
                a value smaller than zero means that the priviledged group is discriminated.
                the treshold for discrimination is a deviation from 0 of 0.15. That means that a for a metric between the values of -0.15 and 0.15 the ml model, in relation to that metric, does not appear to have discrimination.

                BAGGING
                Metric	Division	Subtraction
                GroupFairness	-0.77447	0.10186
                PredictiveParity	0.15079	-0.0936
                PredictiveEquality	-0.89419	0.27707
                EqualOpportunity	-0.31429	-0.18333
                EqualizedOdds	-0.89419	0.18333
                ConditionalUseAccuracyEquality	0.33179	-0.22271
                OverallAccuracyEquality	0.20408	1
                TreatmentEquality	-0.68831	-1
                FORParity	1	0.22271

                RANDOM FOREST
                Metric	Division	Subtraction
                GroupFairness	-0.87113	0.11457
                PredictiveParity	0.61111	-0.37931
                PredictiveEquality	-1	0.30986
                EqualOpportunity	-0.4	-0.26667
                EqualizedOdds	-1	0.26667
                ConditionalUseAccuracyEquality	0.61111	-0.37931
                OverallAccuracyEquality	0.2449	1
                TreatmentEquality	-1	-1
                FORParity	1	0.21283

                For a non-expert user possible routes to navigate could be:
                What is fairness in the context of ML?
                Why so many metrics and which one to use.
                How to interpret the results.
                ....and so on.
                So when the conversation gets stale and the user doesnt know what to ask, suggest these topics.

                From the next message your conversation with the user shall begin.
                """

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=azure_endpoint
)

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
                "content": initialPrompt,
            },
        ]
    #add the up to the last 5 messages as history
    for msg in messagesMemory:
        currentMessages.append(msg)

    chat_completion = client.chat.completions.create(
        model="fairBot",
        messages=currentMessages,
    )

    add_message(messagesMemory, 
            {
                "role": "assistant",
                "content": chat_completion.choices[0].message.content,
            }
        )
    
    return str(chat_completion.choices[0].message.content)


if __name__ == '__main__':
    app.run()
