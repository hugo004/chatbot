import json
import os
import random
import pickle
import spacy

from src.utils import PROJECT_ROOT_PATH
from models.predict import predict_intent, predict_ner
from data.generate_data import get_intents
from keras.models import load_model


def get_trained_data(name: str, intent_path=None):
    model = load_model(
        os.path.join(PROJECT_ROOT_PATH, f"models/{name}.h5")
    )
    words = pickle.load(
        open(os.path.join(PROJECT_ROOT_PATH, f"models/{name}-words.pkl"), "rb")
    )
    labels = pickle.load(
        open(os.path.join(PROJECT_ROOT_PATH,
             f"models/{name}-labels.pkl"), "rb")
    )

    return {
        'model': model,
        'words': words,
        'labels': labels,
        'intents': json.loads(open(os.path.join(PROJECT_ROOT_PATH, intent_path)).read()) if intent_path is not None else get_intents()
    }


chatbot = get_trained_data('chatbot')

form = get_trained_data(name='form',
                        intent_path='data/intents/eform/form-expert.json')
form_upload = get_trained_data(name='upload',
                               intent_path='data/intents/eform/form-upload-expert.json')
book_flight = get_trained_data(name='book-flight',
                               intent_path='data/intents/booking/booking-expert.json')

model_dict = {
    'form-custom': form,
    'upload-fail': form_upload,
    'book-flight': book_flight
}

chatbot_name = "DEV"
context = {}
ie_context = {}
bot_memory = []
user = "123"


def catch_ner_info(ent, context_filter, show_detail=False):
    ner_tag = ent.label_
    if ner_tag in ["FROM", "TO", "DPTL", "DSTT", "RTND", "DPTD"]:
        bot_memory.append(ner_tag)

    latest_memory = bot_memory[-1] if len(bot_memory) > 0 else None

    if show_detail:
        print(
            "current ner",
            ner_tag,
            "bot memory:",
            bot_memory,
            "latest memory:",
            latest_memory,
        )

    if context_filter == "bkd-query":
        if ner_tag == "GPE":
            if latest_memory in ["FROM", "DPTL"]:
                ie_context["destination"] = ent.text
                bot_memory.pop()
            elif latest_memory in ["TO", "DSTT"]:
                ie_context["depature"] = ent.text
                bot_memory.pop()

    elif context_filter in ["bkt-query", "bklt-query"]:
        if ner_tag in ["CARDINAL", "DATE"]:
            if latest_memory in ["FROM", "DPTD"]:
                ie_context["return-date"] = ent.text
                bot_memory.pop()
            elif latest_memory in ["TO", "RTND"]:
                ie_context["depature-date"] = ent.text
                bot_memory.pop()


def get_response(predicted: list, intents: list, userId: str, doc=None, forward_type=False, show_detail=False):
    response = "Sorry, I don't understand"
    result = None

    if show_detail:
        print("\npredicted:", predicted)

    if predicted is None:
        return (response, None)

    for data in predicted:
        tag = data["intent"]

        for i in intents:
            if i["tag"] == tag:
                if "context_set" in i:
                    if show_detail:
                        print("context:", i["context_set"])

                    context[userId] = i["context_set"]

                if (
                    not "context_filter" in i
                    or ("context_filter" in i and "context_set" in i)
                    or (userId in context and i["context_filter"] == context[userId])
                ):
                    if show_detail:
                        print("tag:", i["tag"])

                    result = i
                    tag = i["tag"]
                    if len(i["responses"]) > 0:
                        response = random.choice(i["responses"])

                # perform information extraction (IE)
                if 'type' in i and i['type'] == 'ie':
                    if "context_filter" in i:
                        print("\n")
                        print("-" * 25, "NLP", "-" * 25)
                        print("predicted user intent: ", tag)

                        if doc is not None:
                            print("nlp ents:", doc.ents)
                            for ent in doc.ents:
                                catch_ner_info(
                                    ent, i["context_filter"], show_detail)
                                print("entity:", ent,
                                      "text:", ent.text,
                                      "NER label:", ent.label_)
                                print("context info:", ie_context)

                        break
    if show_detail:
        print('forward type:', forward_type, '\nresponse:')

    return (response, result)


def chatbot_response(sentence: str, userId, show_detail=False):
    ner = predict_ner(sentence)
    predicted = predict_intent(sentence, model=chatbot['model'],
                               words=chatbot['words'], labels=chatbot['labels'])

    response, intent = get_response(predicted=predicted,
                                    intents=chatbot['intents'],
                                    doc=ner,
                                    userId=userId,
                                    show_detail=show_detail)

    if intent is not None and 'type' in intent and intent['type'] == 'forward':
        forward_model = model_dict[intent['tag']]
        predicted = predict_intent(sentence=sentence, model=forward_model['model'],
                                   words=forward_model['words'], labels=forward_model['labels'])

        forward_response, forward_intent = get_response(predicted=predicted,
                                                        intents=forward_model['intents'],
                                                        userId=userId,
                                                        doc=ner,
                                                        forward_type=True,
                                                        show_detail=show_detail)

        return (forward_response, forward_intent)

    return (response, intent)


if __name__ == "__main__":
    print(f"${chatbot_name}: My name is ${chatbot_name}. What can i help you ?")
    nlp = spacy.load('en_core_web_sm')

    while True:
        user_response = input()
        if user_response in ["bye", "exit", "quit"]:
            print(f"${chatbot_name}: bye")
            print("\n")
            if bool(ie_context):
                print("-" * 10, "information", "-" * 10)
                print(ie_context)
            break

        # handle multiple intents
        doc = nlp(user_response)
        for sent in doc.sents:
            response, user_intent = chatbot_response(sentence=sent.text,
                                                     userId=user,
                                                     show_detail=True)

            print("\n")
            print("-" * 25, "RESPONSE", "-" * 25)
            print(f"${chatbot_name}: ${response}\n")
