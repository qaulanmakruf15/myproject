from flask import Flask, request

app = Flask(__name__)

FB_API_URL = 'https://graph.facebook.com/v8.0/me/messages'
VERIFY_TOKEN = 'pass' # <paste your verify token here>
PAGE_ACCESS_TOKEN = 'EAAp2V59oJJ8BAFsVDuQ25Bs3H8Wt3ZCTQOzF31Qjx0tFHwCVGpUZAp9bLCcr69gyVmZCXVxTWive6lHAVDtgvCsJEgSe2LaOf062ZAQfAouZClENh3nlSkFatPKLUMR0kwrJbGc37j8HG958l1E28sfTunLFssCLaPwxW1ShvewZDZD' # paste your page access token here>"


def get_bot_response(message):
    """This is just a dummy function, returning a variation of what
    the user said. Replace this function with one connected to chatbot."""
    return "This is a dummy response to '{}'".format(message)


def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = get_bot_response(message)
    send_message(sender, response)


def is_user_message(message):
    """Check if the message is a message from the user"""
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))


@app.route("/webhook")
def listen():
    """This is the main function flask uses to 
    listen at the `/webhook` endpoint"""
    if request.method == 'GET':
        return verify_webhook(request)

    if request.method == 'POST':
        payload = request.json
        event = payload['entry'][0]['messaging']
        for x in event:
            if is_user_message(x):
                text = x['message']['text']
                sender_id = x['sender']['id']
                respond(sender_id, text)

        return "ok"

import requests

def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {
        'message': {
            'text': text
        },
        'recipient': {
            'id': recipient_id
        },
        'notification_type': 'regular'
    }

    auth = {
        'access_token': PAGE_ACCESS_TOKEN
    }

    response = requests.post(
        FB_API_URL,
        params=auth,
        json=payload
    )

    return response.json()