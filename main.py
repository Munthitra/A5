import dash
from dash import html, dcc, Input, Output
import torch
import torchtext
from bert import *
from flask import Flask

app = dash.Dash(__name__)
server = app.server  # Required for deployment to Heroku

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
vocab = torch.load('model/vocab')

params, state = torch.load('model/s-bert.pt')
model = BERT(**params, device=device).to(device)
model.load_state_dict(state)
model.eval()

app.layout = html.Div([
    html.H1("Text Similarity Calculator"),
    html.Div([
        html.Label("Enter Sentence A:"),
        dcc.Input(id='sentence_a', type='text', placeholder='Enter sentence A', value=''),
        html.Label("Enter Sentence B:"),
        dcc.Input(id='sentence_b', type='text', placeholder='Enter sentence B', value=''),
        html.Button('Calculate Similarity', id='calculate-button', n_clicks=0),
        html.Div(id='output-div')
    ])
])

@app.callback(
    Output('output-div', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [Input('sentence_a', 'value'), Input('sentence_b', 'value')]
)
def prediction(n_clicks, sentence_a, sentence_b):
    if n_clicks > 0:
        score = calculate_similarity(model, tokenizer, vocab, params['max_len'], sentence_a, sentence_b, device)
        return f"Similarity Score: {round(score, 4)}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)