import torch
import torch.autograd as autograd


def predict(text, model, text_field, label_feild):
    model.eval()
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    # x = x.to(device)
    output = model(x)
    a = torch.nn.Softmax()
    output = a(output)
    _, predicted = torch.max(output, 1)

    return label_feild.vocab.itos[predicted.data[0]+1]