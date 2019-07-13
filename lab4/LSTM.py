import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import random
import codecs
from torchtext import data
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 1

def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class MR(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'rt-polarity.neg'),'r','Windows-1252') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with codecs.open(os.path.join(path, 'rt-polarity.pos'),'r','Windows-1252') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./dataset/", **kwargs):

        examples = cls(text_field, label_field, path=path, **kwargs).examples

        #if shuffle: random.shuffle(examples)
        train_index = 4250
        dev_index = 4800
        test_index = 5331

        train_examples = examples[0:train_index] + examples[test_index:][0:train_index]
        dev_examples = examples[train_index:dev_index] + examples[test_index:][train_index:dev_index]
        test_examples = examples[dev_index:test_index] + examples[test_index:][dev_index:]

        random.shuffle(train_examples)
        random.shuffle(dev_examples)
        random.shuffle(test_examples)
        print('train:',len(train_examples),'dev:',len(dev_examples),'test:',len(test_examples))
        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=dev_examples),
                cls(text_field, label_field, examples=test_examples),
                )

# load MR dataset
def load_mr(text_field, label_field, batch_size):
    #print('loading data')
    train_data, dev_data, test_data = MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    #print('building batches')
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_data, dev_data, test_data), batch_sizes=(batch_size, len(dev_data), len(test_data)),repeat=False,
        #device = -1
    )

    return train_iter, dev_iter, test_iter



class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y,dim=1)
        return log_probs

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def parse_embeddings(filename):
    word_to_ix = {}  # index of word in embeddings
    embeds = []
    print("filename:" + filename )
    with open(filename, 'rt', encoding='utf-8') as f:
        i = 0
        for line in f:
            word_and_embed = line.split()
            word = word_and_embed.pop(0)
            word_to_ix[word] = i
            embeds.append([float(val) for val in word_and_embed])
            '''if (i % 50000 == 49999):  # 400,000 lines total
                print("parsed line " + str(i))'''
            i += 1
            if (i > 100000):
                break
    return word_to_ix, embeds

def train():
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 10
    BATCH_SIZE = 10
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter , test_iter = load_mr(text_field, label_field, batch_size=BATCH_SIZE)

    word_to_ix, embeds = parse_embeddings("glove.6B." + str(EMBEDDING_DIM) + "d.txt")
    embeddings = nn.Embedding(len(word_to_ix), len(embeds[0]))  # 50 features per word
    embeddings.weight.data.copy_(torch.FloatTensor(embeds))  # set the weights
    # to the pre-trained vector
    #text_field.vocab.load_vectors('glove.6B.50d')
    '''cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='glove.6B.50d.txt', cache=cache)'''
    #text_field.build_vocab(text_field, vectors=vectors)


    #text_field = word_to_ix
    #print (len(text_field.vocab),(len(label_field.vocab) - 1))

    best_dev_acc = 0.0

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)
    model.word_embeddings.weight.data =embeddings.weight.data
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    no_up = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model,dev_iter,loss_function,'dev')
        test_acc = evaluate(model, test_iter, loss_function,'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()
#
def evaluate(model, eval_iter, loss_function,  name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in eval_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc

def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    for batch in train_iter:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        count += 1
        if count % 100 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.item()))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

train()

