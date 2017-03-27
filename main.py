
# Format of the image is arraged as pertaining to theano
# Will change that to make it in correspondenc with TF
import numpy as np
import argparse
import os

from keras.preprocessing import image
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

from feature_learner import FeatureLearner
from objectives import my_hinge
from utils import generate_embedding_matrix, argrender, lossrender, serialize_ans_embedding_matrix
import word_table

def train(model, embedding_matrix, args):
    for epoch in xrange(args.epochs):
        fd = open(args.data_path,'r')

        print "Epoch %s\n" % str(epoch + 1)
        
        n = 248286 #TODO: CHANGE THIS TO COUNT OF TRAINING EXAMPLES
        count = 0
        it = 0
        
        #TODO: This function will work better if we store the image features previously
        for line in fd.readlines():
            img_features = []
            q_features = []
            labels = []
            count+=1
            img_file, question, answer = line.split('~')
            img_file = args.image_dir + img_file
            
            answer = answer[:-1]

            if args.model_name == 'baseline_classifier':
                if(answer not in args.ClassDict):
                    continue
                ans_vec = np_utils.to_categorical(args.ClassDict[answer], args.max_classes)
            elif args.model_name == 'zsl':
                idx = args.wt.getIdx(answer)
                embedding = embedding_matrix[idx]
                if np.sum(embedding) == 0.0:
                    continue

                ans_vec = np.expand_dims(embedding, axis=0)
            else:
                raise Exception('Model name ' + args.model_name + ' is unrecognized.')

            labels.append(ans_vec)
            
            img = image.load_img(img_file, target_size=(args.img_dims[0], args.img_dims[1]))
            img_vec = image.img_to_array(img)      
            img_features.append(img_vec)
            
            question_vec = args.wt.encodeQ(question)
            q_features.append(question_vec)
        
            if(count % args.batch_size==0):
                it += 1

                q_features = pad_sequences(np.array(q_features), args.max_q_len)

                img_features = np.array(img_features)

                labels = np.array(labels)
                labels = labels.reshape(labels.shape[0], labels.shape[2])

                metrics = model.train_on_batch([q_features, img_features], labels)

                lossrender(args, it, metrics)
        
        if(len(labels) > 0):
            it += 1

            q_features = np.array(q_features)
            q_features = pad_sequences(q_features, args.max_q_len)
            
            img_features = np.array(img_features)
            
            labels = np.array(labels)
            labels = labels.reshape(labels.shape[0],labels.shape[2])

            metrics = model.train_on_batch([q_features, img_features], labels)
            lossrender(args, it, metrics)

def build(args):
    feature_learner = FeatureLearner(args)

    question = Input(shape=(args.max_q_len,), dtype='int32', name='question_input')
    image = Input(shape=args.img_dims, name='image')

    # look up embeddings for question and summarize it
    encoded_question = feature_learner.embed(question)
    question_summary = feature_learner.summarize(encoded_question)

    embedding_matrix = feature_learner.embedding_matrix
    
    # image features from pre-trained model
    img_features = feature_learner.vgg_features(image)

    merged_img_q = merge([img_features, question_summary], mode='concat', concat_axis=-1)

    if args.model_name == 'baseline_classifier':
        answer = Dense(args.max_classes, activation='softmax')(merged_img_q)
    elif args.model_name == 'zsl':
        answer = Dense(args.glove_embed_size)(merged_img_q) # must be same size as glove embeddings
    else:
        raise Exception('No other models available right now. Sorry!')

    return Model(input=[question, image], output=answer), embedding_matrix

def main():
    parser = argparse.ArgumentParser(description='Multimodal VQA Model')

    # model name
    parser.add_argument('--model_name', default='baseline_classifier', help='Model to run.  baseline_classifier or zsl.')

    # data params
    parser.add_argument('--image_dir', default='./data/training/')
    parser.add_argument('--data_path', default='./data_json/train2014.csv')
    parser.add_argument('--embedding_dir', default='./data/glove')

    # training params
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training.')
    parser.add_argument('--dropout', default=False, type=bool, help='Whether or not to use dropout.')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Only applies to dropout==True.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of passes through data during training.')
    parser.add_argument('--loss', default='categorical_crossentropy')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate for training.')
    parser.add_argument('--num_classes', default=1000, type=int, help='Top # of classes to consider.')

    # language model params
    parser.add_argument('--glove_embed_size', default=100, type=int, help='Initial dimensionality of word representation.')
    parser.add_argument('--rep_dims', default=64, type=int, help='Final dimensionality of word representation')
    parser.add_argument('--feature_learning_mode', default='lstm', help='how to represent word embeddings into single vector.  Either lstm, gru, or {avg,max}_pooling.')
    parser.add_argument('--max_classes', default=1000, type=int, help='How many classes/answers to consider for training and testing.')
    parser.add_argument('--max_q_len', default=10, type=int, help='Maximum words for question.  Determines padding/truncation.')

    # global/debugging params
    parser.add_argument('--verbose', default=True, type=bool, help='Affects rendering frequency.  Meant for debugging purposes.')

    args = parser.parse_args()

    args.wt = word_table.WordTable()
    args.vocab_size = args.wt.vocabSize()

    args.ClassDict = args.wt.top_answers(args.data_path, args.max_classes)

    if not os.path.exists('./data/top_answer_embeddings'):
        serialize_ans_embedding_matrix(args.ClassDict, args.embedding_dir, args.glove_embed_size)

    args.img_dims = (224,224,3)

    model, embedding_matrix = build(args)

    optimizer = Adam(lr=args.lr)

    args.METRICS = ['categorical_crossentropy', 'cosine_proximity']

    model.compile(
        optimizer=optimizer,
        loss=my_hinge if args.loss == 'my_hinge' else args.loss,
        metrics=args.METRICS
    )

    if args.verbose:
        model.summary()
        argrender(args)

    print("Training model now...\n")
    train(model, embedding_matrix, args)

if __name__ == '__main__':
    main()



