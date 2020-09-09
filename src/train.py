import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from model import TextCNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import re
import os
from nltk.stem.porter import PorterStemmer



# Parameters
# ==================================================
path = os.path.dirname(os.path.abspath(__file__))
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .5, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", path + "/positive_data_file.csv", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", path + "/negative_data_file.csv", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 8, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

porter = PorterStemmer()
stop = [re.sub(r'[^a-z\s]+', '', word) for word in stopwords.words('english')]
stop_porter = [porter.stem(word) for word in stop]
def decision_tree():
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    x, y = preprocess_2()
    print(x[0], x[1], x[2])
    dt_tfidf = Pipeline([('vect', tfidf),
                         ('dt', RandomForestClassifier(random_state=0))])
    param_grid = [{'vect__ngram_range': [(1, 1), (1, 2)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer],
                   'dt__max_depth': [None, 10, 8],
                   'dt__n_estimators': [10, 50, 100]},
                  {'vect__ngram_range': [(1, 1), (1, 2)],
                   'vect__stop_words': [stop_porter, None],
                   'vect__tokenizer': [tokenizer_porter],
                   'dt__max_depth': [None, 10, 8],
                   'dt__n_estimators': [10, 50, 100]},
                  ]
    gs_dt_tfidf = GridSearchCV(dt_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    gs_dt_tfidf.fit(x, y)
    print('Best set: %s' % gs_dt_tfidf.best_params_)
    with open('decision_tree_result.txt', 'w+') as result:
        result.write(str(gs_dt_tfidf.cv_results_))

def adaboost():
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    x, y = preprocess_2()
    print(x[0], x[1], x[2])
    ada_tfidf = Pipeline([('vect', tfidf),
                         ('ada', AdaBoostClassifier(random_state=0))])
    param_grid = [{'vect__ngram_range': [(1, 1), (1, 2)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer],
                   'ada__learning_rate': [0.1, 0.5, 1, 5],
                   'ada__n_estimators': [50, 100, 200]},
                  {'vect__ngram_range': [(1, 1), (1, 2)],
                   'vect__stop_words': [stop_porter, None],
                   'vect__tokenizer': [tokenizer_porter],
                   'ada__learning_rate': [0.1, 0.5, 1, 5],
                   'ada__n_estimators': [50, 100, 200]}
                  ]
    gs_dt_tfidf = GridSearchCV(ada_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    gs_dt_tfidf.fit(x, y)
    print('Best set: %s' % gs_dt_tfidf.best_params_)
    with open('ada_boost_result.txt', 'w+') as result:
        means = gs_dt_tfidf.cv_results_['mean_test_score']
        stds = gs_dt_tfidf.cv_results_['std_test_score']
        for mean, std, params, rank in zip(means, stds, gs_dt_tfidf.cv_results_['params'], gs_dt_tfidf.cv_results_['rank_test_score']):
            if params['vect__stop_words'] is not None:
                params['vect__stop_words'] = 'stop'
            result.write("%0.3f (+/-%0.03f) for %r, rank %d\n"
                  % (mean, std * 2, params, rank))

def svm():
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    x, y = preprocess_2()
    print(x[0], x[1], x[2])
    ada_tfidf = Pipeline([('vect', tfidf),
                         ('svm', SVC(random_state=0))])
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer],
                   'svm__C': [0.1, 1, 5],
                   'svm__kernel': ['rbf', 'poly', 'sigmoid']},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop_porter, None],
                   'vect__tokenizer': [tokenizer_porter],
                   'svm__C': [0.1, 1, 5],
                   'svm__kernel': ['rbf', 'poly', 'sigmoid'],
                   'svm__gamma': ['scale', 1e-8, 1e-6, 1e-4]}
                  ]
    gs_dt_tfidf = GridSearchCV(ada_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    print('Best set: %s' % gs_dt_tfidf.best_params_)
    with open('nonlinear_svm_result.txt', 'w+') as result:
        result.write(str(gs_dt_tfidf.cv_results_))

def logreg():
    x_train, y_train = preprocess_2()
    x_test, y_test = load_new_data()
    dt_tfidf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), stop_words=None, tokenizer=tokenizer_porter)),
                         ('dt', LogisticRegression())])
    start_time = time.time()
    dt_tfidf.fit(x_train, y_train)
    y_predict = dt_tfidf.predict(x_test)
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    print("Final Accuracy linear: %f, time: %f" % (
    np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time))
    return (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time)



def preprocess_2():
    x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file, number_only=True)
    for i in range(len(x_text)):
        x_text[i] = re.sub(r'[^a-z\s]+', '', x_text[i].lower())
        x_text[i] = re.sub(r'[ ]{2,}', ' ', x_text[i])
        shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text = np.array(x_text)
    y = np.array(y)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_text[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled[:1000], y_shuffled[:1000]

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    yield x_train, y_train, vocab_processor, x_dev, y_dev
    yield x_dev, y_dev, vocab_processor, x_train, y_train

def leprocess():
    x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_eval, y_eval = data_helper.load_data_and_labels('new_positive.csv', 'new_negative.csv')
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length + 10)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    x_eval = np.array(list(vocab_processor.transform(x_eval)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled[:1000], y_shuffled[:1000], vocab_processor, x_eval, y_eval

def train_CNN(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================
    acc = 0
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints, save_relative_paths=True)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                nonlocal acc
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                acc = accuracy
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    return acc

def load_new_data():
    x_text, y = data_helper.load_data_and_labels('new_positive.csv', 'new_negative.csv', number_only=True)
    for i in range(len(x_text)):
        x_text[i] = re.sub(r'[^a-z\s]+', '', x_text[i].lower())
        x_text[i] = re.sub(r'[ ]{2,}', ' ', x_text[i])
        shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text = np.array(x_text)
    y = np.array(y)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_text[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled, y_shuffled


def decision_tree_final():
    x_train, y_train = preprocess_2()
    x_test, y_test = load_new_data()
    dt_tfidf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words=None, tokenizer=tokenizer_porter)),
                         ('dt', RandomForestClassifier(n_estimators=100))])
    start_time = time.time()
    dt_tfidf.fit(x_train, y_train)
    y_predict = dt_tfidf.predict(x_test)
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    print("Final Accuracy Random Forest: %f, time: %f" % (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time))
    return (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time)

def adaboost_final():
    x_train, y_train = preprocess_2()
    x_test, y_test = load_new_data()
    dt_tfidf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words=None, tokenizer=tokenizer_porter)),
                         ('dt', AdaBoostClassifier(n_estimators=200, learning_rate=0.5))])
    start_time = time.time()
    dt_tfidf.fit(x_train, y_train)
    y_predict = dt_tfidf.predict(x_test)
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    print("Final Accuracy Adaboost: %f, time: %f" % (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time))
    return (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time)

def svm_final():
    x_train, y_train = preprocess_2()
    x_test, y_test = load_new_data()
    dt_tfidf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), stop_words=None, tokenizer=tokenizer_porter)),
                         ('dt', SVC(C=5, gamma='scale', kernel='rbf'))])
    start_time = time.time()
    dt_tfidf.fit(x_train, y_train)
    y_predict = dt_tfidf.predict(x_test)
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    print("Final Accuracy SVM: %f, time: %f" % (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time))
    return (np.sum((y_predict == y_test).astype(float)) / len(y_test), time.time() - start_time)


"""def main(argv=None):
    accs = np.array([])
    for x_train, y_train, vocab_processor, x_dev, y_dev in preprocess():
        accs = np.append(accs, train_CNN(x_train, y_train, vocab_processor, x_dev, y_dev))
    print(accs.sum() / 2, accs.std())
    

if __name__ == '__main__':
    tf.compat.v1.app.run()"""


def main(argv=None):
    start_time = time.time()
    print("accuracy=%f" % train_CNN(*final_evaluation_preprocess()))
    print("Time=%f" % (time.time() - start_time))


if __name__ == '__main__':
    tf.compat.v1.app.run()

def final_evaluation():
    RF_accuracies = []
    RF_time = []
    SVM_accuracies = []
    SVM_time = []
    ADA_accuracies = []
    ADA_time = []
    for i in range(5):
        svmacc, svmt = svm_final()
        rfacc, rft = decision_tree_final()
        adaacc, adat = adaboost_final()
        RF_accuracies.append(rfacc)
        RF_time.append(rft)
        SVM_accuracies.append(svmacc)
        SVM_time.append(svmt)
        ADA_accuracies.append(adaacc)
        ADA_time.append(adat)
    RF_accuracies = np.array(RF_accuracies)
    RF_time = np.array(RF_time)
    SVM_accuracies = np.array(SVM_accuracies)
    SVM_time = np.array(SVM_time)
    ADA_accuracies = np.array(ADA_accuracies)
    ADA_time = np.array(ADA_time)
    print("Random Forest acc: %f +/- %f, time: %f +/- %f" % (np.mean(RF_accuracies), 2 * np.std(RF_accuracies), np.mean(RF_time), 2 * np.std(RF_time)))
    print("Ada boost acc: %f +/- %f, time: %f +/- %f" % (np.mean(ADA_accuracies), 2 * np.std(ADA_accuracies), np.mean(ADA_time), 2 * np.std(ADA_time)))
    print("SVM acc: %f +/- %f, time: %f +/- %f" % (np.mean(SVM_accuracies), 2 * np.std(SVM_accuracies), np.mean(SVM_time), 2 * np.std(SVM_time)))


#final_evaluation()