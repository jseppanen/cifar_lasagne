
import time
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne

def train(network, input_var,
          X_train, y_train, X_val, y_val,
          learning_rate, learning_rate_decay=0.95,
          momentum=0.9, momentum_decay=0.95,
          decay_after_epochs=1,
          regu=0.0,
          batch_size=100, num_epochs=10,
          save_path=None):
    print("Compiling...")
    target_var = T.ivector('target')
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    regu_loss = lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)
    loss = loss + regu * regu_loss
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                       dtype=theano.config.floatX)
    learning_rate_var = theano.shared(np.float32(learning_rate))
    momentum_var = theano.shared(np.float32(momentum))
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate_var, momentum=momentum_var)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    train_acc_fn = theano.function([input_var, target_var], train_acc)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    print("Training...")
    best_val_acc = 0.0
    best_model = None
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        # train model for one pass
        train_err = train_batches = 0
        start_time = time.time()
        for X_batch, y_batch in gen_minibatches(X_train, y_train,
                                                batch_size, shuffle=True):
            err = train_fn(X_batch, y_batch)
            train_err += err
            train_batches += 1
            loss_history.append(err)
        # training accuracy
        n_acc = len(y_val)
        trval_err = trval_acc = trval_batches = 0
        for X_batch, y_batch in gen_minibatches(X_train[:n_acc], y_train[:n_acc],
                                                batch_size, shuffle=False):
            err, acc = val_fn(X_batch, y_batch)
            trval_err += err
            trval_acc += acc
            trval_batches += 1
        trval_acc /= trval_batches
        train_acc_history.append(trval_acc)
        # validation accuracy
        val_err = val_acc = val_batches = 0
        for X_batch, y_batch in gen_minibatches(X_val, y_val,
                                                batch_size, shuffle=False):
            err, acc = val_fn(X_batch, y_batch)
            val_err += err
            val_acc += acc
            val_batches += 1
        val_acc /= val_batches
        val_acc_history.append(val_acc)
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = lasagne.layers.get_all_param_values(network)
        print('epoch %d / %d in %.1fs: loss %f, train: %.3f, val %.3f, lr %e mom %e'
              % (epoch + 1, num_epochs, time.time() - start_time,
                 train_err / train_batches, trval_acc, val_acc,
                 learning_rate_var.get_value(), momentum_var.get_value()))
        # decay learning rate
        if (epoch + 1) % decay_after_epochs == 0:
            learning_rate_var.set_value(
                np.float32(learning_rate_var.get_value() * learning_rate_decay))
            momentum = (1.0 - (1.0 - momentum_var.get_value()) * momentum_decay) \
                       .clip(max=0.9999)
            momentum_var.set_value(np.float32(momentum))
        # save model snapshots
        if save_path and (epoch + 1) % 10 == 0:
            model = lasagne.layers.get_all_param_values(network)
            path = '%s_epoch%03d_acc%.4f.pickle' % (save_path, epoch + 1, val_acc)
            with open(path, 'wb') as f:
                pickle.dump({'model': model}, f, -1)
    return best_model, loss_history, train_acc_history, val_acc_history

def gen_minibatches(X, y, batch_size, shuffle=False):
    assert len(X) == len(y), "Training data sizes don't match"
    if shuffle:
        ids = np.random.permutation(len(X))
    else:
        ids = np.arange(len(X))
    for start_idx in range(0, len(X) - batch_size + 1, batch_size):
        ii = ids[start_idx:start_idx + batch_size]
        yield X[ii], y[ii]

#def test
