# RNN average model

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
import os
import pickle

import utils.file2dict as fdt
import utils.read_minibatch as rmb
import utils.data_util as data_util
import utils.confusion_matrix as cm
import utils.data_util as du

from datetime import datetime

import tensorflow as tf
import numpy as np

from AttributionModel import AttributionModel
from proj_rnn_cell import RNNCell
from proj_gru_cell import GRUCell

logger = logging.getLogger("RNN_Author_Attribution")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

tf.reset_default_graph()  #used so that variables gets reinitialized every time

class Config:
    cell_type="gru" # either rnn, gru or lstm

    window_size = 0

    max_length = 70 # longest length of a sentence we will process
    n_classes = 50 # in total, we have 50 classes
    dropout = 0.9

    embed_size = 50

    hidden_size = 300
    batch_size = 64

    n_epochs = 10
    regularization = 0

    max_grad_norm = 10.0 # max gradients norm for clipping
    lr = 0.004 # learning rate

    def __init__(self, args):

        #self.cell = args.cell

        self.cell = GRUCell(Config.embed_size, Config.hidden_size)
        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format("RNN", datetime.now())

        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"

        #self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"

class RNNModel(AttributionModel):
    
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, [None, Config.max_length])
        self.labels_placeholder = tf.placeholder(tf.int32, [None, Config.n_classes])
        self.mask_placeholder = tf.placeholder(tf.float32, [None, Config.max_length])
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):

        feed_dict = {}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch
        return feed_dict

    def add_embedding(self):     
        embeddingTensor = tf.Variable(self.pretrained_embeddings,tf.float32)
        embeddings = tf.nn.embedding_lookup(embeddingTensor, self.input_placeholder)
        return embeddings

    def add_prediction_op(self):
      
        x = self.add_embedding()

        dropout_rate = self.dropout_placeholder
        preds = [] # Predicted output at each timestep should go here!

        if Config.cell_type=="rnn":
            cell = RNNCell(Config.embed_size, Config.hidden_size)
        elif Config.cell_type=="gru":
            cell = GRUCell(Config.embed_size, Config.hidden_size)
        else:
            assert False, "Cell type undefined"

        self.U = tf.get_variable('U',
                              [Config.hidden_size, Config.n_classes],
                              initializer = tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable('b2',
                              [Config.n_classes, ],
                              initializer = tf.contrib.layers.xavier_initializer())
        h = tf.zeros([tf.shape(x)[0], Config.hidden_size])

        with tf.variable_scope("RNN"):

            for time_step in range(Config.max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                o, h = cell(x[:,time_step,:], h)

                o_drop = tf.nn.dropout(o, dropout_rate)
                preds.append(tf.matmul(o_drop, self.U) + self.b2)

        self.raw_preds=tf.stack(preds)
        preds=tf.reshape(tf.transpose(self.raw_preds, [1, 0, 2]),[-1,Config.max_length,Config.n_classes])
        return preds

    def add_loss_op(self, preds):

        pred_mask=tf.reshape(self.mask_placeholder,[-1,Config.max_length,1])
        pred_mask=tf.tile(pred_mask,[1,1,Config.n_classes])
        pred_masked= preds * pred_mask
        pred_masked=tf.reduce_sum(pred_masked,axis=1)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_masked, labels=self.labels_placeholder)

        loss = tf.reduce_mean(loss) + config.regularization * ( tf.nn.l2_loss(self.U) )
        with tf.variable_scope("RNN/cell", reuse= True):
            # add regularization

            loss += config.regularization * (tf.nn.l2_loss(tf.get_variable("W_r"))
                                             + tf.nn.l2_loss(tf.get_variable("U_r"))
                                             + tf.nn.l2_loss(tf.get_variable("W_z"))
                                             + tf.nn.l2_loss(tf.get_variable("U_z"))
                                             + tf.nn.l2_loss(tf.get_variable("W_o"))
                                             + tf.nn.l2_loss(tf.get_variable("U_o")))
        return loss

    def add_training_op(self, loss):        
        train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):

        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss, pred = sess.run([self.train_op, self.loss, self.pred], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
    
        feed = self.create_feed_dict(inputs_batch,mask_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        pred_mask=sess.run(self.mask_placeholder,feed_dict=feed)
        pred_mask=np.reshape(pred_mask,(-1,Config.max_length,1))
        pred_mask=np.tile(pred_mask,(1,1,Config.n_classes))
        pred_masked=predictions*pred_mask
        pred_masked=np.sum(pred_masked,axis=1)
        return pred_masked

    def record_history_init(self,f):
        f.write("###\n")
        f.write("training_model: "+sys.argv[0]+"\n")
        f.write("starting_time: "+str(datetime.now())+"\n")
        f.write("\n")
        f.write("parameters:\n")
        f.write("\tcell_type: "+Config.cell_type+"\n")
        f.write("\tembed_size: {}\n".format(Config.embed_size))
        f.write("\thidden_size: {}\n".format(Config.hidden_size))
        f.write("\tlearning_rate: {0:.4f}\n".format(Config.lr))
        f.write("\tregularization: {0:.7f}\n".format(Config.regularization))
        f.write("\tdropout: {0:.5f}\n".format(Config.dropout))
        f.write("\tn_epochs: {0:.5f}\n".format(Config.n_epochs))
        f.write("\tbatch_size: {}\n".format(Config.batch_size))
        f.write("\n")
        f.write("training history:\n")
        f.write("epoch \t\t loss \t\t train_accu \t\t test_accu\n")

    def record_history_accu(self,f,n_epoch,average_train_loss,train_accu,test_accu):
        f.write("{0:d} \t\t {1:.5f} \t\t {2:.5f} \t\t {3:.5f}\n".format(n_epoch,average_train_loss,train_accu,test_accu))

    def record_history_finish(self,f):
        f.write("END\n")
        f.write("\n")
        f.close()

    def test_model(self, session, batch_list):
        print "Now, testing on the test set..."
        total = 0
        accuCount = 0
        for batch in batch_list:
            batch_feat = np.array(batch[0], dtype = np.int32)
            batch_mask = np.array(batch[1], dtype = np.float32)
            pred = self.predict_on_batch(session, batch_feat, batch_mask)
            accuCount += np.sum(np.argmax(pred,1) == batch[2])
            total += len(batch[2])
        accu = accuCount * 1.0 / total
        logger.info( ("Test accuracy %f" %(accu)) )
        return accu

    def test_trainset_model(self, session, batch_list):
        print "Now, testing on the training set, notice this is only for debugging..."
        total = 0
        accuCount = 0
        for batch in batch_list:
            batch_feat = np.array(batch[0], dtype = np.int32)
            batch_mask = np.array(batch[1], dtype = np.float32)
            pred = self.predict_on_batch(session, batch_feat, batch_mask)
            accuCount += np.sum(np.argmax(pred,1) == batch[2])
            total += len(batch[2])
        accu = accuCount * 1.0 / total
        logger.info( ("Test accuracy on training set is: %f" %(accu)) )
        return accu     
      
    def process_model_output(self):
        pkl_file = open('/content/auth_id/data_article_test.pkl', 'rb')
        batch_list = pickle.load(pkl_file)
        pkl_file.close()

        test_size = int(len(batch_list) /1)   
        training_batch = batch_list[0 : len(batch_list) - test_size]
        print test_size, len(batch_list)
        testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]

        init = tf.global_variables_initializer()   #added initialization
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            #load_path = "./model_weights_article"
            saver.restore(session, "./model_weights_article")

            print "Now, collecting the model outputs..."
            total = 0
            accuCount = 0

            for batch in testing_batch:
                batch_feat = np.array(batch[1], dtype = np.int32)
                batch_mask = np.array(batch[2], dtype = np.float32)

                preds = self.predict_on_batch(session, batch_feat, batch_mask)
                #print len(preds)
                author_find = data_util.find_author(preds)
                if author_find == batch[0]:
                    accuCount += 1
                total += 1
            accuracy = accuCount / total
            print( ("Test accuracy %f" %(accuracy)) )
            return accuracy

    def process_model_confusion_matrix(self):
        pkl_file = open('/content/auth_id/data_test_bundle_seq.pkl', 'rb')
        batch_list = pickle.load(pkl_file)
        pkl_file.close()

        test_size = int(len(batch_list) / 1)
        training_batch = batch_list[0 : len(batch_list) - test_size]
        print test_size, len(batch_list)
        testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]

        init = tf.global_variables_initializer()   #added initialization
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            #load_path = "./model_weights_article"
            saver.restore(session, "./model_weights_article")
            
            print "Now, collecting the model outputs..."
            pred_list = []
            real_label_list = []
            total = 0
            accuCount = 0
            for batch in testing_batch:
                batch_feat = np.array(batch[0], dtype = np.int32)
                batch_mask = np.array(batch[1], dtype = np.float32)

                pred = self.predict_on_batch(session, batch_feat, batch_mask)
                accuCount += np.sum(np.argmax(pred,1) == batch[2])
                pred_list.extend(np.argmax(pred,1).tolist())
                real_label_list.extend(batch[2])                  #changed from batch[0] to batch[2]
                total += len(batch[2])

            accu = accuCount * 1.0 / total
            print( ("Test accuracy %f" %(accu)) )
            t_cm = cm.generate_cm(real_label_list,pred_list, 50)
            x = t_cm.as_matrix().astype(np.uint8)
            du.visualize_cm(x, "C50_word")
            return accu

    def train_model(self):
        level='word' # 'word' or ''
        dataset='c50'
        parameter='hs'
        date='0319'
        training_history_txt_filename='/content/auth_id/results/training_history'+level+'_'+parameter+'_'+dataset+'_'+date  +'.txt'
        # write training_history
        training_history_file = open(training_history_txt_filename,'a+')
        self.record_history_init(training_history_file)

        if not os.path.exists(config.log_output):
            os.makedirs(os.path.dirname(config.log_output))
        handler = logging.FileHandler(config.log_output)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)
        '''
        pkl_file = open('../data/batch_data/C50/data_bundle_seq.pkl', 'rb')
        batch_list = pickle.load(pkl_file)
        pkl_file.close()
        test_size = int(len(batch_list) / 10)
        training_batch = batch_list[0 : len(batch_list) - test_size]
        print test_size
        testing_train_batch = batch_list[test_size : 2 * test_size]
        testing_batch = batch_list[len(batch_list) - test_size : len(batch_list)]
        '''

        train_file = open('/content/auth_id/data_train_bundle_seq.pkl', 'rb')
        training_batch = pickle.load(train_file)
        train_file.close()


        test_file = open('/content/auth_id/data_test_bundle_seq.pkl', 'rb')
        testing_batch = pickle.load(test_file)
        test_file.close()

        testing_train_batch = training_batch[0 : int(len(training_batch) / 2)]

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            #load_path = "results/RNN/20170310_1022/model.weights_20"
            #saver.restore(session, load_path)

            #the following is a test for what in tensor
            batch = training_batch[0]
            batch_label = rmb.convertOnehotLabel(batch[2],  Config.n_classes)
            batch_feat = np.array(batch[0], dtype = np.int32)
            batch_mask = np.array(batch[1], dtype = np.float32)
            feed = self.create_feed_dict(batch_feat, labels_batch=batch_label, mask_batch=batch_mask,
                                     dropout=Config.dropout)
            _, loss, raw_pred, pred = session.run([self.train_op, self.loss, self.raw_preds, self.pred], feed_dict=feed)
            ##############

            for iterTime in range(Config.n_epochs):
                loss_list = []
                smallIter = 0

                for batch in training_batch:
                    batch_label = rmb.convertOnehotLabel(batch[2],  Config.n_classes)
                    batch_feat = np.array(batch[0], dtype = np.int32)
                    batch_mask = np.array(batch[1], dtype = np.float32)
                    loss = self.train_on_batch(session, batch_feat, batch_label, batch_mask)
                    loss_list.append(loss)
                    smallIter += 1

                    if(smallIter % 40 == 0):
                        #self.test_trainset_model(session, testing_train_batch)
                        #self.test_model(session, testing_batch)
                        logger.info(("Intermediate epoch %d Total Iteration %d: loss : %f" %(iterTime, smallIter, np.mean(np.mean(np.array(loss)))) ))
                # record training history on this epoch
                train_accu=self.test_trainset_model(session, testing_train_batch)
                test_accu=self.test_model(session, testing_batch)
                average_train_loss=np.mean(np.array(loss_list))
                self.record_history_accu(training_history_file,iterTime,average_train_loss,train_accu,test_accu)

                if(iterTime % 20 == 0):
                    logger.info(("epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) ))
                    saver.save(session,"./model_weights_article")

                    #if(smallIter % 200 == 0):
                    print ("Intermediate epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) )

            print ("epoch %d : loss : %f" %(iterTime, np.mean(np.mean(np.array(loss)))) )
            self.record_history_finish(training_history_file)

           
    def __init__(self, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(config)
        self.pretrained_embeddings = pretrained_embeddings
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None
        self.raw_preds = None
        self.build()


if __name__ == "__main__":
    args = "gru"
    config = Config(args)
    glove_path = "/content/glove.6B.50d.txt"
    glove_vector = data_util.load_embeddings(glove_path, config.embed_size)
    model = RNNModel(config, glove_vector.astype(np.float32))
    model.train_model()
    model.process_model_output()
    #model.process_model_confusion_matrix()
