from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow.compat.v1 as tf
import scipy.io as scio
from ops import *
import vgg

class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=230,
               image_width=310,
               label_height=230, 
               label_width=310,
               batch_size=2,
               c_dim=3, 
               checkpoint_dir=None, 
               sample_dir=None,
               #test_image_name = None,
               #id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    #self.test_image_name = test_image_name
    #self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.vgg_dir='vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'

    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')

    self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    self.labels_test = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test')
    self.pred_h = self.model()
    self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h * 255))
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.loss_texture1 =tf.reduce_mean(tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))
    
    self.loss_h1= tf.reduce_mean(tf.abs(self.labels_image-self.pred_h))
    self.loss = 0.05*self.loss_texture1+ self.loss_h1
    t_vars = tf.trainable_variables()


    self.saver = tf.train.Saver()
    
  def train(self, config):
    if config.is_train:
      data_train_list = prepare_data(self.sess, dataset="input_train")
      image_train_list = prepare_data(self.sess, dataset="gt_train")

      data_test_list = prepare_data(self.sess, dataset="input_test")
      image_test_list = prepare_data(self.sess, dataset="gt_test")

      seed = 568
      np.random.seed(seed)
      np.random.shuffle(data_train_list)
      np.random.seed(seed)
      np.random.shuffle(image_train_list)
    else:
      data_test_list = prepare_data(self.sess, dataset="input_test")
      #data_wb_test_list = prepare_data(self.sess, dataset="input_wb_test")
      #data_ce_test_list = prepare_data(self.sess, dataset="input_ce_test")
      #data_gc_test_list = prepare_data(self.sess, dataset="input_gc_test")
      image_test_list = prepare_data(self.sess, dataset="gt_test")

    sample_data_files = data_test_list[16:20]
    #sample_wb_data_files = data_wb_test_list[16:20]
    #sample_ce_data_files = data_ce_test_list[16:20]
    #sample_gc_data_files = data_gc_test_list[16:20]
    sample_image_files = image_test_list[16:20]

    sample_data = [
          get_image(sample_data_file,
                    is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
    sample_lable_image = [
          get_image(sample_image_file,
                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

    sample_inputs_data = np.array(sample_data).astype(np.float32)
    sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)

    self.train_op = tf.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)
    tf.global_variables_initializer().run()

    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      loss = np.ones(config.epoch)

      for ep in range(config.epoch):
        # Run by batch images
        
        batch_idxs = len(data_train_list) // config.batch_size
        for idx in range(0, batch_idxs):

          batch_files= data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]

          batch_ = [
          get_image(batch_file,
                    is_grayscale=self.is_grayscale) for batch_file in batch_files]
          batch_labels_image = [
          get_image(batch_image_file,
                    is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]

          batch_input = np.array(batch_).astype(np.float32)
          batch_image_input = np.array(batch_labels_image).astype(np.float32)

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss ], feed_dict={self.images: batch_input, self.labels_image:batch_image_input})
          # print(batch_light)

          if counter % 100 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err ))

          if idx  == batch_idxs-1: 
            batch_test_idxs = len(data_test_list) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):

              sample_data_files = data_train_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_image_files = image_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
             
              sample_data = [get_image(sample_data_file,
                            is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]

              sample_lable_image = [get_image(sample_image_file,
                                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

              sample_inputs_data = np.array(sample_data).astype(np.float32)
              sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)
              

              err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: sample_inputs_data, self.labels_image:sample_inputs_lable_image})    

            loss[ep]=np.mean(err_test)
            print(loss)
            self.save(config.checkpoint_dir, counter) 


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    #image_test =  get_image(self.test_image_name,is_grayscale=False)
    #shape = image_test.shape
    #expand_test = image_test[np.newaxis,:,:,:]
    #expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    #batch_test_image = np.append(expand_test,expand_zero,axis = 0)

    #tf.global_variables_initializer().run()
    
    
    #counter = 0
    #start_time = time.time()

    #if self.load(self.checkpoint_dir):
      #print(" [*] Load SUCCESS")
    #else:
      #print(" [!] Load failed...")
    #result_h = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image})

    #_,h ,w , c = result_h.shape
    #for id in range(0,1):

        #result_h0 = result_h[id,:,:,:].reshape(h , w , 3)
        #image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        #final = (result_h0+1.)/2 
        #image_path = os.path.join(image_path0, "%2dtest_dehaze.bmp"%(self.id))
        #image_path = os.path.join(image_path0, self.test_image_name+'_out.png')
        #imsave_lable(final, image_path)


  def model(self):

    with tf.variable_scope("model_h") as scope:
        #if self.id > 0: 
          #scope.reuse_variables()
        image_conv1 = tf.nn.relu(conv2d(self.images, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze1"))
        image_conv2 = tf.nn.relu(conv2d(image_conv1, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze2"))
        image_conv3 = tf.nn.relu(conv2d(image_conv2, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze3"))
        dehaze_concat1 = tf.concat(axis = 3, values = [image_conv1,image_conv2,image_conv3,self.images])
        image_conv4 = tf.nn.relu(conv2d(dehaze_concat1, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze4"))
        image_conv5 = tf.nn.relu(conv2d(image_conv4, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze5"))
        image_conv6 = tf.nn.relu(conv2d(image_conv5, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze6"))
        dehaze_concat2 = tf.concat(axis = 3, values = [dehaze_concat1,image_conv4,image_conv5,image_conv6])
        image_conv7 = tf.nn.relu(conv2d(dehaze_concat2, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze7"))
        image_conv8 = tf.nn.relu(conv2d(image_conv7, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze8"))
        image_conv9 = tf.nn.relu(conv2d(image_conv8, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze9"))
        dehaze_concat3 = tf.concat(axis = 3, values = [dehaze_concat2,image_conv7,image_conv8,image_conv9])
        image_conv10 = conv2d(dehaze_concat3, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2d_dehaze10")
        out = tf.add(self.images , image_conv10)
    return out

  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
