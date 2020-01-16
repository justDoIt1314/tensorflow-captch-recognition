import tensorflow as tf
import numpy as np 
from train import text2vec,vec2text,MAX_CAPTCHA,CHAR_SET_LEN,convert2gray,IMAGE_HEIGHT,IMAGE_WIDTH
from gen_captch import gen_captcha_text_and_image
import matplotlib.pyplot as plt
import os

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout
# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    #w_c2_alpha = np.sqrt(2.0/(3*3*32)) 
    #w_c3_alpha = np.sqrt(2.0/(3*3*64)) 
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))
    #out_alpha = np.sqrt(2.0/1024)

    with tf.name_scope('conv1') as scope:
        w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]),name='w_c1')
        b_c1 = tf.Variable(b_alpha*tf.random_normal([32]),name='b_c1')
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1),name=scope)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool_1')
        conv1 = tf.nn.dropout(conv1, keep_prob,name='drop_1')
    with tf.name_scope('conv2') as scope:
        w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]),name='w_c2')
        b_c2 = tf.Variable(b_alpha*tf.random_normal([64]),name='b_c2')
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2),name=scope)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool_2')
        conv2 = tf.nn.dropout(conv2, keep_prob,name='drop_2')
    with tf.name_scope('conv3') as scope:
        w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]),name='w_c3')
        b_c3 = tf.Variable(b_alpha*tf.random_normal([64]),name='b_c3')
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3),name=scope)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool_3')
        conv3 = tf.nn.dropout(conv3, keep_prob,name='drop_3')

    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]),name='w_fc1')
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]),name='b_fc1')
    fltten_0 = w_d.get_shape().as_list()[0]
    fltten_1 = conv3.get_shape()[-1].value * conv3.get_shape()[-2].value * conv3.get_shape()[-3].value
    dense = tf.reshape(conv3, [-1, fltten_1],name='fltten')
    
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob,name='fc1_drop')

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]),name='w_fc2')
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]),name='b_fc2')
    out = tf.add(tf.matmul(dense, w_out), b_out,name='fc2_out')
    #out = tf.nn.softmax(out)
    return out
# 向量转回文本
def vec2text_idx(vec):
    
    text=[]
    for char_idx in vec:
        #char_at_pos = i #c/63
        #char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
 
def test_crack_captcha_cnn():
    output = crack_captcha_cnn()   
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    saver = tf.train.Saver()
    sess = tf.Session()
    # writer = tf.summary.FileWriter("logs/", sess.graph) # tensorboard 第一个参数指定生成文件的目录。
    if os.path.exists("model"):
        saver.restore(sess,"model/crack_capcha.model-29700")
    while 1:
        text, image = gen_captcha_text_and_image()
        grayImage = convert2gray(image)
        grayImage = grayImage.flatten() / 255
        res_vec = sess.run(max_idx_p,feed_dict={X:[grayImage],keep_prob:1.0})
        result = vec2text_idx(res_vec[0])
        print(result)
        f = plt.figure()
        f.add_subplot(111)
        plt.imshow(image)
        plt.show()
        
        
if __name__ == "__main__":
    test_crack_captcha_cnn()
    
    #print(ord('Z'))