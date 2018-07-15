import tensorflow as tf
import cv2
### GRAPH ###
tf.reset_default_graph()
saver = tf.train.import_meta_graph('/home/yitao/Documents/fun-project/npy2ckpt/output/openpose_model.ckpt.meta')
inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
body = tf.get_default_graph().get_tensor_by_name('concat_stage7:0')
print('The input placeholder is expecting an array of shape {} and type {}'.format(inputs.shape, inputs.dtype))
### IMAGE ###
img = cv2.imread('/home/yitao/Documents/fun-project/npy2ckpt/tests/dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
res_img = cv2.resize(img, (656, 368))
prep_img = res_img.reshape([1, 368, 656, 3])
### SESSION ###
with tf.Session() as sess:
    saver.restore(sess, '/home/yitao/Documents/fun-project/npy2ckpt/output/openpose_model.ckpt')
    
    output_img = sess.run(body, feed_dict={
            inputs: prep_img
        })
    
    print(output_img.shape)
    # print(output_img)