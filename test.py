import tensorflow as tf

def demo1():
    a = tf.ones([10, 5, 5])
    a_expand = tf.expand_dims(a, -1)

    filter_size = [3,3,1,10]

    w1 = tf.Variable(tf.ones([10, 5, 5]))
    w1_expand = tf.expand_dims(w1, -1)
    cnv1 = tf.nn.conv2d(input=w1_expand, filter=filter_size, strides=[1,1,1,1], padding="VALID")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(a.shape)
        a_expand_out = sess.run(a_expand)
        cnv1_out = sess.run(cnv1)
        print("a_expand_out: ",a_expand_out.shape)
        print("cnv1_out: ", cnv1_out)

    print("demo1 is over")


def main():
    demo1()

if __name__ == '__main__':
    main()