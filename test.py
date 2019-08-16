import tensorflow as tf

def demo1():
    a = tf.ones([10, 5, 5])
    a_expand = tf.expand_dims(a, -1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(a.shape)
        a_expand_out = sess.run(a_expand)
        print(a_expand_out.shape)

    print("demo1 is over")


def main():
    demo1()

if __name__ == '__main__':
    main()