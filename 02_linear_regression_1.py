"""
 3 step 진행
 (1) Build graph 
 (2) feed data and run graph
 (3) update varaible in the graph (and return output values)
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# (1) Build graph

# X and Y data (training data set)
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Variable 은 tensor flow 가 사용하는 W 값 (= trainable varaible) 텐서플로우가 학습하는 과정에서 변경시킴
# [1] 1 rank array, shpae = 1 -> 스칼라
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b


# cost 값
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean : 괄호 안의 value set 을 평균내주는것


# cost minimize 하기 > Gradient Descent 라는 것 사용 (아직 몰라도됨)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)  # W, b 를 바꿔가면서 cost 를 min인 값을 찾음

# (2) Run graph
sess = tf.Session()
# Varaible 사용했다면, 무조건 initialize 해줘야함.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
