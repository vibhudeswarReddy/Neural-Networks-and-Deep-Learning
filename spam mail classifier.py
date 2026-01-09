import math
free = 0
win = 1
often = 1
w_free_n1 = -0.4
w_win_n1 = 0.6
w_often_n1 = 0.2

w_free_n2 = 0.3
w_win_n2 = -0.5
w_often_n2 = 0.4

w_n1_out = 0.5
w_n2_out = 0.8

def relu(x):
  return max(0, x)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

n1 = relu(free*w_free_n1 + win*w_win_n1 + often*w_often_n1)
n2 = relu(free*w_free_n2 + win*w_win_n2 + often*w_often_n2)

output_raw = n1*w_n1_out + n2*w_n2_out

spam_probability = sigmoid(output_raw)

print("n1 =", n1)
print("n2 =", n2)
print("Raw output =", output_raw)
print("Spam probability =", spam_probability)
