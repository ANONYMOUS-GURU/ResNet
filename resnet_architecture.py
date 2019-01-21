import tensorflow as tf 
from ops import conv,fc,batch_norm,maxpool


def placeholders(img_size,img_channel,label_cnt):
	with tf.name_scope('input'):
		X=tf.placeholder(shape=[None,img_size,img_size,img_channel],dtype=tf.float32,name='image')
		y=tf.placeholder(shape=[None,label_cnt],dtype=tf.float32,name='target')

	with tf.name_scope('hparams'):
		learning_rate=tf.placeholder(shape=None,dtype=tf.float32,name='learning_rate')
		dropout_keep_prob=tf.placeholder(shape=None,dtype=tf.float32,name='keep_prob')

	training=tf.placeholder(shape=None,dtype=tf.bool,name='is_train')
	return X,y,learning_rate,dropout_keep_prob,training

def projection_(shortcut,out_channels,training,name):
	shortcut=conv(shortcut,stride_size=2,filter_size=1,padding='VALID',out_channels=out_channels)
	shortcut=batch_norm(shortcut,training,name=name)
	return shortcut

def res_build_block_v1(X,increase=False,init=1.0,stddev=1.0,training=False,projection=None,name=None):
	print('input shape for {} :: {}'.format(name,X.get_shape().as_list()))
	shortcut_projection=X
	stride=1
	in_channels=X.get_shape().as_list()[-1]		
	out_channels=in_channels
	if increase:
		out_channels=2*in_channels
		stride=2
		if projection:
			name_=name+'projection_batch_norm'
			shortcut_projection=projection(shortcut_projection,out_channels=out_channels,training=training,name=name_)
		else:
			shortcut_projection=maxpool(shortcut_projection,filter_size=[1,2,2,1],stride_size=[1,2,2,1],padding='VALID')
			pad=(out_channels-in_channels)
			shortcut_projection=tf.pad(shortcut_projection,[[0,0],[0,0],[0,0],[0,pad]])
	with tf.name_scope('conv1layer'):
		name_=name+'conv1layer_batch_norm'
		X=conv(X,filter_size=3,out_channels=out_channels,stride_size=stride,padding='SAME',init_bias=init,stddev=stddev)
		X=batch_norm(X,training,name=name_)
		X=tf.nn.relu(X)
	with tf.name_scope('conv2layer'):
		name_=name+'conv2layer_batch_norm'
		X=conv(X,filter_size=3,out_channels=out_channels,stride_size=1,padding='SAME',init_bias=init,stddev=stddev)
		X=batch_norm(X,training,name=name_)
	with tf.name_scope('residual'):
		X+=shortcut_projection
		X=tf.nn.relu(X)
	print('output shape for {} :: {}'.format(name,X.get_shape().as_list()))
	return X 

	
def bottleneck_block_v1(X,increase=False,init=1.0,stddev=1.0,training=False,projection=None,name=None):
	shortcut_projection=X
	in_channels=X.get_shape().as_list()[-1]
	out_channels_=[64,128,256,512]
	out_channels=out_channels_[int(name[4])-1]
	bottle_out=4*out_channels
	stride=1
	print(X.get_shape().as_list())
	if increase:
		out_channels=2*out_channels
		stride=2
		if projection:
			name_=name+'projection_batch_norm'
			shortcut_projection=projection(shortcut_projection,out_channels=out_channels,training=training,name=name_)
		else:
			shortcut_projection=tf.nn.maxpool(shortcut_projection,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
			pad=(out_channels-in_channels)
			shortcut_projection=tf.pad(shortcut_projection,[[0,0],[0,0],[0,0],[0,pad]])

	with tf.name_scope('conv1layer'):
		name_=name+'conv1layer_batch_norm'
		X=conv(X,filter_size=1,out_channels=out_channels,stride_size=stride,padding='SAME',init_bias=init,stddev=stddev)
		X=batch_norm(X,training=training,name=name_)
		X=tf.nn.relu(X)

	with tf.name_scope('conv2layer'):
		name_=name+'conv2layer_batch_norm'
		X=conv(X,filter_size=3,out_channels=out_channels,stride_size=1,padding='SAME',init_bias=init,stddev=stddev)
		X=batch_norm(X,training=training,name=name_)
		X=tf.nn.relu(X)

	with tf.name_scope('conv3layer'):
		name_=name+'conv3layer_batch_norm'
		X=conv(X,filter_size=1,out_channels=bottle_out,stride_size=1,padding='SAME',init_bias=init,stddev=stddev)
		X=batch_norm(X,training=training,name=name_)

	with tf.name_scope('residual'):
		pad=(bottle_out-in_channels)
		shortcut_projection=tf.pad(shortcut_projection,[[0,0],[0,0],[0,0],[0,pad]])
		X+=shortcut_projection
		X=tf.nn.relu(X)
	print(X.get_shape().as_list())
	return X

# have a placeholder at for is_train parameter
def network(X,training=False,dropout_keep_prob=1.0,type_=18,label_cnt=1000):
	# need to change pre_res conv for matching output shape of 112,112 i.e. add padding of some amount.
	with tf.name_scope('pre_res'):
		X=conv(X,filter_size=7,out_channels=64,stride_size=2,padding='SAME')
		X=batch_norm(X,training=training,name='pre_res_batch_norm')
		X=tf.nn.relu(X)
		X=tf.pad(X,[[0,0],[1,1],[1,1],[0,0]])
		X=maxpool(X,stride_size=2,filter_size=3,padding='VALID')
	num_units34=[3,4,6,3]
	num_units18=[2,2,2,2]
	num_units50=[3,4,6,3]
	num_units101=[3,4,23,3]
	num_units152=[3,8,36,3]
	if type_==18:
		num_units=num_units18
		res_block=res_build_block_v1
	elif type==34:
		num_units=num_units34
		res_block=res_build_block_v1
	elif type==50:
		num_units=num_units50
		res_block=bottleneck_block_v1
	elif type==101:
		num_units=num_units101
		res_block=bottleneck_block_v1
	else:
		num_units=num_units152
		res_block=bottleneck_block_v1

	for x in range(len(num_units)):								
		name_res_block='res_{}_'.format(x+1)
		with tf.name_scope(name_res_block):
			increase=True
			if x==0:
				increase=False
			inner_block='block_{}'.format(1)
			with tf.name_scope(inner_block):
				name=name_res_block+inner_block
				X=res_block(X,training=training,increase=increase,projection=projection_,name=name)
			for y in range(1,num_units[x]):
				inner_block='block_{}'.format(y+1)
				with tf.name_scope(inner_block):
					name=name_res_block+inner_block
					X=res_block(X,training=training,increase=False,projection=projection_,name=name)
	with tf.name_scope('final_layer'):
		X=fc(X,label_cnt,a=None)

	with tf.name_scope('softmaxlayer'):
		out_probs=tf.nn.softmax(logits=X,axis=-1,name='softmax_op')

	return X,out_probs


def loss(logits,labels):
	with tf.name_scope('loss'):
		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
	tf.summary.scalar('loss',loss)
	return loss

def accuracy(logits,labels):
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
	tf.summary.scalar('accuracy', accuracy)
	return accuracy

def optimizer(loss,learning_rate):
	with tf.name_scope('AdamOptimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)
	return train_op

