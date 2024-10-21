import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import math
from sklearn.svm import SVC

def predict(model,db_test,batchz=None,verbose=True):
    '''
    自定义的model的预测方法，主要是实现配对返回预测值和真实值
    :param model:
    :param db_test:
    :param batchz:当db已经进行了batch，这里就不需要赋值
    :param verbose:
    :return:
    '''
    y_pre = np.array([])
    y_tru = np.array([])
    for elem in db_test.as_numpy_iterator():
    	# 注意，这里的model要非训练模式
        batch_y_pre=model(elem[0],training=False).numpy().flatten()
        print(batch_y_pre.shape)
        print(len(batch_y_pre))
        batch_y_tru=elem[1].flatten()
        y_pre = np.insert(y_pre, len(y_pre), batch_y_pre)
        y_tru = np.insert(y_tru, len(y_tru), batch_y_tru)
    return y_pre,y_tru

##数据预处理
#数据下载
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

##训练前九个图像和标签
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#划分数据集
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

#使用缓冲预提取从磁盘加载图像，以免造成 I/O 阻塞
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#使用数据扩充，将随机但现实的转换应用于训练图像（例如旋转或水平翻转）来人为引入样本多样性。这有助于使模型暴露于训练数据的不同方面并减少过拟合。
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')


#再缩放像素值
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

##从预训练卷积网络创建基础模型
#将 MobileNet V2 模型来创建基础模型
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
image_batch, label_batch = next(iter(train_dataset))
print(image_batch.shape)
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#冻结卷积层，冻结可避免在训练期间更新给定层中的权重。
base_model.trainable = False
base_model.summary()

#要从特征块生成预测，在 5x5 空间位置内取平均值，以将特征转换成每个图像一个向量（包含 1280 个元素）。
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#通过使用 Keras 函数式 API 将数据扩充、重新缩放、base_model 和特征提取程序层链接在一起来构建模型。
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

#编译模型
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)


##训练模型
initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

##画学习曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

##SVM
#冻结卷积基
base_model.trainable=False

#获取X,Y的训练集和测试集，注意这里获取的X以及testX因为被拉直成了一维，需要在后面进行reshape处理
X,Y=predict(base_model,train_dataset)
testX,testY=predict(base_model,test_dataset)

#对X进行数据处理，获取训练集Xave&&测试集testXave
#由于X被拉直成一维，需要reshape还原成高维
X1=X.reshape(2000,5,5,1280)
#池化
Xave=global_average_layer(X1)

#同上
testX1=testX.reshape(192,5,5,1280)
testXave=global_average_layer(testX1)

#定义SVM模型
svm_model=SVC(kernel='poly',C=1)
svm_model.fit(Xave,Y)

#评估模型性能
y_pred = svm_model.predict(testXave)

#混淆矩阵
from sklearn.metrics import classification_report, confusion_matrix
print('Classification Report:\n', classification_report(testY, y_pred))
print('Confusion Matrix:\n', confusion_matrix(testY, y_pred))

#绘制学习曲线
#代表使用SVM的分类模型，输入特征为X，输出label为y，进行10折交叉验证，通过均值平方差的方式计分，
#学习曲线分为5段。 其一共具有3个返回值，分别是train_sizes, train_loss, test_loss，
# 其中train_loss指的是训练集的loss，其shape为(5,10)，第n行对应学习曲线的第n段，第n行的内容代表着第n段的10折交叉验证的结果；test_loss的含义与train_loss类似，其对应的是测试集的loss。
from sklearn.model_selection import learning_curve
Xarray=np.array(Xave)
train_sizes, train_loss, test_loss = learning_curve(
    svm_model, Xarray, Y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(.1, 1.0, 5))

import matplotlib.pyplot as plt
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
# 显示图例
plt.legend(loc="best")
plt.show()

##微调
#解冻顶层
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

#编译模型
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)

#训练模型
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

##提取base_model

#coding=utf-8
import seaborn as sbn
import pylab as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.models import Model
#取某一层的输出为输出新建为model，采用函数模型
base_layer_model = Model(inputs=model.input,outputs=model.get_layer('global_average_pooling2d').output)

#以这个model的预测值作为输出
#base_model_output = predict(base_layer_model,train_dataset)

base_layer_model.summary()

#冻结卷积基
base_layer_model.trainable=False

#获取X,Y的训练集和测试集，注意这里获取的X以及testX因为被拉直成了一维，需要在后面进行reshape处理
X2,Y2=predict(base_layer_model,train_dataset)
testX2,testY2=predict(base_layer_model,test_dataset)

#由于X被拉直成一维，需要reshape还原成高维
Xave2=X2.reshape(2000,1280)
#同上
testXave2=testX2.reshape(192,1280)

#建立SVM模型
svm_model2=SVC(kernel='rbf',C=1.0)

#训练模型
svm_model2.fit(Xave2,Y2)

#模型评估
y_pred2 = svm_model2.predict(testXave2)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print("accuracy:")
print(accuracy_score(testY2, y_pred2))
print("recall:")
print(recall_score(testY2, y_pred2, average='macro'))
print("F1_score:")
print(f1_score(testY2, y_pred2, average='macro'))

#混淆矩阵
from sklearn.metrics import classification_report, confusion_matrix
print('Classification Report:\n', classification_report(testY2, y_pred2))
print('Confusion Matrix:\n', confusion_matrix(testY2, y_pred2))

##绘制学习曲线
from sklearn.model_selection import learning_curve
Xarray2=np.array(Xave2)
train_sizes, train_loss, test_loss = learning_curve(
    svm_model2, Xarray2, Y2, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(.1, 1.0, 5))

import matplotlib.pyplot as plt
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
# 显示图例
plt.legend(loc="best")
plt.show()

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

##评估与预测
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

#预测
# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")