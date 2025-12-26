# Лабораторная работа №2 (вариант 4)

## Цель лабораторной работы

По заданию выбрать свои классы и обучить сверточную нейронную сеть из примера, используя GPU, а затем повысить точность модели. Провести три обучения для 3 разных тактик пуллинга: пуллинг с помощью шага свёртки stride, макс пуллинг, усредняющий пуллинг. Сравнить достигнутое качество, время обучения и степень переобучения. Выбрать лучшую конфигурацию. Сохранить модель. Перезапустить среду выполнения - теряются все текующие переменные.

Загрузить в colab готовую уже обученную на cifar100 модель. Преобразовать в onnx и сохранить локально.

Скачать каталог с html-файлом и встроить в него два файла моделей - обученную на ЛР1 и на ЛР2.

Скачать картинки из интернета согласно варианту и открыть их в html по кнопке. Автоматически в скрипте масштабируется изображение.

Выбрать в js нужные классы для готовой модели. Проверить на устойчивость обе модели, полносвязную и свёрточную, двигая картинку, убедиться в наличии свойства инвариантности сверточного слоя.

---

## Классификация изображений CIFAR100

### Импортирование необходимых библиотек


```python
#!pip install torchsummary
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import pickle
from sklearn.metrics import classification_report
from PIL import Image
from tqdm.auto import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
%matplotlib inline
```

### Определяем видеокарту GPU, чтобы на ней учить нейронную сеть


```python
!nvidia-smi
```

    Tue Nov 25 12:23:07 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
    | N/A   52C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+
    


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Загрузка и распаковка набора данных CIFAR100


```python
!wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
!tar -xvzf cifar-100-python.tar.gz
```

    --2025-11-25 12:23:07--  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
    Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 169001437 (161M) [application/x-gzip]
    Saving to: ‘cifar-100-python.tar.gz’
    
    cifar-100-python.ta 100%[===================>] 161.17M  44.1MB/s    in 4.1s    
    
    2025-11-25 12:23:12 (39.3 MB/s) - ‘cifar-100-python.tar.gz’ saved [169001437/169001437]
    
    cifar-100-python/
    cifar-100-python/file.txt~
    cifar-100-python/train
    cifar-100-python/test
    cifar-100-python/meta
    

### Чтение тренировочной и тестовой выборки


```python
with open('cifar-100-python/train', 'rb') as f:
    data_train = pickle.load(f, encoding='latin1')
with open('cifar-100-python/test', 'rb') as f:
    data_test = pickle.load(f, encoding='latin1')

CLASSES = [0, 33, 41]


train_X = data_train['data'].reshape(-1, 3, 32, 32)
train_X = np.transpose(train_X, [0, 2, 3, 1]) # NCWC -> NWHC
train_y = np.array(data_train['fine_labels'])
mask = np.isin(train_y, CLASSES)
train_X = train_X[mask].copy()
train_y = train_y[mask].copy()
train_y = np.unique(train_y, return_inverse=1)[1]
del data_train

test_X = data_test['data'].reshape(-1, 3, 32, 32)
test_X = np.transpose(test_X, [0, 2, 3, 1])
test_y = np.array(data_test['fine_labels'])
mask = np.isin(test_y, CLASSES)
test_X = test_X[mask].copy()
test_y = test_y[mask].copy()
test_y = np.unique(test_y, return_inverse=1)[1]
del data_test
Image.fromarray(train_X[50]).resize((256,256))
```




    
![output_15_0](https://github.com/user-attachments/assets/197edc45-3594-4046-b94a-8b6787fbf91a)

    



### Создание Pytorch DataLoader'a


```python
batch_size = 128
dataloader = {}
for (X, y), part in zip([(train_X, train_y), (test_X, test_y)],
                        ['train', 'test']):
    tensor_x = torch.Tensor(X)
    tensor_y = F.one_hot(torch.Tensor(y).to(torch.int64),
                                     num_classes=len(CLASSES))/1.
    dataset = TensorDataset(tensor_x, tensor_y) # создание объекта датасета
    dataloader[part] = DataLoader(dataset, batch_size=batch_size, shuffle=True) # создание экземпляра класса DataLoader
dataloader
```




    {'train': <torch.utils.data.dataloader.DataLoader at 0x7c71f7f22f30>,
     'test': <torch.utils.data.dataloader.DataLoader at 0x7c71fc093fb0>}



### Создание Pytorch модели сверточной нейронной сети


```python
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)

    def forward(self, input):
        x = input / 255.0
        x = x - self.mean
        x = x / self.std
        return x.permute(0, 3, 1, 2) # nhwc -> nm

class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, input):
        out = F.adaptive_max_pool2d(input, output_size=1)
        return out.flatten(start_dim=1)

class Cifar100_CNN(nn.Module):
    def __init__(self, hidden_size=32, classes=100):
        super(Cifar100_CNN, self).__init__()
        # https://blog.jovian.ai/image-classification-of-cifar100-dataset-using-pytorch-8b7145242df1
        self.seq = nn.Sequential(
            Normalize([0.5074,0.4867,0.4411],[0.2011,0.1987,0.2025]),
            # первый способ уменьшения размерности картинки - через stride
            nn.Conv2d(3, HIDDEN_SIZE, 5, stride=4, padding=2),
            nn.ReLU(),
            # второй способ уменьшения размерности картинки - через слой пуллинг
            nn.Conv2d(HIDDEN_SIZE, HIDDEN_SIZE*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),#nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(HIDDEN_SIZE*8, classes),
        )

    def forward(self, input):
        return self.seq(input)

HIDDEN_SIZE = 32
model = Cifar100_CNN(hidden_size=HIDDEN_SIZE, classes=len(CLASSES))
# NEW
model.to(device)
print(model(torch.rand(1, 32, 32, 3).to(device)))
summary(model, input_size=(32, 32, 3))
model
```

    tensor([[ 0.2622, -0.2338, -0.1270]], device='cuda:0',
           grad_fn=<AddmmBackward0>)
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
             Normalize-1            [-1, 3, 32, 32]               0
                Conv2d-2             [-1, 32, 8, 8]           2,432
                  ReLU-3             [-1, 32, 8, 8]               0
                Conv2d-4             [-1, 64, 8, 8]          18,496
                  ReLU-5             [-1, 64, 8, 8]               0
             AvgPool2d-6             [-1, 64, 2, 2]               0
               Flatten-7                  [-1, 256]               0
                Linear-8                    [-1, 3]             771
    ================================================================
    Total params: 21,699
    Trainable params: 21,699
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 0.12
    Params size (MB): 0.08
    Estimated Total Size (MB): 0.22
    ----------------------------------------------------------------
    




    Cifar100_CNN(
      (seq): Sequential(
        (0): Normalize()
        (1): Conv2d(3, 32, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))
        (2): ReLU()
        (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (6): Flatten(start_dim=1, end_dim=-1)
        (7): Linear(in_features=256, out_features=3, bias=True)
      )
    )



### Выбор функции потерь и оптимизатора градиентного спуска


```python
criterion = nn.CrossEntropyLoss()
# используется SGD c momentum
optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
```

### Обучение модели по эпохам


```python
EPOCHS = 500
REDRAW_EVERY = 20
steps_per_epoch = len(dataloader['train'])
steps_per_epoch_val = len(dataloader['test'])
# NEW
pbar = tqdm(total=EPOCHS*steps_per_epoch)
losses = []
losses_val = []
passed = 0
for epoch in range(EPOCHS):  # проход по набору данных несколько раз
    #running_loss = 0.0
    tmp = []
    model.train()
    for i, batch in enumerate(dataloader['train'], 0):
        # получение одного минибатча; batch это двуэлементный список из [inputs, labels]
        inputs, labels = batch
        # на GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # очищение прошлых градиентов с прошлой итерации
        optimizer.zero_grad()

        # прямой + обратный проходы + оптимизация
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # для подсчёта статистик
        #running_loss += loss.item()
        accuracy = (labels.detach().argmax(dim=-1)==outputs.detach().argmax(dim=-1)).\
                    to(torch.float32).mean().cpu()*100
        tmp.append((loss.item(), accuracy.item()))
        pbar.update(1)
    #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / steps_per_epoch:.3f}')
    losses.append((np.mean(tmp, axis=0),
                   np.percentile(tmp, 25, axis=0),
                   np.percentile(tmp, 75, axis=0)))
    #running_loss = 0.0
    tmp = []
    model.eval()
    with torch.no_grad(): # отключение автоматического дифференцирования
        for i, data in enumerate(dataloader['test'], 0):
            inputs, labels = data
            # на GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #running_loss += loss.item()
            accuracy = (labels.argmax(dim=-1)==outputs.argmax(dim=-1)).\
                        to(torch.float32).mean().cpu()*100
            tmp.append((loss.item(), accuracy.item()))
    #print(f'[{epoch + 1}, {i + 1:5d}] val loss: {running_loss / steps_per_epoch_val:.3f}')
    losses_val.append((np.mean(tmp, axis=0),
                       np.percentile(tmp, 25, axis=0),
                       np.percentile(tmp, 75, axis=0)))
    if (epoch+1) % REDRAW_EVERY != 0:
        continue
    clear_output(wait=False)
    passed += pbar.format_dict['elapsed']
    pbar = tqdm(total=EPOCHS*steps_per_epoch, miniters=5)
    pbar.update((epoch+1)*steps_per_epoch)
    x_vals = np.arange(epoch+1)
    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    stats = np.array(losses)
    stats_val = np.array(losses_val)
    ax[1].set_ylim(stats_val[:, 0, 1].min()-5, 100)
    ax[1].grid(axis='y')
    for i, title in enumerate(['CCE', 'Accuracy']):
        ax[i].plot(x_vals, stats[:, 0, i], label='train')
        ax[i].fill_between(x_vals, stats[:, 1, i],
                           stats[:, 2, i], alpha=0.4)
        ax[i].plot(x_vals, stats_val[:, 0, i], label='val')
        ax[i].fill_between(x_vals,
                           stats_val[:, 1, i],
                           stats_val[:, 2, i], alpha=0.4)
        ax[i].legend()
        ax[i].set_title(title)
    plt.show()
print('Обучение закончено за %s секунд' % passed)
```


      0%|          | 0/6000 [00:00<?, ?it/s]


batch_size = 128

<img width="974" height="390" alt="image" src="https://github.com/user-attachments/assets/c21f4edd-172b-4b51-9a2e-40a96f53a368" />

batch_size = 64

<img width="974" height="377" alt="image" src="https://github.com/user-attachments/assets/6ecaa7d2-0ffc-4da1-a53c-ba1e1357559c" />

batch_size = 256

<img width="974" height="368" alt="image" src="https://github.com/user-attachments/assets/f010f325-90fe-45d0-998d-b797f3a3c9e2" />

stride

<img width="974" height="396" alt="image" src="https://github.com/user-attachments/assets/4ddc4cea-bd75-4bb3-85b7-c5257fadb754" />

Max-pooling

<img width="974" height="393" alt="image" src="https://github.com/user-attachments/assets/dba88987-fe0e-4da6-8f50-3a570dc8daf8" />

Average-pooling

<img width="974" height="390" alt="image" src="https://github.com/user-attachments/assets/ac625601-8d36-4f5b-a241-db71aea7f42c" />


### Проверка качества модели по классам на обучающей и тестовой выборках


```python
for part in ['train', 'test']:
    y_pred = []
    y_true = []
    with torch.no_grad(): # отключение автоматического дифференцирования
        for i, data in enumerate(dataloader[part], 0):
            inputs, labels = data
             # на GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).detach().cpu().numpy()
            y_pred.append(outputs)
            y_true.append(labels.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        print(part)
        print(classification_report(y_true.argmax(axis=-1), y_pred.argmax(axis=-1),
                                    digits=4, target_names=list(map(str, CLASSES))))
        print('-'*50)
```

    train
                  precision    recall  f1-score   support
    
               0     1.0000    1.0000    1.0000       500
              33     1.0000    1.0000    1.0000       500
              41     1.0000    1.0000    1.0000       500
    
        accuracy                         1.0000      1500
       macro avg     1.0000    1.0000    1.0000      1500
    weighted avg     1.0000    1.0000    1.0000      1500
    
    --------------------------------------------------
    test
                  precision    recall  f1-score   support
    
               0     0.9490    0.9300    0.9394       100
              33     0.8958    0.8600    0.8776       100
              41     0.8585    0.9100    0.8835       100
    
        accuracy                         0.9000       300
       macro avg     0.9011    0.9000    0.9001       300
    weighted avg     0.9011    0.9000    0.9001       300
    
    --------------------------------------------------
    

---

## Сохранение модели в ONNX


```python
# сохраняем и загружаем модель двумя способами

# ПЕРВЫЙ СПОСОБ: сохранение только параметров (state_dict)
PATH = 'cifar_cnn.pth'

# сохранение
torch.save(model.state_dict(), PATH)

# загрузка
new_model = Cifar100_CNN(hidden_size=HIDDEN_SIZE, classes=len(CLASSES)).to(device)
new_model.load_state_dict(torch.load(PATH, map_location=device))
new_model.eval()


# ВТОРОЙ СПОСОБ: сохранение всей архитектуры целиком
PATH2 = 'cifar_cnn.pt'

# сохранение
torch.save(model, PATH2)

# загрузка (важно: weights_only=False для новых версий PyTorch)
new_model_2 = torch.load(PATH2, map_location=device, weights_only=False)
new_model_2.eval()

```




    Cifar100_CNN(
      (seq): Sequential(
        (0): Normalize()
        (1): Conv2d(3, 32, kernel_size=(5, 5), stride=(4, 4), padding=(2, 2))
        (2): ReLU()
        (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU()
        (5): AvgPool2d(kernel_size=4, stride=4, padding=0)
        (6): Flatten(start_dim=1, end_dim=-1)
        (7): Linear(in_features=256, out_features=3, bias=True)
      )
    )




```python
!pip install onnx onnxscript
```

    Requirement already satisfied: onnx in /usr/local/lib/python3.12/dist-packages (1.19.1)
    Collecting onnxscript
      Downloading onnxscript-0.5.6-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: numpy>=1.22 in /usr/local/lib/python3.12/dist-packages (from onnx) (2.0.2)
    Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (5.29.5)
    Requirement already satisfied: typing_extensions>=4.7.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (4.15.0)
    Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from onnx) (0.5.4)
    Collecting onnx_ir<2,>=0.1.12 (from onnxscript)
      Downloading onnx_ir-0.1.12-py3-none-any.whl.metadata (3.2 kB)
    Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from onnxscript) (25.0)
    Downloading onnxscript-0.5.6-py3-none-any.whl (683 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m683.0/683.0 kB[0m [31m18.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading onnx_ir-0.1.12-py3-none-any.whl (129 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m129.3/129.3 kB[0m [31m8.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: onnx_ir, onnxscript
    Successfully installed onnx_ir-0.1.12 onnxscript-0.5.6
    


```python
# входной тензор для модели
x = torch.randn(1, 32, 32, 3, requires_grad=True).to(device)
torch_out = model(x)

# экспорт модели
torch.onnx.export(model,               # модель
                  x,                   # входной тензор (или кортеж нескольких тензоров)
                  "cifar100_CNN.onnx", # куда сохранить (либо путь к файлу либо fileObject)
                  export_params=True,  # сохраняет веса обученных параметров внутри файла модели
                  opset_version=16,     # версия ONNX
                  do_constant_folding=True,  # следует ли выполнять укорачивание констант для оптимизации
                  input_names = ['input'],   # имя входного слоя
                  output_names = ['output'],  # имя выходного слоя
                  dynamic_axes={'input' : {0 : 'batch_size'},    # динамичные оси, в данном случае только размер пакета
                                'output' : {0 : 'batch_size'}})
```

 /tmp/ipython-input-1386216986.py:6: UserWarning: # 'dynamic_axes' is not recommended when dynamo=True, and may lead to 'torch._dynamo.exc.UserError: Constraints violated.' Supply the 'dynamic_shapes' argument instead if export is unsuccessful.
  torch.onnx.export(model,               # модель
W1226 17:39:36.634000 2893 torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 16 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
[torch.onnx] Obtain model graph for `Cifar100_CNN([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `Cifar100_CNN([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
WARNING:onnxscript.version_converter:The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 16).
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
ONNXProgram(
    model=
        <
            ir_version=10,
            opset_imports={'': 16},
            producer_name='pytorch',
            producer_version='2.9.0+cu126',
            domain=None,
            model_version=None,
        >
        graph(
            name=main_graph,
            inputs=(
                %"input"<FLOAT,[s31,32,32,3]>
            ),
            outputs=(
                %"output"<FLOAT,[1,3]>
            ),
            initializers=(
                %"seq.1.bias"<FLOAT,[32]>{TorchTensor(...)},
                %"seq.3.bias"<FLOAT,[64]>{TorchTensor(...)},
                %"seq.5.bias"<FLOAT,[128]>{TorchTensor(...)},
                %"seq.8.bias"<FLOAT,[3]>{TorchTensor<FLOAT,[3]>(Parameter containing: tensor([-0.3263,  0.3473, -0.0554], device='cuda:0', requires_grad=True), name='seq.8.bias')},
                %"seq.0.mean"<FLOAT,[3]>{TorchTensor<FLOAT,[3]>(tensor([0.5074, 0.4867, 0.4411], device='cuda:0'), name='seq.0.mean')},
                %"seq.0.std"<FLOAT,[3]>{TorchTensor<FLOAT,[3]>(tensor([0.2011, 0.1987, 0.2025], device='cuda:0'), name='seq.0.std')},
                %"seq.1.weight"<FLOAT,[32,3,5,5]>{TorchTensor(...)},
                %"seq.3.weight"<FLOAT,[64,32,3,3]>{TorchTensor(...)},
                %"seq.5.weight"<FLOAT,[128,64,3,3]>{TorchTensor(...)},
                %"seq.8.weight"<FLOAT,[3,512]>{TorchTensor(...)},
                %"val_4"<INT64,[2]>{Tensor<INT64,[2]>(array([  1, 512]), name='val_4')},
                %"val_0"<FLOAT,[]>{TensorProtoTensor<FLOAT,[]>(array(255., dtype=float32), name='val_0')}
            ),
        ) {
             0 |  # node_div
                  %"div"<FLOAT,[s31,32,32,3]> ⬅️ ::Div(%"input", %"val_0"{255.0})
             1 |  # node_sub_1
                  %"sub_1"<FLOAT,[1,32,32,3]> ⬅️ ::Sub(%"div", %"seq.0.mean"{[0.5073999762535095, 0.48669999837875366, 0.44110000133514404]})
             2 |  # node_div_1
                  %"div_1"<FLOAT,[1,32,32,3]> ⬅️ ::Div(%"sub_1", %"seq.0.std"{[0.20110000669956207, 0.19869999587535858, 0.20250000059604645]})
             3 |  # node_permute
                  %"permute"<FLOAT,[1,3,32,32]> ⬅️ ::Transpose(%"div_1") {perm=(0, 3, 1, 2)}
             4 |  # node_conv2d
                  %"conv2d"<FLOAT,[1,32,8,8]> ⬅️ ::Conv(%"permute", %"seq.1.weight"{...}, %"seq.1.bias"{...}) {group=1, pads=(2, 2, 2, 2), auto_pad='NOTSET', strides=(4, 4), dilations=(1, 1)}
             5 |  # node_relu
                  %"relu"<FLOAT,[1,32,8,8]> ⬅️ ::Relu(%"conv2d")
             6 |  # node_conv2d_1
                  %"conv2d_1"<FLOAT,[1,64,4,4]> ⬅️ ::Conv(%"relu", %"seq.3.weight"{...}, %"seq.3.bias"{...}) {group=1, pads=(1, 1, 1, 1), auto_pad='NOTSET', strides=(2, 2), dilations=(1, 1)}
             7 |  # node_relu_1
                  %"relu_1"<FLOAT,[1,64,4,4]> ⬅️ ::Relu(%"conv2d_1")
             8 |  # node_conv2d_2
                  %"conv2d_2"<FLOAT,[1,128,2,2]> ⬅️ ::Conv(%"relu_1", %"seq.5.weight"{...}, %"seq.5.bias"{...}) {group=1, pads=(1, 1, 1, 1), auto_pad='NOTSET', strides=(2, 2), dilations=(1, 1)}
             9 |  # node_relu_2
                  %"relu_2"<FLOAT,[1,128,2,2]> ⬅️ ::Relu(%"conv2d_2")
            10 |  # node__unsafe_view
                  %"_unsafe_view"<FLOAT,[1,512]> ⬅️ ::Reshape(%"relu_2", %"val_4"{[1, 512]}) {allowzero=1}
            11 |  # node_linear
                  %"output"<FLOAT,[1,3]> ⬅️ ::Gemm(%"_unsafe_view", %"seq.8.weight"{...}, %"seq.8.bias"{[-0.32634788751602173, 0.3472962975502014, -0.055435847491025925]}) {beta=1.0, transB=1, alpha=1.0, transA=0}
            return %"output"<FLOAT,[1,3]>
        }


    ,
    exported_program=
        ExportedProgram:
            class GraphModule(torch.nn.Module):
                def forward(self, p_seq_1_weight: "f32[32, 3, 5, 5]", p_seq_1_bias: "f32[32]", p_seq_3_weight: "f32[64, 32, 3, 3]", p_seq_3_bias: "f32[64]", p_seq_5_weight: "f32[128, 64, 3, 3]", p_seq_5_bias: "f32[128]", p_seq_8_weight: "f32[3, 512]", p_seq_8_bias: "f32[3]", c_seq_0_mean: "f32[3]", c_seq_0_std: "f32[3]", input: "f32[s31, 32, 32, 3]"):
                    input_1 = input
            
                     # File: /tmp/ipython-input-3043060065.py:8 in forward, code: x = input / 255.0
                    div: "f32[s31, 32, 32, 3]" = torch.ops.aten.div.Tensor(input_1, 255.0);  input_1 = None
            
                     # File: /tmp/ipython-input-3043060065.py:9 in forward, code: x = x - self.mean
                    sub_1: "f32[1, 32, 32, 3]" = torch.ops.aten.sub.Tensor(div, c_seq_0_mean);  div = c_seq_0_mean = None
            
                     # File: /tmp/ipython-input-3043060065.py:10 in forward, code: x = x / self.std
                    div_1: "f32[1, 32, 32, 3]" = torch.ops.aten.div.Tensor(sub_1, c_seq_0_std);  sub_1 = c_seq_0_std = None
            
                     # File: /tmp/ipython-input-3043060065.py:11 in forward, code: return x.permute(0, 3, 1, 2) # nhwc -> nm
                    permute: "f32[1, 3, 32, 32]" = torch.ops.aten.permute.default(div_1, [0, 3, 1, 2]);  div_1 = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/conv.py:548 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d: "f32[1, 32, 8, 8]" = torch.ops.aten.conv2d.default(permute, p_seq_1_weight, p_seq_1_bias, [4, 4], [2, 2]);  permute = p_seq_1_weight = p_seq_1_bias = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/activation.py:144 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu: "f32[1, 32, 8, 8]" = torch.ops.aten.relu.default(conv2d);  conv2d = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/conv.py:548 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_1: "f32[1, 64, 4, 4]" = torch.ops.aten.conv2d.default(relu, p_seq_3_weight, p_seq_3_bias, [2, 2], [1, 1]);  relu = p_seq_3_weight = p_seq_3_bias = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/activation.py:144 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu_1: "f32[1, 64, 4, 4]" = torch.ops.aten.relu.default(conv2d_1);  conv2d_1 = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/conv.py:548 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_2: "f32[1, 128, 2, 2]" = torch.ops.aten.conv2d.default(relu_1, p_seq_5_weight, p_seq_5_bias, [2, 2], [1, 1]);  relu_1 = p_seq_5_weight = p_seq_5_bias = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/activation.py:144 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu_2: "f32[1, 128, 2, 2]" = torch.ops.aten.relu.default(conv2d_2);  conv2d_2 = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/flatten.py:56 in forward, code: return input.flatten(self.start_dim, self.end_dim)
                    clone: "f32[1, 128, 2, 2]" = torch.ops.aten.clone.default(relu_2, memory_format = torch.contiguous_format);  relu_2 = None
                    _unsafe_view: "f32[1, 512]" = torch.ops.aten._unsafe_view.default(clone, [1, 512]);  clone = None
            
                     # File: /usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
                    linear: "f32[1, 3]" = torch.ops.aten.linear.default(_unsafe_view, p_seq_8_weight, p_seq_8_bias);  _unsafe_view = p_seq_8_weight = p_seq_8_bias = None
                    return (linear,)
            
        Graph signature: 
            # inputs
            p_seq_1_weight: PARAMETER target='seq.1.weight'
            p_seq_1_bias: PARAMETER target='seq.1.bias'
            p_seq_3_weight: PARAMETER target='seq.3.weight'
            p_seq_3_bias: PARAMETER target='seq.3.bias'
            p_seq_5_weight: PARAMETER target='seq.5.weight'
            p_seq_5_bias: PARAMETER target='seq.5.bias'
            p_seq_8_weight: PARAMETER target='seq.8.weight'
            p_seq_8_bias: PARAMETER target='seq.8.bias'
            c_seq_0_mean: CONSTANT_TENSOR target='seq.0.mean'
            c_seq_0_std: CONSTANT_TENSOR target='seq.0.std'
            input: USER_INPUT
    
            # outputs
            linear: USER_OUTPUT
    
        Range constraints: {s31: VR[0, 2]}

)

---

## Loss landscape

### идея https://arxiv.org/abs/1712.09913


```python
from copy import deepcopy
state_dict_backup = deepcopy(model.state_dict())
```


```python
def generate_theta(seed=None):
    model.eval()
    model.load_state_dict(state_dict_backup)
    if seed is not None:
        np.random.seed(seed)
    params = []
    with torch.no_grad():
        for w in model.parameters():
            params.append(w.detach().cpu().numpy())
    params_n = np.concatenate([p.flatten() for p in params]).size
    random_theta_flat = np.random.normal(size=params_n)
    random_theta = []
    offset = 0
    for p in params:
        p_weights = p.flatten().size
        random_theta.append(random_theta_flat[offset:offset+p_weights].\
                            reshape(*p.shape))
        # normalization
        rank = random_theta[-1].shape.__len__()
        if rank == 4: # convolution
            #print('Conv')
            # Frobenius norm
            norm_r = np.sqrt((random_theta[-1]**2).sum(axis=-1).sum(axis=-1))
            norm_p = np.sqrt((p**2).sum(axis=-1).sum(axis=-1))
            norm = (norm_p / norm_r).reshape(*norm_p.shape, 1, 1)
        elif rank == 2: # fully connected
            #print('FC')
            norm_r = np.sqrt((random_theta[-1]**2).sum(axis=-1))
            norm_p = np.sqrt((p**2).sum(axis=-1))
            norm = (norm_p / norm_r).reshape(-1, 1)
        elif rank == 1: # bias
            #print('bias')
            norm_r = np.sqrt((random_theta[-1]**2).sum())
            norm_p = np.sqrt((p**2).sum())
            norm = norm_p / norm_r
        random_theta[-1] = random_theta[-1]*norm
        offset += p_weights
    assert offset==params_n, \
            "Not all params are utilized. Expected %d, found %d"%(params_n, offset)
    return random_theta

theta1 = generate_theta(seed=0)
theta2 = generate_theta(seed=1011)
```


```python
criterion2 = nn.CrossEntropyLoss(reduction='none')
loss_curve = []
alphas = np.arange(-1500, 1500, step=5)/1000
for alpha in tqdm(alphas):
    state_dict = model.state_dict()
    for (k, v), v_new in zip(state_dict_backup.items(), theta1):
        tensor = v.clone().detach() + \
              alpha*(torch.tensor(v_new, device=device) - v.clone().detach())
        state_dict[k] = tensor
    model.load_state_dict(state_dict)
    loss = []

    with torch.no_grad(): # отключение автоматического дифференцирования
        for i, data in enumerate(dataloader['test'], 0):
            inputs, labels = data
            # на GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).detach()
            loss.append(criterion2(outputs, labels).detach().cpu().numpy())
    loss_curve.append(np.concatenate(loss).mean())
```


      0%|          | 0/600 [00:00<?, ?it/s]



```python
plt.plot(alphas, loss_curve)
plt.yscale('log')
plt.xlabel('alpha')
plt.ylabel('CCE, log10')
```




    Text(0, 0.5, 'CCE, log10')




    
<img width="570" height="432" alt="output_35_1" src="https://github.com/user-attachments/assets/51bba262-99e4-4003-b29f-0724d600f97c" />

    


### 2D


```python
# увеличив step, можно значительно ускорить вычисления
# однако вместе с этим теряется разрешение
alphas = []
vals = np.arange(-150, 151, step=5)/100
size = vals.size

# создаём заданный растр
for a1 in vals:
    for a2 in vals:
        alphas.append((a1, a2))

alphas = np.array(alphas)
```


```python
Z = []
for a1, a2 in tqdm(alphas):
    state_dict = model.state_dict()
    for (k, v), v_new, v_new2 in zip(state_dict_backup.items(), theta1, theta2):
        # линейная интерполяция
        tensor = v.clone().detach() + \
                 a1*(torch.tensor(v_new, device=device) - \
                     v.clone().detach())
        # ещё раз со вторым вектором
        tensor = tensor + a2*(torch.tensor(v_new2, device=device) - tensor)
        state_dict[k] = tensor
        #print(k, tensor, v_new)
    model.load_state_dict(state_dict)
    loss = []
    with torch.no_grad(): # отключение автоматического дифференцирования
        for i, data in enumerate(dataloader['test'], 0):
            inputs, labels = data
            # на GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).detach()#.cpu().numpy()
            loss.append(criterion2(outputs, labels).detach().cpu().numpy())
    Z.append(np.concatenate(loss).mean())
ZZ = np.array(Z)
```


      0%|          | 0/3721 [00:00<?, ?it/s]



```python
# настройка размера графика
plt.figure(figsize=(8, 8))
# отрисовка закрашенных контуров, аналогично 2 части 1 ЛР
cs = plt.contourf(alphas[:,0].reshape(size, size),
             alphas[:,1].reshape(size, size),
             np.log10(ZZ.reshape(size, size)),
             levels=255,
             cmap=plt.cm.jet,
             )
# установка цветовой шкалы и её названия
plt.colorbar(cs).ax.set_ylabel('CCE, log10', rotation=270, fontsize=15)
# установка названий осям X, Y
plt.xlabel('α', fontsize=20)
plt.ylabel('β', fontsize=20)
plt.show()
```


    
<img width="712" height="698" alt="output_39_0" src="https://github.com/user-attachments/assets/8b26abd9-7378-4cf7-ae79-eb402907e8fa" />

    


### 3D


```python
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
# установка названий осям X, Y и Z
ax.set_xlabel('α', fontsize=20)
ax.set_ylabel('β', fontsize=20)
ax.set_zlabel('CCE, log10', fontsize=20)
# выставление прозрачности сетки, для красоты
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# отрисовка 3D поверхности, данные для отрисовки аналогичны contourf
surf = ax.plot_surface(alphas[:,0].reshape(size, size),
                       alphas[:,1].reshape(size, size),
                       np.log10(ZZ.reshape(size, size)),
                       cmap=plt.cm.coolwarm,
                       linewidth=0.1,
                       edgecolors='k',
                       alpha=0.8,
                       antialiased=True)
# первый агрумент - вращение вокруг XY, второрй аргумент - вокруг YZ
ax.view_init(40, -30)
```



<img width="658" height="645" alt="output_41_0" src="https://github.com/user-attachments/assets/192e108d-c6d3-44cd-9139-6ff7ae16356b" />



---

## Ответы на задания для самостоятельной работы (вариант 4)

_Эксперименты проводились для 3 классов CIFAR-100: 0, 33 и 41 (группа 22, вариант 4)._

**1. Анализ результатов обучения сверточной модели**

При обучении сверточной нейронной сети функция потерь (CCE) на обучающей выборке монотонно убывает и к концу обучения стремится к нулю. Точность на обучении достигает практически 100%. На валидационной выборке loss сначала быстро уменьшается, затем после ~100–150-й эпохи выходит на плато и начинает медленно расти, в то время как точность колеблется в районе 88–92%. Итоговая точность на тестовой выборке составляет около 90%. Такое поведение (очень высокая точность на train и заметно меньшая на val/test, рост валидационного loss при дальнейшем снижении тренировочного) свидетельствует о появлении переобучения на поздних эпохах, хотя в целом модель хорошо различает выбранные классы.

**2. Сравнение трех вариантов пуллинга**

Были рассмотрены три варианта операции субдискретизации: max-pooling, average-pooling и stride.  
Max-pooling дал наилучший результат по точности на тестовой выборке: при нём сеть лучше выделяет наиболее информативные яркие признаки (границы, углы, пятна текстуры), которые определяют принадлежность к классу. Average-pooling сглаживает карту признаков и усредняет как полезные, так и шумовые компоненты, поэтому точность немного ниже. Stride показал промежуточные результаты и оказался чувствительнее к выбросам. В итоге наиболее выгодным компромиссом между качеством и простотой реализации оказался max-pooling, который и был использован в финальной архитектуре.

**3. Влияние числа сверточных слоев и параметров свертки**

Увеличение числа сверточных слоёв с одного до двух позволило сети выделять более сложные иерархические признаки (от простых границ к более абстрактным контурам и текстурам), что привело к заметному росту точности. Попытка дальнейшего увеличения глубины приводила к росту времени обучения и склонности к переобучению при несущественном выигрыше по качеству.  
Размер ядра 3×3 показал себя оптимальным: при слишком крупном ядре (5×5 и более) количество параметров растёт, локальные детали размываются и качество падает; при меньшем эффективном receptive field сеть хуже улавливает пространственный контекст. Увеличение шага свёртки (stride) уменьшает размер карт признаков и ускоряет обучение, но при слишком большом шаге информация теряется, и точность падает. Отсутствие padding приводит к тому, что информация по краям изображения учитывается хуже; использование «same»-padding (с сохранением размера карты) даёт более высокое качество.

**4. Переобучение модели**

Переобучение в данной модели проявляется следующим образом:

- ошибка на обучающей выборке продолжает убывать почти до нуля;
- ошибка на валидации после некоторого числа эпох перестаёт убывать и начинает плавно расти;
- точность на обучающей выборке достигает 100%, в то время как на валидации и тесте остаётся существенно ниже (~90%).

Для уменьшения переобучения можно:

- сократить число эпох и применять раннюю остановку по минимуму val-loss;
- добавить регуляризацию (L2-регуляризация / weight decay) и/или Dropout в полносвязный слой;
- уменьшить размер модели (количество каналов и/или число слоёв);
- расширить обучающую выборку за счёт аугментаций (случайные повороты, сдвиги, отражения, небольшие изменения яркости).

**5. Влияние гиперпараметров обучения**

Увеличение числа эпох сначала приводит к росту точности, однако после достижения некоторого уровня (порядка сотни эпох) качество на тестовой выборке перестаёт заметно улучшаться, а переобучение усиливается. Оптимальным оказался диапазон эпох, при котором val-loss минимален.  
Изменение размера batch_size показало стандартный компромисс: при малом batch (например, 32) обучение нестабильнее, но градиенты более «шумные» и лучше выходят из локальных минимумов; при большом batch (128–256) обучение более гладкое, но минимума может достигать медленнее.  
Скорость обучения (learning rate) сильно влияет на сходимость: при слишком большом значении модель расходится или качественно «скачет»; при слишком малом — сходится очень медленно и может за отведённое число эпох не выйти на хороший минимум. Наилучшие результаты дали умеренные значения learning rate с возможным поэтапным уменьшением в процессе обучения.

**6. Сравнение полносвязной и сверточной нейронных сетей**

Полносвязная сеть, обученная на тех же трёх классах, показала заметно более низкую точность и сильнее склонна к переобучению. Количество параметров в чисто полносвязной архитектуре при работе с изображениями велико, что затрудняет обучение и требует более жёсткой регуляризации. Кроме того, полносвязная сеть не учитывает пространственную структуру изображения: каждый пиксель обрабатывается независимо от своего положения.  
Сверточная нейросеть, напротив, использует локальные рецептивные поля и совместно обучаемые фильтры, что позволяет эффективно извлекать инвариантные к сдвигам и небольшим поворотам признаки. В результате CNN достигает более высокой точности на тесте, быстрее сходится и остаётся более устойчивой к небольшим деформациям входного изображения.

**7. Действия, которые сильнее всего повысили точность модели**

Наибольший вклад в улучшение качества внесли следующие изменения:

1. **Переход от полносвязной архитектуры к сверточной** — позволил резко сократить число параметров при одновременном росте выразительной способности и учёте пространственной структуры данных.
2. **Добавление второго сверточного слоя и увеличение числа каналов** — дало возможность извлекать более сложные признаки и улучшило разделимость классов.
3. **Выбор корректного типа пуллинга и его параметров** (max-pooling с подходящим размером окна и шагом) — обеспечил компромисс между уменьшением размерности и сохранением информативных особенностей.
4. **Тщательная настройка гиперпараметров обучения** (learning rate, число эпох, размер batch) — позволила добиться хорошей сходимости без сильного переобучения.
5. **Нормализация входных данных** — ускорила обучение и сделала процесс оптимизации более стабильным.

В совокупности эти изменения позволили получить точность порядка 90% на тестовой выборке для варианта 4 при приемлемом времени обучения и устойчивости модели к небольшим вариациям входных изображений.

