# Audio-Generation-using-Neutral-Net

For this project, I worked on Kaggler to execute and run my program: 
> Kaggle URL : https://www.kaggle.com/code/louisminguet/audio-generation-using-neural-net

According to the various tests that I was able to carry out, we observe that by simply using **LSTM** the results are the least interesting because **many notes are repeated**.<br>
By using **Embedded LSTM** we get **less repetitions of notes**, the result is more convincing, but **remains approximate**.<br>
With **GAN**, one obtains slightly **more harmonious results** than Embedded LSTM, but progress remains to be made.

## Audio Generation Using :
* LSTM
* LSTM using Embedding
* GAN

## Default parameters used 

#### LSTM :
- Epoch : 70
- Dropout : 0.5
- Activation function : Relu (dense: 3)
- Loss : Mae
- Optimizer : Adam
- Batch size : 256

#### Embedded LSTM :
- 128 notes
- Epoch : 200
- Embed size : 100
- Dropout : 0.3
- Activation function : Relu (dense: 1)
- Optimizer : RMSProp

#### GAN : 
- 128 notes
- Epoch : 60

--  RESULTS FILES

/test1 : Default parameters
/test2 : Default parameters
/test3 : LSTM changes : Epoch=200, Dropout=0.3, Optimizer=RMSProp | No changes on Embed LSTM & GAN
/test4 : 
