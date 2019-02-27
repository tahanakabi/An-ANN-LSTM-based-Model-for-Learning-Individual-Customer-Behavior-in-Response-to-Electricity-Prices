import lstm
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import numpy as np



# This script is for training an LSTM network with given hyperparameters.



def train_score(network={}):
    print(network)

    global seq_len
    global num_features
    # initialize model according to the given values of the network


    model = lstm.build_model(input_shape=[seq_len, num_features],
                             lstm_size=30,
                             num_lstm=1,
                             dropout=0.2,
                             activation='tanh',
                             recurrent_activation='selu',
                             optimizer='rmsprop')

    model.fit(
        dataset[0],
        dataset[1],
        validation_split=0.2)

    print('Training duration (s) : ', time.time() - global_start_time)
    # model.save('model.h5')

    predictions = lstm.predict(model, dataset[2])
    global scaler
    try:
        predicted_load = lstm.inverse_transform(dataset[2], predictions, scaler)
        true_load = lstm.inverse_transform(dataset[2], dataset[3], scaler)

        rmse = sqrt(mean_squared_error(true_load, predicted_load))
        mape = np.mean(np.abs((true_load - predicted_load) / true_load)) * 100
    except Exception as e:
        print(e)
        rmse=100.0
        mape=100.0
    print('Test RMSE: %.3f' % rmse)
    print('Test MAPE: %.3f ' % mape)

    pyplot.plot(true_load, label='True')
    pyplot.plot(predicted_load,'--', label='predicted')
    pyplot.legend()
    pyplot.show()
    return predicted_load, true_load



if __name__ == '__main__':
    global_start_time = time.time()
    # # epochs  = 10
    seq_len = 2

    print('> Loading data... ')

    X_train, y_train, X_test, y_test, scaler = lstm.load_data('fuzzy_out.csv', seq_len)
    num_features = X_train.shape[2]
    dataset = [X_train, y_train, X_test, y_test]
    predicted_load, true_load = train_score()

    # Plot results
    pyplot.plot(true_load, label='True')
    pyplot.plot(predicted_load, label='predicted')
    pyplot.legend()
    pyplot.show()

