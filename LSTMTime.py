# 导入相关包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LSTMTimePredictor:

    def __init__(self, df, test_ratio=0.2, n_past=30, optimizer='adam'):
        '''
        df:DataFrame时间序列数据;
        test_ratio:测试比率
        n_past:预测的窗口数;
        optimizer:优化器;
        n_features:特征数;
        feature_names:特征名称;
        '''
        self.df = df
        self.test_ratio = test_ratio
        self.n_past = n_past
        self.optimizer = optimizer
        self.n_features = self.df.shape[1]
        self.feature_names = self.df.columns

    def _train_test_split(self):
        '''
        训练测试划分;
        '''
        test_split = round(len(self.df) * self.test_ratio)  # 计算测试集中的样本数量
        df_training = self.df[:-test_split]
        df_testing = self.df[-test_split:]
        # 进行最小最大归一化
        scaler = MinMaxScaler()
        df_training_scaled = scaler.fit_transform(df_training)
        df_testing_scaled = scaler.transform(df_testing)

        # 获取训练集和测试集的样本数量
        self.train_length = len(df_training_scaled)
        self.test_length = len(df_testing_scaled)

        # 获取归一化后的训练样本和测试样本
        # self.df_training_scaled = df_training_scaled
        # self.df_training_scaled = df_testing_scaled
        self.scaler = scaler
        return df_training_scaled, df_testing_scaled

    def createXY(self, datasets):
        '''
        生成用于LSTM输入的多元数据,例如时间窗口n_past=30,则一个样本的维度为(30,5)
        30代表时间窗口,5代表特征数量
        '''
        dataX = []
        dataY = []
        for i in range(self.n_past, len(datasets)):
            dataX.append(datasets[i - self.n_past:i, 0:datasets.shape[1]])
            dataY.append(datasets[i, 0])
        return np.array(dataX), np.array(dataY)

    def _build_model(self, ):
        '''
        建立模型
        '''
        grid_model = Sequential()
        grid_model.add(LSTM(50, return_sequences=True, input_shape=(self.n_past, self.n_features)))
        grid_model.add(LSTM(50))
        grid_model.add(Dropout(0.2))
        grid_model.add(Dense(1))
        grid_model.compile(loss='mse', optimizer=self.optimizer)
        # 封装为scikit-learn模型
        return grid_model

    def fit(self, ):

        df_training_scaled = self._train_test_split()[0]
        df_testing_scaled = self._train_test_split()[1]
        X_train, y_train = self.createXY(df_training_scaled)
        X_test, y_test = self.createXY(df_testing_scaled)

        grid_model = KerasRegressor(build_fn=self._build_model, verbose=1, validation_data=(X_test, y_test))

        grid_model.fit(X_train, y_train)
        self.model = grid_model

    def evaluate(self, plot=True):
        df_testing_scaled = self._train_test_split()[1]
        X_test, y_test = self.createXY(df_testing_scaled)
        # 预测值
        prediction = self.model.predict(X_test)
        prediction_copy_array = np.repeat(prediction, self.n_features, axis=-1)
        pred = self.scaler.inverse_transform(np.reshape(prediction_copy_array, (len(prediction), self.n_features)))[:,
               0]
        # 实际值
        original_copies_array = np.repeat(y_test, self.n_features, axis=-1)
        original = self.scaler.inverse_transform(np.reshape(original_copies_array, (len(y_test), self.n_features)))[:,
                   0]
        if plot:
            fig, ax = plt.subplots(figsize=(20, 8))
            ax.plot(original, color='red', label='Real Values')
            ax.plot(pred, color='blue', label='Predicted Values')
            ax.set_title('Time Series Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Values')
            ax.legend()
            plt.show()
        mae = mean_absolute_error(original, pred)
        mse = mean_squared_error(original, pred)
        mape = np.mean(np.abs(original - pred) / original)
        print("MSE is {},MAE is {}, MAPE is {}".format(mse, mae, mape))
        return pred

    def predict(self, df_unknown):
        df_days_past = self.df.iloc[-self.n_past:, :]
        df_unknown[self.feature_names[0]] = 0
        df_unknown = df_unknown[self.feature_names]

        old_scaled_array = self.scaler.transform(df_days_past)
        new_scaled_array = self.scaler.transform(df_unknown)

        new_scaled_df = pd.DataFrame(new_scaled_array)
        new_scaled_df.iloc[:, 0] = np.nan

        full_df = pd.concat([pd.DataFrame(old_scaled_array), new_scaled_df]).reset_index().drop(["index"], axis=1)

        full_df_scaled_array = full_df.values
        all_data = []
        time_step = self.n_past
        for i in range(time_step, len(full_df_scaled_array)):
            data_x = []
            data_x.append(full_df_scaled_array[i - time_step:i, 0:full_df_scaled_array.shape[1]])
            data_x = np.array(data_x)
            prediction = self.model.predict(data_x)
            all_data.append(prediction)
            full_df.iloc[i, 0] = prediction

        new_array = np.array(all_data)
        new_array = new_array.reshape(-1, 1)
        prediction_copies_array = np.repeat(new_array, self.n_features, axis=-1)
        y_pred_future_days = self.scaler.inverse_transform(
            np.reshape(prediction_copies_array, (len(new_array), self.n_features)))[:, 0]
        return y_pred_future_days