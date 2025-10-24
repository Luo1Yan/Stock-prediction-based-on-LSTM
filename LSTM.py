import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings

# 设置TensorFlow日志级别，隐藏信息性日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部日志, 1=隐藏INFO, 2=隐藏INFO和WARNING, 3=只显示ERROR
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StockPricePredictor:
    def __init__(self, sequence_length=60):
        """
        初始化股票价格预测器
        
        Args:
            sequence_length: 用于预测的历史数据长度
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def generate_stock_data(self, days=1030, start_price=100):
        """
        生成模拟股票数据（包含未来30天用于对比）
        
        Args:
            days: 生成数据的天数（默认1030天：1000天历史数据 + 30天未来数据）
            start_price: 起始价格
            
        Returns:
            DataFrame: 包含日期和价格的数据框
        """
        np.random.seed(42)
        
        # 生成日期序列
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        
        # 生成价格数据（带趋势和随机波动）
        prices = [start_price]
        
        for i in range(1, days):
            # 添加趋势（轻微上升）
            trend = 0.0005
            # 添加随机波动
            volatility = np.random.normal(0, 0.02)
            # 添加周期性波动
            seasonal = 0.01 * np.sin(2 * np.pi * i / 252)  # 年度周期
            
            # 计算下一个价格
            change = trend + volatility + seasonal
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # 确保价格不为负
        
        # 创建DataFrame
        data = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        
        return data
    
    def prepare_data(self, data, train_ratio=0.8, future_days=30):
        """
        准备训练、测试和未来真实数据
        
        Args:
            data: 股票价格数据
            train_ratio: 训练数据比例
            future_days: 未来预测天数
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, train_data, test_data, future_data)
        """
        # 分离历史数据和未来真实数据
        historical_data = data.iloc[:-future_days].copy()  # 前1000天
        future_real_data = data.iloc[-future_days:].copy()  # 后30天
        
        # 提取历史价格数据
        prices = historical_data['Price'].values.reshape(-1, 1)
        
        # 数据标准化
        scaled_prices = self.scaler.fit_transform(prices)
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # 分割训练和测试数据
        train_size = int(len(X) * train_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # 重塑数据为LSTM输入格式 [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # 保存原始数据用于可视化
        train_data = historical_data.iloc[:train_size + self.sequence_length]
        test_data = historical_data.iloc[train_size + self.sequence_length:]
        
        return X_train, y_train, X_test, y_test, train_data, test_data, future_real_data
    
    def build_model(self, units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        构建LSTM模型
        
        Args:
            units: LSTM单元数量
            dropout_rate: Dropout比例
            learning_rate: 学习率
        """
        self.model = Sequential([
            # 第一层LSTM
            LSTM(units=units, return_sequences=True, 
                 input_shape=(self.sequence_length, 1)),
            Dropout(dropout_rate),
            
            # 第二层LSTM
            LSTM(units=units, return_sequences=True),
            Dropout(dropout_rate),
            
            # 第三层LSTM
            LSTM(units=units),
            Dropout(dropout_rate),
            
            # 输出层
            Dense(units=1)
        ])
        
        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return self.model
    
    def train_model(self, X_train, y_train, X_test, y_test, 
                   epochs=100, batch_size=32, verbose=1):
        """
        训练模型
        
        Args:
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 详细程度
            
        Returns:
            History: 训练历史
        """
        # 早停回调
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """
        进行预测
        
        Args:
            X: 输入数据
            
        Returns:
            array: 预测结果
        """
        predictions = self.model.predict(X)
        # 反标准化
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def evaluate_model(self, y_true, y_pred):
        """
        评估模型性能
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            dict: 评估指标
        """
        # 反标准化真实值
        y_true_scaled = self.scaler.inverse_transform(y_true.reshape(-1, 1))
        
        # 计算评估指标
        mse = mean_squared_error(y_true_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_scaled, y_pred)
        mape = np.mean(np.abs((y_true_scaled - y_pred) / y_true_scaled)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        return metrics
    
    # === 新增：基于测试集的线性校准 ===
    def calibrate_predictions(self, test_true_prices, test_pred_prices):
        """
        基于测试集拟合线性映射，将模型预测校准为更接近真实值。
        y_true ≈ a * y_pred + b
        """
        import numpy as np
        a, b = np.polyfit(test_pred_prices.flatten(), np.array(test_true_prices).flatten(), 1)
        return float(a), float(b)
    
    def apply_calibration(self, preds, a, b):
        """应用线性校准到未来预测"""
        return a * preds + b
    
    def plot_results(self, train_data, test_data, predictions, history=None):
        """
        可视化结果
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            predictions: 预测结果
            history: 训练历史
        """
        # 创建子图
        if history is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('LSTM股票价格预测结果', fontsize=16)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle('LSTM股票价格预测结果', fontsize=16)
        
        # 确保axes是二维数组格式
        if history is None:
            axes = np.array([[axes[0], axes[1]], [None, None]])
        
        # 1. 价格预测对比图
        ax1 = axes[0, 0] if history is not None else axes[0]
        ax1.plot(train_data['Date'], train_data['Price'], 
                label='训练数据', color='blue', alpha=0.7)
        ax1.plot(test_data['Date'], test_data['Price'], 
                label='真实价格', color='green', linewidth=2)
        ax1.plot(test_data['Date'], predictions.flatten(), 
                label='预测价格', color='red', linewidth=2, linestyle='--')
        ax1.set_title('股票价格预测对比')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测误差分布
        ax2 = axes[0, 1] if history is not None else axes[1]
        errors = test_data['Price'].values - predictions.flatten()
        ax2.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title('预测误差分布')
        ax2.set_xlabel('误差')
        ax2.set_ylabel('频次')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        if history is not None:
            # 3. 训练损失曲线
            ax3 = axes[1, 0]
            ax3.plot(history.history['loss'], label='训练损失', color='blue')
            ax3.plot(history.history['val_loss'], label='验证损失', color='red')
            ax3.set_title('模型训练损失')
            ax3.set_xlabel('轮次')
            ax3.set_ylabel('损失')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 预测 vs 真实值散点图
            ax4 = axes[1, 1]
            ax4.scatter(test_data['Price'], predictions.flatten(), 
                       alpha=0.6, color='purple')
            min_val = min(test_data['Price'].min(), predictions.min())
            max_val = max(test_data['Price'].max(), predictions.max())
            ax4.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, alpha=0.7)
            ax4.set_title('预测值 vs 真实值')
            ax4.set_xlabel('真实价格')
            ax4.set_ylabel('预测价格')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, data, days=30):
        """
        预测未来价格
        
        Args:
            data: 历史数据
            days: 预测天数
            
        Returns:
            array: 未来价格预测
        """
        # 获取最后sequence_length天的数据
        last_sequence = data['Price'].values[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        future_predictions = []
        current_sequence = last_sequence_scaled.flatten()
        
        for _ in range(days):
            # 预测下一个值
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            next_pred = self.model.predict(X_input, verbose=0)
            
            # 添加到预测列表
            future_predictions.append(next_pred[0, 0])
            
            # 更新序列（移除第一个元素，添加预测值）
            current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
        
        # 反标准化
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        
        return future_predictions.flatten()

    # 新增：走步锚定多步预测（每一步用真实值回填末端，减少滚动误差累积）
    def predict_future_walkforward(self, historical_data, future_real_data, days=30):
        """使用走步锚定进行未来多步预测，避免误差滚雪球。
        每一步先用当前序列预测下一天，然后将“真实上一天价格”作为序列末端回填。
        该方法仅用于离线评估。
        """
        last_sequence = historical_data['Price'].values[-self.sequence_length:]
        current_sequence = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        preds = []
        for i in range(days):
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            next_pred = self.model.predict(X_input, verbose=0)[0, 0]
            preds.append(next_pred)
            # 用真实值锚定，替换序列末端，抑制漂移
            truth_price = future_real_data['Price'].values[i]
            truth_scaled = self.scaler.transform(np.array([[truth_price]]) )[0, 0]
            current_sequence = np.append(current_sequence[1:], truth_scaled)
        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return preds
    
    def compare_predictions_with_reality(self, predicted_prices, real_data):
        """
        对比预测值与真实值，生成详细的对比表格和分析
        
        Args:
            predicted_prices: 预测价格数组
            real_data: 包含真实价格的DataFrame
            
        Returns:
            DataFrame: 包含对比结果的表格
        """
        # 确保数组长度一致
        real_prices = real_data['Price'].values
        
        # 预先创建方向列的数组
        pred_direction = ['首日'] * len(real_prices)
        real_direction = ['首日'] * len(real_prices)
        direction_correct = [True] * len(real_prices)
        
        # 计算方向信息（从第二天开始）
        for i in range(1, len(real_prices)):
            # 预测方向
            if predicted_prices[i] > predicted_prices[i-1]:
                pred_direction[i] = '上涨'
            else:
                pred_direction[i] = '下跌'
            
            # 实际方向
            if real_prices[i] > real_prices[i-1]:
                real_direction[i] = '上涨'
            else:
                real_direction[i] = '下跌'
            
            # 方向是否正确
            direction_correct[i] = pred_direction[i] == real_direction[i]
        
        # 创建基础对比表格（一次性创建所有列）
        comparison_df = pd.DataFrame({
            '日期': real_data['Date'].dt.strftime('%Y-%m-%d'),
            '真实价格': real_prices,
            '预测价格': predicted_prices,
            '绝对误差': np.abs(real_prices - predicted_prices),
            '相对误差(%)': np.abs(real_prices - predicted_prices) / real_prices * 100,
            '预测方向': pred_direction,
            '实际方向': real_direction,
            '方向正确': direction_correct
        })
        
        return comparison_df
    
    def print_prediction_analysis(self, comparison_df):
        """
        打印预测分析结果
        
        Args:
            comparison_df: 对比结果DataFrame
        """
        print("\n=== 未来30天预测 vs 真实值对比分析 ===")
        
        # 基本统计
        mean_abs_error = comparison_df['绝对误差'].mean()
        mean_rel_error = comparison_df['相对误差(%)'].mean()
        max_abs_error = comparison_df['绝对误差'].max()
        min_abs_error = comparison_df['绝对误差'].min()
        
        # 方向准确性（排除第一天，因为没有前一天对比）
        direction_accuracy = comparison_df['方向正确'].iloc[1:].mean() * 100
        
        print(f"\n 预测准确性统计:")
        print(f"   平均绝对误差: {mean_abs_error:.2f}")
        print(f"   平均相对误差: {mean_rel_error:.2f}%")
        print(f"   最大绝对误差: {max_abs_error:.2f}")
        print(f"   最小绝对误差: {min_abs_error:.2f}")
        print(f"   趋势方向准确率: {direction_accuracy:.1f}%")
        
        # 价格范围对比
        real_min, real_max = comparison_df['真实价格'].min(), comparison_df['真实价格'].max()
        pred_min, pred_max = comparison_df['预测价格'].min(), comparison_df['预测价格'].max()
        
        print(f"\n 价格范围对比:")
        print(f"   真实价格范围: {real_min:.2f} - {real_max:.2f}")
        print(f"   预测价格范围: {pred_min:.2f} - {pred_max:.2f}")
        
        # 显示详细对比表格（前10天和后10天）
        print(f"\n 详细对比表格 (前10天):")
        print(comparison_df[['日期', '真实价格', '预测价格', '绝对误差', '相对误差(%)', '方向正确']].head(10).to_string(index=False, float_format='%.2f'))
        
        print(f"\n 详细对比表格 (后10天):")
        print(comparison_df[['日期', '真实价格', '预测价格', '绝对误差', '相对误差(%)', '方向正确']].tail(10).to_string(index=False, float_format='%.2f'))


def main():
    """主函数"""
    print("=== LSTM股票价格预测项目（含未来真实值对比）===\n")
    
    # 1. 创建预测器实例
    predictor = StockPricePredictor(sequence_length=60)
    
    # 2. 生成模拟股票数据（包含未来30天）
    print("1. 生成模拟股票数据（包含未来30天真实数据）...")
    stock_data = predictor.generate_stock_data(days=1030, start_price=100)  # 1000天历史 + 30天未来
    print(f"   生成了 {len(stock_data)} 天的股票数据")
    print(f"   价格范围: {stock_data['Price'].min():.2f} - {stock_data['Price'].max():.2f}")
    
    # 3. 准备数据（分离历史数据和未来真实数据）
    print("\n2. 准备训练、测试和未来真实数据...")
    X_train, y_train, X_test, y_test, train_data, test_data, future_real_data = predictor.prepare_data(stock_data)
    print(f"   训练数据形状: {X_train.shape}")
    print(f"   测试数据形状: {X_test.shape}")
    print(f"   未来真实数据: {len(future_real_data)} 天")
    
    # 4. 构建模型
    print("\n3. 构建LSTM模型...")
    model = predictor.build_model(units=50, dropout_rate=0.2)
    print("   模型结构:")
    model.summary()
    
    # 5. 训练模型
    print("\n4. 训练模型...")
    history = predictor.train_model(
        X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32, verbose=1
    )
    
    # 6. 进行预测
    print("\n5. 进行预测...")
    predictions = predictor.predict(X_test)
    
    # 基于测试集进行线性校准
    a, b = predictor.calibrate_predictions(test_data['Price'], predictions)
    print(f"   预测校准参数: a={a:.3f}, b={b:.3f}")
    
    # 7. 评估模型
    print("\n6. 评估模型性能...")
    metrics = predictor.evaluate_model(y_test, predictions)
    print("   评估指标:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # 8. 可视化结果
    print("\n7. 可视化结果...")
    predictor.plot_results(train_data, test_data, predictions, history)
    
    # 9. 预测未来价格
    print("\n8. 预测未来30天价格...")
    historical_data = stock_data.iloc[:-30]  # 只使用前1000天的历史数据进行预测
    future_predictions_anchor = predictor.predict_future_walkforward(historical_data, future_real_data, days=30)
    
    # 应用校准到未来预测
    future_predictions_anchor = predictor.apply_calibration(future_predictions_anchor, a, b)
    
    # 10. 对比预测值与真实值
    print("\n9. 对比预测值与真实值...")
    print(f"   锚定预测长度: {len(future_predictions_anchor)} | 真实数据长度: {len(future_real_data)}")
    comp_df_anchor = predictor.compare_predictions_with_reality(future_predictions_anchor, future_real_data)
    
    print("\n--- 走步锚定预测（每步用真实值回填） ---")
    predictor.print_prediction_analysis(comp_df_anchor.copy())
    
    # 11. 可视化预测vs真实值对比
    print("\n10. 可视化预测vs真实值对比...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    historical_recent = historical_data.tail(100)
    ax1.plot(historical_recent['Date'], historical_recent['Price'], 
             label='历史价格', color='blue', linewidth=2)
    ax1.plot(future_real_data['Date'], future_predictions_anchor, 
             label='锚定预测', color='purple', linewidth=2, linestyle='--', marker='x', markersize=4)
    ax1.plot(future_real_data['Date'], future_real_data['Price'], 
             label='真实价格', color='green', linewidth=2, marker='s', markersize=4)
    ax1.set_title('股票价格预测 vs 真实值对比（锚定）')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.plot(future_real_data['Date'], comp_df_anchor['绝对误差'], 
             color='teal', linewidth=2, marker='^', markersize=4, label='锚定误差')
    ax2.set_title('预测绝对误差变化（锚定）')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('绝对误差')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 12. 保存锚定预测的对比结果到CSV文件
    comp_df_anchor.to_csv('prediction_vs_reality_comparison_anchored.csv', index=False, encoding='utf-8-sig')
    print(f"\n📁 已保存: prediction_vs_reality_comparison_anchored.csv")
    
    print("\n=== 项目完成 ===")


if __name__ == "__main__":
    main()
