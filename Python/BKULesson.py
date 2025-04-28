import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 1. 生成模拟的股票高频数据
def generate_simulated_high_frequency_data(start_date, end_date, freq='s'):
    """
    生成模拟的股票高频数据（包括价格、买卖盘口等）。
    """
    # 生成时间戳
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)
    # 生成模拟的价格数据（假设初始价格为100）
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))  # 使用对数正态分布
    # 生成模拟的买卖盘口数据
    bid_prices = prices - np.random.uniform(0.01, 0.05, n)   # 买盘价格
    bid_volumes = np.random.randint(100, 1000, n)            # 买盘数量
    ask_prices = prices + np.random.uniform(0.01, 0.05, n)   # 卖盘价格
    ask_volumes = np.random.randint(100, 1000, n)            # 卖盘数量
    # 构建 DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'bid_price': bid_prices,
        'bid_volume': bid_volumes,
        'ask_price': ask_prices,
        'ask_volume': ask_volumes
    })
    return df

# 2. 高频做市策略
def market_making_strategy(df, initial_capital=100000, commission=0.0002, slippage=0.0001, max_inventory=1000):
    """
    高频做市策略。
    """
    # 初始化交易信号和交易记录
    df['signal'] = 0  # 交易信号：0 表示不交易，1 表示买入，-1 表示卖出
    trades = []       # 交易记录
    # 初始化做市商状态
    
    inventory = 0  # 当前库存
    capital = initial_capital  # 初始资金
    print(len(df),"aaaaa")
    for i in range(1, len(df)):
        # 获取当前市场数据
        bid_price = df.loc[df.index[i], 'bid_price']
        ask_price = df.loc[df.index[i], 'ask_price']
        mid_price = (bid_price + ask_price) / 2  # 中间价

        # 挂单逻辑
        if inventory < max_inventory:  # 库存未达到上限，挂买单
            buy_price = bid_price * (1 - slippage)  # 考虑滑点
            buy_volume = min(df.loc[df.index[i], 'bid_volume'], max_inventory - inventory)
            if buy_volume > 0:
                df.loc[df.index[i], 'signal'] = 1
                trade_cost = buy_price * buy_volume * (1 + commission)
                capital -= trade_cost
                inventory += buy_volume
                # 记录交易
                trade = {
                    'timestamp': df.loc[df.index[i], 'timestamp'],
                    'price': buy_price,
                    'volume': buy_volume,
                    'signal': 1  # 买入
                }
                trades.append(trade)
                print(f"挂买单: 时间 {df.loc[df.index[i], 'timestamp']}, 价格 {buy_price:.2f}, 数量 {buy_volume}, 资金 {capital:.2f}, 库存 {inventory}")

        if inventory > -max_inventory:  # 库存未达到下限，挂卖单
            sell_price = ask_price * (1 + slippage)  # 考虑滑点
            sell_volume = min(df.loc[df.index[i], 'ask_volume'], max_inventory + inventory)
            if sell_volume > 0:
                df.loc[df.index[i], 'signal'] = -1
                trade_cost = sell_price * sell_volume * (1 + commission)
                capital += trade_cost
                inventory -= sell_volume
                # 记录交易
                trade = {
                    'timestamp': df.loc[df.index[i], 'timestamp'],
                    'price': sell_price,
                    'volume': sell_volume,
                    'signal': -1  # 卖出
                }
                trades.append(trade)
                print(f"挂卖单: 时间 {df.loc[df.index[i], 'timestamp']}, 价格 {sell_price:.2f}, 数量 {sell_volume}, 资金 {capital:.2f}, 库存 {inventory}")
        # 风险管理：平仓逻辑
        if abs(inventory) > max_inventory * 0.8:  # 库存超过 80% 上限，强制平仓
            if inventory > 0:  # 卖出平仓
                sell_price = ask_price * (1 + slippage)
                sell_volume = inventory
                trade_cost = sell_price * sell_volume * (1 + commission)
                capital += trade_cost
                inventory = 0
                # 记录交易
                trade = {
                    'timestamp': df.loc[df.index[i], 'timestamp'],
                    'price': sell_price,
                    'volume': sell_volume,
                    'signal': -1  # 卖出
                }
                trades.append(trade)
                print(f"强制平仓（卖出）：时间 {df.loc[df.index[i], 'timestamp']}, 价格 {sell_price:.2f}, 数量 {sell_volume}, 资金 {capital:.2f}, 库存 {inventory}")
            else:  # 买入平仓
                buy_price = bid_price * (1 - slippage)
                buy_volume = -inventory
                trade_cost = buy_price * buy_volume * (1 + commission)
                capital -= trade_cost
                inventory = 0
                # 记录交易
                trade = {
                    'timestamp': df.loc[df.index[i], 'timestamp'],
                    'price': buy_price,
                    'volume': buy_volume,
                    'signal': 1  # 买入
                }
                trades.append(trade)
                print(f"强制平仓（买入）：时间 {df.loc[df.index[i], 'timestamp']}, 价格 {buy_price:.2f}, 数量 {buy_volume}, 资金 {capital:.2f}, 库存 {inventory}")

    # 计算策略收益率
    df['strategy_return'] = (capital + inventory * df['price'].iloc[-1]) / initial_capital - 1
    # 计算累计收益率
    df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()
    # 将交易记录转化为DataFrame
    df_trades = pd.DataFrame(trades)
    return df, df_trades
    

# 3. 可视化结果并保存为 HTML 和 PNG
def visualize_and_save_results(df, df_trades, save_dir):
    """  
    可视化策略结果并保存为 HTML 和 PNG 文件。  
    """  
    # 创建子图  
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[2, 1, 1])  
    
    # 绘制价格 (第一行)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name='价格', line=dict(color='blue')), row=1, col=1)  
    
    # 绘制交易信号点 (如果 df_trades 不为空)
    if not df_trades.empty:  
        buy_signals = df_trades[df_trades['signal'] == 1]  
        sell_signals = df_trades[df_trades['signal'] == -1]  
        fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['price'], mode='markers', name='买入信号', marker=dict(color='green', symbol='triangle-up')), row=1, col=1)  
        fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['price'], mode='markers', name='卖出信号', marker=dict(color='red', symbol='triangle-down')), row=1, col=1)  
    
    # 绘制累计收益率 (第二行)
    if 'cumulative_return' in df.columns:  # 确保累计收益率列存在  
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cumulative_return'], mode='lines', name='累计收益率', line=dict(color='green')), row=2, col=1)  
    else:  
        print("警告: 未找到累计收益率列！")  
    
    # 可以添加其他指标在第三行
    
    # 设置布局  
    fig.update_layout(  
        title='高频做市策略回测结果',  
        xaxis_title='时间',  
        yaxis_title='值',
        hovermode='x unified',  
        showlegend=True,  
        xaxis_rangeslider_visible=True
    )  
    
    # 为每个子图设置y轴标题
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="累计收益率", row=2, col=1)
    
    # 保存为 HTML 文件  
    html_path = os.path.join(save_dir, 'market_making_backtest.html')  
    fig.write_html(html_path)  
    print(f"HTML 文件已保存至：{html_path}")  
    
    # 保存为 PNG 文件  
    png_path = os.path.join(save_dir, 'market_making_backtest.png')  
    fig.write_image(png_path, scale=2)  
    print(f"PNG 文件已保存至：{png_path}")
    # 显示图形
    fig.show()

# 4. 保存数据到本地
def save_data_to_local(df, df_trades, save_dir):
    """  
    将模拟数据和策略结果保存到本地。  
    """  
    # 保存模拟的高频数据
    df.to_csv(os.path.join(save_dir, 'simulated_high_frequency_data.csv'), index=False)  
    # 保存交易信号  
    df[['timestamp', 'signal']].to_csv(os.path.join(save_dir, 'trade_signals.csv'), index=False)  
    # 保存交易记录（如果 df_trades 不为空）  
    if not df_trades.empty:  
        df_trades.to_csv(os.path.join(save_dir, 'trades_record.csv'), index=False)  

# 5. 主程序
if __name__ == '__main__':  
    # 设置保存路径  
    save_dir = './Python/Strategy'  
    if not os.path.exists(save_dir):  
        os.makedirs(save_dir)  
    # 生成模拟数据  
    start_date = '2023-10-01 09:30:00'  
    end_date = '2023-10-01 15:00:00'  
    df = generate_simulated_high_frequency_data(start_date, end_date)  
    # 运行高频做市策略  
    initial_capital = 100000  # 初始资金  
    commission = 0.0002  # 佣金  
    slippage = 0.0001  # 滑点  
    max_inventory = 1000  # 最大库存  
    df, df_trades = market_making_strategy(df, initial_capital, commission, slippage, max_inventory)  
    # 输出回测结果  
    final_capital = initial_capital * (1 + df['strategy_return'].iloc[-1])  
    print(f"初始资金: {initial_capital:.2f}")  
    print(f"最终资金: {final_capital:.2f}")  
    print(f"收益率: {(final_capital - initial_capital) / initial_capital * 100:.2f}%")  
    # 保存数据到本地

    # 可视化结果并保存为 HTML 和 PNG  
    visualize_and_save_results(df, df_trades, save_dir)
    save_data_to_local(df, df_trades, save_dir)


        