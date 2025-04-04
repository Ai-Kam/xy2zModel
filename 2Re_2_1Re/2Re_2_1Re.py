import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ReLUMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_units=[64, 32], output_dim=1, negative_slope=0.01):
        """
        Parameters:
        - input_dim: 入力層のユニット数（デフォルトは2）
        - hidden_units: 各隠れ層のユニット数を格納したリスト
        - output_dim: 出力層のユニット数（デフォルトは1）
        - negative_slope: LeakyReLU の negative_slope パラメータ（デフォルトは0.01）
        """
        super(ReLUMLP, self).__init__()
        
        layers = []
        in_features = input_dim
        
        # 各隠れ層の定義
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            in_features = units
        
        # 出力層の定義
        layers.append(nn.Linear(in_features, output_dim))
        
        # Sequential で層をまとめる
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, train_loader, num_epochs=100, learning_rate=0.001, verbose=True, return_metrics=False):
        """
        モデルの学習を行う関数
        
        Parameters:
        - train_loader: 学習データの DataLoader
        - num_epochs: エポック数（デフォルトは100）
        - learning_rate: 学習率（デフォルトは0.001）
        - verbose: 途中経過を出力するかどうか（デフォルトは True）
        - return_metrics: 最終的な損失関数の値を返すかどうか（デフォルトは False）
        """
        self.train()  # 学習モードに設定
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        prev_avg_loss = float('inf')
        epsilon_fact = 1e-4

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)

            if verbose and (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            if prev_avg_loss - avg_loss < epsilon_fact*prev_avg_loss:
                break
            prev_avg_loss = avg_loss
        
        if return_metrics:
            return epoch, avg_loss
        
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def compare_unit_num(train_loader, test_x, test_y, test_z):
    epochs_list = []
    loss_list = []
    for i in (1,2,4,8,16,32,64,128,256):
        model = ReLUMLP(input_dim=2, hidden_units=[i], output_dim=1, negative_slope=0.01)
    
        # 学習の実行
        metrics = model.fit(train_loader, num_epochs=1000, learning_rate=0.001, verbose=True, return_metrics=True)
        epochs_list.append(metrics[0])
        loss_list.append(metrics[1])
        
        # 学習後のダミー入力による出力例


        test_input = torch.cat([test_x.reshape(-1, 1), test_y.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            pred_z = model(test_input).reshape(test_x.shape).detach().numpy()

        # 3D 曲面プロット
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(test_x.numpy(), test_y.numpy(), pred_z, cmap='viridis')  # cmapはお好みで変更可能
        ax.plot_surface(test_x.numpy(), test_y.numpy(), test_z, alpha=0.5, cmap='coolwarm')
        plt.suptitle(f'xy to z by flat {i} LeakyReLU units')
        plt.tight_layout()
        plt.savefig(f'2Re_2_1Re_{i}.png')
        plt.close()
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.plot([1,2,4,8,16,32,64,128,256], loss_list)
    plt.xlabel('hidden units')
    plt.ylabel('loss')
    plt.title('hidden units to loss')
    plt.tight_layout()
    plt.savefig('2Re_2_1Re_loss.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xscale('log')
    plt.plot([1,2,4,8,16,32,64,128,256], epochs_list)
    plt.xlabel('hidden units')
    plt.ylabel('epochs')
    plt.title('hidden units to epochs')
    plt.tight_layout()
    plt.savefig('2Re_2_1Re_epochs.png')
    plt.close()


def compare_model_shape(train_loader, test_x, test_y, test_z):
    epochs_list = []
    loss_list = []
    param_count_list = []
    ses = ([32], [16, 16], [8,8,8,8], [4]*8, [2]*16)
    for s in ses:
        model = ReLUMLP(input_dim=2, hidden_units=s, output_dim=1, negative_slope=0.01)
    
        # 学習の実行
        param_count_list.append(model.count_trainable_params())
        metrics = model.fit(train_loader, num_epochs=1000, learning_rate=0.001, verbose=True, return_metrics=True)
        epochs_list.append(metrics[0])
        loss_list.append(metrics[1])
        
        # 学習後のダミー入力による出力例


        test_input = torch.cat([test_x.reshape(-1, 1), test_y.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            pred_z = model(test_input).reshape(test_x.shape).detach().numpy()

        # 3D 曲面プロット
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(test_x.numpy(), test_y.numpy(), pred_z, cmap='viridis')  # cmapはお好みで変更可能
        ax.plot_surface(test_x.numpy(), test_y.numpy(), test_z, alpha=0.5, cmap='coolwarm')
        plt.suptitle(f'xy to z by {s} LeakyReLU units')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'2Re_2_1Re_{s}.png')
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(['32', '16*2', '8*4', '4*8', '2*16'], param_count_list)
    plt.xlabel('model shape')
    plt.ylabel('param count')
    plt.suptitle('param count')
    plt.tight_layout()
    plt.savefig('2Re_2_1Re_param_count.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(['32', '16*2', '8*4', '4*8', '2*16'], loss_list)
    plt.yscale('log')
    plt.xlabel('model shape')
    plt.ylabel('loss')
    plt.suptitle('loss')
    plt.tight_layout()
    plt.savefig('2Re_2_1Re_loss_2.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(['32', '16*2', '8*4', '4*8', '2*16'], epochs_list)
    plt.xlabel('model shape')
    plt.ylabel('epochs')
    plt.suptitle('epochs')
    plt.tight_layout()
    plt.savefig('2Re_2_1Re_epochs2.png')
    plt.close()

# 使用例
if __name__ == "__main__":
    # ダミーデータの作成（例としてランダムなデータを使用）
    # 入力: 2変数、出力: 1変数
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fact = 100
    data_num = 10000
    batch_size = 100

    x = fact * (torch.rand(data_num, 1) - 0.5)
    y = fact * (torch.rand(data_num, 1) - 0.5)
    z = x**2 + y**2
    
    test_x = torch.linspace(-int(fact/2), int(fact/2), 21).float()
    test_y = torch.linspace(-int(fact/2), int(fact/2), 21).float()
    test_x, test_y = torch.meshgrid(test_x, test_y)
    test_z = (test_x**2 + test_y**2).numpy()

    # TensorDataset と DataLoader を利用してバッチ処理可能にする
    dataset = TensorDataset(torch.cat([x, y], dim=1), z)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    compare_unit_num(train_loader, test_x, test_y, test_z)

    fact = 10
    x = fact * (torch.rand(data_num, 1) - 0.5)
    y = fact * (torch.rand(data_num, 1) - 0.5)
    z = torch.exp(torch.sqrt(x**2 + y**2))

    test_x = torch.linspace(-int(fact/2), int(fact/2), 21).float()
    test_y = torch.linspace(-int(fact/2), int(fact/2), 21).float()
    test_x, test_y = torch.meshgrid(test_x, test_y)
    test_z = torch.exp(torch.sqrt(test_x**2 + test_y**2)).numpy()

    # TensorDataset と DataLoader を利用してバッチ処理可能にする
    dataset = TensorDataset(torch.cat([x, y], dim=1), z)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    #compare_model_shape(train_loader, test_x, test_y, test_z)





