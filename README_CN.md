# TCPELIA：丘脑—皮层预测误差侧向抑制注意力

[English README](README.md) · [Ultralytics 集成说明](docs/ultralytics_integration.md)

**TCPELIA**（Thalamo-Cortical Predictive Error Lateral-Inhibition Attention）是一个基于 PyTorch 的研究型视觉注意力模块。它将来自当前层的特征与反馈特征映射到统一通道空间，通过预测偏差、侧向抑制和动态资源门控迭代构建空间注意力，并对原始当前特征进行乘性残差增强。

> **公开范围声明：** 本仓库基于上传的单个 TCPELIA 模块源码整理，包含独立模块、示例、单元测试与框架集成说明；暂不包含完整 YOLO26 项目、训练数据、模型权重、已经验证的检测指标或论文 DOI。“丘脑—皮层”属于算法设计的生物启发命名，并不表示该模块已经获得神经生理学验证。

## 研究动机与主要特点

针对弱纹理、稀疏响应和复杂背景下的视觉表征，TCPELIA 以**预测不一致性**作为空间调节线索，并通过邻域竞争与动态资源因子约束注意力更新。模块支持不同通道数、不同空间尺度的双输入特征；输出张量与当前输入形状完全一致。

- **反馈条件预测：** 对反馈特征先对齐，再利用上一轮注意力产生的反馈响应进行调节。
- **预测误差：** 计算当前特征与预测特征的通道均方差，得到单通道误差图 `q`。
- **侧向抑制：** 使用归一化的 `5×5`、中心权重为零的固定核从上一轮注意力 `a` 中计算竞争抑制图 `h`。
- **资源门控：** 递推资源变量 `r` 和 `u`，通过 `g=r·u` 调节误差激励。
- **迭代闭环：** 联合预测误差、侧向抑制和资源增益，更新注意力图 `a`。
- **特征调制：** `y=x_cur·(1+lam·a)`，保持当前输入通道与分辨率。

## 直接运行

推荐 Python 3.9+、PyTorch 2.0+：

```bash
python -m pip install -e .
python -m examples.quickstart
python -m pytest -q
```

```python
import torch
from tcpelia import TCPELIA

x_cur = torch.randn(1, 128, 40, 40)
z_fb = torch.randn(1, 256, 20, 20)

model = TCPELIA(c_x=128, c_z=256, c_r=64, iters=2)
y = model([x_cur, z_fb])
print(y.shape)                  # [1,128,40,40]
print(model.attn_map.shape)     # [1,1,40,40]
```

**注意两个容易混淆的细节：** 直接创建 `TCPELIA(...)` 时，`iters` 的 Python 默认值为 **5**；上传源码中的 YAML 示例显式指定为 **2**。另外，`c_r` 必须能被 8 整除，因为代码使用 `nn.GroupNorm(8, c_r)`。

## 算法主要关系

1. 特征对齐：`x_r = GN(Conv1×1(x_cur))`，`z_r = Resize(GN(Conv1×1(z_fb)))`。
2. 反馈预测：`fb = sigmoid(Conv3×3(a))`，`x_hat = z_r·(1+rho·fb)`。
3. 预测误差：`q = mean_channel((x_r-x_hat)^2)`。
4. 侧向抑制：`h_new = (1-1/tau_h)·h+(1/tau_h)·Conv5×5(a)`。
5. 资源递推：`r_new = clamp(r+(1-r)/tau_d-u·r·q, 0, 1)`，`u_new = clamp(u+(U0-u)/tau_f+Uf·(1-u)·q, U0, 1)`。
6. 注意力更新：`a_prop=sigmoid(alpha·r_new·u_new·q-beta·h_new)`；`a_new=(1-1/tau_a)·a+(1/tau_a)·a_prop`。
7. 输出：`y=x_cur·(1+lam·a)`。

以上更新发生在同一次前向传播内部；`model.attn_map` 保存的是**已分离计算图**的最终注意力图，可用于可视化，但损失经输出 `y` 反传仍能更新模块参数。

## 集成 Ultralytics / YOLO26

上传代码的 YAML 用法示例：

```yaml
- [[15, 10], 1, TCPELIA, [64, 2]]
```

这行不能直接用于未经修改的 Ultralytics 安装。需要在相应版本的模块注册和 `parse_model` 中支持**两个输入特征、两个对应输入通道**，正确设置输出通道 `c2=c_x`；层索引 `15,10` 必须根据具体网络结构核对。完整的逻辑边界与示意修补方式见 [集成指南](docs/ultralytics_integration.md)。

## 仓库文件

- `tcpelia/attention.py`：上传的核心计算实现，保留其参数、迭代公式与输出逻辑。
- `examples/quickstart.py`：可运行的前向/反向传播示例。
- `examples/visualize_attention.py`：**随机特征**上的注意力可视化演示（不代表真实训练结果）。
- `tests/test_tcpelia.py`：形状、梯度、注意力范围、卷积核、权重序列化等测试。
- `docs/ultralytics_integration.md`：Ultralytics 改造提示，**不是已测试的完整插件**。

## GitHub 发布说明

Windows PowerShell 上传教程：[如何发布仓库](docs/publish_to_github.md)。发布前建议逐项检查 [发布检查清单](docs/release_checklist.md)。

## 论文引用与许可证

论文作者、正式题目、发表情况、DOI 尚未从此次上传的代码确认，因此本仓库不虚构 BibTeX 论文条目。建议论文确定后补充 `CITATION.cff` 或 BibTeX。

目前附带的是**建议版 MIT 许可证**。正式发布前应核实项目归属、合作者授权、软件许可选择，并根据需要修改 `LICENSE` 的版权主体。参考 [发布检查清单](docs/release_checklist.md)。
