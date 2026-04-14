# Chat Screenshot Mirroring Tool

一个基于 OpenCV / NumPy / Pillow 的聊天截图重排原型。

这个项目的目标是把聊天截图中的“自己”隐藏成观察者视角：  
识别右侧绿色聊天气泡与对应头像，将其镜像到左侧，重绘为白色气泡，并支持可选昵称。

## 项目动机

灵感来自一类第一视角聊天截图场景。

- 不讨论事件本身的是非
- 重点是降低分享聊天记录时暴露自身身份的风险
- 希望在保留聊天内容的同时，把分享者处理成“观察者”视角

希望这个工具被谨慎且正确地使用。

## 主要功能

- 识别右侧绿色聊天气泡
- 识别对应头像
- 将聊天对象镜像到左侧
- 将绿色气泡重绘为白色气泡
- 可选在头像上方添加昵称

当前代码更接近“可工作的算法原型”，而不是完整产品。公开仓库默认只保留核心实现和展示资源；本地调试时用到的大量 `test_*` / `debug_*` / `measure_*` 脚本没有一并公开。

## 使用说明

### 1. 选择是否显示昵称

使用前，可以先在微信中决定是否开启“显示群成员昵称”。

<p align="center">
  <img src="assets/readme/usage-toggle-nickname.PNG" alt="显示群成员昵称设置" width="360" />
</p>

### 2. 准备输入图片

下面分别是带昵称和不带昵称的输入示例：

<table>
  <tr>
    <td align="center">
      <strong>PIC1：带昵称</strong><br/>
      <img src="assets/readme/example-with-nickname.jpg" alt="带昵称输入示例" width="320" />
    </td>
    <td align="center">
      <strong>PIC2：不带昵称</strong><br/>
      <img src="assets/readme/example-without-nickname.jpg" alt="不带昵称输入示例" width="320" />
    </td>
  </tr>
</table>

### 3. 运行命令

程序通过 `yesnickname` 或 `nonickname` 来区分输入是否带昵称。

带昵称示例命令：

```bash
python3 main.py yesnickname Tony TEST.PNG output/result_with_name.png
```

<p align="center">
  <img src="assets/readme/run-with-nickname.png" alt="带昵称运行示例" width="760" />
</p>

不带昵称示例命令：

```bash
python3 main.py nonickname TEST.PNG output/result.png
```

<p align="center">
  <img src="assets/readme/run-without-nickname.png" alt="不带昵称运行示例" width="760" />
</p>

说明：

- `TEST.PNG` 为待处理图片
- `output/result.png` 为处理后的输出图片
- `Tony` 为昵称示例，可替换为你自己的昵称

如果不写 `nonickname` 或 `yesnickname`，程序默认按无昵称模式处理：

```bash
python3 main.py TEST.PNG output/result.png
```

## 效果展示

### 处理结果

<table>
  <tr>
    <td align="center">
      <strong>PIC1：带昵称结果</strong><br/>
      <img src="assets/readme/result-with-nickname.png" alt="带昵称处理结果" width="360" />
    </td>
    <td align="center">
      <strong>PIC2：不带昵称结果</strong><br/>
      <img src="assets/readme/result-without-nickname.png" alt="不带昵称处理结果" width="360" />
    </td>
  </tr>
</table>

### 参考图与生成结果对比

<table>
  <tr>
    <td align="center">
      <strong>真实参考图：单人</strong><br/>
      <img src="real_example/real_example_of_single.png" alt="真实参考图 单人" width="260" />
    </td>
    <td align="center">
      <strong>生成结果：单人</strong><br/>
      <img src="output/output_of_single.png" alt="生成结果 单人" width="260" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>真实参考图：多人连续消息</strong><br/>
      <img src="real_example/real_example_of_connection_with_others.png" alt="真实参考图 多人连续消息" width="260" />
    </td>
    <td align="center">
      <strong>生成结果：多人连续消息</strong><br/>
      <img src="output/output_of_connection_with_others.png" alt="生成结果 多人连续消息" width="260" />
    </td>
  </tr>
</table>

也可以直接打开 [compare.html](compare.html) 做本地对比查看。

## 核心流程

主入口在 [`main.py`](main.py)。

算法大致分成 4 步：

1. 使用 HSV 阈值检测右侧绿色聊天气泡。
2. 在气泡附近搜索头像，并提取头像可见区域。
3. 擦除原始头像和气泡区域，用背景 patch 回填。
4. 将头像和气泡镜像到左侧，同时重绘白色气泡；若开启昵称，则在消息组首条上方绘制昵称。

为了避免气泡整体翻转后出现反字，代码会单独提取原气泡内文字区域，并在镜像后的气泡上重新贴回可读内容。

## 运行环境

- Python 3.10+
- macOS 推荐

主要依赖：

- `opencv-python`
- `numpy`
- `Pillow`

安装示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy Pillow
```

说明：

- 昵称模式默认优先使用 macOS 系统字体路径，中文昵称在 macOS 上效果更稳定。
- 如果找不到对应字体，ASCII 文本会退回到 OpenCV 文本绘制，效果会差一些。

## 仓库结构

```text
.
├── main.py                  # Python 主实现
├── C++/main.cpp             # C++ 对照实现
├── assets/readme/           # README 展示用图片
├── real_example/            # 参考效果图
├── output/                  # 示例输出图
├── compare.html             # 本地效果对比页
├── LICENSE
├── .gitignore
└── README.md
```

## 当前限制

- 参数目前主要偏向“纯文本聊天”场景；发送表情、图片、视频、文件等特殊消息时，检测与重排的稳定性不足。
- 昵称字号、颜色和粗细目前仍依赖启发式规则，不是完整的字体测量与版式系统。
- 在测试中发现，不同设备上的微信昵称粗细和字号并不完全一致，即使手机型号和微信版本相同，显示效果也可能不同。
- 本地实验脚本和校准脚本未纳入公开仓库，也不构成正式测试体系。
- 对背景多样性影响的测试还不充分，目前主要在浅色背景下验证；复杂或多彩背景预计会降低准确率。
- 当前仅支持较有限的系统字体路径，尚未完整支持第三方字体，也尚未系统测试非 iPhone 设备。

昵称显示差异示例：

<table>
  <tr>
    <td align="center">
      <strong>来自朋友手机的截图</strong><br/>
      <img src="assets/readme/nickname-friend-phone.png" alt="朋友手机昵称显示" width="220" />
    </td>
    <td align="center">
      <strong>来自本人手机的截图</strong><br/>
      <img src="assets/readme/nickname-my-phone.png" alt="本人手机昵称显示" width="220" />
    </td>
  </tr>
</table>

可见昵称的字体大小和粗细并不完全一致，差异比较明显。

## 版本说明

- `main.py` 是当前主要维护版本。
- `C++/main.cpp` 保留了更完整的早期实现思路，适合作为算法对照参考。

## 后续可改进方向

- 增加 `requirements.txt` 和更明确的环境说明
- 将 `main.py` 拆分成检测、重建、排版等模块
- 引入更多元化的测试样例，包括不同背景、不同字体、不同消息类型
- 对更多消息类型进行处理，包括表情、图片、视频、文件等
- 将硬编码的字体路径改成可配置项，并支持多字体与不同类型设备
