# Chat Screenshot Mirroring Tool

一个基于 OpenCV / NumPy / Pillow 的聊天截图重排原型。

灵感来源：

来自“求真书院”舆论贴，由于是第一视角截图

曝光者很快受到处分

不讨论事件本身的正确与否

但是希望曝光者可以隐藏一下自己，分享聊天记录时把自己变成“观察者”

从而降低曝光风险和保护自身权益

希望能够被正确使用

它的主要功能是：

- 识别右侧绿色聊天气泡
- 识别对应头像
- 将聊天对象镜像到左侧
- 将绿色气泡重绘为白色气泡
- 可选在头像上方添加昵称

当前代码更接近“可工作的算法原型”，而不是完整产品。公开仓库默认只保留核心实现和展示资源；本地调试时用到的大量 `test_*` / `debug_*` / `measure_*` 脚本没有一并公开。

## 使用说明

在使用时，使用者可以先准备一份微信原聊天记录，可以选择是否带昵称

<table>
  <tr>
    <td align="center" colspan="2">
    <strong>在“显示群成员昵称处”选择是否显示昵称</strong><br/>
    <img src="assets/readme/usage-toggle-nickname.PNG" width="360" />
<table>
  <tr>
    <td align="center" colspan="2">
    <strong>PIC1:带昵称</strong><br/>
    <img src="assets/readme/example-with-nickname.jpg" width="360" />


<table>
  <tr>
    <td align="center" colspan="2">
    <strong>PIC2:不带昵称</strong><br/>
    <img src="assets/readme/example-without-nickname.jpg" width="360" />

在使用时，可以通过在命令中选择yesnickname或者nonickname来知会程序是否要处理带昵称的聊天记录

以PIC1和PIC2分别举例

PIC1的运行命令为：

python3 main.py yesnickname Tony TEST.PNG output/result_with_name.png


<table>
  <tr>
    <td align="center" colspan="2">
    <strong>PIC1运行示例</strong><br/>
    <img src="assets/readme/run-with-nickname.png" width="5000" />

PIC2的运行命令为：

python3 main.py nonickname TEST.PNG output/result.png

<table>
  <tr>
    <td align="center" colspan="2">
    <strong>PIC2运行示例</strong><br/>
    <img src="assets/readme/run-without-nickname.png" width="5000" />

注：此处TEST.PNG为待处理的图片，output/result.png为处理后的图片，Tony为昵称

可以替换为自己想处理的图片路径以及昵称

另外，如果不带nonickname和yesnickname，默认无昵称

python3 main.py TEST.PNG output/result.png





## 效果展示
<table>
  <tr>
    <td align="center" colspan="2">
    <strong>PIC1:带昵称</strong><br/>
    <img src="assets/readme/result-with-nickname.png" width="5000" />

<table>
  <tr>
    <td align="center" colspan="2">
    <strong>PIC2:不带昵称</strong><br/>
    <img src="assets/readme/result-without-nickname.png" width="5000" />


<table>
  <tr>
    <td align="center">
      <strong>真实参考图：单人</strong><br/>
      <img src="real_example/real_example_of_single.png" width="260" />
    </td>
    <td align="center">
      <strong>生成结果：单人</strong><br/>
      <img src="output/output_of_single.png" width="260" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>真实参考图：多人连续消息</strong><br/>
      <img src="real_example/real_example_of_connection_with_others.png" width="260" />
    </td>
    <td align="center">
      <strong>生成结果：多人连续消息</strong><br/>
      <img src="output/output_of_connection_with_others.png" width="260" />
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

## 用法

不加昵称：

```bash
python3 main.py nonickname TEST.PNG output/result.png
```

加昵称：

```bash
python3 main.py yesnickname Tony TEST.PNG output/result_with_name.png
```

也支持最简参数形式：

```bash
python3 main.py TEST.PNG output/result.png
```

程序运行后会在终端输出一些诊断信息，例如：

- 中线位置
- 检测到的气泡数量
- 检测到的头像数量
- 是否启用昵称模式

## 仓库结构

```text
.
├── main.py                  # Python 主实现
├── C++/main.cpp             # C++ 对照实现
├── real_example/            # 参考效果图
├── output/                  # 示例输出图
├── compare.html             # 本地效果对比页
├── LICENSE
├── .gitignore
└── README.md
```

## 当前限制

- 参数仅仅偏向被修改方的纯文本聊天，即发送表情等特殊消息时，会难以检索，尚需要进一步升级，临时想法是通过第一句出现的文字检索到头像，再利用头像本身倒推被修改人发送的全部信息
- 昵称字号字体颜色仍有待调整和限制，昵称排版目前是启发式规则，不是完整的字体测量与版式系统。同时，在测试的过程中发现不同人的微信之间的昵称字号与颜色并不相同，且尚未确定原因（调整iphone系统文字大小，微信系统文字大小仍无法统一昵称的字号，手机型号统一，微信版本统一）由于本人微信显示为相对粗体，所以用粗体处理yesnickname中的昵称，如果昵称不是粗体，则需要调整代码
<table>
  <tr>
    <td align="center">
      <strong>来自朋友手机的截图</strong><br/>
      <img src="assets/readme/nickname-friend-phone.png" width="200" />
    </td>
    <td align="center">
      <strong>来自本人手机的截图</strong><br/>
      <img src="assets/readme/nickname-my-phone.png" width="200" />
    </td>
  </tr>
<table>
可见昵称的字体大小和粗细并不一致，甚至可以说有很大的差别



- 本地实验脚本和校准脚本未纳入公开仓库，也不构成正式测试体系。
- 对背景的多样性产生的可能影响并未测试，目前只在白色背景下测试，预计背景的不同会影响程序的精准度，尤其是多彩（绿色）背景
- 当前仅支持文字的单一字体，不支持第三方字体，且开发与测试均基于iphone，尚未测试其他设备
## 版本说明

- `main.py` 是当前主要维护版本。
- `C++/main.cpp` 保留了更完整的早期实现思路，适合作为算法对照参考。

## 后续可改进方向

- 增加 `requirements.txt` 和更明确的环境说明
- 将 `main.py` 拆分成检测、重建、排版等模块
- 引入更加多元化的测试样例，包括但不限于不同背景，不同字体，不同消息类型等
- 对多种类型的消息进行处理，包括但不限于表情，图片，视频，文件等
- 将硬编码的字体路径改成可配置项，并支持多字体与不同类型的设备
