# 德州扑克 CLI 决策器

本项目是一个本地德州扑克练习/决策工具。它有两类入口：

- 人用练习 UI：`python gto.py ui`
- 程序接口 JSON：`python gto.py deal`、`python gto.py stream`

## 安装

基础的屏幕识别、复盘和测试环境：

```powershell
python -m pip install -r requirements.txt
```

该命令也会安装文字识别组件，用于读取底池、跟注额和操作按钮文字。安装完成后可用下面的命令确认：

```powershell
python -c "from rapidocr_onnxruntime import RapidOCR; print('OCR OK')"
```

需要运行深度牌面模型或 Hugging Face 教师模型时，再安装：

```powershell
python -m pip install -r requirements-ml.txt
```

现场录像、截图、人工标注队列、校准坐标和本地模型均不会提交到仓库；它们会生成在 `video_frames` 下。

`deal` 默认输出 JSON，是给后续 UI、CV、脚本调用的接口；如果你只是想练习，不要直接用 `deal`，用 `ui`。

## 直接开始练习

如果完全没接触过，先看规则速查：

```powershell
python gto.py rules
```

然后进入练习：

```powershell
python gto.py ui
```

如果想用网页玩：

```powershell
python gto.py web
```

然后打开：

```text
http://127.0.0.1:8765/
```

网页里有两个模式：

- `单题练习`: 一道一道练决策。简单档比较单一，中等/高级会出现更多局面。
- `完整对局`: 从拿到两张牌开始，一路和电脑玩到结束。

德州扑克不能换牌。你的两张手牌从开始到结束都不变，后面只会往桌上发大家都能用的公共牌。

进去后会让你选择难度：

- `simple` / 简单：新手档，主要练翻前清晰点
- `medium` / 中等：加入翻牌权益和底池赔率
- `advanced` / 高级：加入转河、3bet/4bet、更多筹码深度
- `master` / 大师：全随机高混合点

每道题都会先显示规则提示和思考提示。答题时可以输入：

- `fold` / `call` / `raise` / `check` / `bet` / `limp`: 选择动作
- `h`: 重新显示提示
- `q`: 退出练习

也可以不用菜单，直接开始：

```powershell
python gto.py practice --level simple -n 10
python gto.py practice --level medium -n 20
python gto.py practice --level advanced -n 20
python gto.py practice --level master -n 30
```

## 生成单题

给人看：

```powershell
python gto.py deal --level simple --with-answer --format text
```

给程序/UI/CV 看：

```powershell
python gto.py deal --level simple
python gto.py deal --level medium --with-answer --compact
```

## CV 接入

CV 每识别到一个局面，就向 stdin 写入一行 JSON；CLI 会返回一行 JSON 决策。

```powershell
python gto.py stream
```

输入示例：

```json
{"hero":{"cards":["As","Ks"],"position":"BTN","stack_bb":100},"table":{"pot_bb":1.5,"to_call_bb":0,"effective_stack_bb":100,"board":[]},"action":{"scenario":"rfi","street":"preflop"},"villain":{"profile":"standard"}}
```

### 截图识别 D 庄家按钮

`pict/D.png` 是 D 庄家按钮模板。截图里你的 UI 固定在最下面，程序会识别 D 的位置，把最近的座位当作庄家，再按顺时针判断你离庄家几格，以及翻前/翻后你第几个行动。

```powershell
python gto.py cv pict/aaeb2a5d-1789-4754-bf09-9e19c6e2112b.png --format text
python gto.py cv pict/d77de0d8-8e84-4d66-8d6b-d32a667ce257.png --annotate pict/out.png
```

默认按 8 人桌识别；如果你的桌子是 6 人桌：

```powershell
python gto.py cv your-table.png --seats 6
```

长视频可以直接抽帧分析。下面这个命令会只取视频中间 25%-75%，每 5 秒抽一帧，保存原始帧、标注帧、`analysis.json` 和 `summary.csv`：

```powershell
python gto.py video-cv "C:\path\to\table.mp4" --middle --every 5 --output-dir video_frames/run1
```

输出里会包含每一帧的庄家座位、你离庄家的顺时针距离、翻前/翻后行动顺序、每个座位是否还持牌、底池金额、识别到的下注金额，以及可见的我方手牌/公共牌。实际有人弃牌后，行动顺序需要跳过 `folded_or_empty` 的座位。牌面里的 `?` 表示花色或牌面置信度不足，程序不会强行猜。

## 单次决策

```powershell
python gto.py advise --state samples/preflop_btn_rfi.json --format text
python gto.py advise --state samples/flop_facing_bet.json
```

## 输入字段

```powershell
python gto.py schema
```

核心字段：

- `hero.cards`: Hero 手牌，例如 `["As", "Ks"]`
- `hero.position`: `UTG`、`HJ`、`CO`、`BTN`、`SB`、`BB`
- `table.board`: 公牌；翻前为空，翻牌三张，转牌四张，河牌五张
- `table.pot_bb`: 当前底池，单位 BB
- `table.to_call_bb`: 当前需要跟注的额度；无人下注时为 `0`
- `table.effective_stack_bb`: 有效筹码，单位 BB
- `action.scenario`: `rfi`、`vs_open`、`vs_3bet`
- `villain.profile`: `tight`、`standard`、`wide`、`current`
- `practice.level`: 模拟盘难度
- `seed`: 可选，固定随机种子，方便复盘

## 模型边界

当前不是商业 solver 输出。翻前是位置/场景/筹码深度的启发式范围模型；翻后是牌力评估、对手范围抽样、蒙特卡洛权益和底池赔率组合出的近似建议。适合朋友私局、训练、复盘和模拟。
