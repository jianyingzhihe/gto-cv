# Realtime CV State Stream

这份文档描述当前的实时牌局状态流原型。腾讯会议截图源还没接入时，先用视频模拟实时输入。

## Commands

快速模式不跑 OCR，适合先看庄家、位置、手牌、公共牌、弃牌状态：

```powershell
python gto.py live-cv "C:\path\to\table.mp4" --start 309 --end 390 --every 1 --trigger state-change --no-ocr --output-dir video_frames/live_fast
```

完整模式会跑 OCR，额外输出底池和下注金额，但速度会慢一些：

```powershell
python gto.py live-cv "C:\path\to\table.mp4" --start 309 --end 329 --every 5 --trigger state-change --output-dir video_frames/live_ocr
```

输出每个采样帧，而不是只在状态变化时输出：

```powershell
python gto.py live-cv "C:\path\to\table.mp4" --start 309 --end 329 --every 1 --trigger frame --no-ocr --output-dir video_frames/live_every_frame
```

## Output Files

- `events.jsonl`: 实时事件流，一行一个状态，适合程序消费
- `events.json`: 同样事件的格式化 JSON 数组，适合人工查看
- `current_state.json`: 最近一次状态
- `realtime_summary.json`: 本次运行的统计和文件路径

## JSON Shape

每个事件是一份完整状态：

```json
{
  "ok": true,
  "source": {
    "kind": "video",
    "path": "C:\\path\\to\\table.mp4",
    "timestamp_sec": 319.0,
    "frame_index": 6061,
    "sample_index": 2,
    "frame_path": "",
    "annotated_path": ""
  },
  "event": {
    "index": 2,
    "trigger": "state-change",
    "reason": "state_changed",
    "signature": "..."
  },
  "table": {
    "seat_count": 8,
    "street": "flop",
    "dealer_seat_index": 4,
    "dealer_seat": "top",
    "dealer_position": "BTN",
    "pot_bb": 12.8,
    "to_call_bb": 0.0,
    "board": ["Kh", "9d", "4s"]
  },
  "hero": {
    "seat_index": 0,
    "seat": "bottom_hero",
    "position": "UTG+1",
    "gto_position": "UTG",
    "cards": ["Qs", "Ah"],
    "distance_from_dealer_clockwise": 4,
    "preflop_action_order": 2,
    "postflop_action_order": 4,
    "status": "active_or_showdown",
    "has_cards": true,
    "bet_bb": null
  },
  "seats": [],
  "bets": [],
  "confidence": {
    "dealer_button": 0.9994,
    "pot_ocr": 0.859,
    "cards": {
      "hero": [],
      "board": []
    }
  }
}
```

## Tencent Meeting Integration Point

后面接腾讯会议时，保持 `build_realtime_state()` 输出结构不变，只需要替换帧来源：

1. 当前：`cv2.VideoCapture(video_path)` 读取视频帧。
2. 下一步：从腾讯会议窗口截图得到同样的 OpenCV BGR frame。
3. 继续调用 `analyze_video_frame(frame, template, ocr=...)`。
4. 用 `state_signature()` 判断是否触发新事件。

这样下游只需要订阅 `events.jsonl` 或读取 `current_state.json`，不需要关心帧来自视频还是腾讯会议截图。

## Realtime Performance Notes

`live-cv --trigger frame` now uses the realtime fast path:

- Sequential video reads instead of random seeking.
- Dealer button cache. Frame mode defaults to refreshing dealer detection every 30 sampled frames.
- Hero/board card ROI cache. Card templates are only re-run when the card regions visually change.
- OCR remains optional. Use `--no-ocr` for high-frequency frame mode, and refresh OCR from `visual-change` or a slower background pass.

Example:

```powershell
python gto.py live-cv "C:\path\to\table.mp4" --start 950 --every 0.0526316 --max-frames 115 --trigger frame --no-ocr --output-dir video_frames/frame_fast
```

Measured on the `d68958e...mp4` test clip:

- Before optimization: about `1119 ms/frame` for visual-only frame detection.
- After dealer + card ROI caching: `68.63 ms/frame` across 115 real source frames.
- Short stable segment: `41.71 ms/frame` across 60 frames.

If the dealer cannot change during a short hand segment, `--dealer-refresh-frames 9999` avoids periodic dealer re-detection. Keep the default or lower value across hand boundaries.
