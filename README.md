# ContextShift

面向代码讲解、技术会议和面试的实时短提示助手。音频在本地通过
FSMN-VAD / SenseVoice 转写；云端模型只接收文本。模型建议不进入真实对话历史。

## 本地运行

```sh
uv sync --extra speech
uv run contextshift audio-devices
uv run contextshift live --input-device MIC_INDEX --system-device LOOPBACK_INDEX
```

将 `MIC_INDEX` 替换为候选人麦克风编号，将 `LOOPBACK_INDEX` 替换为承载会议声音的
回环输入设备编号。两者必须是独立输入；系统音频设备必须显式指定。
回环通道应只包含会议对方声音，避免把自己的麦克风混入其中。

默认模型目录：`models/asr/SenseVoiceSmall`、`models/vad/fsmn-vad`。
可用 `--asr-model`、`--vad-model` 指定已有本地模型；云端配置沿用
`DEEPSEEK_API_KEY`。`--input-device` 省略时使用默认麦克风，若它与回环设备相同则拒绝启动。

当前系统通道使用 **已经配置好的 PortAudio 回环输入**，要求支持单声道、16 kHz。
本补丁不创建回环设备，也不包含 macOS 原生系统音频捕获或自动重采样。
后续平台采集适配器应沿用相同的 `source` / `role` 事件协议。

## 音频路由

| 事件来源 | 角色 | 讲话开始 | 最终转写 |
| --- | --- | --- | --- |
| `microphone` | `candidate` | 记录讲话状态，继续现有建议 | 仅写入候选人上下文 |
| `system` | `interviewer` | 同步取消并使旧模型输出失效 | 写入面试官上下文，最新问题触发建议 |

每路拥有独立的采集流、VAD 状态、语句编号、ASR 实例和转写队列。
语句编号只在各自来源内比较；每路最终转写按队列顺序交付，重复或已处理的编号被忽略。
系统新问题开始后，旧问题的转写可进入真实历史，但不会启动旧问题的模型请求。
两路中任一路异常退出时，另一通道也会关闭；停止时不会继续提交积压的转写。

模型输入保留真实发言的角色标签，使用请求构建时已完成转写的最近上下文。
尚未完成 ASR 的候选人发言不会被猜测或等待，也不会在完成后自动重启正在生成的建议。
生成建议仅经 `Session` 的流式限制器展示，不回写任何角色的历史。

`listen` 仍是独立麦克风本地转写命令；`interactive` 仍接受手工输入的问题。

## 验证

```sh
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

路由和取消行为使用模拟音频、ASR 与 Provider 验证，不需要真实模型或云端请求。
真实设备兼容性、端到端延迟及 macOS 采集需要在目标机器上另行测量。
