## 8. API 接口使用说明

服务运行后，您可以通过 HTTP 请求调用以下接口。

### 8.1 语音合成

- **功能**: 将指定的文本转换为语音文件（Base64编码）。
- **URL**: `/api/tts`
- **方法**: `POST`
- **请求体 (JSON)**:

```json
{
  "text": "你好，欢迎使用语音合成服务。",
  "voice": "longxiaochun_v2",
  "rate": 1,
  "volume": 50
}
```

- **参数说明**:
  - `text` (string, 必需): 需要合成的文本内容。
  - `voice` (string, 必需): 使用的音色。可以是内置音色（如 `longxiaochun_v2 知性积极女，longshu_v2 沉稳青年男`）或您自己克隆的音色ID。
  - `rate` (integer, 可选): 语速，取值范围参考阿里云文档，默认为1。
  - `volume` (integer, 可选): 音量，取值范围参考阿里云文档，默认为50。

- **成功响应 (JSON)**:

```json
{
  "audio_base64": "UklGRi...<音频文件的Base64编码数据>..."
}
```

### 8.2 克隆音色

- **功能**: 通过一个 base64音频 克隆一个新的音色。
- **URL**: `/api/clone-voice`
- **方法**: `POST`
- **请求体 (JSON)**:

```json
{
  "audio_base64": "",
  "prefix": "myvoice"
}
```

- **参数说明**:
  - `audio_base64` (string, 必需): 用于复刻音色的音频文件base64格式。
  - `prefix` (string, 必需): 音色的自定义前缀，仅允许数字和小写字母，小于十个字符。

- **成功响应 (JSON)**:

```json
{
  "voice_id": "index-tts-v1.5-prefix-myvoice-xxxxxxxx"
}
```

### 8.3 获取所有克隆的音色

- **功能**: 返回数据库中存储的所有克隆音色列表。
- **URL**: `/api/voices`
- **方法**: `GET`
- **成功响应 (JSON)**:

```json
[
    {
        "id": 1,
        "voice_id": "index-tts-v1.5-prefix-myvoice-xxxxxxxx",
        "prefix": "myvoice",
        "status": "deploying",
        "created_at": "2023-10-27 10:00:00"
    }
]
```

### 8.4 删除指定的克隆音色

- **功能**: 根据音色ID删除一个克隆的音色。
- **URL**: `/api/voices/{voice_id}`
- **方法**: `DELETE`
- **URL参数**:
  - `voice_id` (string, 必需): 要删除的音色ID。

- **成功响应 (JSON)**:

```json
{
  "message": "音色删除成功"
}
```
