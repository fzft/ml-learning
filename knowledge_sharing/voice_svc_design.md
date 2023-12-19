### Text-to-Speech and Speech-to-Text Interfaces
这个类图展示了音频录制基类及其继承的文本转语音和语音转文本接口。每个接口包含了相关的方法，如添加声音、获取声音列表、转录音频等。

```mermaid
classDiagram
    class AudioRecordingBase {
      +recordAudio() bytes
      +saveAudioFile(audioData) str
      +deleteAudioFile(fileID) bool
    }

    class TextToSpeechInterface {
      <<interface>>
      +addVoice(voiceName, inputFile, labels, description) str
      +getVoices() list
      +getVoice(voiceID) dict
      +deleteVoice(voiceID) bool
      +editVoice(voiceID, newDetails) bool
    }

    class SpeechToTextInterface {
      <<interface>>
      +transcribe(audioFile, language, additionalParams) str
      +getTranscriptionHistory(userID) list
      +getTranscription(transcriptionID) dict
      +deleteTranscription(transcriptionID) bool
    }

    class LocalModelTextToSpeech {
      +addVoice(voiceName, inputFile, labels, description) str
      +getVoices() list
      +getVoice(voiceID) dict
      +deleteVoice(voiceID) bool
      +editVoice(voiceID, newDetails) bool
    }

    class ThirdPartyAPITextToSpeech {
      +addVoice(voiceName, inputFile, labels, description) str
      +getVoices() list
      +getVoice(voiceID) dict
      +deleteVoice(voiceID) bool
      +editVoice(voiceID, newDetails) bool
    }

    class LocalModelSpeechToText {
      +transcribe(audioFile, language, additionalParams) str
      +getTranscriptionHistory(sessionID) list
      +getTranscription(messageID) dict
      +deleteTranscription(sessionID|messageID) bool
    }

    class ThirdPartyAPISpeechToText {
      +transcribe(audioFile, language, additionalParams) str
      +getTranscriptionHistory(sessionID) list
      +getTranscription(messageID) dict
      +deleteTranscription(sessionID|messageID) bool
    }

    AudioRecordingBase <|-- TextToSpeechInterface
    AudioRecordingBase <|-- SpeechToTextInterface
    TextToSpeechInterface <|-- LocalModelTextToSpeech
    TextToSpeechInterface <|-- ThirdPartyAPITextToSpeech
    SpeechToTextInterface <|-- LocalModelSpeechToText
    SpeechToTextInterface <|-- ThirdPartyAPISpeechToText

```

### Data Storage in MySQL and S3
此类图描述了在 MySQL 和 S3 中存储数据的结构。MySQL 存储会话和消息的元数据，而 S3 用于存储音频文件。会话和消息之间的关系也被表示出来。

```mermaid
classDiagram
    class MySQL {
        +Sessions
        +Messages
        +VoiceModels
    }

    class S3 {
        +AudioFiles
    }

    class Session {
        +SessionID str
        +VoiceID str
        +StartTime datetime
        +EndTime datetime
    }

    class Message {
        +MessageID str
        +SessionID str
        +Timestamp datetime
        +AudioFileURL str
        +Transcription str
        +AdditionalMetadata str
    }

    class VoiceModel {
        +ModelID str
        +VoiceID str
        +ModelFilePath str
        +TrainingAudioFileURL str
        +AdditionalMetadata str
        +LocalTraining bool
    }

    MySQL --|> Session : stores
    MySQL --|> Message : stores
    MySQL --|> VoiceModel : stores
    S3 --|> Message : stores audio files referenced in
    S3 --|> VoiceModel : stores training audio files
```

### Asynchronous Voice Character Addition
此顺序图展示了异步添加声音特征的过程。用户请求添加声音后，API 将任务加入消息队列，然后由工作处理器处理。用户还可以查询任务的状态。
```mermaid
sequenceDiagram
    participant User
    participant API
    participant JobQueue
    participant VoiceService

    User->>API: Request to Add Voice Character
    API->>JobQueue: Spawn Add Voice Job
    JobQueue->>VoiceService: Process Voice Addition
    API-->>User: Job Spawned (Job ID)

    User->>API: Query Job Status
    API->>JobQueue: Check Job Status
    JobQueue-->>API: Job Status
    API-->>User: Job Status (Done/In Progress)
```
### TTS and STT Processes with S3 Storage
这个顺序图说明了文本转语音和语音转文本过程中与 S3 存储的交互。在这两个过程中，生成的音频文件被存储在 S3 中，并返回相关的消息 ID 和音频 URL。
```mermaid
sequenceDiagram
    participant 用户 as User
    participant TTS_API as TTS_API
    participant STT_API as STT_API
    participant S3 as S3

    用户->>TTS_API: 发送文本及会话信息 (Send Text with Session Info)
    TTS_API->>S3: 存储音频文件 (Store Audio File)
    TTS_API-->>用户: 返回消息ID和音频URL (Return MessageID & Audio URL)

    用户->>STT_API: 发送音频及会话信息 (Send Audio with Session Info)
    STT_API->>S3: 存储音频文件 (Store Audio File)
    STT_API-->>用户: 返回消息ID和转录文本 (Return MessageID & Transcription)

```
### Retrieving Audio History with Pagination
用户通过 sessionID 和 userID 请求批量历史音频数据，可能包括分页信息。
API 返回相关音频文件的批量数据，包括音频URL、转录文本和元数据。
对于单个消息，用户提供 messageID，API 返回相应的音频文件数据
```mermaid
sequenceDiagram
    participant 用户 as User
    participant AudioHistoryAPI as AudioHistoryAPI

    用户->>AudioHistoryAPI: 请求批量历史音频 (Request Batch Audio History)
    Note over AudioHistoryAPI: 参数: sessionID, voiceID, 分页信息
    AudioHistoryAPI->>用户: 返回音频文件批量数据 (Return Batch Audio Data)

    用户->>AudioHistoryAPI: 请求单个消息音频 (Request Single Message Audio)
    Note over AudioHistoryAPI: 参数: messageID
    AudioHistoryAPI->>用户: 返回单个音频文件数据 (Return Single Audio Data)
```
### Deleting Messages
这个顺序图展示了删除消息的API调用流程。用户可以请求删除特定会话中的所有消息或删除特定的单个消息。
```mermaid
sequenceDiagram
    participant 用户 as User
    participant API as API

    用户->>API: 请求删除会话中的消息 (Request to Delete Messages in Session)
    Note over API: 参数: sessionID
    API-->>用户: 确认消息已删除 (Confirm Messages Deleted)

    用户->>API: 请求删除特定消息 (Request to Delete Specific Message)
    Note over API: 参数: messageID
    API-->>用户: 确认消息已删除 (Confirm Message Deleted)
```
