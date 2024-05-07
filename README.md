# GPT WebBot

GPT WebBot 是一個基於 Azure OpenAI 模型的網頁界面，允許用戶透過瀏覽器與之互動，進行即時的對話、代碼解釋和圖像分析。

## 專案目的

此專案旨在創建一個簡單易用的網頁界面讓用戶能夠進行以下操作：

- 與多個配置的 GPT 模型進行互動
- 通過文字進行對話，求助於 AI 進行問題解答
- 使用Assistant，傳遞代碼並得到執行結果
- 透過視覺模型解讀圖片內容

## 程式架構

專案包含以下主要檔案：

- `web_gpt.py` - 主要的 Web 伺服器文件，用於設置和運行 Gradio 網頁界面。
- `call_gpt.py` - 定義 `ChatGPT` 類，處理與 GPT 模型的通信和回應邏輯。
- `model_config.json.example` - 模型配置範例文件，用戶須根據自己的配置需求進行修改，並將檔案重新命名為 `model_config.json `。
- `user_config.json.example` - 登入帳號與密碼配置處，以及不同使用者的System message內容，其中需保留 `default`，修改後將檔案重新命名為 `user_config.json `。
- `requirements.txt` - 列舉了進行專案所需的所有 Python 依賴包。
- 專案中使用的GPT模型都以Azure OpenAI部署。

## 使用者手冊

### 安裝配置

建議使用Anaconda管理環境：
- `python `：version = 3.11
- `openai `：version = 1.20.0

建立conda環境：
```bash
conda create -n web-gpt-env python=3.11
conda activate web-gpt-env
```

安裝所需的依賴包，可以通過以下命令完成：

```bash
python -m pip install -U pip
```

```bash
pip install -r requirements.txt
```

### 使用說明

- 運行後會在終端機持續運行。
- 網頁模組以gradio套件建立。
- GPT-3.5 Turbo模型響應速度較快。
- GPT-4 Turbo模型較新回應效果佳，能夠接受較大的輸入，響應速度較差。
- GPT-4 Vison可以輸入圖片進行分析，加上Computer Vision進行 OCR與Object detection。
- Dall-E-3能夠根據輸入的prompt生成圖片，並且自定義參數。
- Assistant可以撰寫代碼並執行，目前以Python與JavaScript成果較佳。
- 模型會輸入過往所有歷史紀錄，因此建議定期清空歷史紀錄，避免響應時間過長。
- 使用者可以自定義System message使模型扮演不同角色，並且可以儲存與刪除。
- Max tokens能夠限制模型回應的token數量，設置較低的值能夠加快模型響應速度，但可能會導致模型回應被截斷。
