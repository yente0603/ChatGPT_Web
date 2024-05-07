import base64
import json
import requests


from openai import AzureOpenAI


import litellm
import traceback

from PIL import Image
import io



class ChatGPT:
    def __init__(self, model_config, init_system=None):
        """初始化ChatGPT的配置、訊息的queue以及初始系統訊息。

        Args:
            model_config (dict): 模型相關配置的字典。
            init_system (dict, optional): 初始化時的系統訊息。如果為None，則使用預設值。
        """
        if init_system is None:
            init_system = {
                "role": "system",
                "content": "你是一个人工智能助手，帮助人们查找信息。"
            }
        self.messages = [init_system]  # 初始化訊息列表，包含初始系統訊息
        self.model_config = model_config  # 儲存模型配置
        self.assistant_sys = """你是一個可以根據用戶問題撰寫代碼並執行的有用AI助手。請注意代碼與所有檔案名稱以及圖像的文字應該以英文命名與展示。敘述與說明的部分應該以繁體中文展示。
                            你擁有一個隔離的環境用於編寫和測試代碼。若是遇到編碼錯誤請優先考慮UTF-8的編碼。一個簡單的案例如下:
                            - 當要求你創建視覺化時，你應該遵循以下步驟：
                            1. 寫代碼。
                            2. 每次寫新代碼時顯示代碼的預覽，以展示你的工作。
                            3. 運行代碼以確認運行情況。
                            4. 如果代碼運行成功，顯示視覺化。
                            5. 如果代碼運行失敗，顯示錯誤消息，並嘗試修改代碼並重新運行，再次進行上述步驟。
                            - 當要求你打開一個檔案進行分析或是總結時，你應該遵循以下步驟：
                            1. 寫代碼確認檔案的格式，例如是pdf、word、excel、csv、txt、image或代碼檔案。
                            2. 確認檔案格式時也要確認檔案內的結構，確保不會有讀取失敗的行為。
                            3. 運行代碼以確認運行情況。
                            4. 根據所獲得的內容進行分析或是總結。
                            5. 如果代碼運行失敗，顯示錯誤消息，並嘗試修改代碼並重新運行，再次進行上述步驟。
                            """
        if model_config["model_name"] == "Assistants":
            self.use_model = "GPT-3.5 Turbo"  # 根據配置設定使用的模型
            self.client, self.assistant, self.thread = self.create_assistant()  # 設定並創建AI助手


    def _handle_default_model(self, question, max_tokens, image_path, user):
        """專門處理標準GPT模型流式輸出。
        
        Args:
            question (str): 用戶的提問。
            max_tokens (int): 最大 token 數。
            image_path (str, optional): 圖片的路徑，如果有的話。
            user (str): 使用者身份標識。
        """
        partial_message = ""
        if "翻譯" in self.messages[0]["content"]:
            self.messages.append({"role": "user", "content": "翻譯下列內容:\n\n#####"+question+"#####"})  # 增加翻譯請求的訊息
        else:
            self.messages.append({"role": "user", "content": question})  # 添加用戶的問題到訊息列表
        try:
            # 若有圖片路徑，處理圖片和訊息
            if image_path:
                message_content = [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{self._get_base64_from_image(path)}"} for path in image_path] + [{"type": "text", "text": question}]
                self.messages.append({"role": "user", "content": message_content})
                for chunk in AzureOpenAI(
                    api_key=self.model_config["key"], 
                    api_version=self.model_config["api-version"], 
                    base_url=self.model_config["endpoint"]+"/openai/deployments/"+self.model_config["deployment"]
                    ).chat.completions.create(
                        model=self.model_config["deployment"], 
                        messages=self.messages, 
                        max_tokens=max_tokens, 
                        stream=True
                        ):
                    if chunk.choices and chunk.choices[0].delta.content:
                        partial_message += chunk.choices[0].delta.content
                    yield partial_message
                self.messages.append({"role": "assistant", "content": partial_message})
                yield partial_message
            else:
                # 處理純文字訊息
                for chunk in litellm.completion(
                    model="azure/"+self.model_config["deployment"], 
                    api_base=self.model_config["endpoint"], 
                    api_key=self.model_config["key"], 
                    api_version=self.model_config["api-version"], 
                    max_tokens=max_tokens, 
                    messages=self.messages, 
                    stream=True, 
                    user=user
                    ):
                    if chunk['choices'][0]['delta']['content']:
                        partial_message += chunk['choices'][0]['delta']['content']
                    yield partial_message
                self.messages.append({"role": "assistant", "content": partial_message})
                yield partial_message
        except:
            e = f"error occurred: {traceback.format_exc()}"
            for i in e:
                partial_message += i
                yield partial_message

    def _handle_vision_model(self, question, max_tokens, image_path, user):
        """專門處理有視覺輸入的GPT模型。
        
        Args:
            question (str): 用戶的提問。
            max_tokens (int): 最大 token 數。
            image_path (str, optional): 圖片的路徑，如果有的話。
            user (str): 使用者身份標識。
        """
        partial_message = ""
        try:
            # 處理圖片訊息以及文字內容
            if image_path:
                message_content = [{"type": "image_url", "image_url": f"data:image/jpeg;base64,{self._get_base64_from_image(path)}"} for path in image_path] + [{"type": "text", "text": question}]
                self.messages.append({"role": "user", "content": message_content})
                url = self.model_config["endpoint"] + "openai/deployments/"+self.model_config["deployment"]+"/extensions"
                response = litellm.completion(
                        model="azure/gpt4-v",
                        api_key=self.model_config["key"],
                        api_version=self.model_config["api-version"],
                        messages=self.messages,
                        max_tokens=max_tokens,
                        base_url=url,
                        enhancements={"ocr": {"enabled": True}, "grounding": {"enabled": True}},
                        dataSources=[
                            {
                                "type": "AzureComputerVision",
                                "parameters": {
                                    "endpoint": self.model_config["cv_endpoint"],
                                    "key": self.model_config["cv_key"],
                                },
                            }
                        ],
                        user=user
                )
                for item in response['choices'][0]['message']['content']:
                    partial_message += item
                    yield partial_message
                self.messages.append({"role": "assistant", "content": partial_message})
        except:
            e = f"error occurred: {traceback.format_exc()}"
            for i in e:
                partial_message += i
                yield partial_message

    def get_image(self, prompt, Image_size, Image_style, Image_Quality, user):
        """根據提示生成圖像。

        Args:
            prompt (str): 圖像生成的提示語。
            Image_size (str): 圖像尺寸。
            Image_style (str): 圖像風格。
            Image_Quality (str): 圖像質量。
            user (str): 使用者身份。
        
        Returns:
            tuple: 包含修正後的提示和生成的圖像對象。
        """
        try:
            response = litellm.image_generation(  # 調用圖像生成模型
                model="azure/dall-e-3",
                prompt=prompt,
                api_key=self.model_config["key"],
                api_base=self.model_config["endpoint"],
                api_version=self.model_config["api-version"],
                size=Image_size,
                quality=Image_Quality,
                style=Image_style,
                n=1,
                user=user
            )
            img_response = requests.get(response.data[0]["url"])  # 從返回的URL獲取圖像數據
            img = Image.open(io.BytesIO(img_response.content))  # 打開並轉換圖像數據為PIL圖像
            return response.data[0]["revised_prompt"], img  # 返回修正後的提示和圖像對象
        except Exception as e:
            return e, None  # 异常處理，返回錯誤信息和None作為圖像對象
        
    def create_assistant(self, use_model="GPT-3.5 Turbo"):
        """創建並初始化一個AI助手。

        Args:
            use_model (str, optional): 使用的模型配置名稱。預設為 "GPT-3.5 Turbo"。
        
        Returns:
            tuple: 包含客戶端、助手和線程的對象。
        """
        client = AzureOpenAI(  # 初始化客戶端
            api_key=self.model_config[use_model]["key"],
            api_version=self.model_config[use_model]["api-version"],
            azure_endpoint=self.model_config[use_model]["endpoint"]
        )
        assistant = client.beta.assistants.create(  # 創建AI助手
            name="code interpreter",
            tools=[{"type": "code_interpreter"}],
            model=self.model_config[use_model]["deployment"]
        )
        thread = client.beta.threads.create()  # 創建處理線程
        return client, assistant, thread  # 返回組成元素
    
    def upload_file(self, file):
        """上傳文件至AI助手。

        Args:
            file (str): 文件的路徑。
        
        Returns:
            list: 包含上傳文件的ID。
        """
        file = self.client.files.create(  # 上傳文件
            file=open(file, "rb"),
            purpose='assistants'
        )
        return [file.id]  # 返回文件的ID列表
    
    def get_response(self, question, max_tokens, user, image_path=None, system_message=""):
        """根據提問得到GPT的回答。

        Args:
            question (str): 用戶的提問。
            max_tokens (int): 最大 token 數。
            user (str): 使用者身份標識。
            image_path (str, optional): 圖片的路徑，如果有的話。
            system_message (str): 需要傳遞給模型的系統訊息。
        """
        if system_message:
            if self.messages == []:
                self.messages = [{"role": "system", "content": system_message}]
            else:
                self.messages[0] = {"role": "system", "content": system_message}
        if self.model_config["model_name"] == "GPT-4 Vision":
            return self._handle_vision_model(question, max_tokens, image_path, user)
        else:
            return self._handle_default_model(question, max_tokens, image_path, user)
        
    def _get_base64_from_image(self, image_path):
        """將圖片轉換為base64編碼。

        Args:
            image_path (str): 圖片的檔案路徑。

        Returns:
            str: 圖片的base64編碼字符串。
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def assistant_stream_output(self, prompt, file, use_model, sys_message):
        """從助理進行串流輸出。

        Args:
            prompt (str): 輸入提示給助理。
            file (list): 包含文件ID的列表。
            use_model (str): 使用的模型。
            sys_message (str): 系統訊息。
        
        Yields:
            tuple: 包含輸出文本、文件ID、文件類型和文件路徑的元組。
        """
        try:
            # 若模型變化，則重新創建助理
            if use_model != self.use_model:
                self.create_assistant(use_model)

            # 創建消息和運行流
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=prompt,
                file_ids=file
            )
            stream = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=sys_message,
                stream=True,
            )

            # 處理事件流
            file_type = None
            file_path = None
            file_id = None
            for event in stream:
                if event.event == "thread.run.step.created":
                    details = event.data.step_details
                    if details.type == "tool_calls":
                        # print("Generating code to interpret:\n\n```py")
                        yield "Generating code to interpret:\n\n```py", file_id, file_type, file_path
                elif event.event == "thread.message.created":
                    # print("\nResponse:\n")
                    yield "\nResponse:\n", file_id, file_type, file_path
                elif event.event == "thread.message.delta":
                    if event.data.delta.content[0].type == 'text':
                        if event.data.delta.content[0].text.value:
                            # print(event.data.delta.content[0].text.value, end="", flush=True)
                            yield event.data.delta.content[0].text.value, file_id, file_type, file_path
                        elif event.data.delta.content[0].text.annotations:
                            # print(event.data.delta.content[0].text.annotations[0].file_path.file_id, event.data.delta.content[0].text.annotations[0].type)
                            yield None, event.data.delta.content[0].text.annotations[0].file_path.file_id, event.data.delta.content[0].text.annotations[0].type, file_path
                    elif event.data.delta.content[0].type == 'image_file':
                        # print("image", event.data.delta.content[0].image_file.file_id)
                        file_type = "image"
                        yield None, event.data.delta.content[0].image_file.file_id, file_type, file_path
                elif event.event == "thread.run.step.completed":
                    details = event.data.step_details
                    if details.type == "tool_calls":
                        for tool in details.tool_calls:
                            if tool.type == "code_interpreter":
                                # print("\n```\nExecuting code...")
                                yield "\n```\nExecuting code...", file_id, file_type, file_path
                elif event.event == "thread.run.step.delta":
                    details = event.data.delta.step_details
                    if details is not None and details.type == "tool_calls":
                        for tool in details.tool_calls or []:
                            if tool.type == "code_interpreter" and tool.code_interpreter and tool.code_interpreter.input:
                                # print(tool.code_interpreter.input, end="", flush=True)
                                yield tool.code_interpreter.input, file_id, file_type, file_path
                            elif tool.type == "code_interpreter" and tool.code_interpreter and tool.code_interpreter.outputs and tool.code_interpreter.outputs[0].type == "image":
                                # print(tool.code_interpreter.outputs[0].type, tool.code_interpreter.outputs[0].image.file_id)
                                yield None, tool.code_interpreter.outputs[0].image.file_id, tool.code_interpreter.outputs[0].type, file_path
        except Exception as e:
            output = f"An error occurred: {traceback.format_exc()}"
            yield output, None, None, None


    def show_output(self, prompt):
        """展示助理的輸出結果。

        Args:
            prompt (str): 輸入提示。
        
        Yields:
            tuple: 包含輸出文本和保存文件名的元組。
        """
        output = ""
        save_file_name = None
        for text_output, file_id, file_type, file_path in self.assistant_stream_output(prompt):
            if text_output:
                output += text_output
                yield output, save_file_name
            
            if file_id and file_type:
                # 處理從助理回傳的文件內容
                content = self.client.files.content(file_id)
                if file_type == "image":
                    save_file_name = file_id + ".png"
                else:
                    save_file_name = file_id + "." + file_type
                with open(save_file_name, "wb") as file:
                    file.write(content.read())
                yield None, save_file_name