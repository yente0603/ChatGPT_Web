import json
import time
import os
import gradio as gr
from ChatGPT_Web.call_gpt import ChatGPT

class User:
    def __init__(self, name, ip):
        """初始化用戶類，設定用戶基本信息。

        Args:
            name (str): 用戶名稱。
            ip (str): 用戶IP地址。
        """
        self.ip = ip
        self.username = name
        self.chatgpt = None
        self.chat_history = None
        self.model_deployment_list = None
        self.system_message_dict = None
        self.system_message_list = None
        self.system_message = None
        self.max_tokens = None
        self.download_path = None
        self.dalle_index = 0

class WebBot:
    def __init__(self, config_path='model_config.json', web_name='Nick GPT', web_server=None):
        """初始化WebBot的配置並加載模型。

        Args:
            config_path (str): 模型配置檔案的路徑。
            web_name (str): 網站的名稱。
            web_server (Optional[bool]): 是否運行在web服務器模式。
        """
        self.config_path = config_path
        self.web_name = web_name
        self.web_server = web_server
        self.init_setting()

    def init_setting(self):
        """初始化設定，加載用戶和系統訊息配置文件。"""
        self.user = {}

        # 讀取模型列表配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.model_list = json.load(f)

        # 讀取用戶配置文件
        with open('user_config.json', 'r', encoding='utf-8') as f:
            self.user_config = json.load(f)
            self.user_list = self.user_config[0]
            self.system_message = self.user_config[1]['nick']

        # 建立系統訊息列表
        self.system_message_list = [(key, value) for key, value in self.system_message.items()]

        # 初始化系統訊息
        self.init_system = {
            "role": "system",
            "content": (self.system_message["default"])
        }

        # 初始化助理系統訊息
        self.init_assistants_system = {
            "role": "system",
            "content": (self.system_message["Assistants"])
        }
        
        # 根據模型配置創建ChatGPT實例
        self.chatgpt = {
            config["model_name"]: ChatGPT(config, self.init_system)
            for config in self.model_list
        }
        
        # 初始化聊天歷史記錄
        self.chat_history = {
            config["model_name"]: [] for config in self.model_list
        }
        
        # 創建模型部署列表
        self.model_deployment_list = [
            model["model_name"] for model in self.model_list
        ]

    def new_user_setting(self, username, ip):
        """為新用戶建立配置並儲存到用戶字典中。

        Args:
            username (str): 用戶名。
            ip (str): 用戶的 IP 地址。

        Returns:
            User: 初始化完成的用戶對象。
        """
        new_user = User(username, ip)
        # 讀取用戶專屬的系統消息配置
        with open('user_config.json', 'r', encoding='utf-8') as f:
            new_user.system_message_dict = json.load(f)[1][username]
        new_user.system_message_list = [(key, value) for key, value in new_user.system_message_dict.items()]
        
        # 設定初始系統消息
        init_system = {
            "role": "system",
            "content": new_user.system_message_dict["default"]
        }
        new_user.system_message = new_user.system_message_dict["default"]
        
        # 為新用戶創建每個模型的 ChatGPT 實例
        new_user.chatgpt = {
            config["model_name"]: ChatGPT(config, init_system)
            for config in self.model_list
        }
        
        # 初始化新用戶的聊天歷史
        new_user.chat_history = {
            config["model_name"]: [] for config in self.model_list
        }
        
        # 創建新用戶的模型部署列表
        new_user.model_deployment_list = [
            model["model_name"] for model in self.model_list
        ]
        self.user[username] = new_user
        return new_user
    
    def auth_user(self, username, password):
        if username in self.user_list.keys():
            if self.user_list[username] == password:
                return True  
        return False
    
    def dalle(self, prompt, Image_size, Image_style, Image_Quality, image_1, image_2, image_3, image_4, image_5
              , image_6, image_7, image_8, image_9, image_10, request: gr.Request):
        response, image_url = self.user[request.username].chatgpt["Dall-E-3"].get_image(prompt, Image_size, Image_style, Image_Quality, request.username)
        image_list = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10]
        image_list[self.user[request.username].dalle_index] = gr.Image(value=image_url, label=prompt, height=512, show_label=True, visible=True)
        if self.user[request.username].dalle_index+1 == len(image_list):
            self.user[request.username].dalle_index = 0
        else:
            self.user[request.username].dalle_index += 1
        return [response]+image_list



    def delete_system_message(self, system_message, sys_message_select, request: gr.Request):
        """從用戶配置中刪除特定的系統消息。

        Args:
            system_message (str): 要刪除的系統消息。
            sys_message_select (str): 選擇的系統消息標籤。
            request (gr.Request): Gradio的請求對象，包含用戶信息。

        Returns:
            tuple: 更新後的系統消息 widgets。
        """
        del self.user_config[1][request.username][sys_message_select]
        # 更新設定文件
        with open('user_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.user_config, f, ensure_ascii=False, indent=4)
        
        gr.Info("Delete")
        
        self.user[request.username].system_message_dict = self.user_config[1][request.username]
        self.user[request.username].system_message_list = [(key, value) for key, value in self.user[request.username].system_message_dict.items()]
        
        # 創建更新後的下拉選擇器
        sys_message_select = gr.Dropdown(
            label="Choose system message", 
            choices=self.user[request.username].system_message_list,
            allow_custom_value=True,
            value="default",
        )
        # 更新文本框內容
        system_message = gr.Textbox(
            label="System Message", 
            placeholder="System message...", 
            lines=8, 
            max_lines=8, 
            value=self.user[request.username].system_message_dict["default"]
        )
        return system_message, sys_message_select, sys_message_select, sys_message_select, sys_message_select

    def save_system_message(self, system_message, system_message_name, request: gr.Request):
        """保存或更新用戶定義的系統消息。
        Args:
            system_message (str): 要保存的系統消息文本。
            system_message_name (str): 系統消息的命名。
            request (gr.Request): Gradio的請求對象，包含用戶信息。

        Returns:
            tuple: 更新後的系統消息下拉選項控件。
        """
        # 更新用戶配置並保存到檔案
        self.user_config[1][request.username][system_message_name] = system_message
        self.user[request.username].system_message_dict[system_message_name] = system_message
        self.user[request.username].system_message = system_message
        self.user[request.username].system_message_list = [(key, value) for key, value in self.user[request.username].system_message_dict.items()]
        
        # 創建更新後的下拉選擇器
        sys_message_select = gr.Dropdown(
            label="Choose system message", 
            choices=self.user[request.username].system_message_list,
            allow_custom_value=True,
            value=system_message_name,
        )
        
        gr.Info("Saved")
        
        # 將更新寫回配置文件
        with open('user_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.user_config, f, ensure_ascii=False, indent=4)

        return sys_message_select, sys_message_select, sys_message_select, sys_message_select
    
    def get_request_ip(self, sys_message_select, request: gr.Request):
        """處理來自Gradio前端的請求，獲取請求用戶的IP並初始化設置。

        Args:
            sys_message_select (gr.Dropdown): 系統消息下拉選項控件。
            request (gr.Request): Gradio的請求對象。

        Returns:
            tuple: 更新後的系統消息下拉選項控件。
        """
        # 創建新用戶並初始化設置
        new_user = self.new_user_setting(request.username, request.client.host)
        
        # 更新系統消息下拉選項控件
        sys_message_select = gr.Dropdown(
            label="Choose system message", 
            choices=new_user.system_message_list,
            allow_custom_value=True,
            value=0,
        )
        
        gr.Info(f"Welcome {request.username}")
        
        return sys_message_select, sys_message_select, sys_message_select, sys_message_select
    
    def update_max_tokens(self, number, request: gr.Request):
        """更新用戶的最大tokens設定。

        Args:
            number (int): 新的最大token數量。
            request (gr.Request): Gradio的請求對象，包含用戶信息。

        Returns:
            int: 更新後的token數量。
        """
        # 根據請求信息更新指定用戶的max_tokens
        self.user[request.username].max_tokens = number
        return number
    
    def update_system_message(self, sys_message_select, system_message, request: gr.Request):
        """根據用戶選擇更新系統訊息。

        Args:
            sys_message_select (str): 選擇的系統訊息標識。
            system_message (str): 系統訊息內容。
            request (gr.Request): Gradio的請求對象，包含用戶信息。

        Returns:
            str: 根據選擇更新後的系統訊息內容。
        """
        # 檢查選擇的系統消息是否存在於用戶設定中
        if sys_message_select in self.user[request.username].system_message_dict.keys():
            # 如果存在，返回該訊息內容
            return system_message
        else:
            # 如果不在，更新用戶系統消息選擇
            self.user[request.username].system_message = sys_message_select
            return sys_message_select
        
    def reset_input(self):
        """送出訊息後清除選擇框。"""
        time.sleep(1)  # 暫停1秒以確保前端JS能正常更新，防止操作過快導致前端未能即時刷新
        return gr.update(value=None)  # 清除輸入框中的內容，讓用戶有更好的交互體驗
    
    def reset_history(self, model_select, chatbot, request: gr.Request):
        """清除指定模型的聊天歷史記錄。

        Args:
            model_select (str): 選擇的模型名稱。
            chatbot (list): 聊天界面組件列表，用於顯示清除後的結果。
            request (gr.Request): 包含用戶信息的請求對象。

        Returns:
            list: 更新後的聊天組件列表。
        """
        if model_select == "Assistants":  # 特定模型可能需要重置較多資源
            # 重新創建相關助理實例，并重置相關數據
            self.user[request.username].client, self.user[request.username].assistant, self.user[request.username].thread = self.user[request.username].create_assistant()
        else:
            # 對其他模型進行聊天記錄的清空
            self.user[request.username].chatgpt[model_select].messages = []
            self.user[request.username].chat_history[model_select] = []
        chatbot = []  # 清空聊天機器人顯示組件
        return chatbot  # 返回更新後的聊天機器人組件
    
    def download_file(self, request: gr.Request):
        """提供文件下載的按鈕與動作。

        Args:
            request (gr.Request): 包含用戶信息的請求對象。

        Returns:
            list: 包含按鈕和下載按鈕的列表，方便用戶進行操作。
        """
        return [
            gr.Button("Click to get file", variant="primary", interactive=True),  # 提供主動獲取文件的按鈕
            gr.DownloadButton("Download file", variant="primary", interactive=False)  # 提供文件下載按鈕
        ]
    
    def get_file(self, request: gr.Request):
        """處理文件獲取的請求。

        Args:
            request (gr.Request): 包含用戶信息的請求對象。

        Returns:
            list: 更新後的按鈕列表，用於顯示文件獲取與下載的狀態。
        """
        # 檢查用戶是否有待下載的文件路徑
        if self.user[request.username].download_path:
            path = self.user[request.username].download_path
            self.user[request.username].download_path = None  # 清除已記錄的下載路徑以防重複下載
            return [
                gr.Button("Click to get file", variant="primary", interactive=False),  # 更新按鈕狀態為不可交互
                gr.DownloadButton("Download file", value=path, variant="primary", interactive=True)  # 更新下載按鈕，提供文件路徑以供下載
            ]
        else:
            # 若無可下載文件，顯示警告並返回按鈕
            gr.Warning("No files were found to be downloaded")
            return [
                gr.Button("Click to get file", variant="primary", interactive=True),  # 重新激活獲取文件按鈕
                gr.DownloadButton("Download file", variant="primary", interactive=False)  # 保持下載按鈕為不可交互狀態
            ]
        
    def assistant_echo(self, message, history, model, use_model, sys_message, request: gr.Request):
        """處理和回應用戶的互動，包括檔案的處理和文字的反饋。

        Args:
            message (dict): 包含文本和文件的信息。
            history (list): 聊天歷史記錄。
            model (str): 使用的模型名稱。
            use_model (str): 啟用模型進行處理的選項。
            sys_message (str): 系統訊息。
            request (gr.Request): 包含用戶信息的請求對象。

        Returns:
            list: 更新後的聊天歷史。
        """
        history = self.user[request.username].chat_history[model]
        question = message['text']
        if len(message['files']) > 0:
            file_path = message['files'][0]
            file = self.user[request.username].chatgpt[model].upload_file(file_path)
        else:
            file = []
        response = ""
        output_file = True  # 標記是否需要處理文件輸出
        for text_output, file_id, file_type, file_path in self.user[request.username].chatgpt[model].assistant_stream_output(question, file, use_model, sys_message):
            if text_output:
                response += text_output
                yield response
            if file_id and file_type and output_file:
                content = self.user[request.username].chatgpt[model].client.files.content(file_id)
                if file_type == "image":
                    save_file_name = os.path.join(os.path.dirname(__file__), 'image', file_id + '.png')
                    image = content.write_to_file(save_file_name)
                else:
                    output_file_object = self.user[request.username].chatgpt[model].client.files.retrieve(file_id)
                    save_file_name = os.path.join(os.path.dirname(__file__), 'image', output_file_object.filename.split("/")[-1])
                    with open(save_file_name, "wb") as file:
                        file.write(self.user[request.username].chatgpt[model].client.files.content(file_id).read())
                self.user[request.username].download_path = save_file_name
                if response.endswith("```"):
                    response += '\n\n'
                else:
                    response += '\n```\n\n'
                output_file = False
                yield response
        history.append((message, response))
        self.user[request.username].chat_history[model] = history
        return history
    
    def slow_echo(self, message, history, model, max_tokens, system_message, request: gr.Request):
        """處理接收到的訊息並透過GPT模型生成回答，然後返回一個生成回應的生成器。

        Args:
            message (dict): 包含文本和可能包含文件的訊息。
            history (list): 當前對話的歷史列表。
            model (str): 使用的模型名稱。
            max_tokens (int): 生成回應的最大token數。
            system_message (str): 系統訊息。
            request (gr.Request): 包含用戶信息的請求對象。

        Yields:
            str: 生成的單次回應。
        """
        history = self.user[request.username].chat_history[model]
        question = message['text']
        image = message.get('files', None)  # 檢查是否有文件附帶
        responses = self.user[request.username].chatgpt[model].get_response(question, max_tokens, request.username, image, system_message=system_message)
        response = ""
        for response_part in responses:
            response += response_part
            yield response_part
        history.append((message, response))
        self.user[request.username].chat_history[model] = history
        return history
    
    def run_web(self):
        """啟動Gradio網頁介面。"""
        model_select = [gr.Dropdown(
                            choices=self.model_deployment_list, 
                            value=deplotment
                        ) for deplotment in self.model_deployment_list]

        bot_list = [gr.Chatbot(
                            render=False, 
                            avatar_images=[None, "IMG_9954.jpg"]
                        ) for _ in self.model_deployment_list]

        
        with gr.Blocks(fill_height=True) as demo:
            gr.HTML(f"<h1 align='center'>{self.web_name}</h1>")
            with gr.Tab(self.model_list[0]["model_name"]): #gpt35
                with gr.Row(0, equal_height=True):
                    model_info = self.model_list[0]["model_info"]
                    deployment_info = self.model_list[0]["deployment_info"]
                    gr.HTML(f"<p>{model_info}</p>")
                    gr.HTML(f"<p>{deployment_info}</p>")

                    with gr.Column(scale=2):
                        system_message_box_1 = gr.Textbox(
                            placeholder="Add new system message name",
                            label="System message name"
                            )
                        with gr.Row(1):
                            number = gr.Number(
                                label="Max tokens", 
                                minimum=1, 
                                maximum=16385, 
                                step=1,
                                value=300
                            )
                            
                            sys_message_select_1 = gr.Dropdown(
                                label="Choose system message", 
                                choices=self.system_message_list,
                                allow_custom_value=True,
                                value=0,
                            )
                    with gr.Column(scale=2):
                        with gr.Row(1):
                            delete_btn_1 = gr.Button(
                                "Delete system message",
                                variant="stop"
                            )
                            save_btn_1 = gr.Button(
                                "Save system message",
                                variant="primary"
                            )
                            
                        system_message_1 = gr.Textbox(
                            label="System Message", 
                            placeholder="System message...", 
                            lines=6, 
                            max_lines=6, 
                            value=self.init_system["content"],
                            scale=2
                        )
                x=gr.ChatInterface(fn=self.slow_echo,
                            chatbot=bot_list[0],
                            additional_inputs=[model_select[0], number, system_message_1],
                            concurrency_limit=10,
                            multimodal=True).queue()
                x.textbox.autoscroll = False
                
            with gr.Tab(self.model_list[1]["model_name"]): #gpt4
                with gr.Row(0, equal_height=True):
                    model_info = self.model_list[1]["model_info"]
                    deployment_info = self.model_list[1]["deployment_info"]
                    gr.HTML(f"<p>{model_info}</p>")
                    gr.HTML(f"<p>{deployment_info}</p>")

                    with gr.Column(scale=2):
                        system_message_box_2 = gr.Textbox(
                            placeholder="Add new system message name",
                            label="System message name"
                            )
                        with gr.Row(1):
                            number = gr.Number(
                                label="Max tokens \n(Image input requires over 800)", 
                                minimum=1, 
                                maximum=4096, 
                                step=1,
                                value=300
                            )
                            
                            sys_message_select_2 = gr.Dropdown(
                                label="Choose system message", 
                                choices=self.system_message_list,
                                allow_custom_value=True,
                                value=0,
                            )
                    with gr.Column(scale=2):
                        with gr.Row(1):
                            delete_btn_2 = gr.Button(
                                "Delete system message",
                                variant="stop"
                            )
                            save_btn_2 = gr.Button(
                                "Save system message",
                                variant="primary"
                            )
                            
                        system_message_2 = gr.Textbox(
                            label="System Message", 
                            placeholder="System message...", 
                            lines=6, 
                            max_lines=6, 
                            value=self.init_system["content"],
                            scale=2
                        )
                gr.ChatInterface(fn=self.slow_echo,
                            chatbot=bot_list[1],
                            additional_inputs=[model_select[1], number, system_message_2],
                            concurrency_limit=10,
                            multimodal=True).queue()
                
            with gr.Tab(self.model_list[2]["model_name"]): #gpt4-v
                with gr.Row(0, equal_height=True):
                    model_info = self.model_list[2]["model_info"]
                    deployment_info = self.model_list[2]["deployment_info"]
                    gr.HTML(f"<p>{model_info}</p>")
                    gr.HTML(f"<p>{deployment_info}</p>")

                    with gr.Column(scale=2):
                        system_message_box_3 = gr.Textbox(
                            placeholder="Add new system message name",
                            label="System message name"
                            )
                        with gr.Row(1):
                            number = gr.Number(
                                label="Max tokens", 
                                minimum=1, 
                                maximum=4096, 
                                step=1,
                                value=800
                            )
                            
                            sys_message_select_3 = gr.Dropdown(
                                label="Choose system message", 
                                choices=self.system_message_list,
                                allow_custom_value=True,
                                value=0,
                            )
                    with gr.Column(scale=2):
                        with gr.Row(1):
                            delete_btn_3 = gr.Button(
                                "Delete system message",
                                variant="stop"
                            )
                            save_btn_3 = gr.Button(
                                "Save system message",
                                variant="primary"
                            )
                            
                        system_message_3 = gr.Textbox(
                            label="System Message", 
                            placeholder="System message...", 
                            lines=6, 
                            max_lines=6, 
                            value=self.init_system["content"],
                            scale=2
                        )
                gr.ChatInterface(fn=self.slow_echo,
                            chatbot=bot_list[2],
                            additional_inputs=[model_select[2], number, system_message_3],
                            concurrency_limit=10,
                            multimodal=True).queue()
                
            with gr.Tab(self.model_list[3]["model_name"]): #dall-e-3
                with gr.Row(0, equal_height=True):
                    model_info = self.model_list[3]["model_info"]
                    deployment_info = self.model_list[3]["deployment_info"]
                    gr.HTML(f"<p>{model_info}</p>")
                    gr.HTML(f"<p>{deployment_info}</p>")
                    with gr.Column(scale=2):
                        Image_size = gr.Radio(["1024x1024", "1024x1792"], value="1024x1024", label="Image size")
                        Image_style = gr.Radio(["vivid", "natural"], value="vivid", label="Image style")
                        Image_Quality = gr.Radio(["standard", "hd"], value="hd", label="Image Quality")
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to create.", autofocus=True, show_label=True, scale=2)
                            send_request = gr.Button("Generate", variant="primary", scale=0)
                        revised_prompt = gr.Textbox(label="Revised prompt", show_label=True, show_copy_button=True, scale=4)
                with gr.Row(0, equal_height=True):
                    image_list = []
                    [image_list.append(gr.Image(visible=False)) for _ in range(10)]
                    send_request.click(self.dalle, [prompt, Image_size, Image_style, Image_Quality]+image_list, [revised_prompt]+image_list)

            with gr.Tab(self.model_list[4]["model_name"]): #assistant
                with gr.Row(equal_height=True):
                    model_info = self.model_list[4]["model_info"]
                    deployment_info = self.model_list[4]["deployment_info"]
                    gr.HTML(f"<p>{model_info}</p>")
                    gr.HTML(f"<p>{deployment_info}</p>")
                        
                    with gr.Column(scale=2):
                        with gr.Row():
                            choose_model = gr.Radio(self.model_deployment_list[:2], value=self.model_deployment_list[0], show_label=False)
                            system_message_box_4 = gr.Textbox(
                            label="System message name"
                            )
                        with gr.Row():
                            delete_btn_4 = gr.Button(
                                "Delete",
                                variant="stop",
                                # size='sm'
                            )
                            save_btn_4 = gr.Button(
                                "Save",
                                variant="primary",
                                # size='sm'
                            )
                        with gr.Row():
                            get_file_btn = gr.Button("Get file", variant="primary")
                            download_btn = gr.DownloadButton("Download file", variant="primary", interactive=False)
                    with gr.Column(scale=2):
                        sys_message_select_4 = gr.Dropdown(
                                label="Choose system message", 
                                choices=self.system_message_list,
                                allow_custom_value=True
                            )
                        system_message_4 = gr.Textbox(
                                label="System Message", 
                                placeholder="System message...", 
                                lines=6, 
                                max_lines=6, 
                                value=self.init_assistants_system["content"],
                                scale=2
                            )
                gr.ChatInterface(fn=self.assistant_echo,
                            chatbot=bot_list[4],
                            additional_inputs=[model_select[4], choose_model, system_message_4],
                            concurrency_limit=5,
                            multimodal=True).queue()
                get_file_btn.click(self.get_file, None, [get_file_btn, download_btn])
                download_btn.click(self.download_file, None, [get_file_btn, download_btn])
            with gr.Tab("setting"):
                logout_button = gr.Button("Logout", link="/logout")
                refresh_btn = gr.Button(value="Refresh the page")
            refresh_btn.click(self.init_setting, js="window.location.reload()")
            demo.load(self.get_request_ip, [sys_message_select_1], [sys_message_select_1, sys_message_select_2, sys_message_select_3, sys_message_select_4])

            save_btn_1.click(self.save_system_message, [system_message_1, system_message_box_1], [sys_message_select_1, sys_message_select_2, sys_message_select_3])
            delete_btn_1.click(self.delete_system_message, [system_message_1, sys_message_select_1], [system_message_1, sys_message_select_1, sys_message_select_2, sys_message_select_3])
            sys_message_select_1.change(
                fn=self.update_system_message, 
                inputs=[sys_message_select_1, system_message_1], 
                outputs=[system_message_1])

            save_btn_2.click(self.save_system_message, [system_message_2, system_message_box_2], [sys_message_select_1, sys_message_select_2, sys_message_select_3])
            delete_btn_2.click(self.delete_system_message, [system_message_2, sys_message_select_2], [system_message_2, sys_message_select_1, sys_message_select_2, sys_message_select_3])
            sys_message_select_2.change(
                fn=self.update_system_message, 
                inputs=[sys_message_select_2, system_message_2], 
                outputs=[system_message_2])
            
            save_btn_3.click(self.save_system_message, [system_message_3, system_message_box_3], [sys_message_select_1, sys_message_select_2, sys_message_select_3])
            delete_btn_3.click(self.delete_system_message, [system_message_3, sys_message_select_3], [system_message_3, sys_message_select_1, sys_message_select_2, sys_message_select_3])
            sys_message_select_3.change(
                fn=self.update_system_message, 
                inputs=[sys_message_select_3, system_message_3], 
                outputs=[system_message_3])
                        
            save_btn_4.click(self.save_system_message, [system_message_4, system_message_box_4], [sys_message_select_1, sys_message_select_2, sys_message_select_3, sys_message_select_4])
            delete_btn_4.click(self.delete_system_message, [system_message_4, sys_message_select_4], [system_message_4, sys_message_select_1, sys_message_select_2, sys_message_select_3, sys_message_select_4])
            sys_message_select_4.change(
                fn=self.update_system_message, 
                inputs=[sys_message_select_4, system_message_4], 
                outputs=[system_message_4])

            if self.web_server:
                demo.launch(inbrowser=True, server_name="0.0.0.0", auth=self.auth_user)
            else:
                demo.launch(inbrowser=True, auth=self.auth_user)
            

if __name__ == '__main__':
    web = WebBot(web_server=True)
    web.run_web()